import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))

from tracking import (  # noqa: E402
    MODEL_NAME,
    PACKAGES,
    REDACTED,
    REQUIRED_TAGS,
    collect_run_metadata,
    compare_runs,
    find_best_run,
    load_champion,
    log_training_result,
    register_and_promote,
    reproduce_command,
    start_run,
)

# Предупреждения, которые выдаёт сам MLflow 3.18 с SQLAlchemy 2.1 и pandas 3.1: к коду решения не относятся
pytestmark = [
    pytest.mark.filterwarnings("ignore:The ``noload`` loader strategy is deprecated"),
    pytest.mark.filterwarnings("ignore:Operation with range is deprecated"),
    pytest.mark.filterwarnings("ignore:.*Any type hint is inferred as AnyType"),
]

COMMAND = "uv run python train.py --alpha 1.0"
METRIC = "valid_mae"
MEAN = "valid_mae_mean"


# ---------- фикстуры и помощники ----------


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """Локальное хранилище MLflow на весь модуль: база SQLite и артефакты во временном каталоге.

    Артефакты по умолчанию пишутся в ./mlruns, поэтому на время тестов текущим становится временный каталог.
    """
    root = tmp_path_factory.mktemp("mlflow")
    previous = Path.cwd()
    os.chdir(root)
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    yield root
    os.chdir(previous)


@pytest.fixture(autouse=True)
def no_active_run(store):
    """После каждого теста не остаётся активного запуска."""
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()


@pytest.fixture
def names(request):
    """Имена эксперимента и модели, уникальные для теста: хранилище общее на модуль."""
    base = re.sub(r"[^A-Za-z0-9_]+", "-", request.node.name).strip("-")
    return f"exp-{base}", f"model-{base}"


@pytest.fixture
def data_dir(tmp_path):
    path = tmp_path / "data"
    (path / "raw" / "date=2024-06-02").mkdir(parents=True)
    (path / "raw" / "date=2024-06-01").mkdir(parents=True)
    (path / "raw" / "date=2024-06-02" / "sales.csv").write_text("store_id,units\nS1,7\n")
    (path / "raw" / "date=2024-06-01" / "sales.csv").write_text("store_id,units\nS1,5\n")
    (path / "calendar.csv").write_text("date,holiday\n2024-06-12,1\n")
    return path


@pytest.fixture
def repo(tmp_path):
    """Репозиторий Git с одним коммитом и чистым рабочим деревом."""
    path = tmp_path / "repo"
    path.mkdir()
    (path / "train.py").write_text("print('train')\n")
    (path / ".gitignore").write_text("mlruns/\n*.db\n")
    git(path, "init", "-q", "-b", "main")
    git(path, "add", "-A")
    identity = ["-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"]
    git(path, *identity, "commit", "-q", "-m", "init")
    return path


def git(path: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


def make_metadata(**overrides: str) -> dict[str, str]:
    metadata = {
        "git_commit": "a" * 40,
        "git_dirty": "false",
        "data_hash": "d" * 64,
        "python": "3.12.0",
        "seed": "0",
        "command": COMMAND,
        "version.numpy": "2.0.0",
    }
    metadata.update(overrides)
    return metadata


def make_data(n: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "price": rng.uniform(50, 500, n),
            "promo": rng.integers(0, 2, n).astype(float),
            "dow": rng.integers(0, 7, n).astype(float),
        }
    )
    y = 20 - 0.03 * X["price"] + 4 * X["promo"] + rng.normal(0, 1, n)
    return X, y


def fit_model(alpha: float = 1.0) -> tuple[Pipeline, pd.DataFrame]:
    X, y = make_data()
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))]).fit(X, y), X


def log_run(experiment, run_name, value, *, metric=METRIC, alpha=1.0, metadata=None, tags=None):
    """Запуск с моделью: метрика по трём фолдам со средним value."""
    model, X = fit_model(alpha)
    with start_run(experiment, run_name, metadata or make_metadata(), tags) as run:
        log_training_result(model, X, {"model": {"alpha": alpha}}, {metric: [value - 0.1, value, value + 0.1]})
    return run.info.run_id, model, X


def light_run(experiment, run_name, metrics=None, params=None, metadata=None, tags=None, fail=False) -> str:
    """Запуск без модели: только теги, параметры и метрики. С fail=True завершается исключением."""
    try:
        with start_run(experiment, run_name, metadata or make_metadata(), tags) as run:
            mlflow.log_params(params or {})
            mlflow.log_metrics(metrics or {})
            if fail:
                raise RuntimeError("обучение упало")
    except RuntimeError:
        if not fail:
            raise
    return run.info.run_id


def versions(name: str) -> list[int]:
    return sorted(int(found.version) for found in MlflowClient().search_model_versions(f"name = '{name}'"))


def alias_version(name: str, alias: str = "champion") -> int:
    return int(MlflowClient().get_model_version_by_alias(name, alias).version)


# ---------- collect_run_metadata ----------


def test_metadata_keys_and_values(repo, data_dir):
    metadata = collect_run_metadata(repo, [data_dir], seed=7, command=COMMAND)
    assert set(metadata) == set(REQUIRED_TAGS) | {f"version.{name}" for name in PACKAGES}
    assert all(isinstance(value, str) for value in metadata.values())
    assert metadata["python"] == platform.python_version()
    assert metadata["seed"] == "7"
    assert metadata["command"] == COMMAND
    for name in PACKAGES:
        assert metadata[f"version.{name}"] == version(name)


def test_metadata_clean_repo(repo, data_dir):
    metadata = collect_run_metadata(repo, [data_dir], seed=0, command=COMMAND)
    assert metadata["git_commit"] == git(repo, "rev-parse", "HEAD")
    assert len(metadata["git_commit"]) == 40
    assert metadata["git_dirty"] == "false"


def test_metadata_modified_file_is_dirty(repo, data_dir):
    (repo / "train.py").write_text("print('другой код')\n")
    metadata = collect_run_metadata(repo, [data_dir], seed=0, command=COMMAND)
    assert metadata["git_dirty"] == "true"
    assert metadata["git_commit"] == git(repo, "rev-parse", "HEAD")


def test_metadata_untracked_file_is_dirty(repo, data_dir):
    (repo / "features.py").write_text("# новый модуль, не добавленный в Git\n")
    assert collect_run_metadata(repo, [data_dir], seed=0, command=COMMAND)["git_dirty"] == "true"


def test_metadata_ignored_file_is_not_dirty(repo, data_dir):
    (repo / "mlruns").mkdir()
    (repo / "mlruns" / "artifact.txt").write_text("артефакт")
    (repo / "mlflow.db").write_text("база")
    assert collect_run_metadata(repo, [data_dir], seed=0, command=COMMAND)["git_dirty"] == "false"


def test_metadata_outside_repository(tmp_path, data_dir):
    plain = tmp_path / "plain"
    plain.mkdir()
    metadata = collect_run_metadata(plain, [data_dir], seed=0, command=COMMAND)
    assert metadata["git_commit"] == "unknown"
    assert metadata["git_dirty"] == "true"


def test_metadata_data_hash(repo, data_dir):
    files = sorted((path for path in data_dir.rglob("*") if path.is_file()), key=str)
    assert len(files) == 3
    expected = hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()
    assert collect_run_metadata(repo, [data_dir], seed=0, command=COMMAND)["data_hash"] == expected

    raw, calendar = data_dir / "raw", data_dir / "calendar.csv"
    direct = collect_run_metadata(repo, [raw, calendar], seed=0, command=COMMAND)["data_hash"]
    swapped = collect_run_metadata(repo, [calendar, raw], seed=0, command=COMMAND)["data_hash"]
    assert direct == swapped == expected

    only_raw = collect_run_metadata(repo, [raw], seed=0, command=COMMAND)["data_hash"]
    assert only_raw != expected


def test_metadata_data_hash_changes_with_data(repo, data_dir):
    before = collect_run_metadata(repo, [data_dir], seed=0, command=COMMAND)["data_hash"]
    (data_dir / "raw" / "date=2024-06-01" / "sales.csv").write_text("store_id,units\nS1,6\n")
    after = collect_run_metadata(repo, [data_dir], seed=0, command=COMMAND)["data_hash"]
    assert before != after


def test_metadata_missing_data(repo, data_dir):
    with pytest.raises(FileNotFoundError):
        collect_run_metadata(repo, [data_dir, data_dir / "нет_такого"], seed=0, command=COMMAND)


# ---------- start_run ----------


def test_start_run_writes_tags(names):
    experiment, _ = names
    metadata = make_metadata()
    with start_run(experiment, "ridge-alpha-1", metadata, {"stage": "candidate"}) as run:
        assert mlflow.active_run().info.run_id == run.info.run_id
    assert mlflow.active_run() is None

    saved = MlflowClient().get_run(run.info.run_id)
    assert saved.info.status == "FINISHED"
    assert saved.info.run_name == "ridge-alpha-1"
    assert mlflow.get_experiment(saved.info.experiment_id).name == experiment
    for key, value in metadata.items():
        assert saved.data.tags[key] == value
    assert saved.data.tags["stage"] == "candidate"


def test_start_run_extra_tags_override_metadata(names):
    experiment, _ = names
    with start_run(experiment, "run", make_metadata(seed="1"), {"seed": "2"}) as run:
        pass
    assert MlflowClient().get_run(run.info.run_id).data.tags["seed"] == "2"


@pytest.mark.parametrize("missing", REQUIRED_TAGS)
def test_start_run_requires_metadata(names, missing):
    experiment, _ = names
    metadata = make_metadata()
    del metadata[missing]
    with pytest.raises(ValueError):
        with start_run(experiment, "run", metadata):
            pass
    assert mlflow.active_run() is None
    found = mlflow.get_experiment_by_name(experiment)
    assert found is None or mlflow.search_runs(experiment_ids=[found.experiment_id], output_format="list") == []


def test_start_run_failure_is_recorded(names):
    experiment, _ = names
    with pytest.raises(ZeroDivisionError):
        with start_run(experiment, "broken", make_metadata()) as run:
            mlflow.log_param("alpha", 1.0)
            1 / 0  # noqa: B018
    saved = MlflowClient().get_run(run.info.run_id)
    assert saved.info.status == "FAILED"
    assert saved.data.tags["git_commit"] == "a" * 40
    assert saved.data.params == {"alpha": "1.0"}
    assert mlflow.active_run() is None


# ---------- log_training_result ----------


def test_log_requires_active_run():
    model, X = fit_model()
    with pytest.raises(RuntimeError) as error:
        log_training_result(model, X, {"alpha": 1.0}, {METRIC: [1.0]})
    assert error.type is RuntimeError  # NotImplementedError — подкласс RuntimeError и не засчитывается


def test_log_params_are_flattened(names):
    experiment, _ = names
    model, X = fit_model()
    params = {"model": {"alpha": 1.0, "solver": {"name": "auto"}}, "train_days": 56, "seed": 0}
    with start_run(experiment, "run", make_metadata()) as run:
        log_training_result(model, X, params, {METRIC: [1.0]})
    expected = {"model.alpha": "1.0", "model.solver.name": "auto", "train_days": "56", "seed": "0"}
    assert MlflowClient().get_run(run.info.run_id).data.params == expected
    config = mlflow.artifacts.load_dict(f"runs:/{run.info.run_id}/config/params.json")
    assert config == {"model.alpha": 1.0, "model.solver.name": "auto", "train_days": 56, "seed": 0}


def test_log_fold_metrics(names):
    experiment, _ = names
    model, X = fit_model()
    folds = {"valid_mae": [1.5, 1.2, 1.8, 1.1], "valid_r2": [0.6, 0.7]}
    with start_run(experiment, "run", make_metadata()) as run:
        log_training_result(model, X, {"alpha": 1.0}, folds)
    client = MlflowClient()
    saved = client.get_run(run.info.run_id).data.metrics
    assert set(saved) == {"valid_mae", "valid_mae_mean", "valid_mae_std", "valid_r2", "valid_r2_mean", "valid_r2_std"}
    assert saved["valid_mae_mean"] == pytest.approx(1.4)
    assert saved["valid_mae_std"] == pytest.approx(np.std([1.5, 1.2, 1.8, 1.1]))
    assert saved["valid_r2_mean"] == pytest.approx(0.65)
    history = sorted(client.get_metric_history(run.info.run_id, "valid_mae"), key=lambda item: item.step)
    assert [(item.step, item.value) for item in history] == [(0, 1.5), (1, 1.2), (2, 1.8), (3, 1.1)]


def test_log_empty_fold_list(names):
    experiment, _ = names
    model, X = fit_model()
    with pytest.raises(ValueError):
        with start_run(experiment, "run", make_metadata()):
            log_training_result(model, X, {"alpha": 1.0}, {METRIC: []})


def test_log_model_with_signature(names):
    experiment, _ = names
    model, X = fit_model()
    with start_run(experiment, "run", make_metadata()) as run:
        model_uri = log_training_result(model, X, {"alpha": 1.0}, {METRIC: [1.0]})
    assert model_uri == f"runs:/{run.info.run_id}/{MODEL_NAME}"

    info = mlflow.models.get_model_info(model_uri)
    assert info.signature.inputs.input_names() == ["price", "promo", "dow"]
    assert info.run_id == run.info.run_id
    loaded = mlflow.pyfunc.load_model(model_uri)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
    with pytest.raises(mlflow.exceptions.MlflowException):
        loaded.predict(X.drop(columns="price"))


def test_log_environment_artifacts(names, tmp_path):
    experiment, _ = names
    model, X = fit_model()
    lock = tmp_path / "uv.lock"
    lock.write_text('version = 1\nrequires-python = ">=3.12"\n')
    with start_run(experiment, "run", make_metadata()) as run:
        log_training_result(model, X, {"alpha": 1.0}, {METRIC: [1.0]}, environment_files=[lock])

    client = MlflowClient()
    assert {item.path for item in client.list_artifacts(run.info.run_id)} >= {"config", "environment"}
    stored = {item.path for item in client.list_artifacts(run.info.run_id, "environment")}
    assert stored == {"environment/environment.json", "environment/uv.lock"}
    environment = mlflow.artifacts.load_dict(f"runs:/{run.info.run_id}/environment/environment.json")
    assert environment["python"] == platform.python_version()
    assert environment["platform"] == platform.platform()
    assert environment["packages"] == {name: version(name) for name in PACKAGES}
    copied = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="environment/uv.lock")
    assert Path(copied).read_text() == lock.read_text()


def test_secrets_are_not_logged(names, monkeypatch):
    experiment, _ = names
    secret = "hunter2-Zx9"
    monkeypatch.setenv("WAREHOUSE_PASSWORD", secret)
    monkeypatch.setenv("SHORT_TOKEN", "100")
    model, X = fit_model()
    params = {
        "model": {"alpha": 1.0},
        "n_estimators": "100",
        "db": {"host": "db.internal", "password": secret, "dsn": f"postgresql://ml:{secret}@db.internal/sales"},
        "api_token": "abc123",
    }
    tags = {"stage": "candidate", "auth_token": "abc123", "note": f"подключение с паролем {secret}"}
    with start_run(experiment, "run", make_metadata(), tags) as run:
        log_training_result(model, X, params, {METRIC: [1.0]})

    saved = MlflowClient().get_run(run.info.run_id).data
    assert saved.params == {"model.alpha": "1.0", "n_estimators": "100", "db.host": "db.internal", "db.dsn": REDACTED}
    assert "auth_token" not in saved.tags
    assert saved.tags["note"] == REDACTED
    assert saved.tags["stage"] == "candidate"
    assert saved.tags["git_commit"] == "a" * 40
    config = mlflow.artifacts.load_dict(f"runs:/{run.info.run_id}/config/params.json")
    everything = json.dumps([saved.params, saved.tags, config], ensure_ascii=False)
    assert secret not in everything
    assert "abc123" not in everything


# ---------- find_best_run ----------


def test_find_best_run_both_directions(names):
    experiment, _ = names
    low = light_run(experiment, "low", {METRIC: 1.1, "valid_r2": 0.5})
    light_run(experiment, "middle", {METRIC: 1.3, "valid_r2": 0.7})
    high = light_run(experiment, "high", {METRIC: 1.6, "valid_r2": 0.9})
    assert find_best_run(experiment, METRIC) == low
    assert find_best_run(experiment, METRIC, direction="min") == low
    assert find_best_run(experiment, METRIC, direction="max") == high
    assert find_best_run(experiment, "valid_r2", direction="max") == high
    assert find_best_run(experiment, "valid_r2", direction="min") == low


def test_find_best_run_skips_failed_and_incomplete(names):
    experiment, _ = names
    light_run(experiment, "failed", {METRIC: 0.01}, fail=True)
    light_run(experiment, "other-metric", {"valid_r2": 0.9})
    good = light_run(experiment, "good", {METRIC: 1.4})
    assert find_best_run(experiment, METRIC) == good
    assert find_best_run(experiment, "valid_mape") is None


def test_find_best_run_tie_goes_to_earlier(names):
    experiment, _ = names
    first = light_run(experiment, "first", {METRIC: 1.2})
    light_run(experiment, "second", {METRIC: 1.2})
    assert find_best_run(experiment, METRIC) == first
    assert find_best_run(experiment, METRIC, direction="max") == first


def test_find_best_run_filters(names):
    experiment, _ = names
    debug = light_run(experiment, "debug", {METRIC: 1.0}, {"model": "ridge"}, tags={"stage": "debug"})
    ridge = light_run(experiment, "ridge", {METRIC: 1.3}, {"model": "ridge"}, tags={"stage": "candidate"})
    lgbm = light_run(experiment, "lgbm", {METRIC: 1.2}, {"model": "lgbm"}, tags={"stage": "candidate"})
    dirty = make_metadata(git_dirty="true")
    light_run(experiment, "dirty", {METRIC: 1.1}, {"model": "lgbm"}, metadata=dirty, tags={"stage": "candidate"})

    assert find_best_run(experiment, METRIC, filters={"tags.stage": "candidate", "tags.git_dirty": "false"}) == lgbm
    assert find_best_run(experiment, METRIC, filters={"tags.stage": "candidate", "params.model": "ridge"}) == ridge
    assert find_best_run(experiment, METRIC, filters={"tags.version.numpy": "2.0.0", "params.model": "ridge"}) == debug
    assert find_best_run(experiment, METRIC, filters={"params.model": "catboost"}) is None


def test_find_best_run_bad_arguments(names):
    experiment, _ = names
    light_run(experiment, "run", {METRIC: 1.0})
    with pytest.raises(ValueError):
        find_best_run(experiment, METRIC, direction="up")
    with pytest.raises(ValueError):
        find_best_run(experiment, METRIC, filters={"metrics.valid_mae": "1.0"})
    with pytest.raises(ValueError):
        find_best_run(experiment, METRIC, filters={"stage": "candidate"})


def test_find_best_run_without_experiment(names):
    experiment, _ = names
    assert find_best_run(experiment, METRIC) is None


# ---------- register_and_promote ----------


def test_first_version_becomes_champion(names):
    experiment, name = names
    run_id, _, _ = log_run(experiment, "first", 1.30)
    result = register_and_promote(run_id, name, MEAN)
    assert result == {"version": 1, "promoted": True, "champion_version": 1}
    assert versions(name) == [1]
    assert alias_version(name) == 1


def test_small_improvement_keeps_champion(names):
    experiment, name = names
    first, _, _ = log_run(experiment, "first", 1.30)
    second, _, _ = log_run(experiment, "second", 1.28)
    register_and_promote(first, name, MEAN, min_improvement=0.05)
    result = register_and_promote(second, name, MEAN, min_improvement=0.05)
    assert result == {"version": 2, "promoted": False, "champion_version": 1}
    assert versions(name) == [1, 2]
    assert alias_version(name) == 1


@pytest.mark.parametrize("value", [1.30, 1.45])
def test_equal_or_worse_model_keeps_champion(names, value):
    experiment, name = names
    first, _, _ = log_run(experiment, "first", 1.30)
    second, _, _ = log_run(experiment, "second", value)
    register_and_promote(first, name, MEAN)
    result = register_and_promote(second, name, MEAN)
    assert result == {"version": 2, "promoted": False, "champion_version": 1}
    assert alias_version(name) == 1


def test_enough_improvement_moves_alias(names):
    experiment, name = names
    first, _, _ = log_run(experiment, "first", 1.30)
    second, _, _ = log_run(experiment, "second", 1.10)
    register_and_promote(first, name, MEAN, min_improvement=0.05)
    result = register_and_promote(second, name, MEAN, min_improvement=0.05)
    assert result == {"version": 2, "promoted": True, "champion_version": 2}
    assert versions(name) == [1, 2]
    assert alias_version(name) == 2


def test_promotion_with_metric_to_maximize(names):
    experiment, name = names
    first, _, _ = log_run(experiment, "first", 0.80, metric="valid_r2")
    worse, _, _ = log_run(experiment, "worse", 0.70, metric="valid_r2")
    better, _, _ = log_run(experiment, "better", 0.90, metric="valid_r2")
    register_and_promote(first, name, "valid_r2_mean", direction="max")
    assert register_and_promote(worse, name, "valid_r2_mean", direction="max")["promoted"] is False
    assert alias_version(name) == 1
    result = register_and_promote(better, name, "valid_r2_mean", direction="max", min_improvement=0.05)
    assert result == {"version": 3, "promoted": True, "champion_version": 3}


def test_repeated_registration_keeps_history(names):
    experiment, name = names
    first, _, _ = log_run(experiment, "first", 1.30)
    second, _, _ = log_run(experiment, "second", 1.10)
    register_and_promote(first, name, MEAN)
    register_and_promote(second, name, MEAN)

    again_first = register_and_promote(first, name, MEAN)
    again_second = register_and_promote(second, name, MEAN)
    assert again_first == {"version": 1, "promoted": False, "champion_version": 2}
    assert again_second == {"version": 2, "promoted": False, "champion_version": 2}
    assert versions(name) == [1, 2]
    assert alias_version(name) == 2
    client = MlflowClient()
    assert client.get_model_version(name, "1").run_id == first
    assert client.get_model_version(name, "2").run_id == second


def test_version_tags(names):
    experiment, name = names
    metadata = make_metadata(git_commit="c" * 40, data_hash="e" * 64)
    run_id, _, _ = log_run(experiment, "first", 1.30, metadata=metadata)
    register_and_promote(run_id, name, MEAN)
    found = MlflowClient().get_model_version(name, "1")
    assert found.run_id == run_id
    assert float(found.tags[MEAN]) == pytest.approx(1.30)
    assert found.tags["git_commit"] == "c" * 40
    assert found.tags["data_hash"] == "e" * 64


def test_separate_alias(names):
    experiment, name = names
    first, _, _ = log_run(experiment, "first", 1.30)
    second, _, _ = log_run(experiment, "second", 1.50)
    register_and_promote(first, name, MEAN)
    result = register_and_promote(second, name, MEAN, alias="challenger")
    assert result == {"version": 2, "promoted": True, "champion_version": 2}
    assert alias_version(name, "champion") == 1
    assert alias_version(name, "challenger") == 2


def test_dirty_run_is_not_registered(names):
    experiment, name = names
    run_id, _, _ = log_run(experiment, "dirty", 1.00, metadata=make_metadata(git_dirty="true"))
    with pytest.raises(ValueError):
        register_and_promote(run_id, name, MEAN)
    assert versions(name) == []


def test_register_rejects_bad_arguments(names):
    experiment, name = names
    run_id, _, _ = log_run(experiment, "good", 1.30)
    failed = light_run(experiment, "failed", {MEAN: 1.0}, fail=True)
    with pytest.raises(ValueError):
        register_and_promote(run_id, name, MEAN, direction="down")
    with pytest.raises(ValueError):
        register_and_promote(run_id, name, MEAN, min_improvement=-0.1)
    with pytest.raises(ValueError):
        register_and_promote(run_id, name, "valid_mape")
    with pytest.raises(ValueError):
        register_and_promote(failed, name, MEAN)
    assert versions(name) == []


# ---------- load_champion ----------


def test_champion_gives_same_predictions(names):
    experiment, name = names
    first, model_first, X = log_run(experiment, "first", 1.30, alpha=1.0)
    register_and_promote(first, name, MEAN)
    np.testing.assert_array_equal(load_champion(name).predict(X), model_first.predict(X))

    second, model_second, _ = log_run(experiment, "second", 1.10, alpha=500.0)
    assert not np.array_equal(model_first.predict(X), model_second.predict(X))
    register_and_promote(second, name, MEAN)
    np.testing.assert_array_equal(load_champion(name).predict(X), model_second.predict(X))

    weak, _, _ = log_run(experiment, "weak", 1.90, alpha=5000.0)
    register_and_promote(weak, name, MEAN)
    np.testing.assert_array_equal(load_champion(name).predict(X), model_second.predict(X))


def test_champion_is_missing(names):
    experiment, name = names
    with pytest.raises(LookupError):
        load_champion(name)
    run_id, _, _ = log_run(experiment, "first", 1.30)
    register_and_promote(run_id, name, MEAN)
    with pytest.raises(LookupError):
        load_champion(name, alias="challenger")
    assert load_champion(name, alias="champion") is not None


# ---------- reproduce_command ----------


def test_reproduce_command(names):
    experiment, _ = names
    metadata = make_metadata(git_commit="b" * 40, data_hash="f" * 64, command="uv run python train.py --seed 3")
    run_id = light_run(experiment, "run", {METRIC: 1.0}, metadata=metadata)
    lines = reproduce_command(run_id).split("\n")
    assert lines == [
        f"git checkout {'b' * 40}",
        "uv sync --locked",
        "uv run python train.py --seed 3",
        f"# data_hash: {'f' * 64}",
    ]


@pytest.mark.parametrize("broken", [{"git_dirty": "true"}, {"git_commit": "unknown"}])
def test_reproduce_command_refuses_unreproducible_run(names, broken):
    experiment, _ = names
    run_id = light_run(experiment, "run", {METRIC: 1.0}, metadata=make_metadata(**broken))
    with pytest.raises(ValueError):
        reproduce_command(run_id)


# ---------- compare_runs ----------


def test_compare_runs_table(names):
    experiment, _ = names
    middle = light_run(experiment, "ridge-1", {METRIC: 1.3, "valid_r2": 0.6}, {"model": "ridge", "alpha": 1.0})
    best = light_run(experiment, "ridge-10", {METRIC: 1.2, "valid_r2": 0.7}, {"model": "ridge", "alpha": 10.0})
    no_metric = light_run(experiment, "lgbm", {"valid_r2": 0.5}, {"model": "lgbm"})
    light_run(experiment, "failed", {METRIC: 0.1}, {"model": "ridge"}, fail=True)

    table = compare_runs(experiment, [METRIC, "valid_r2"], params=["model", "alpha"])
    assert list(table.columns) == ["run_name", "model", "alpha", METRIC, "valid_r2", "git_commit", "data_hash"]
    assert table.index.name == "run_id"
    assert list(table.index) == [best, middle, no_metric]
    assert list(table["run_name"]) == ["ridge-10", "ridge-1", "lgbm"]
    assert list(table["alpha"][:2]) == ["10.0", "1.0"]
    assert table.loc[no_metric, "alpha"] is None or pd.isna(table.loc[no_metric, "alpha"])
    assert table.loc[best, METRIC] == pytest.approx(1.2)
    assert np.isnan(table.loc[no_metric, METRIC])
    assert table.loc[no_metric, "valid_r2"] == pytest.approx(0.5)
    assert set(table["git_commit"]) == {"a" * 40}
    assert set(table["data_hash"]) == {"d" * 64}

    only_ridge = compare_runs(experiment, [METRIC], params=["alpha"], filters={"params.model": "ridge"})
    assert list(only_ridge.index) == [best, middle]
    assert list(only_ridge.columns) == ["run_name", "alpha", METRIC, "git_commit", "data_hash"]


def test_compare_runs_empty(names):
    experiment, _ = names
    table = compare_runs(experiment, [METRIC], params=["alpha"])
    assert len(table) == 0
    assert list(table.columns) == ["run_name", "alpha", METRIC, "git_commit", "data_hash"]
    light_run(experiment, "failed", {METRIC: 0.1}, fail=True)
    assert len(compare_runs(experiment, [METRIC], params=["alpha"])) == 0
