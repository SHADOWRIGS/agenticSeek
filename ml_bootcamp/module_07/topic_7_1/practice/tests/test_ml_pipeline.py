import copy
import hashlib
import json
import platform
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))

from ml_pipeline import (  # noqa: E402
    DEFAULT_CONFIG,
    KEYS,
    MAX_MISSING_SHARE,
    OTHER,
    TARGET,
    DataValidationError,
    RareCategoryGrouper,
    build_pipeline,
    evaluate,
    extract,
    partition_path,
    publish,
    run,
    train,
    validate,
    window_days,
)

RUN_DATE = date(2024, 6, 30)
CONFIG = {**DEFAULT_CONFIG, "train_days": 28, "valid_days": 7, "min_category_count": 5}
STORES = {"S1": 1.6, "S2": 1.0, "S3": 0.7, "S4": 0.5}
# товар -> (цена, базовый спрос)
PRODUCTS = {
    "bread": (60, 12.0),
    "milk": (90, 9.0),
    "tea": (300, 6.0),
    "coffee": (700, 4.0),
    "cheese": (650, 3.0),
    "jam": (250, 2.0),
}
RARE, NOVELTY = "saffron", "novelty"


def make_source(seed=0, signal=True, new_product_from=None):
    """Источник данных: функция «день -> таблица продаж». Результат зависит только от дня и seed.

    Каждый десятый день в магазине S1 продаётся редкий товар RARE; раз в неделю у одной строки пропущен
    units. При signal=False продажи не зависят от признаков. С new_product_from появляется товар NOVELTY.
    """

    def source(day: date) -> pd.DataFrame:
        rng = np.random.default_rng([seed, day.toordinal()])
        rows = [(store, product) for store in STORES for product in PRODUCTS]
        if day.toordinal() % 10 == 0:
            rows.append(("S1", RARE))
        if new_product_from is not None and day >= new_product_from:
            rows.append(("S2", NOVELTY))
        frame = pd.DataFrame(rows, columns=KEYS)
        frame.insert(0, "date", pd.Timestamp(day))
        frame["city"] = "Город"
        frame["promo"] = rng.random(len(frame)) < 0.15
        base_price = frame["product"].map(lambda name: PRODUCTS.get(name, (500, 1.0))[0])
        frame["price"] = np.where(frame["promo"], base_price * 0.85, base_price).round(0)
        demand = frame["product"].map(lambda name: PRODUCTS.get(name, (500, 1.0))[1]) * frame["store_id"].map(STORES)
        demand = demand * (1.4 if day.weekday() >= 5 else 1.0) * np.where(frame["promo"], 1.8, 1.0)
        frame[TARGET] = rng.poisson(demand if signal else np.full(len(frame), 5.0)).astype(float)
        if day.toordinal() % 7 == 3:
            frame.loc[0, TARGET] = np.nan
        return frame

    return source


def all_days(run_date=RUN_DATE, config=CONFIG) -> list[date]:
    train_days, valid_days = window_days(run_date, config)
    return [*train_days, *valid_days]


def write_partition(frame: pd.DataFrame, data_dir: Path, day: date) -> Path:
    path = partition_path(data_dir, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def read_days(data_dir: Path, days: list[date]) -> pd.DataFrame:
    frames = [pd.read_csv(partition_path(data_dir, day), parse_dates=["date"]) for day in days]
    return pd.concat(frames, ignore_index=True)


def listing(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def window_hash(data_dir: Path, run_date=RUN_DATE, config=CONFIG) -> str:
    digest = hashlib.sha256()
    for day in all_days(run_date, config):
        digest.update(partition_path(data_dir, day).read_bytes())
    return digest.hexdigest()


def fake_run_dir(root: Path, name: str, mae_ratio: float) -> Path:
    """Каталог запуска с готовыми артефактами: publish проверяется независимо от train и evaluate."""
    run_dir = root / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "model.joblib").write_bytes(f"model {name}".encode())
    (run_dir / "metrics.json").write_text(json.dumps({"mae_ratio": mae_ratio, "mae": 1.0, "baseline_mae": 2.0}))
    return run_dir


@pytest.fixture
def source():
    return make_source()


@pytest.fixture
def data_dir(tmp_path, source):
    """Каталог данных с партициями обоих окон за RUN_DATE; записан без участия extract."""
    root = tmp_path / "data"
    for day in all_days():
        write_partition(source(day), root, day)
    return root


@pytest.fixture
def cats():
    return pd.DataFrame(
        {"store_id": ["a"] * 6 + ["b"] * 3 + ["c"], "product": ["x"] * 5 + ["y"] * 4 + [None]},
        index=range(100, 110),
    )


@pytest.fixture
def sample(source):
    """Строки, на которых сравниваются прогнозы разных моделей."""
    return source(RUN_DATE).dropna(subset=[TARGET]).reset_index(drop=True)


# ---------- RareCategoryGrouper: контракт scikit-learn ----------


def test_grouper_init_only_stores_params():
    grouper = RareCategoryGrouper(min_count=0)  # неверное значение: __init__ его не проверяет
    assert grouper.get_params() == {"min_count": 0}
    assert not [name for name in vars(grouper) if name.endswith("_")]


def test_grouper_checks_params_in_fit(cats):
    with pytest.raises(ValueError):
        RareCategoryGrouper(min_count=0).fit(cats)


def test_grouper_not_fitted(cats):
    with pytest.raises(NotFittedError):
        RareCategoryGrouper().transform(cats)
    with pytest.raises(NotFittedError):
        RareCategoryGrouper().get_feature_names_out()


def test_grouper_fit_returns_self_and_learns(cats):
    grouper = RareCategoryGrouper(min_count=3)
    assert grouper.fit(cats) is grouper
    assert grouper.categories_ == {"store_id": ["a", "b"], "product": ["x", "y"]}
    assert list(grouper.feature_names_in_) == ["store_id", "product"]
    assert grouper.n_features_in_ == 2


def test_grouper_transform(cats):
    out = RareCategoryGrouper(min_count=3).fit(cats).transform(cats)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["store_id", "product"]
    assert list(out.index) == list(cats.index)
    assert out["store_id"].tolist() == ["a"] * 6 + ["b"] * 3 + [OTHER]
    assert out["product"].tolist() == ["x"] * 5 + ["y"] * 4 + [OTHER]


def test_grouper_unseen_category(cats):
    grouper = RareCategoryGrouper(min_count=3).fit(cats)
    new = pd.DataFrame({"store_id": ["a", "zzz"], "product": ["new", "x"]})
    out = grouper.transform(new)
    assert out["store_id"].tolist() == ["a", OTHER]
    assert out["product"].tolist() == [OTHER, "x"]


def test_grouper_does_not_modify_input(cats):
    backup = cats.copy(deep=True)
    grouper = RareCategoryGrouper(min_count=3).fit(cats)
    out = grouper.transform(cats)
    assert out is not cats
    pd.testing.assert_frame_equal(cats, backup)


def test_grouper_statistics_come_from_fit_only(cats):
    """Утечка: transform не должен пересчитывать частоты по новым данным."""
    grouper = RareCategoryGrouper(min_count=3).fit(cats)
    learned = copy.deepcopy(grouper.categories_)
    test = pd.DataFrame({"store_id": ["c"] * 50, "product": ["new"] * 50})
    out = grouper.transform(test)
    assert out["store_id"].tolist() == [OTHER] * 50  # в тесте "c" частая, но при обучении была редкой
    assert out["product"].tolist() == [OTHER] * 50
    assert grouper.categories_ == learned


def test_grouper_rejects_wrong_input(cats):
    grouper = RareCategoryGrouper(min_count=3)
    with pytest.raises(TypeError):
        grouper.fit(cats.to_numpy())
    grouper.fit(cats)
    with pytest.raises(TypeError):
        grouper.transform(cats.to_numpy())
    with pytest.raises(ValueError):
        grouper.transform(cats.rename(columns={"product": "item"}))


def test_grouper_feature_names_out(cats):
    names = RareCategoryGrouper(min_count=3).fit(cats).get_feature_names_out()
    assert list(names) == ["store_id", "product"]


def test_grouper_clone(cats):
    grouper = RareCategoryGrouper(min_count=3).fit(cats)
    twin = clone(grouper)
    assert twin is not grouper
    assert twin.get_params() == {"min_count": 3}
    assert not hasattr(twin, "categories_")


def test_grouper_in_pipeline_and_search(data_dir):
    table = read_days(data_dir, all_days()).dropna(subset=[TARGET])
    model = Pipeline(
        [("rare", RareCategoryGrouper()), ("onehot", OneHotEncoder(handle_unknown="ignore")), ("model", Ridge())]
    )
    search = GridSearchCV(model, {"rare__min_count": [1, 5, 500]}, cv=3, scoring="neg_mean_absolute_error")
    search.fit(table[KEYS], table[TARGET])
    assert search.best_params_["rare__min_count"] in (1, 5)  # при 500 все категории сливаются в одну
    assert RARE not in search.best_estimator_["rare"].set_params(min_count=5).fit(table[KEYS]).categories_["product"]


# ---------- build_pipeline ----------


def test_pipeline_structure():
    model = build_pipeline(CONFIG)
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["features", "model"]
    features = model.named_steps["features"]
    assert isinstance(features, ColumnTransformer)
    assert features.remainder == "drop"
    assert [name for name, _, _ in features.transformers] == ["date", "cat", "num"]
    assert not hasattr(features, "transformers_")  # пайплайн возвращается необученным
    params = model.get_params()
    assert isinstance(params["features__cat__rare"], RareCategoryGrouper)
    assert params["features__cat__rare__min_count"] == CONFIG["min_category_count"]
    assert params["features__cat__onehot__handle_unknown"] == "ignore"
    assert params["model__regressor__alpha"] == CONFIG["alpha"]


def test_pipeline_uses_config():
    params = build_pipeline({**CONFIG, "alpha": 7.0, "min_category_count": 2}).get_params()
    assert params["model__regressor__alpha"] == 7.0
    assert params["features__cat__rare__min_count"] == 2


def test_pipeline_fit_predict(data_dir):
    train_days, valid_days = window_days(RUN_DATE, CONFIG)
    train_part = read_days(data_dir, train_days).dropna(subset=[TARGET])
    valid_part = read_days(data_dir, valid_days).dropna(subset=[TARGET])
    backup = train_part.copy(deep=True)
    model = build_pipeline(CONFIG).fit(train_part, train_part[TARGET])
    pd.testing.assert_frame_equal(train_part, backup)
    pred = model.predict(valid_part)
    assert pred.shape == (len(valid_part),)
    assert np.isfinite(pred).all()
    mae = np.abs(pred - valid_part[TARGET]).mean()
    baseline = np.abs(train_part[TARGET].mean() - valid_part[TARGET]).mean()
    assert mae < 0.7 * baseline


def test_pipeline_target_is_not_a_feature(data_dir, sample):
    train_part = read_days(data_dir, window_days(RUN_DATE, CONFIG)[0]).dropna(subset=[TARGET])
    model = build_pipeline(CONFIG).fit(train_part, train_part[TARGET])
    changed = sample.assign(**{TARGET: sample[TARGET] * 100, "city": "Другой"})
    assert np.array_equal(model.predict(sample), model.predict(changed))
    assert np.array_equal(model.predict(sample), model.predict(sample.drop(columns=[TARGET, "city"])))


def test_pipeline_unseen_and_rare_categories(data_dir, sample):
    """Статистики редких категорий — только по обучающим данным; незнакомые значения не ломают прогноз."""
    train_part = read_days(data_dir, window_days(RUN_DATE, CONFIG)[0]).dropna(subset=[TARGET])
    model = build_pipeline(CONFIG).fit(train_part, train_part[TARGET])
    rare = model.named_steps["features"].named_transformers_["cat"].named_steps["rare"]
    assert (train_part["product"] == RARE).sum() > 0
    assert sorted(rare.categories_["product"]) == sorted(PRODUCTS)  # RARE встречается реже 5 раз
    learned = copy.deepcopy(rare.categories_)
    strangers = sample.assign(store_id="S9", product=NOVELTY)
    assert np.isfinite(model.predict(pd.concat([strangers] * 20, ignore_index=True))).all()
    assert rare.categories_ == learned


def test_pipeline_clone(data_dir):
    train_part = read_days(data_dir, window_days(RUN_DATE, CONFIG)[0]).dropna(subset=[TARGET])
    model = build_pipeline(CONFIG).fit(train_part, train_part[TARGET])
    twin = clone(model)
    assert not hasattr(twin.named_steps["features"], "transformers_")
    assert twin.get_params()["model__regressor__alpha"] == CONFIG["alpha"]


# ---------- extract ----------


def test_extract_writes_partition(tmp_path, source):
    path = extract(RUN_DATE, source, tmp_path)
    assert Path(path) == partition_path(tmp_path, RUN_DATE)
    saved = pd.read_csv(path, parse_dates=["date"])
    expected = source(RUN_DATE)
    assert list(saved.columns) == list(expected.columns)  # без столбца индекса
    assert len(saved) == len(expected)
    assert saved[TARGET].sum() == expected[TARGET].sum()
    assert set(saved["date"].dt.date) == {RUN_DATE}


def test_extract_is_idempotent(tmp_path, source):
    path = Path(extract(RUN_DATE, source, tmp_path))
    first = path.read_bytes()
    extract(RUN_DATE, source, tmp_path)
    assert path.read_bytes() == first
    assert [item.name for item in path.parent.iterdir()] == ["sales.csv"]  # ни дубликатов, ни временных файлов


def test_extract_replaces_partition(tmp_path, source):
    """Данные в источнике исправили: повторная выгрузка заменяет партицию, а не дописывает её."""
    path = Path(extract(RUN_DATE, source, tmp_path))
    fixed = make_source(seed=1)
    extract(RUN_DATE, fixed, tmp_path)
    saved = pd.read_csv(path)
    assert len(saved) == len(fixed(RUN_DATE))
    assert saved[TARGET].sum() == fixed(RUN_DATE)[TARGET].sum()
    assert listing(tmp_path) == [str(path.relative_to(tmp_path))]


def test_extract_separate_partitions(tmp_path, source):
    first = extract(RUN_DATE, source, tmp_path)
    second = extract(RUN_DATE - timedelta(days=1), source, tmp_path)
    assert Path(first) != Path(second)
    assert len(listing(tmp_path)) == 2


def test_extract_empty_source(tmp_path, source):
    with pytest.raises(ValueError):
        extract(RUN_DATE, lambda day: source(day).iloc[:0], tmp_path)
    assert not partition_path(tmp_path, RUN_DATE).exists()


# ---------- validate ----------


def test_validate_good_partition(tmp_path, source):
    clean_day = next(day for day in all_days() if day.toordinal() % 7 != 3)
    frame = source(clean_day)
    report = validate(write_partition(frame, tmp_path, clean_day), clean_day)
    assert report == {"rows": len(frame), "missing_share": 0.0}


def test_validate_reports_missing_share(tmp_path, source):
    day = next(day for day in all_days() if day.toordinal() % 7 == 3)
    frame = source(day)
    path = write_partition(frame, tmp_path, day)
    before = path.read_bytes()
    report = validate(path, day)
    assert report["rows"] == len(frame)
    assert report["missing_share"] == pytest.approx(1 / len(frame))
    assert report["missing_share"] <= MAX_MISSING_SHARE
    assert path.read_bytes() == before


def drop_price(frame):
    return frame.drop(columns="price")


def wrong_date(frame):
    frame = frame.copy()
    frame.loc[2, "date"] = frame.loc[2, "date"] - pd.Timedelta(days=1)
    return frame


def duplicate_key(frame):
    return pd.concat([frame, frame.iloc[[0]]], ignore_index=True)


def negative_units(frame):
    frame = frame.copy()
    frame.loc[5, TARGET] = -1.0
    return frame


def zero_price(frame):
    frame = frame.copy()
    frame.loc[1, "price"] = 0
    return frame


def many_missing(frame):
    frame = frame.copy()
    frame.loc[:5, TARGET] = np.nan
    return frame


def no_rows(frame):
    return frame.iloc[:0]


@pytest.mark.parametrize(
    "corrupt", [drop_price, wrong_date, duplicate_key, negative_units, zero_price, many_missing, no_rows]
)
def test_validate_catches_bad_data(tmp_path, source, corrupt):
    path = write_partition(corrupt(source(RUN_DATE)), tmp_path, RUN_DATE)
    with pytest.raises(DataValidationError):
        validate(path, RUN_DATE)


def test_validate_checks_run_date(tmp_path, source):
    path = write_partition(source(RUN_DATE), tmp_path, RUN_DATE)
    with pytest.raises(DataValidationError):
        validate(path, RUN_DATE + timedelta(days=1))
    assert issubclass(DataValidationError, ValueError)


# ---------- train ----------


def test_train_saves_model(tmp_path, data_dir, sample):
    run_dir = tmp_path / "runs" / "one"
    path = train(RUN_DATE, data_dir, run_dir, CONFIG)
    assert Path(path) == run_dir / "model.joblib"
    model = joblib.load(path)
    assert isinstance(model, Pipeline)
    assert np.isfinite(model.predict(sample)).all()


def test_train_requires_all_partitions(tmp_path, data_dir):
    partition_path(data_dir, window_days(RUN_DATE, CONFIG)[0][3]).unlink()
    with pytest.raises(FileNotFoundError):
        train(RUN_DATE, data_dir, tmp_path / "run", CONFIG)


def test_train_is_reproducible(tmp_path, data_dir, sample):
    first = joblib.load(train(RUN_DATE, data_dir, tmp_path / "a", CONFIG))
    second = joblib.load(train(RUN_DATE, data_dir, tmp_path / "b", CONFIG))
    assert np.array_equal(first.predict(sample), second.predict(sample))


def test_train_does_not_see_validation_window(tmp_path, data_dir, sample):
    """Утечка: изменение проверочных партиций не должно менять модель."""
    before = joblib.load(train(RUN_DATE, data_dir, tmp_path / "a", CONFIG))
    for day in window_days(RUN_DATE, CONFIG)[1]:
        frame = pd.read_csv(partition_path(data_dir, day))
        frame[TARGET] = frame[TARGET] * 50
        frame["price"] = frame["price"] * 3
        frame["product"] = NOVELTY + frame.index.astype(str)
        write_partition(frame, data_dir, day)
    after = joblib.load(train(RUN_DATE, data_dir, tmp_path / "b", CONFIG))
    assert np.array_equal(before.predict(sample), after.predict(sample))
    rare_before = before.named_steps["features"].named_transformers_["cat"].named_steps["rare"]
    rare_after = after.named_steps["features"].named_transformers_["cat"].named_steps["rare"]
    assert rare_before.categories_ == rare_after.categories_


def test_train_depends_on_training_window(tmp_path, data_dir, sample):
    before = joblib.load(train(RUN_DATE, data_dir, tmp_path / "a", CONFIG))
    day = window_days(RUN_DATE, CONFIG)[0][-1]
    frame = pd.read_csv(partition_path(data_dir, day))
    frame[TARGET] = frame[TARGET] * 5
    write_partition(frame, data_dir, day)
    after = joblib.load(train(RUN_DATE, data_dir, tmp_path / "b", CONFIG))
    assert not np.allclose(before.predict(sample), after.predict(sample))


# ---------- evaluate ----------


def test_evaluate_metrics(tmp_path, data_dir):
    run_dir = tmp_path / "run"
    model_path = train(RUN_DATE, data_dir, run_dir, CONFIG)
    metrics = evaluate(model_path, RUN_DATE, data_dir, run_dir, CONFIG)
    train_days, valid_days = window_days(RUN_DATE, CONFIG)
    train_target = read_days(data_dir, train_days)[TARGET].dropna()
    valid_part = read_days(data_dir, valid_days).dropna(subset=[TARGET])
    mae = np.abs(joblib.load(model_path).predict(valid_part) - valid_part[TARGET]).mean()
    baseline = np.abs(train_target.mean() - valid_part[TARGET]).mean()
    assert set(metrics) == {"mae", "baseline_mae", "mae_ratio", "n_train", "n_valid"}
    assert metrics["mae"] == pytest.approx(mae)
    assert metrics["baseline_mae"] == pytest.approx(baseline)
    assert metrics["mae_ratio"] == pytest.approx(mae / baseline)
    assert (metrics["n_train"], metrics["n_valid"]) == (len(train_target), len(valid_part))
    assert metrics["mae_ratio"] < 0.7


def test_evaluate_writes_json(tmp_path, data_dir):
    run_dir = tmp_path / "run"
    metrics = evaluate(train(RUN_DATE, data_dir, run_dir, CONFIG), RUN_DATE, data_dir, run_dir, CONFIG)
    assert all(type(metrics[key]) is float for key in ("mae", "baseline_mae", "mae_ratio"))
    assert all(type(metrics[key]) is int for key in ("n_train", "n_valid"))
    assert (run_dir / "metrics.json").read_text() == json.dumps(metrics, indent=2, sort_keys=True)


# ---------- publish ----------


def test_publish_good_model(tmp_path):
    run_dir = fake_run_dir(tmp_path, "2024-06-30", mae_ratio=0.5)
    registry = tmp_path / "registry"
    assert publish(run_dir, registry, RUN_DATE, 0.8) is True
    assert (registry / "2024-06-30" / "model.joblib").read_bytes() == (run_dir / "model.joblib").read_bytes()
    assert (registry / "2024-06-30" / "metrics.json").read_bytes() == (run_dir / "metrics.json").read_bytes()
    assert json.loads((registry / "current.json").read_text()) == {
        "run_date": "2024-06-30",
        "model": "2024-06-30/model.joblib",
    }


def test_publish_rejects_bad_model(tmp_path):
    run_dir = fake_run_dir(tmp_path, "2024-06-30", mae_ratio=0.95)
    assert publish(run_dir, tmp_path / "registry", RUN_DATE, 0.8) is False
    assert not (tmp_path / "registry").exists()


def test_publish_threshold_is_inclusive(tmp_path):
    run_dir = fake_run_dir(tmp_path, "2024-06-30", mae_ratio=0.8)
    assert publish(run_dir, tmp_path / "registry", RUN_DATE, 0.8) is True


def test_publish_is_idempotent(tmp_path):
    run_dir = fake_run_dir(tmp_path, "2024-06-30", mae_ratio=0.5)
    registry = tmp_path / "registry"
    publish(run_dir, registry, RUN_DATE, 0.8)
    first = listing(registry)
    publish(run_dir, registry, RUN_DATE, 0.8)
    assert listing(registry) == first == ["2024-06-30/metrics.json", "2024-06-30/model.joblib", "current.json"]


def test_publish_newer_version_moves_pointer(tmp_path):
    registry = tmp_path / "registry"
    publish(fake_run_dir(tmp_path, "2024-06-29", 0.5), registry, date(2024, 6, 29), 0.8)
    publish(fake_run_dir(tmp_path, "2024-06-30", 0.6), registry, RUN_DATE, 0.8)
    assert json.loads((registry / "current.json").read_text())["run_date"] == "2024-06-30"
    assert (registry / "2024-06-29" / "model.joblib").exists()  # прошлая версия остаётся для отката


def test_publish_older_version_keeps_pointer(tmp_path):
    """Пересчёт за прошлую дату не должен подменять текущую модель старой."""
    registry = tmp_path / "registry"
    publish(fake_run_dir(tmp_path, "2024-06-30", 0.5), registry, RUN_DATE, 0.8)
    assert publish(fake_run_dir(tmp_path, "2024-06-20", 0.4), registry, date(2024, 6, 20), 0.8) is True
    assert json.loads((registry / "current.json").read_text())["run_date"] == "2024-06-30"
    assert (registry / "2024-06-20" / "model.joblib").read_bytes() == b"model 2024-06-20"


def test_publish_rejected_model_keeps_current(tmp_path):
    registry = tmp_path / "registry"
    publish(fake_run_dir(tmp_path, "2024-06-29", 0.5), registry, date(2024, 6, 29), 0.8)
    before = listing(registry)
    assert publish(fake_run_dir(tmp_path, "2024-06-30", 0.9), registry, RUN_DATE, 0.8) is False
    assert listing(registry) == before
    assert json.loads((registry / "current.json").read_text())["run_date"] == "2024-06-29"


# ---------- run ----------


def test_run_artifacts(tmp_path, source):
    result = run(RUN_DATE, tmp_path, source, CONFIG)
    run_dir = tmp_path / "runs" / "2024-06-30"
    assert set(result) == {"run_dir", "metrics", "published"}
    assert Path(result["run_dir"]) == run_dir
    assert sorted(item.name for item in run_dir.iterdir()) == ["metadata.json", "metrics.json", "model.joblib"]
    assert result["published"] is True
    assert json.loads((tmp_path / "registry" / "current.json").read_text())["run_date"] == "2024-06-30"
    assert len(listing(tmp_path / "data")) == CONFIG["train_days"] + CONFIG["valid_days"]
    assert result["metrics"] == json.loads((run_dir / "metrics.json").read_text())


def test_run_metadata(tmp_path, source):
    result = run(RUN_DATE, tmp_path, source, CONFIG)
    metadata = json.loads((tmp_path / "runs" / "2024-06-30" / "metadata.json").read_text(encoding="utf-8"))
    assert set(metadata) == {
        "run_date", "config", "train_period", "valid_period", "data_hash", "metrics", "published", "versions",
    }  # fmt: skip
    assert metadata["run_date"] == "2024-06-30"
    assert metadata["config"] == CONFIG
    assert metadata["train_period"] == ["2024-05-27", "2024-06-23"]
    assert metadata["valid_period"] == ["2024-06-24", "2024-06-30"]
    assert metadata["data_hash"] == window_hash(tmp_path / "data")
    assert metadata["metrics"] == result["metrics"]
    assert metadata["published"] is True
    assert metadata["versions"] == {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def test_run_is_idempotent(tmp_path, source, sample):
    run(RUN_DATE, tmp_path, source, CONFIG)
    files = listing(tmp_path)
    run_dir = tmp_path / "runs" / "2024-06-30"
    texts = {name: (run_dir / name).read_bytes() for name in ("metrics.json", "metadata.json")}
    partitions = {name: (tmp_path / name).read_bytes() for name in files if name.startswith("data")}
    pred = joblib.load(run_dir / "model.joblib").predict(sample)

    run(RUN_DATE, tmp_path, source, CONFIG)
    assert listing(tmp_path) == files  # ни дубликатов, ни временных файлов
    assert {name: (run_dir / name).read_bytes() for name in texts} == texts
    assert {name: (tmp_path / name).read_bytes() for name in partitions} == partitions
    assert np.array_equal(joblib.load(run_dir / "model.joblib").predict(sample), pred)


def test_run_is_reproducible(tmp_path, source):
    """Тот же код, те же данные, другой каталог — те же артефакты."""
    run(RUN_DATE, tmp_path / "a", source, CONFIG)
    run(RUN_DATE, tmp_path / "b", make_source(), CONFIG)
    for name in ("metrics.json", "metadata.json"):
        first = (tmp_path / "a" / "runs" / "2024-06-30" / name).read_bytes()
        assert first == (tmp_path / "b" / "runs" / "2024-06-30" / name).read_bytes()


def test_run_for_another_date(tmp_path, source):
    run(RUN_DATE, tmp_path, source, CONFIG)
    earlier = RUN_DATE - timedelta(days=3)
    result = run(earlier, tmp_path, source, CONFIG)
    assert result["published"] is True
    first = json.loads((tmp_path / "runs" / "2024-06-30" / "metadata.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "runs" / "2024-06-27" / "metadata.json").read_text(encoding="utf-8"))
    assert first["data_hash"] != second["data_hash"]
    assert second["valid_period"] == ["2024-06-21", "2024-06-27"]
    assert (tmp_path / "registry" / "2024-06-27" / "model.joblib").exists()
    assert json.loads((tmp_path / "registry" / "current.json").read_text())["run_date"] == "2024-06-30"


def test_run_config_defaults(tmp_path, source):
    run(RUN_DATE, tmp_path, source, {"alpha": 3.0})
    metadata = json.loads((tmp_path / "runs" / "2024-06-30" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["config"] == {**DEFAULT_CONFIG, "alpha": 3.0}
    assert len(listing(tmp_path / "data")) == DEFAULT_CONFIG["train_days"] + DEFAULT_CONFIG["valid_days"]


def test_run_does_not_publish_weak_model(tmp_path):
    result = run(RUN_DATE, tmp_path, make_source(signal=False), CONFIG)
    assert result["metrics"]["mae_ratio"] > CONFIG["max_mae_ratio"]
    assert result["published"] is False
    assert not (tmp_path / "registry").exists()
    run_dir = tmp_path / "runs" / "2024-06-30"
    assert sorted(item.name for item in run_dir.iterdir()) == ["metadata.json", "metrics.json", "model.joblib"]
    assert json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))["published"] is False


def test_run_threshold_from_config(tmp_path, source):
    result = run(RUN_DATE, tmp_path, source, {**CONFIG, "max_mae_ratio": 0.01})
    assert result["published"] is False
    assert not (tmp_path / "registry").exists()


def test_run_stops_on_bad_data(tmp_path, source):
    bad_day = window_days(RUN_DATE, CONFIG)[0][10]

    def broken(day):
        frame = source(day)
        if day == bad_day:
            frame.loc[4, TARGET] = -3.0
        return frame

    with pytest.raises(DataValidationError):
        run(RUN_DATE, tmp_path, broken, CONFIG)
    assert not (tmp_path / "runs" / "2024-06-30" / "model.joblib").exists()
    assert not (tmp_path / "registry").exists()


def test_run_after_data_fix(tmp_path, source):
    """Данные за прошлые дни исправили: перезапуск за ту же дату заменяет артефакты, а не добавляет новые."""
    run(RUN_DATE, tmp_path, source, CONFIG)
    files = listing(tmp_path)
    old = json.loads((tmp_path / "runs" / "2024-06-30" / "metadata.json").read_text(encoding="utf-8"))
    run(RUN_DATE, tmp_path, make_source(seed=1), CONFIG)
    new = json.loads((tmp_path / "runs" / "2024-06-30" / "metadata.json").read_text(encoding="utf-8"))
    assert listing(tmp_path) == files
    assert new["data_hash"] != old["data_hash"]
    assert new["data_hash"] == window_hash(tmp_path / "data")
    assert new["metrics"] != old["metrics"]


def test_run_cleans_previous_attempt(tmp_path, source):
    """В каталоге запуска остались файлы прерванной попытки: после запуска их быть не должно."""
    run_dir = tmp_path / "runs" / "2024-06-30"
    run_dir.mkdir(parents=True)
    (run_dir / "model.joblib.partial").write_bytes(b"stale")
    run(RUN_DATE, tmp_path, source, CONFIG)
    assert sorted(item.name for item in run_dir.iterdir()) == ["metadata.json", "metrics.json", "model.joblib"]


def test_run_with_new_product_in_validation_window(tmp_path):
    first_valid_day = window_days(RUN_DATE, CONFIG)[1][0]
    result = run(RUN_DATE, tmp_path, make_source(new_product_from=first_valid_day), CONFIG)
    assert np.isfinite(result["metrics"]["mae"])
