import json

from {{ cookiecutter.package_name }}.config import TrainConfig, load_config
from {{ cookiecutter.package_name }}.train import run, train


def test_run_saves_model_metrics_and_config(tmp_path):
    config = TrainConfig(output_dir=tmp_path)

    run_dir = run(config)

    assert (run_dir / "model.joblib").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["accuracy"] > 0.9
    # Сохранённый конфиг должен полностью восстанавливать исходный
    assert load_config(run_dir / "config.yaml") == config


def test_same_seed_gives_same_metrics():
    config = TrainConfig(seed=7)

    _, first = train(config)
    _, second = train(config)

    assert first == second
