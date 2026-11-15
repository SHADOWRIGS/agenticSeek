"""Точка входа обучения.

Запуск: uv run python -m {{ cookiecutter.package_name }}.train --config configs/base.yaml
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from {{ cookiecutter.package_name }}.config import TrainConfig, load_config, save_config

DATASET_LOADERS = {
    "breast_cancer": datasets.load_breast_cancer,
    "wine": datasets.load_wine,
    "iris": datasets.load_iris,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def train(config: TrainConfig) -> tuple[Pipeline, dict[str, float]]:
    """Обучает модель по конфигу и возвращает её вместе с метриками на отложенной выборке."""
    set_seed(config.seed)
    features, target = DATASET_LOADERS[config.data.dataset](return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.data.test_size,
        random_state=config.seed,
        stratify=target,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=config.model.C, max_iter=config.model.max_iter, random_state=config.seed),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
    }
    return model, metrics


def run(config: TrainConfig) -> Path:
    """Выполняет один запуск: обучение и сохранение модели, метрик и конфига в папку запуска."""
    model, metrics = train(config)
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + config.experiment_name
    run_dir = config.output_dir / run_name
    # exist_ok=False: лучше упасть, чем перезаписать результаты другого запуска
    run_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, run_dir / "model.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_config(config, run_dir / "config.yaml")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение модели по конфигурационному файлу")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"), help="путь к YAML-конфигу")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = run(config)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    print(f"Запуск сохранён в {run_dir}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
