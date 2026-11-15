"""Схема конфигурации эксперимента и её загрузка из YAML."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    # Опечатка в имени параметра должна приводить к ошибке, а не молча игнорироваться
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    dataset: Literal["breast_cancer", "wine", "iris"] = "breast_cancer"
    test_size: float = Field(default=0.2, gt=0, lt=1)


class ModelConfig(StrictModel):
    C: float = Field(default=1.0, gt=0)
    max_iter: int = Field(default=1000, ge=1)


class TrainConfig(StrictModel):
    experiment_name: str = "baseline"
    seed: int = 42
    output_dir: Path = Path("models")
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()


def load_config(path: str | Path) -> TrainConfig:
    """Читает YAML-файл и проверяет его по схеме TrainConfig."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return TrainConfig.model_validate(raw)


def save_config(config: TrainConfig, path: str | Path) -> None:
    """Сохраняет итоговый конфиг вместе со значениями по умолчанию."""
    data = config.model_dump(mode="json")
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
