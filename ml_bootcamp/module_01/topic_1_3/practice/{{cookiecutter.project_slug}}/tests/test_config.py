from pathlib import Path

import pytest
from pydantic import ValidationError

from {{ cookiecutter.package_name }}.config import TrainConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_base_config_is_valid():
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    assert config.seed == 42
    assert 0 < config.data.test_size < 1


def test_invalid_value_is_rejected():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"data": {"test_size": 1.5}})


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"model": {"max_itre": 100}})
