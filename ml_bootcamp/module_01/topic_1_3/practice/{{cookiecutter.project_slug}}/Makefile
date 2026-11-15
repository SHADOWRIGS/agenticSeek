.DEFAULT_GOAL := help
.PHONY: help install train test lint format clean

CONFIG ?= configs/base.yaml

help:  ## Показать список команд
	@grep -E '^[a-z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "%-10s %s\n", $$1, $$2}'

install:  ## Создать окружение и установить зависимости
	uv sync

train:  ## Обучить модель: make train CONFIG=configs/base.yaml
	uv run python -m {{ cookiecutter.package_name }}.train --config $(CONFIG)

test:  ## Запустить тесты
	uv run pytest

lint:  ## Проверить код линтером и форматтером
	uv run ruff check .
	uv run ruff format --check .

format:  ## Исправить замечания линтера и отформатировать код
	uv run ruff check --fix .
	uv run ruff format .

clean:  ## Удалить кеши
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
