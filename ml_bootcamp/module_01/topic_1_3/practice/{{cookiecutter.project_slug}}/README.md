# {{ cookiecutter.project_name }}

{{ cookiecutter.description }}

Автор: {{ cookiecutter.author }}

## Быстрый старт

Нужны `uv` и `make`. Python {{ cookiecutter.python_version }} `uv` установит сам.

```bash
make install   # создать .venv и установить зависимости
make train     # обучить модель по configs/base.yaml
make test      # запустить тесты
make lint      # проверить код
```

Список всех команд: `make help`.

## Структура проекта

```text
├── configs/            конфигурации экспериментов (YAML)
├── data/
│   ├── raw/            исходные данные, не изменяются
│   ├── interim/        промежуточные результаты обработки
│   └── processed/      готовые к обучению наборы данных
├── models/             результаты запусков: модель, метрики и конфиг
├── notebooks/          исследовательские ноутбуки
├── src/{{ cookiecutter.package_name }}/
│   ├── config.py       схема конфига и его загрузка
│   └── train.py        точка входа обучения
├── tests/              тесты
├── Makefile            команды проекта
└── pyproject.toml      зависимости и настройки инструментов
```

## Эксперименты

Параметры запуска описаны в `configs/base.yaml`. Для нового эксперимента скопируйте файл, измените значения
и передайте путь к нему:

```bash
make train CONFIG=configs/my_experiment.yaml
```

Каждый запуск создаёт папку `models/<дата-время>_<experiment_name>/` с файлами `model.joblib`, `metrics.json`
и `config.yaml` — по ним запуск можно повторить.

Содержимое `data/` и `models/` в Git не попадает.
