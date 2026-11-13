# Практика 7.1.4: материалы

Постановка задачи: [lesson_7_1_4.md](../lesson_7_1_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `ml_pipeline.py` | заготовка: трансформер `RareCategoryGrouper`, `build_pipeline`, этапы `extract`, `validate`, `train`, `evaluate`, `publish` и сквозной запуск `run`. Готовые части: константы, `DataValidationError`, `DateFeatures`, `partition_path`, `window_days`, `sales_source`, `main` |
| `tests/test_ml_pipeline.py` | 59 тестов: контракт трансформера, отсутствие утечки, идемпотентность, воспроизводимость, проверка данных, правило публикации, метаданные |

Тесты не зависят от генератора данных курса и от сети: источник данных в них — маленькая функция
«день → таблица» (4 магазина, 6 товаров, 24–26 строк в день), все файлы создаются во временных каталогах
pytest (`tmp_path`).

## Запуск тестов

Версии, на которых проверены заготовка и тесты: Python 3.12, `scikit-learn 1.9.1`, `pandas 3.1.1`,
`numpy 2.5.4`, `joblib 1.6.0`, `pytest 9.2.0`.

```bash
uv add numpy pandas scikit-learn
uv add --dev pytest ruff
uv run pytest -q          # на заготовке: 58 failed, 1 passed
```

Единственный тест, который проходит на заготовке, проверяет готовый `__init__` трансформера. Остальные
падают с `NotImplementedError`.

По шагам:

```bash
uv run pytest -q -k grouper
uv run pytest -q -k test_pipeline
uv run pytest -q -k test_extract
uv run pytest -q -k test_validate
uv run pytest -q -k test_train
uv run pytest -q -k test_evaluate
uv run pytest -q -k test_publish
uv run pytest -q -k test_run_
```

Фикстуры объявлены на уровне модуля: `source` (источник данных), `data_dir` (каталог с партициями обоих
окон, записанными без участия вашей `extract`), `cats` (маленькая таблица категорий), `sample` (строки
для сравнения прогнозов). Тесты `publish` создают каталог запуска с готовыми файлами сами, поэтому
`publish` можно реализовать и проверить раньше, чем `train` и `evaluate`.

Эталонное решение проходит все тесты за 8 секунд; самые долгие — тесты сквозного запуска, по 0.4–0.5
секунды. Строгая проверка — без предупреждений библиотек:

```bash
uv run python -W error -m pytest -q
```

Проверка стиля:

```bash
uv run ruff check .
uv run ruff format --check ml_pipeline.py tests/test_ml_pipeline.py
```

## Запуск на данных темы

Настоящий источник — генератор продаж темы 2.1. Скопируйте файл
[`module_02/topic_2_1/sales_data.py`](../../../module_02/topic_2_1/sales_data.py) без изменений в папку
с `ml_pipeline.py`:

```bash
uv run python ml_pipeline.py --date 2024-06-30 --workdir work
```

Данные генератора покрывают 2023-01-01 … 2024-12-31; с параметрами по умолчанию запуску нужны 63 дня
истории, поэтому самая ранняя дата запуска — 2023-03-04. Каталог `work/` в Git не добавляйте.

## Замечания

- В метаданных нет времени запуска: тесты требуют, чтобы повторный запуск давал те же `metrics.json`
  и `metadata.json` байт в байт. Время запуска — содержимое лога.
- `test_train_does_not_see_validation_window` переписывает проверочные партиции (продажи × 50, цены × 3,
  новые товары) и требует, чтобы модель не изменилась. Если тест падает, обучение читает лишние дни.
- `test_publish_older_version_keeps_pointer`: версия за прошлую дату публикуется, но `current.json`
  остаётся на более поздней дате. Даты в формате `YYYY-MM-DD` сравниваются как строки.
- `test_run_cleans_previous_attempt` оставляет в каталоге запуска посторонний файл: после запуска в каталоге
  должны быть ровно три артефакта.
