# Практика 2.1.4: материалы

Постановка задачи: [lesson_2_1_4.md](../lesson_2_1_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `make_raw_data.py` | создаёт сырую выгрузку `data/raw_sales.csv` (использует `../sales_data.py`) |
| `sales_analysis.py` | заготовка: функции `load_raw`, `fill_calendar`, `add_features`, `evaluate_baselines` |
| `tests/test_sales_analysis.py` | тесты, описывающие ожидаемое поведение функций |

Файл `sales_data.py` должен лежать на уровень выше папки `practice` — так же, как в материалах курса.

## Запуск

```bash
uv add pandas numpy pyarrow
uv add --dev pytest
cd practice
uv run python make_raw_data.py     # data/raw_sales.csv, около 65 тысяч строк
uv run pytest -q                   # на заготовке все тесты падают
```

По мере реализации функций тесты начинают проходить. Удобно работать по одной функции:

```bash
uv run pytest -q -k TestLoadRaw
uv run pytest -q -k TestFillCalendar
uv run pytest -q -k TestAddFeatures
uv run pytest -q -k TestEvaluateBaselines
```

Папку `data/` в Git не добавляйте: данные воспроизводятся командой выше.
