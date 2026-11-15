# Практика 2.4.4: материалы

Постановка задачи: [lesson_2_4_4.md](../lesson_2_4_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `dashboard_data.py` | подготовка данных: фильтрация, агрегаты, показатели. Три функции — образец, пять — заготовки |
| `dashboard_figures.py` | графики Plotly. Две функции — образец, четыре — заготовки |
| `app.py` | рабочий каркас приложения Dash: макет, фильтры и один готовый callback |
| `tests/test_dashboard_data.py` | тесты функций данных |
| `tests/test_dashboard_figures.py` | тесты графиков |
| `tests/test_app.py` | тесты приложения через тестовый клиент Flask, без браузера |
| `pyproject.toml` | зависимости, настройки pytest и Ruff |

Данные создаёт `sales_data.py` из темы 2.1. Файл должен лежать на уровень выше папки `practice`:

```
p_2_4_dashboard/
├── sales_data.py
└── practice/
    ├── app.py
    └── ...
```

В репозитории курса копировать его не нужно: модуль найдёт файл в папке `topic_2_1`.

## Запуск

```bash
cd practice
uv sync                      # создаст окружение по pyproject.toml
uv run pytest -q             # на заготовке: 19 тестов проходят, 62 падают
uv run python app.py         # http://127.0.0.1:8050/
```

На заготовке проходят тесты функций-образцов и `TestSkeleton` из `tests/test_app.py`. Остальные тесты
падают с `NotImplementedError`, а четыре теста `TestCallbacks` — с сообщением о том, для каких графиков
ещё нет callback-функций.

Удобно работать по одной функции:

```bash
uv run pytest -q -k TestKpiSummary
uv run pytest -q -k TestTimeSeries
uv run pytest -q tests/test_dashboard_figures.py
uv run pytest -q tests/test_app.py
```

Проверка стиля:

```bash
uv run ruff check .
uv run ruff format --check .
```

Проверено на Python 3.12, dash 4.5.0, plotly 7.1.0, pandas 3.1.1, pytest 9.2.0.

## Если что-то не работает

| Симптом | Что делать |
|---|---|
| `ModuleNotFoundError: No module named 'sales_data'` | положите `sales_data.py` на уровень выше папки `practice` |
| `Address already in use` при запуске | порт 8050 занят: остановите прежний запуск или вызовите `app.run(debug=True, port=8051)` |
| Страница открылась, но график пуст и подписан «Здесь будет график» | для этого графика ещё не написан callback |
| Ошибка callback видна только в терминале | запускайте с `debug=True`: ошибка появится и в браузере |
