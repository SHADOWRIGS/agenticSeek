# Практика 4.2.4: материалы

Постановка задачи: [lesson_4_2_4.md](../lesson_4_2_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `my_forest.py` | заготовка: `r2_score`, `MyRandomForestRegressor`, `permutation_importance`, `partial_dependence` |
| `tests/test_my_forest.py` | 52 теста: корректность, свойства, сверка со scikit-learn, скорость |

Данные для второй части практики — генератор [`housing_data.py`](../housing_data.py) в папке темы.
Тестам он не нужен: они работают на синтетической задаче Фридмана из scikit-learn.

## Запуск

```bash
uv add numpy pandas scikit-learn
uv add --dev pytest
uv run pytest -q          # на заготовке падают все тесты, кроме проверки импортов
```

По шагам:

```bash
uv run pytest -q -k TestR2
uv run pytest -q -k "TestFit or TestRandomness"
uv run pytest -q -k TestPredict
uv run pytest -q -k TestOob
uv run pytest -q -k TestPermutationImportance
uv run pytest -q -k TestPartialDependence
uv run pytest -q -k "TestAgainstSklearn or TestEngineering"
```

Общие фикстуры (`friedman`, `forest`, `sklearn_forest`) объявлены на уровне модуля. Фикстура `forest`
обучает `MyRandomForestRegressor`, поэтому до реализации `fit` тесты, которые её используют, завершаются
ошибкой на этапе подготовки (в отчёте pytest — `ERROR`, а не `FAILED`).

Эталонное решение проходит все тесты за 7 секунд. Тесты скорости (`test_fit_speed`,
`test_importance_and_dependence_speed`) имеют многократный запас и отсекают только решения, которые
вызывают `predict` отдельно для каждого объекта.

Проверка стиля:

```bash
uv run ruff check .
uv run ruff format --check my_forest.py tests/test_my_forest.py
```
