# Практика 3.1.4: материалы

Постановка задачи: [lesson_3_1_4.md](../lesson_3_1_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `churn_report.py` | заготовка: `prepare_features`, `fit_logit`, `odds_ratio_table`, `average_marginal_effects`, `predict_scenarios`, `bootstrap_coef_intervals`, `evaluate` |
| `tests/test_churn_report.py` | тесты, описывающие ожидаемое поведение функций |

Файл `churn_data.py` должен лежать на уровень выше папки `practice` — так же, как в материалах курса.

## Запуск

```bash
uv add pandas numpy statsmodels scikit-learn
uv add --dev pytest
cd practice
uv run pytest -q          # на заготовке все тесты падают
```

Удобно работать по одной функции:

```bash
uv run pytest -q -k TestPrepareFeatures
uv run pytest -q -k TestFitLogit
uv run pytest -q -k TestOddsRatioTable
uv run pytest -q -k TestAverageMarginalEffects
uv run pytest -q -k TestPredictScenarios
uv run pytest -q -k TestBootstrap
uv run pytest -q -k TestEvaluate
```

Полный прогон на правильном решении занимает несколько секунд. Если тесты бутстрепа идут минуты —
смотрите совет про выбор оптимизатора в уроке.
