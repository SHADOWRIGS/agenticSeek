# Практика 3.2.4: материалы

Постановка задачи: [lesson_3_2_4.md](../lesson_3_2_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `business_metrics.py` | заготовка: `savings`, `savings_curve`, `best_threshold`, `break_even_threshold`, `expected_value_decision`, `bootstrap_savings_interval`, `compare_strategies` |
| `tests/test_business_metrics.py` | 37 тестов, описывающих ожидаемое поведение функций |

Файл `fraud_data.py` должен лежать на уровень выше папки `practice` — так же, как в материалах курса.
Класс `Costs` и константы `DEFAULT_THRESHOLDS` и `STRATEGIES` в заготовке уже определены — менять их
не нужно.

## Запуск

```bash
uv add pandas numpy scikit-learn
uv add --dev pytest
cd practice
uv run pytest -q          # на заготовке все тесты падают с NotImplementedError
```

Удобно работать по одной функции, в таком порядке:

```bash
uv run pytest -q -k TestSavings
uv run pytest -q -k TestSavingsCurve
uv run pytest -q -k TestBestThreshold
uv run pytest -q -k TestBreakEvenThreshold
uv run pytest -q -k TestExpectedValueDecision
uv run pytest -q -k TestBootstrap
uv run pytest -q -k TestCompareStrategies
```

Часть тестов — маленькие примеры, которые легко проверить на бумаге; остальные обучают логистическую
регрессию на данных генератора (разбиение по времени) и проверяют свойства результата. Полный прогон
на правильном решении занимает несколько секунд. Если тесты бутстрепа идут дольше — смотрите совет
про вклады операций в уроке.
