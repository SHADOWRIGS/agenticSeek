# Практика 4.1.4: материалы

Постановка задачи: [lesson_4_1_4.md](../lesson_4_1_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `my_tree.py` | заготовка: `impurity`, `best_split`, `MyDecisionTreeClassifier`; класс `Node` готов |
| `tests/test_my_tree.py` | 52 теста: корректность, сверка со scikit-learn, скорость |

## Запуск

```bash
uv add numpy
uv add --dev pytest scikit-learn
uv run pytest -q          # на заготовке падают все тесты, кроме проверки импортов
```

По шагам:

```bash
uv run pytest -q -k TestImpurity
uv run pytest -q -k TestBestSplit
uv run pytest -q -k TestTreeStructure
uv run pytest -q -k TestPrediction
uv run pytest -q -k TestAgainstSklearn
uv run pytest -q -k TestEngineering
```

Тесты скорости (`test_speed`, `test_fit_speed`, `test_predict_speed`) отсекают решения с циклом
по порогам и с обходом дерева отдельно для каждого объекта. Пока решение медленное, их удобно исключать:

```bash
uv run pytest -q -k "not speed"
```
