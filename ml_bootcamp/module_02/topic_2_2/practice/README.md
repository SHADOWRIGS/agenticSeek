# Практика 2.2.4: материалы

Постановка задачи: [lesson_2_2_4.md](../lesson_2_2_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `numpy_algorithms.py` | заготовка: восемь функций с описанием поведения в docstring и `raise NotImplementedError` |
| `tests/test_numpy_algorithms.py` | 52 теста: сверка со scikit-learn и SciPy, формы, численная устойчивость, память, скорость, запрет циклов |

Данные скачивать не нужно: тесты используют встроенные наборы scikit-learn (`load_iris`, `load_wine`,
`load_digits`) и синтетические данные (`make_blobs`, `make_regression`).

## Запуск

```bash
uv add numpy
uv add --dev pytest scipy scikit-learn
cd practice
uv run pytest -q                   # на заготовке все 52 теста падают с NotImplementedError
```

По мере реализации функций тесты начинают проходить. Удобно работать по одной функции:

```bash
uv run pytest -q -k TestStandardize
uv run pytest -q -k TestPairwiseSqDistances
uv run pytest -q -k TestKnnPredict
uv run pytest -q -k TestKmeans
uv run pytest -q -k TestLinearRegression
uv run pytest -q -k TestPca
uv run pytest -q -k TestSoftmax
uv run pytest -q -k TestFitSoftmaxRegression
```

Два последних класса проверяют модуль целиком и начинают проходить, когда реализованы все восемь функций:

```bash
uv run pytest -q -k TestImplementationRules    # нет лишних циклов и посторонних импортов
uv run pytest -q -k TestPerformance            # векторная реализация укладывается в пороги с большим запасом
```

Полный прогон на эталонном решении занимает 10–20 секунд.

Проверка стиля:

```bash
uv run ruff check .
uv run ruff format --check .
```
