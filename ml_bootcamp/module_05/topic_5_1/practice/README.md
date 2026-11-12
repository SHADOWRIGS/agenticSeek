# Практика 5.1.4: материалы

Постановка задачи: [lesson_5_1_4.md](../lesson_5_1_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `my_nn.py` | заготовка: `numerical_gradient`, `Linear`, `ReLU`, `Tanh`, `softmax_cross_entropy`, `Sequential`, `SGD`, `Adam`, `train` |
| `tests/test_my_nn.py` | 59 тестов: формы, ручные примеры, численная проверка градиентов, оптимизаторы, обучение, скорость |

## Запуск

```bash
uv add numpy
uv add --dev pytest scikit-learn
uv run pytest -q          # на заготовке падают все тесты, кроме трёх
```

PyTorch для тестов не нужен. В `my_nn.py` разрешены только NumPy и стандартная библиотека.

По шагам:

```bash
uv run pytest -q -k TestNumericalGradient
uv run pytest -q -k TestSoftmaxCrossEntropy
uv run pytest -q -k TestLinear
uv run pytest -q -k TestActivations
uv run pytest -q -k TestSequential
uv run pytest -q -k TestOptimizers
uv run pytest -q -k TestTrain
uv run pytest -q -k TestDigits
uv run pytest -q -k TestEngineering
```

Тесты слоёв и сети используют вашу функцию `numerical_gradient`, поэтому начинать нужно с неё.

Тест `test_speed` отсекает решения с циклом по объектам батча в любом из слоёв или в функции потерь.
Пока решение медленное, его удобно исключать:

```bash
uv run pytest -q -k "not speed"
```

Итоговая проверка — без предупреждений NumPy:

```bash
uv run pytest -q -W error
```
