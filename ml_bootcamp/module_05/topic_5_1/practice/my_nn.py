"""Практика 5.1.4: нейронная сеть с нуля на NumPy.

Реализуйте функции и методы, в которых стоит raise NotImplementedError. Ожидаемое поведение описано
в docstring и проверяется тестами: uv run pytest -q

Общие соглашения:

- батч — двумерный массив формы (n, d): строка — объект; все вычисления в float64;
- слой — объект с методами forward(x) и backward(grad_out) и атрибутами params и grads:
  списками массивов одинаковой длины (у слоя без параметров оба списка пустые);
- forward запоминает всё, что понадобится обратному проходу; входной массив при этом не изменяется;
- backward получает градиент потерь по выходу слоя, записывает градиенты по параметрам в grads
  (перезаписывает, а не накапливает) и возвращает градиент потерь по входу слоя;
- циклов по объектам батча быть не должно — только по слоям, параметрам, батчам и эпохам.

Использовать torch и scikit-learn в этом модуле нельзя.
"""

from collections.abc import Callable

import numpy as np


def numerical_gradient(f: Callable[[], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Градиент скалярной функции по массиву x методом центральных разностей.

    f — функция без аргументов; она возвращает число и читает массив x при каждом вызове.
    Для каждого элемента x: прибавить eps, вычислить f, вычесть eps, вычислить f, вернуть элементу
    исходное значение. Оценка производной — (f_plus - f_minus) / (2 * eps).

    Возвращает массив той же формы, что и x (x может быть любой размерности). После вызова x содержит
    исходные значения.
    """
    raise NotImplementedError


class Linear:
    """Полносвязный слой: out = x @ W + b.

    W — матрица формы (n_in, n_out), b — вектор формы (n_out,).
    Инициализация Хе: элементы W — из нормального распределения со средним 0 и стандартным отклонением
    sqrt(2 / n_in), полученные из генератора rng (если rng не задан — np.random.default_rng()); b — нули.

    Атрибуты: W, b — параметры; dW, db — градиенты потерь по ним (тех же форм, до первого backward — нули).
    params возвращает [W, b], grads — [dW, db]: сами массивы, а не копии.
    """

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator | None = None) -> None:
        raise NotImplementedError

    @property
    def params(self) -> list[np.ndarray]:
        raise NotImplementedError

    @property
    def grads(self) -> list[np.ndarray]:
        raise NotImplementedError

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x формы (n, n_in) -> результат формы (n, n_out)."""
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """grad_out формы (n, n_out) -> градиент по входу формы (n, n_in).

        Записывает dW = x.T @ grad_out и db = сумма grad_out по объектам, где x — вход последнего forward.
        """
        raise NotImplementedError


class ReLU:
    """Поэлементно max(x, 0). Производная в точке 0 считается равной 0. Параметров нет."""

    params: list[np.ndarray] = []
    grads: list[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Tanh:
    """Поэлементно гиперболический тангенс. Параметров нет."""

    params: list[np.ndarray] = []
    grads: list[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Средняя по батчу кросс-энтропия после softmax и её градиент по logits.

    logits — форма (n, K); y — целые номера классов от 0 до K - 1, форма (n,).
    Возвращает (loss, grad): loss — число типа float, grad — массив формы (n, K), равный
    (softmax(logits) - one_hot(y)) / n.

    Вычисление должно быть численно устойчивым: логиты порядка ±1000 не должны давать nan, inf
    и предупреждений NumPy. Массив logits не изменяется.
    """
    raise NotImplementedError


class Sequential:
    """Последовательность слоёв.

    layers — список слоёв; хранится в атрибуте layers в том же порядке.
    """

    def __init__(self, layers: list) -> None:
        raise NotImplementedError

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Пропускает x через все слои по порядку."""
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Пропускает градиент через слои в обратном порядке; возвращает градиент по входу сети."""
        raise NotImplementedError

    def parameters(self) -> list[np.ndarray]:
        """Параметры всех слоёв одним списком, в порядке слоёв: сами массивы, а не копии."""
        raise NotImplementedError

    def gradients(self) -> list[np.ndarray]:
        """Градиенты в том же порядке, что и parameters(). Вызывается после backward."""
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Номер класса с наибольшим выходом сети для каждого объекта, форма (n,)."""
        raise NotImplementedError


class SGD:
    """Стохастический градиентный спуск с импульсом (в формулировке PyTorch).

    Для каждого параметра p с градиентом g:
        v = momentum * v + g        (v в начале — нули)
        p = p - lr * v
    Параметры обновляются на месте: массивы из списка params остаются теми же объектами.
    """

    def __init__(self, params: list[np.ndarray], lr: float = 0.01, momentum: float = 0.0) -> None:
        raise NotImplementedError

    def step(self, grads: list[np.ndarray]) -> None:
        """Один шаг; grads — градиенты в том же порядке, что и params."""
        raise NotImplementedError


class Adam:
    """Adam с поправкой смещения (Kingma, Ba).

    На шаге t = 1, 2, ... для каждого параметра p с градиентом g:
        m = beta1 * m + (1 - beta1) * g          (m и v в начале — нули)
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        p = p - lr * m_hat / (sqrt(v_hat) + eps)
    Параметры обновляются на месте.
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        raise NotImplementedError

    def step(self, grads: list[np.ndarray]) -> None:
        """Один шаг; grads — градиенты в том же порядке, что и params."""
        raise NotImplementedError


def train(
    model: Sequential,
    optimizer,
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 10,
    batch_size: int = 32,
    seed: int = 0,
) -> list[float]:
    """Обучение классификатора мини-батчами.

    В начале создаётся генератор np.random.default_rng(seed). В каждой эпохе объекты перемешиваются
    этим генератором и делятся на батчи по batch_size подряд; последний батч может быть неполным
    и тоже используется. Для каждого батча: forward, softmax_cross_entropy, backward,
    optimizer.step(model.gradients()).

    Возвращает список длины n_epochs: средние потери на объект за каждую эпоху — сумма потерь батчей,
    умноженных на размер батча, делённая на число объектов. Потери батча считаются до шага оптимизатора.

    Два вызова с одинаковыми seed, данными и одинаково инициализированными моделями дают одинаковый
    результат.
    """
    raise NotImplementedError
