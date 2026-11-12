import ast
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_digits, make_blobs
from sklearn.model_selection import train_test_split

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))

from my_nn import (  # noqa: E402
    SGD,
    Adam,
    Linear,
    ReLU,
    Sequential,
    Tanh,
    numerical_gradient,
    softmax_cross_entropy,
    train,
)

# Три шага оптимизатора из одной точки: значения проверены на torch.optim
START = [1.0, -2.0, 3.0]
STEP_GRADS = [[0.5, -1.0, 2.0], [-0.3, 0.4, 1.0], [0.1, 0.1, -2.0]]


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.abs(a - b).max() / max(1e-8, np.abs(a).max() + np.abs(b).max()))


def away_from_zero(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Случайные числа с модулем не меньше 0.1: рядом с изломом ReLU численный градиент не определён."""
    x = rng.normal(size=shape)
    return x + 0.1 * np.sign(x)


def check_layer(layer, x: np.ndarray, tol: float = 1e-6) -> None:
    """Сверяет backward слоя с численным градиентом функции sum(forward(x) * r)."""
    r = np.random.default_rng(1).normal(size=layer.forward(x).shape)

    def f() -> float:
        return float((layer.forward(x) * r).sum())

    layer.forward(x)
    grad_x = layer.backward(r)
    analytic = [g.copy() for g in layer.grads]
    assert grad_x.shape == x.shape
    assert rel_error(grad_x, numerical_gradient(f, x)) < tol
    for p, g in zip(layer.params, analytic, strict=True):
        assert g.shape == p.shape
        assert rel_error(g, numerical_gradient(f, p)) < tol


def check_network(model: Sequential, x: np.ndarray, y: np.ndarray, tol: float = 1e-6) -> None:
    # Ненулевые смещения: при b = 0 объект, у которого погасли все нейроны слоя, попадает точно на излом ReLU
    rng = np.random.default_rng(2)
    for p in model.parameters():
        if p.ndim == 1:
            p[:] = rng.normal(size=p.shape)

    def f() -> float:
        return softmax_cross_entropy(model.forward(x), y)[0]

    _, grad = softmax_cross_entropy(model.forward(x), y)
    grad_x = model.backward(grad)
    analytic = [g.copy() for g in model.gradients()]
    assert rel_error(grad_x, numerical_gradient(f, x)) < tol
    for p, g in zip(model.parameters(), analytic, strict=True):
        assert rel_error(g, numerical_gradient(f, p)) < tol


def make_mlp(seed: int, sizes: tuple[int, ...] = (64, 64, 10), activation=ReLU) -> Sequential:
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(Linear(sizes[i], sizes[i + 1], rng))
        if i < len(sizes) - 2:
            layers.append(activation())
    return Sequential(layers)


def run_steps(optimizer_factory) -> np.ndarray:
    w = np.array(START)
    optimizer = optimizer_factory([w])
    trajectory = []
    for g in STEP_GRADS:
        optimizer.step([np.array(g)])
        trajectory.append(w.copy())
    return np.array(trajectory)


class Identity:
    """Слой-шпион: ничего не меняет и записывает первые столбцы всех батчей."""

    params: list[np.ndarray] = []
    grads: list[np.ndarray] = []

    def __init__(self) -> None:
        self.batches: list[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.batches.append(x[:, 0].copy())
        return x

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out


class CountingOptimizer:
    def __init__(self) -> None:
        self.n_steps = 0

    def step(self, grads: list[np.ndarray]) -> None:
        self.n_steps += 1


@pytest.fixture(scope="module")
def digits():
    X, y = load_digits(return_X_y=True)
    return train_test_split(X / 16.0, y, test_size=0.25, random_state=0, stratify=y)


@pytest.fixture(scope="module")
def blobs():
    X, y = make_blobs(n_samples=100, centers=3, n_features=5, random_state=0)
    return X / 10.0, y


class TestNumericalGradient:
    def test_cubic(self):
        x = np.array([1.0, -2.0, 0.5])
        grad = numerical_gradient(lambda: float((x**3).sum()), x)
        assert rel_error(grad, 3 * x**2) < 1e-7

    def test_matrix(self):
        x = np.random.default_rng(0).normal(size=(3, 4))
        grad = numerical_gradient(lambda: float(np.sin(x).sum()), x)
        assert grad.shape == (3, 4)
        assert rel_error(grad, np.cos(x)) < 1e-7

    def test_restores_x(self):
        x = np.random.default_rng(0).normal(size=(2, 3, 2))
        before = x.copy()
        numerical_gradient(lambda: float((x**2).sum()), x)
        np.testing.assert_array_equal(x, before)

    def test_central_differences(self):
        # Для квадратичной функции центральная разность точна при любом eps, односторонняя — нет
        x = np.array([1.0, 2.0])
        grad = numerical_gradient(lambda: float((x**2).sum()), x, eps=0.1)
        np.testing.assert_allclose(grad, [2.0, 4.0], atol=1e-10)


class TestLinear:
    def test_init(self):
        layer = Linear(400, 300, np.random.default_rng(0))
        assert layer.W.shape == (400, 300)
        assert layer.W.dtype == np.float64
        assert layer.b.shape == (300,)
        assert not layer.b.any()
        assert layer.W.std() == pytest.approx(np.sqrt(2 / 400), rel=0.03)
        assert abs(layer.W.mean()) < 0.001

    def test_init_depends_only_on_rng(self):
        a = Linear(5, 3, np.random.default_rng(7))
        b = Linear(5, 3, np.random.default_rng(7))
        c = Linear(5, 3, np.random.default_rng(8))
        np.testing.assert_array_equal(a.W, b.W)
        assert not np.array_equal(a.W, c.W)

    def test_init_without_rng(self):
        assert Linear(5, 3).W.shape == (5, 3)

    def test_params_and_grads(self):
        layer = Linear(5, 3, np.random.default_rng(0))
        assert layer.params[0] is layer.W
        assert layer.params[1] is layer.b
        assert [g.shape for g in layer.grads] == [(5, 3), (3,)]

    def test_forward_manual(self):
        layer = Linear(3, 2, np.random.default_rng(0))
        layer.W[:] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        layer.b[:] = [0.5, -0.5]
        out = layer.forward(np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]]))
        np.testing.assert_allclose(out, [[-3.5, -4.5], [5.5, 7.5]])

    def test_backward_manual(self):
        layer = Linear(3, 2, np.random.default_rng(0))
        layer.W[:] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        layer.forward(np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]]))
        grad_in = layer.backward(np.array([[1.0, 0.0], [0.0, 2.0]]))
        np.testing.assert_allclose(grad_in, [[1.0, 3.0, 5.0], [4.0, 8.0, 12.0]])
        np.testing.assert_allclose(layer.grads[0], [[1.0, 4.0], [0.0, 2.0], [-1.0, 0.0]])
        np.testing.assert_allclose(layer.grads[1], [1.0, 2.0])

    def test_gradients_numerically(self):
        rng = np.random.default_rng(0)
        check_layer(Linear(6, 4, rng), rng.normal(size=(5, 6)))

    def test_backward_overwrites_grads(self):
        rng = np.random.default_rng(0)
        layer = Linear(4, 3, rng)
        x, g = rng.normal(size=(5, 4)), rng.normal(size=(5, 3))
        layer.forward(x)
        layer.backward(g)
        first = [a.copy() for a in layer.grads]
        layer.forward(x)
        layer.backward(g)
        for a, b in zip(first, layer.grads, strict=True):
            np.testing.assert_allclose(a, b)

    def test_single_object_batch(self):
        layer = Linear(4, 3, np.random.default_rng(0))
        assert layer.forward(np.ones((1, 4))).shape == (1, 3)
        assert layer.backward(np.ones((1, 3))).shape == (1, 4)


class TestActivations:
    def test_relu_manual(self):
        layer = ReLU()
        x = np.array([[-1.0, 0.0, 2.0], [3.0, -0.5, 0.0]])
        np.testing.assert_array_equal(layer.forward(x), [[0.0, 0.0, 2.0], [3.0, 0.0, 0.0]])
        grad = layer.backward(np.full((2, 3), 5.0))
        np.testing.assert_array_equal(grad, [[0.0, 0.0, 5.0], [5.0, 0.0, 0.0]])

    def test_relu_does_not_modify_input(self):
        x = np.array([[-1.0, 2.0]])
        ReLU().forward(x)
        np.testing.assert_array_equal(x, [[-1.0, 2.0]])

    def test_relu_gradient_numerically(self):
        check_layer(ReLU(), away_from_zero(np.random.default_rng(0), (5, 7)))

    def test_tanh_manual(self):
        layer = Tanh()
        out = layer.forward(np.array([[0.0, 1.0, -2.0]]))
        np.testing.assert_allclose(out, [[0.0, 0.7615941559557649, -0.9640275800758169]])
        grad = layer.backward(np.array([[2.0, 1.0, 1.0]]))
        np.testing.assert_allclose(grad, [[2.0, 0.41997434161402614, 0.07065082485316443]])

    def test_tanh_gradient_numerically(self):
        check_layer(Tanh(), np.random.default_rng(0).normal(size=(5, 7)))

    @pytest.mark.parametrize("layer_class", [ReLU, Tanh])
    def test_no_parameters(self, layer_class):
        layer = layer_class()
        assert len(layer.params) == 0
        assert len(layer.grads) == 0


class TestSoftmaxCrossEntropy:
    def test_uniform(self):
        y = np.array([0, 3, 9, 3])
        loss, grad = softmax_cross_entropy(np.zeros((4, 10)), y)
        assert isinstance(loss, float)
        assert loss == pytest.approx(np.log(10))
        expected = np.full((4, 10), 0.1)
        expected[np.arange(4), y] -= 1.0
        np.testing.assert_allclose(grad, expected / 4)

    def test_manual(self):
        loss, grad = softmax_cross_entropy(np.array([[0.0, np.log(3.0)]]), np.array([1]))
        assert loss == pytest.approx(-np.log(0.75))
        np.testing.assert_allclose(grad, [[0.25, -0.25]])

    def test_gradient_rows_sum_to_zero(self):
        rng = np.random.default_rng(0)
        _, grad = softmax_cross_entropy(rng.normal(size=(6, 5)), rng.integers(0, 5, 6))
        assert grad.shape == (6, 5)
        np.testing.assert_allclose(grad.sum(axis=1), 0.0, atol=1e-12)

    def test_shift_invariance(self):
        rng = np.random.default_rng(0)
        logits, y = rng.normal(size=(6, 5)), rng.integers(0, 5, 6)
        shifted = logits + rng.normal(size=(6, 1)) * 10
        assert softmax_cross_entropy(shifted, y)[0] == pytest.approx(softmax_cross_entropy(logits, y)[0])

    def test_mean_over_batch(self):
        rng = np.random.default_rng(0)
        logits, y = rng.normal(size=(6, 5)), rng.integers(0, 5, 6)
        loss, grad = softmax_cross_entropy(logits, y)
        loss2, grad2 = softmax_cross_entropy(np.vstack([logits, logits]), np.concatenate([y, y]))
        assert loss2 == pytest.approx(loss)
        np.testing.assert_allclose(grad2[:6], grad / 2)

    @pytest.mark.parametrize(("y", "expected"), [([0, 2], 0.0), ([2, 0], 2000.0), ([1, 1], 1000.0)])
    def test_large_logits(self, y, expected):
        logits = np.array([[1000.0, 0.0, -1000.0], [-1000.0, 0.0, 1000.0]])
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            loss, grad = softmax_cross_entropy(logits, np.array(y))
        assert np.isfinite(loss)
        assert np.isfinite(grad).all()
        assert loss == pytest.approx(expected)

    def test_gradient_numerically(self):
        rng = np.random.default_rng(0)
        logits, y = rng.normal(size=(5, 4)) * 3, rng.integers(0, 4, 5)
        _, grad = softmax_cross_entropy(logits, y)
        numeric = numerical_gradient(lambda: softmax_cross_entropy(logits, y)[0], logits)
        assert rel_error(grad, numeric) < 1e-7

    def test_does_not_modify_logits(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        softmax_cross_entropy(logits, np.array([2]))
        np.testing.assert_array_equal(logits, [[1.0, 2.0, 3.0]])


class TestSequential:
    def test_forward_is_composition(self):
        rng = np.random.default_rng(0)
        first, second = Linear(4, 6, rng), Linear(6, 3, rng)
        model = Sequential([first, Tanh(), second])
        x = rng.normal(size=(5, 4))
        expected = np.tanh(x @ first.W + first.b) @ second.W + second.b
        np.testing.assert_allclose(model.forward(x), expected)
        assert model.layers[0] is first

    def test_parameters_and_gradients(self):
        model = make_mlp(0, (4, 6, 3))
        params = model.parameters()
        assert [p.shape for p in params] == [(4, 6), (6,), (6, 3), (3,)]
        assert params[0] is model.layers[0].W
        assert params[3] is model.layers[2].b
        x = np.random.default_rng(1).normal(size=(5, 4))
        _, grad = softmax_cross_entropy(model.forward(x), np.array([0, 1, 2, 0, 1]))
        model.backward(grad)
        assert [g.shape for g in model.gradients()] == [(4, 6), (6,), (6, 3), (3,)]

    def test_tanh_network_gradients_numerically(self):
        rng = np.random.default_rng(0)
        model = make_mlp(0, (5, 8, 6, 3), activation=Tanh)
        check_network(model, rng.normal(size=(7, 5)), rng.integers(0, 3, 7))

    def test_relu_network_gradients_numerically(self):
        rng = np.random.default_rng(0)
        model = make_mlp(0, (5, 8, 6, 3), activation=ReLU)
        check_network(model, rng.normal(size=(7, 5)), rng.integers(0, 3, 7))

    def test_gradient_is_zero_at_perfect_prediction(self):
        model = make_mlp(0, (2, 3))
        model.layers[0].W[:] = 0.0
        model.layers[0].b[:] = [500.0, 0.0, -500.0]
        _, grad = softmax_cross_entropy(model.forward(np.ones((4, 2))), np.zeros(4, dtype=int))
        model.backward(grad)
        for g in model.gradients():
            np.testing.assert_allclose(g, 0.0, atol=1e-12)

    def test_predict(self):
        model = make_mlp(0, (4, 6, 3))
        x = np.random.default_rng(1).normal(size=(9, 4))
        predicted = model.predict(x)
        assert predicted.shape == (9,)
        assert np.issubdtype(predicted.dtype, np.integer)
        np.testing.assert_array_equal(predicted, model.forward(x).argmax(axis=1))


class TestOptimizers:
    def test_sgd(self):
        expected = [[0.95, -1.9, 2.8], [0.98, -1.94, 2.7], [0.97, -1.95, 2.9]]
        np.testing.assert_allclose(run_steps(lambda p: SGD(p, lr=0.1)), expected, rtol=1e-12)

    def test_sgd_momentum(self):
        expected = [[0.95, -1.9, 2.8], [0.935, -1.85, 2.52], [0.9115, -1.815, 2.468]]
        np.testing.assert_allclose(run_steps(lambda p: SGD(p, lr=0.1, momentum=0.9)), expected, rtol=1e-12)

    def test_adam(self):
        expected = [
            [0.900000002, -1.900000001, 2.9000000005],
            [0.8808501989, -1.8654394167, 2.8067820372],
            [0.8554536806, -1.8447499007, 2.7957037301],
        ]
        np.testing.assert_allclose(run_steps(lambda p: Adam(p, lr=0.1)), expected, rtol=1e-9)

    def test_adam_first_step_does_not_depend_on_gradient_scale(self):
        w = np.zeros(3)
        Adam([w], lr=0.01).step([np.array([1e-4, -1.0, 1e4])])
        np.testing.assert_allclose(w, [-0.01, 0.01, -0.01], rtol=1e-3)

    @pytest.mark.parametrize("factory", [lambda p: SGD(p, lr=0.1, momentum=0.9), lambda p: Adam(p, lr=0.1)])
    def test_updates_in_place(self, factory):
        params = [np.ones((2, 3)), np.ones(3)]
        originals = list(params)
        optimizer = factory(params)
        optimizer.step([np.ones((2, 3)), np.ones(3)])
        optimizer.step([np.ones((2, 3)), np.ones(3)])
        for p, original in zip(params, originals, strict=True):
            assert p is original
            assert (p < 1.0).all()

    def test_state_is_separate_for_each_parameter(self):
        a, b = np.array([1.0]), np.array([1.0])
        optimizer = SGD([a, b], lr=0.1, momentum=0.9)
        optimizer.step([np.array([1.0]), np.array([0.0])])
        optimizer.step([np.array([0.0]), np.array([0.0])])
        assert a[0] == pytest.approx(1.0 - 0.1 - 0.09)
        assert b[0] == 1.0

    def test_does_not_modify_gradients(self):
        g = np.array([1.0, 2.0])
        for factory in (lambda p: SGD(p, lr=0.1, momentum=0.9), lambda p: Adam(p)):
            factory([np.zeros(2)]).step([g])
        np.testing.assert_array_equal(g, [1.0, 2.0])


class TestTrain:
    def test_history(self, blobs):
        X, y = blobs
        model = make_mlp(0, (5, 16, 3))
        history = train(model, SGD(model.parameters(), lr=0.1, momentum=0.9), X, y, n_epochs=20, batch_size=16)
        assert len(history) == 20
        assert all(isinstance(value, float) for value in history)
        assert history[-1] < 0.5 * history[0]

    def test_reproducible(self, blobs):
        X, y = blobs
        results = []
        for seed in (3, 3, 4):
            model = make_mlp(0, (5, 16, 3))
            history = train(model, Adam(model.parameters(), lr=0.01), X, y, n_epochs=5, batch_size=16, seed=seed)
            results.append((history, model.parameters()))
        assert results[0][0] == results[1][0]
        for a, b in zip(results[0][1], results[1][1], strict=True):
            np.testing.assert_array_equal(a, b)
        assert results[0][0] != results[2][0]

    def test_history_is_mean_over_objects(self, blobs):
        # При нулевом шаге веса не меняются, и средние потери за эпоху равны потерям на всей выборке
        X, y = blobs
        model = make_mlp(0, (5, 16, 3))
        expected = softmax_cross_entropy(model.forward(X), y)[0]
        history = train(model, SGD(model.parameters(), lr=0.0), X, y, n_epochs=2, batch_size=32)
        assert history == pytest.approx([expected, expected], rel=1e-10)

    @pytest.mark.parametrize(("batch_size", "steps_per_epoch"), [(32, 4), (50, 2), (100, 1), (1000, 1), (1, 100)])
    def test_number_of_steps(self, blobs, batch_size, steps_per_epoch):
        X, y = blobs
        optimizer = CountingOptimizer()
        train(make_mlp(0, (5, 16, 3)), optimizer, X, y, n_epochs=3, batch_size=batch_size)
        assert optimizer.n_steps == 3 * steps_per_epoch

    def test_epoch_is_shuffled_partition(self, blobs):
        X, y = blobs
        X = np.column_stack([np.arange(100.0), X])
        spy = Identity()
        model = Sequential([spy, Linear(6, 3, np.random.default_rng(0))])
        train(model, CountingOptimizer(), X, y, n_epochs=2, batch_size=32, seed=0)
        assert [len(batch) for batch in spy.batches] == [32, 32, 32, 4] * 2
        first, second = np.concatenate(spy.batches[:4]), np.concatenate(spy.batches[4:])
        np.testing.assert_array_equal(np.sort(first), np.arange(100))
        np.testing.assert_array_equal(np.sort(second), np.arange(100))
        assert not np.array_equal(first, np.arange(100))
        assert not np.array_equal(first, second)

    def test_full_batch_epoch_is_one_gradient_step(self, blobs):
        X, y = blobs
        model, twin = make_mlp(0, (5, 16, 3)), make_mlp(0, (5, 16, 3))
        _, grad = softmax_cross_entropy(twin.forward(X), y)
        twin.backward(grad)
        expected = [p - 0.5 * g for p, g in zip(twin.parameters(), twin.gradients(), strict=True)]
        train(model, SGD(model.parameters(), lr=0.5), X, y, n_epochs=1, batch_size=100)
        for p, e in zip(model.parameters(), expected, strict=True):
            np.testing.assert_allclose(p, e, rtol=1e-9, atol=1e-12)


class TestDigits:
    @pytest.mark.parametrize(
        "factory",
        [lambda p: Adam(p, lr=0.003), lambda p: SGD(p, lr=0.05, momentum=0.9)],
        ids=["adam", "sgd_momentum"],
    )
    def test_accuracy(self, digits, factory):
        X_train, X_test, y_train, y_test = digits
        model = make_mlp(0)
        start = time.perf_counter()
        history = train(model, factory(model.parameters()), X_train, y_train, n_epochs=30, batch_size=32, seed=0)
        elapsed = time.perf_counter() - start
        assert history[-1] < 0.15
        assert (model.predict(X_test) == y_test).mean() >= 0.95
        assert elapsed < 20

    def test_overfits_small_sample(self, digits):
        # Сеть без ошибок в градиентах обязана запомнить 64 объекта
        X_train, _, y_train, _ = digits
        X, y = X_train[:64], y_train[:64]
        model = make_mlp(0)
        history = train(model, Adam(model.parameters(), lr=0.01), X, y, n_epochs=100, batch_size=64)
        assert history[-1] < 0.01
        assert (model.predict(X) == y).all()


class TestEngineering:
    def test_speed(self):
        # Проход по 200 000 объектов сравнивается с самым дешёвым циклом по объектам: одно умножение
        # на объект. Решение с циклом по объектам в любом из слоёв или в функции потерь не уложится.
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(200_000, 4)), rng.integers(0, 3, 200_000)
        model = make_mlp(0, (4, 8, 3))
        W = model.layers[0].W.copy()
        timings, loop_timings = [], []
        for _ in range(3):
            start = time.perf_counter()
            _, grad = softmax_cross_entropy(model.forward(X), y)
            model.backward(grad)
            timings.append(time.perf_counter() - start)
        for _ in range(2):
            start = time.perf_counter()
            for row in X:
                row @ W
            loop_timings.append(time.perf_counter() - start)
        assert min(timings) < 0.7 * min(loop_timings)

    def test_no_torch_or_sklearn_in_solution(self):
        tree = ast.parse((PRACTICE_DIR / "my_nn.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & {"torch", "sklearn", "scipy", "autograd", "jax", "tensorflow"}
