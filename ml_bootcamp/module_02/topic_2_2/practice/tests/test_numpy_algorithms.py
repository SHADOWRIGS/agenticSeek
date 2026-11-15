import ast
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.special import softmax as scipy_softmax
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, load_iris, load_wine, make_blobs, make_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))

import numpy_algorithms as na  # noqa: E402

LOADERS = {"iris": load_iris, "wine": load_wine, "digits": load_digits}
STRICT_FP = {"over": "raise", "invalid": "raise", "divide": "raise"}


def load_split(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Стандартизированные обучающая и тестовая части встроенного датасета (тест — четверть выборки)."""
    X, y = LOADERS[name](return_X_y=True)
    perm = np.random.default_rng(0).permutation(len(y))
    n_test = len(y) // 4
    test, train = perm[:n_test], perm[n_test:]
    scaler = StandardScaler().fit(X[train])
    return scaler.transform(X[train]), y[train], scaler.transform(X[test]), y[test]


def cross_entropy(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray) -> float:
    log_p = np.log(scipy_softmax(X @ W + b, axis=1))
    return float(-log_p[np.arange(len(y)), y].mean())


class TestStandardize:
    def test_matches_standard_scaler(self):
        X, _ = load_wine(return_X_y=True)
        Z, mean, std = na.standardize(X)
        scaler = StandardScaler().fit(X)
        assert Z.shape == X.shape
        assert mean.shape == std.shape == (X.shape[1],)
        np.testing.assert_allclose(Z, scaler.transform(X), atol=1e-10)
        np.testing.assert_allclose(mean, scaler.mean_)
        np.testing.assert_allclose(std, scaler.scale_)

    def test_uses_given_statistics(self):
        X, _ = load_wine(return_X_y=True)
        train, test = X[:120], X[120:]
        _, mean, std = na.standardize(train)
        Z_test, mean_out, std_out = na.standardize(test, mean, std)
        np.testing.assert_allclose(Z_test, StandardScaler().fit(train).transform(test), atol=1e-10)
        np.testing.assert_allclose(mean_out, mean)
        np.testing.assert_allclose(std_out, std)

    def test_constant_column(self):
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        with np.errstate(**STRICT_FP):
            Z, mean, std = na.standardize(X)
        np.testing.assert_allclose(Z[:, 1], 0.0)
        np.testing.assert_allclose(mean, [2.0, 5.0])
        np.testing.assert_allclose(std, [np.sqrt(2.0 / 3.0), 1.0])

    def test_integer_input_is_not_modified(self):
        X = np.arange(12).reshape(4, 3)
        original = X.copy()
        Z, _, _ = na.standardize(X)
        assert Z.dtype == np.float64
        np.testing.assert_array_equal(X, original)
        np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(Z.std(axis=0), 1.0)


class TestPairwiseSqDistances:
    def test_small_example(self):
        D = na.pairwise_sq_distances(np.array([[0.0, 0.0], [3.0, 4.0]]), np.array([[0.0, 0.0]]))
        np.testing.assert_allclose(D, [[0.0], [25.0]], atol=1e-12)

    def test_matches_cdist(self):
        rng = np.random.default_rng(1)
        A, B = rng.normal(size=(50, 7)), rng.normal(size=(30, 7))
        D = na.pairwise_sq_distances(A, B)
        assert D.shape == (50, 30)
        assert D.dtype == np.float64
        np.testing.assert_allclose(D, cdist(A, B, "sqeuclidean"), atol=1e-9)

    def test_self_distances_are_non_negative(self):
        # На данных большого масштаба разложение через скалярные произведения даёт значения чуть ниже нуля.
        P = np.random.default_rng(2).random((300, 20)) * 1000.0
        D = na.pairwise_sq_distances(P, P)
        assert (D >= 0.0).all()
        with np.errstate(**STRICT_FP):
            assert np.isfinite(np.sqrt(D)).all()
        assert np.diag(D).max() < 1e-4

    def test_no_three_dimensional_temporary(self):
        rng = np.random.default_rng(3)
        A, B = rng.random((1000, 64)), rng.random((500, 64))
        na.pairwise_sq_distances(A[:10], B[:10])
        tracemalloc.start()
        na.pairwise_sq_distances(A, B)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # Результат занимает 4 МБ; массив (n, m, d) занял бы 256 МБ.
        assert peak < 64 * 2**20, f"пиковая память {peak / 2**20:.0f} МБ"


class TestKnnPredict:
    def test_small_example(self):
        X_train = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
        y_train = np.array([0, 0, 0, 1, 1, 1])
        pred = na.knn_predict(X_train, y_train, np.array([[1.5], [10.5], [5.0]]), k=3)
        np.testing.assert_array_equal(pred, [0, 1, 0])

    @pytest.mark.parametrize("name", ["iris", "wine", "digits"])
    @pytest.mark.parametrize("k", [1, 5])
    def test_matches_sklearn(self, name, k):
        X_train, y_train, X_test, _ = load_split(name)
        pred = na.knn_predict(X_train, y_train, X_test, k)
        expected = KNeighborsClassifier(n_neighbors=k, algorithm="brute").fit(X_train, y_train).predict(X_test)
        assert pred.shape == expected.shape
        # Одно расхождение на сотню объектов допустимо: при равных расстояниях соседи могут выбираться по-разному.
        assert (pred != expected).sum() <= max(1, len(expected) // 100)

    def test_arbitrary_labels_and_dtype(self):
        X, y = make_blobs(n_samples=300, centers=3, n_features=4, random_state=0)
        labels = np.array([30, 10, 20], dtype=np.int32)[y]
        pred = na.knn_predict(X[:200], labels[:200], X[200:], k=5)
        expected = KNeighborsClassifier(n_neighbors=5, algorithm="brute").fit(X[:200], labels[:200]).predict(X[200:])
        assert pred.dtype == np.int32
        np.testing.assert_array_equal(pred, expected)

    def test_vote_tie_goes_to_smallest_label(self):
        X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
        X_test = np.array([[1.5]])
        assert na.knn_predict(X_train, np.array([1, 0, 1, 0]), X_test, k=2)[0] == 0
        assert na.knn_predict(X_train, np.array([0, 1, 0, 1]), X_test, k=2)[0] == 0
        assert na.knn_predict(X_train, np.array([7, 7, 4, 4]), X_test, k=4)[0] == 4

    def test_k_equal_to_train_size(self):
        X_train = np.array([[0.0], [1.0], [100.0]])
        pred = na.knn_predict(X_train, np.array([0, 0, 1]), np.array([[99.0], [0.5]]), k=3)
        np.testing.assert_array_equal(pred, [0, 0])


class TestKmeans:
    def test_small_example(self):
        X = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]])
        centers, labels, inertia, n_iter = na.kmeans(X, np.array([[0.0, 0.0], [10.0, 0.0]]))
        np.testing.assert_allclose(centers, [[0.0, 0.5], [10.0, 0.5]])
        np.testing.assert_array_equal(labels, [0, 0, 1, 1])
        assert inertia == pytest.approx(1.0)
        assert n_iter == 2

    def test_output_types_and_shapes(self):
        X, _ = make_blobs(n_samples=200, centers=3, n_features=4, random_state=1)
        init = X[:3].copy()
        centers, labels, inertia, n_iter = na.kmeans(X, init)
        assert centers.shape == (3, 4)
        assert labels.shape == (200,)
        assert np.issubdtype(labels.dtype, np.integer)
        assert isinstance(inertia, float)
        assert isinstance(n_iter, int)
        np.testing.assert_array_equal(init, X[:3])

    def test_matches_sklearn_on_blobs(self):
        X, _ = make_blobs(n_samples=600, centers=4, n_features=5, random_state=3)
        init = X[np.random.default_rng(5).choice(len(X), 4, replace=False)]
        centers, labels, inertia, _ = na.kmeans(X, init, max_iter=300, tol=0.0)
        reference = KMeans(n_clusters=4, init=init, n_init=1, algorithm="lloyd", tol=0, max_iter=300).fit(X)
        np.testing.assert_allclose(centers, reference.cluster_centers_, atol=1e-6)
        np.testing.assert_array_equal(labels, reference.labels_)
        assert inertia == pytest.approx(reference.inertia_, rel=1e-8)

    @pytest.mark.parametrize("name", ["iris", "wine"])
    def test_matches_sklearn_on_datasets(self, name):
        X, y, _, _ = load_split(name)
        k = len(np.unique(y))
        init = X[np.random.default_rng(1).choice(len(X), k, replace=False)]
        centers, labels, inertia, _ = na.kmeans(X, init, max_iter=300, tol=0.0)
        reference = KMeans(n_clusters=k, init=init, n_init=1, algorithm="lloyd", tol=0, max_iter=300).fit(X)
        np.testing.assert_allclose(centers, reference.cluster_centers_, atol=1e-6)
        assert inertia == pytest.approx(reference.inertia_, rel=1e-8)

    def test_single_iteration(self):
        X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=7)
        init = X[:3].copy()
        centers, labels, inertia, n_iter = na.kmeans(X, init, max_iter=1)
        first_labels = cdist(X, init).argmin(axis=1)
        expected = np.array([X[first_labels == j].mean(axis=0) for j in range(3)])
        assert n_iter == 1
        np.testing.assert_allclose(centers, expected)
        # Метки и инерция относятся к итоговым центрам, а не к начальным.
        distances = cdist(X, centers, "sqeuclidean")
        np.testing.assert_array_equal(labels, distances.argmin(axis=1))
        assert inertia == pytest.approx(distances.min(axis=1).sum())

    def test_empty_cluster_keeps_its_center(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0]])
        init = np.array([[0.0, 0.0], [1.0, 0.0], [100.0, 100.0]])
        with np.errstate(**STRICT_FP):
            centers, labels, inertia, _ = na.kmeans(X, init)
        np.testing.assert_allclose(centers, init)
        np.testing.assert_array_equal(labels, [0, 1])
        assert inertia == pytest.approx(0.0, abs=1e-12)


def regression_data() -> tuple[np.ndarray, np.ndarray]:
    return make_regression(n_samples=300, n_features=12, n_informative=6, noise=5.0, bias=100.0, random_state=0)


class TestLinearRegression:
    def test_matches_sklearn_ols(self):
        X, y = regression_data()
        coef, intercept = na.linear_regression(X, y)
        reference = LinearRegression().fit(X, y)
        assert coef.shape == (12,)
        assert np.ndim(intercept) == 0
        np.testing.assert_allclose(coef, reference.coef_, atol=1e-8)
        assert intercept == pytest.approx(reference.intercept_)

    @pytest.mark.parametrize("alpha", [0.1, 10.0, 1000.0])
    def test_matches_sklearn_ridge(self, alpha):
        X, y = regression_data()
        coef, intercept = na.linear_regression(X, y, alpha=alpha)
        reference = Ridge(alpha=alpha).fit(X, y)
        np.testing.assert_allclose(coef, reference.coef_, atol=1e-8)
        assert intercept == pytest.approx(reference.intercept_)

    def test_intercept_is_not_regularized(self):
        X, y = regression_data()
        coef, intercept = na.linear_regression(X, y, alpha=1e12)
        np.testing.assert_allclose(coef, 0.0, atol=1e-6)
        assert intercept == pytest.approx(y.mean(), rel=1e-6)

    def test_collinear_columns(self):
        X, y = regression_data()
        X_dup = np.column_stack([X, X[:, 0]])
        coef, intercept = na.linear_regression(X_dup, y)
        reference = LinearRegression().fit(X_dup, y)
        assert np.isfinite(coef).all()
        np.testing.assert_allclose(X_dup @ coef + intercept, reference.predict(X_dup), atol=1e-6)


class TestPca:
    @pytest.mark.parametrize("name", ["iris", "wine", "digits"])
    def test_matches_sklearn(self, name):
        X, _ = LOADERS[name](return_X_y=True)
        components, explained_variance, transformed = na.pca(X, 3)
        reference = PCA(n_components=3, svd_solver="full").fit(X)
        assert components.shape == (3, X.shape[1])
        assert explained_variance.shape == (3,)
        assert transformed.shape == (len(X), 3)
        # Сравнение по модулю: соглашение о знаке компонент проверяется отдельным тестом.
        np.testing.assert_allclose(np.abs(components), np.abs(reference.components_), atol=1e-8)
        np.testing.assert_allclose(explained_variance, reference.explained_variance_)
        np.testing.assert_allclose(np.abs(transformed), np.abs(reference.transform(X)), atol=1e-8)

    def test_sign_convention(self):
        X, _ = load_wine(return_X_y=True)
        components, _, _ = na.pca(X, 5)
        pivot = np.abs(components).argmax(axis=1)
        assert (components[np.arange(5), pivot] > 0).all()

    def test_components_are_orthonormal(self):
        X, _ = load_digits(return_X_y=True)
        components, _, _ = na.pca(X, 10)
        np.testing.assert_allclose(components @ components.T, np.eye(10), atol=1e-10)

    def test_projection_is_consistent(self):
        X, _ = load_iris(return_X_y=True)
        original = X.copy()
        components, explained_variance, transformed = na.pca(X, 2)
        np.testing.assert_allclose(transformed, (X - X.mean(axis=0)) @ components.T, atol=1e-10)
        np.testing.assert_allclose(transformed.var(axis=0, ddof=1), explained_variance)
        assert explained_variance[0] >= explained_variance[1]
        np.testing.assert_array_equal(X, original)


class TestSoftmax:
    def test_matches_scipy(self):
        Z = np.random.default_rng(4).normal(size=(20, 5)) * 3.0
        P = na.softmax(Z)
        assert P.shape == Z.shape
        np.testing.assert_allclose(P, scipy_softmax(Z, axis=1), atol=1e-12)
        np.testing.assert_allclose(P.sum(axis=1), 1.0)

    def test_large_values(self):
        Z = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0], [1e4, -1e4, 0.0]])
        with np.errstate(**STRICT_FP):
            P = na.softmax(Z)
        assert np.isfinite(P).all()
        np.testing.assert_allclose(P, scipy_softmax(Z, axis=1), atol=1e-12)
        np.testing.assert_allclose(P[0], [0.09003057, 0.24472847, 0.66524096], atol=1e-8)

    def test_shift_invariance(self):
        Z = np.random.default_rng(5).normal(size=(10, 4))
        np.testing.assert_allclose(na.softmax(Z + 123.0), na.softmax(Z), atol=1e-12)

    def test_input_is_not_modified(self):
        Z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        original = Z.copy()
        P = na.softmax(Z)
        np.testing.assert_array_equal(Z, original)
        np.testing.assert_allclose(P[1], 1.0 / 3.0)


class TestFitSoftmaxRegression:
    def test_zero_iterations(self):
        X, y, _, _ = load_split("iris")
        W, b = na.fit_softmax_regression(X, y, 3, n_iter=0)
        assert W.shape == (4, 3)
        assert b.shape == (3,)
        assert not W.any() and not b.any()

    def test_single_step_matches_formula(self):
        X, y, _, _ = load_split("wine")
        W, b = na.fit_softmax_regression(X, y, 3, lr=0.5, n_iter=1, l2=0.1)
        # При нулевых параметрах все вероятности равны 1/3, а слагаемое регуляризации равно нулю.
        diff = np.full((len(y), 3), 1.0 / 3.0) - np.eye(3)[y]
        np.testing.assert_allclose(W, -0.5 * X.T @ diff / len(y), atol=1e-12)
        np.testing.assert_allclose(b, -0.5 * diff.mean(axis=0), atol=1e-12)

    def test_loss_decreases(self):
        X, y, _, _ = load_split("digits")
        losses = [cross_entropy(X, y, *na.fit_softmax_regression(X, y, 10, lr=0.5, n_iter=n)) for n in (0, 10, 100)]
        assert losses[0] == pytest.approx(np.log(10))
        assert losses[0] > losses[1] > losses[2]
        assert losses[2] < 0.2

    def test_regularization_shrinks_weights(self):
        X, y, _, _ = load_split("wine")
        W_weak, _ = na.fit_softmax_regression(X, y, 3, lr=0.5, n_iter=300, l2=0.001)
        W_strong, _ = na.fit_softmax_regression(X, y, 3, lr=0.5, n_iter=300, l2=1.0)
        assert np.abs(W_strong).sum() < 0.5 * np.abs(W_weak).sum()

    @pytest.mark.parametrize("name", ["iris", "wine"])
    def test_matches_sklearn(self, name):
        X_train, y_train, X_test, y_test = load_split(name)
        l2 = 0.01
        W, b = na.fit_softmax_regression(X_train, y_train, 3, lr=0.5, n_iter=3000, l2=l2)
        # sklearn минимизирует сумму потерь + ||W||^2 / (2C), отсюда C = 1 / (l2 * n).
        reference = LogisticRegression(C=1.0 / (l2 * len(y_train)), max_iter=10_000, tol=1e-10)
        reference.fit(X_train, y_train)
        np.testing.assert_allclose(W.T, reference.coef_, atol=1e-2)
        np.testing.assert_allclose(b, reference.intercept_, atol=1e-2)
        probabilities = scipy_softmax(X_test @ W + b, axis=1)
        np.testing.assert_allclose(probabilities, reference.predict_proba(X_test), atol=5e-3)
        assert (probabilities.argmax(axis=1) == y_test).mean() >= 0.9


LOOP_NODES = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
ALLOWED_LOOPS = {"kmeans": 1, "fit_softmax_regression": 1}
FORBIDDEN_NAMES = {"vectorize", "apply_along_axis", "frompyfunc", "fromiter", "nditer", "ndenumerate"}
FORBIDDEN_MODULES = {"sklearn", "scipy", "pandas", "numba"}


def call_every_function() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.5], [4.0, 4.0], [5.0, 5.5]])
    y = np.array([0, 0, 1, 1])
    na.standardize(X)
    na.pairwise_sq_distances(X, X)
    na.knn_predict(X, y, X, k=1)
    na.kmeans(X, X[[0, 3]])
    na.linear_regression(X, y.astype(float), alpha=1.0)
    na.pca(X, 1)
    na.softmax(X)
    na.fit_softmax_regression(X, y, 2, n_iter=2)


def parse_implemented_module() -> ast.Module:
    """Дерево разбора модуля; на незаполненной заготовке завершается NotImplementedError."""
    call_every_function()
    return ast.parse((PRACTICE_DIR / "numpy_algorithms.py").read_text(encoding="utf-8"))


class TestImplementationRules:
    def test_loops_only_over_iterations(self):
        tree = parse_implemented_module()
        violations = {}
        for node in tree.body:
            name = node.name if isinstance(node, ast.FunctionDef | ast.ClassDef) else "<код вне функций>"
            loops = sum(isinstance(child, LOOP_NODES) for child in ast.walk(node))
            if loops > ALLOWED_LOOPS.get(name, 0):
                violations[name] = loops
        assert not violations, f"лишние циклы (функция: число циклов): {violations}"

    def test_no_hidden_loops(self):
        tree = parse_implemented_module()
        used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        used |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert not used & FORBIDDEN_NAMES, f"скрытые циклы Python: {sorted(used & FORBIDDEN_NAMES)}"

    def test_only_numpy_is_imported(self):
        tree = parse_implemented_module()
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & FORBIDDEN_MODULES, f"запрещённые импорты: {sorted(imported & FORBIDDEN_MODULES)}"


class TestPerformance:
    """Пороги в десятки раз выше времени векторной реализации; цикл по объектам в них не укладывается.

    BLAS ограничен одним потоком, чтобы результат не зависел от загрузки остальных ядер.
    """

    def test_pairwise_sq_distances_speed(self):
        rng = np.random.default_rng(6)
        A, B = rng.random((1500, 32)), rng.random((1500, 32))
        na.pairwise_sq_distances(A[:10], B[:10])
        with threadpool_limits(limits=1):
            start = time.perf_counter()
            D = na.pairwise_sq_distances(A, B)
            elapsed = time.perf_counter() - start
        assert D.shape == (1500, 1500)
        assert elapsed < 1.0, f"{elapsed:.2f} с: похоже на цикл по объектам"

    def test_fit_softmax_regression_speed(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(20_000, 20))
        y = rng.integers(0, 5, size=20_000)
        na.fit_softmax_regression(X[:100], y[:100], 5, n_iter=2)
        with threadpool_limits(limits=1):
            start = time.perf_counter()
            W, _ = na.fit_softmax_regression(X, y, 5, n_iter=30)
            elapsed = time.perf_counter() - start
        assert W.shape == (20, 5)
        assert elapsed < 2.0, f"{elapsed:.2f} с: похоже на цикл по объектам"
