import sys
import time
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))

from my_tree import MyDecisionTreeClassifier, best_split, impurity  # noqa: E402

# Двенадцать клиентов из урока 4.1.1: признаки annual, calls, tenure
CLIENTS_X = np.array(
    [[0, 4, 2], [0, 0, 30], [1, 1, 24], [0, 3, 5], [1, 0, 40], [0, 5, 3], [1, 2, 12], [0, 1, 8], [0, 3, 20], [1, 4, 6]]
    + [[0, 0, 15], [1, 0, 50]],
    dtype=float,
)
CLIENTS_Y = np.array([1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0])

DATASETS = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer}


def split_dataset(name: str):
    X, y = DATASETS[name](return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


def all_nodes(node):
    yield node
    if not node.is_leaf:
        yield from all_nodes(node.left)
        yield from all_nodes(node.right)


class TestImpurity:
    @pytest.mark.parametrize(
        ("counts", "expected"),
        [([5, 5], 0.5), ([10, 0], 0.0), ([8, 2], 0.32), ([1, 1, 1], 2 / 3), ([0, 0], 0.0)],
    )
    def test_gini(self, counts, expected):
        assert impurity(np.array(counts), "gini") == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("counts", "expected"),
        [([5, 5], 1.0), ([10, 0], 0.0), ([8, 2], 0.7219281), ([1, 1, 1, 1], 2.0), ([0, 0], 0.0)],
    )
    def test_entropy(self, counts, expected):
        result = impurity(np.array(counts), "entropy")
        assert not np.isnan(result)
        assert result == pytest.approx(expected)

    def test_default_is_gini(self):
        assert impurity(np.array([3, 1])) == pytest.approx(0.375)

    def test_unknown_criterion(self):
        with pytest.raises(ValueError):
            impurity(np.array([1, 1]), "unknown")


class TestBestSplit:
    def test_obvious_split(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        feature, threshold, gain = best_split(X, np.array([0, 0, 1, 1]), n_classes=2)
        assert (feature, threshold) == (0, 2.5)
        assert gain == pytest.approx(0.5)

    def test_threshold_is_midpoint(self):
        feature, threshold, _ = best_split(np.array([[0.0], [10.0]]), np.array([0, 1]), n_classes=2)
        assert (feature, threshold) == (0, 5.0)

    def test_lesson_example(self):
        # Таблица из урока 4.1.1: calls <= 2.5 и tenure <= 10 дают одинаковый выигрыш, выбирается calls
        feature, threshold, gain = best_split(CLIENTS_X, CLIENTS_Y, n_classes=2)
        assert (feature, threshold) == (1, 2.5)
        assert gain == pytest.approx(0.5 - 1 / 7)

    def test_equal_gain_prefers_smaller_feature_index(self):
        column = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.6])
        X = np.column_stack([column, column])
        feature, _, _ = best_split(X, np.array([1, 0, 1, 0, 1, 0]), n_classes=2)
        assert feature == 0

    def test_equal_gain_prefers_smaller_threshold(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        feature, threshold, gain = best_split(X, np.array([0, 1, 0, 1]), n_classes=2)
        assert (feature, threshold) == (0, 1.5)
        assert gain == pytest.approx(1 / 6)

    def test_min_samples_leaf(self):
        X = np.arange(1.0, 7.0).reshape(-1, 1)
        y = np.array([0, 1, 1, 1, 1, 1])
        assert best_split(X, y, n_classes=2)[1] == 1.5
        feature, threshold, gain = best_split(X, y, n_classes=2, min_samples_leaf=2)
        assert (feature, threshold) == (0, 2.5)
        assert gain == pytest.approx(1 / 9)
        assert best_split(X, y, n_classes=2, min_samples_leaf=4) is None

    def test_no_split_for_identical_objects(self):
        X = np.ones((6, 3))
        assert best_split(X, np.array([0, 1, 0, 1, 0, 1]), n_classes=2) is None

    def test_zero_gain_split_is_allowed(self):
        X = np.array([[1.0], [1.0], [2.0], [2.0]])
        feature, threshold, gain = best_split(X, np.array([0, 1, 0, 1]), n_classes=2)
        assert (feature, threshold) == (0, 1.5)
        assert gain == pytest.approx(0.0, abs=1e-12)

    def test_threshold_never_between_equal_values(self):
        X = np.array([[1.0], [1.0], [1.0], [2.0]])
        _, threshold, _ = best_split(X, np.array([0, 1, 1, 1]), n_classes=2)
        assert threshold == 1.5

    def test_entropy(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        feature, threshold, gain = best_split(X, np.array([0, 0, 1, 1]), n_classes=2, criterion="entropy")
        assert (feature, threshold) == (0, 2.5)
        assert gain == pytest.approx(1.0)

    def test_three_classes(self):
        X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 5.0], [4.0, 5.0], [5.0, 9.0], [6.0, 9.0]])
        y = np.array([0, 0, 1, 1, 2, 2])
        feature, threshold, gain = best_split(X, y, n_classes=3)
        assert (feature, threshold) == (0, 2.5)
        assert gain == pytest.approx(2 / 3 - 4 / 6 * 0.5)

    def test_matches_sklearn_root_on_real_data(self):
        X, y = load_breast_cancer(return_X_y=True)
        feature, threshold, gain = best_split(X, y, n_classes=2)
        tree = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y).tree_
        left, right = tree.children_left[0], tree.children_right[0]
        n = tree.n_node_samples
        sklearn_gain = tree.impurity[0] - (n[left] * tree.impurity[left] + n[right] * tree.impurity[right]) / n[0]
        assert gain == pytest.approx(sklearn_gain)
        assert feature == tree.feature[0]
        assert threshold == pytest.approx(tree.threshold[0])

    def test_speed(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20_000, 10))
        y = (X[:, 3] + rng.normal(size=20_000) > 0).astype(int)
        start = time.perf_counter()
        feature, _, _ = best_split(X, y, n_classes=2)
        elapsed = time.perf_counter() - start
        assert feature == 3
        assert elapsed < 3.0, f"best_split работал {elapsed:.1f} с: нужен один проход с накопленными счётчиками"


class TestTreeStructure:
    def test_fit_returns_self_and_sets_attributes(self):
        tree = MyDecisionTreeClassifier()
        assert tree.fit(CLIENTS_X, CLIENTS_Y) is tree
        assert tree.classes_.tolist() == [0, 1]
        assert tree.n_features_in_ == 3

    def test_lesson_tree(self):
        tree = MyDecisionTreeClassifier().fit(CLIENTS_X, CLIENTS_Y)
        root = tree.root_
        assert (root.feature, root.threshold, root.n_samples) == (1, 2.5, 12)
        assert root.impurity == pytest.approx(0.5)
        assert root.value.tolist() == pytest.approx([0.5, 0.5])

        assert root.right.is_leaf
        assert root.right.n_samples == 5
        assert root.right.value.tolist() == pytest.approx([0.0, 1.0])

        assert (root.left.feature, root.left.threshold, root.left.n_samples) == (2, 10.0, 7)
        assert root.left.left.value.tolist() == pytest.approx([0.0, 1.0])
        assert root.left.right.value.tolist() == pytest.approx([1.0, 0.0])

        assert tree.get_depth() == 2
        assert tree.get_n_leaves() == 3

    def test_pure_data_gives_single_leaf(self):
        tree = MyDecisionTreeClassifier().fit(np.arange(10.0).reshape(-1, 1), np.zeros(10, dtype=int))
        assert tree.root_.is_leaf
        assert tree.get_depth() == 0
        assert tree.get_n_leaves() == 1

    def test_max_depth_zero_predicts_class_shares(self):
        tree = MyDecisionTreeClassifier(max_depth=0).fit(CLIENTS_X, np.array([1] * 9 + [0] * 3))
        assert tree.get_n_leaves() == 1
        assert np.allclose(tree.predict_proba(CLIENTS_X[:2]), [[0.25, 0.75], [0.25, 0.75]])

    @pytest.mark.parametrize("max_depth", [1, 2, 4])
    def test_max_depth(self, max_depth):
        X_train, _, y_train, _ = split_dataset("breast_cancer")
        tree = MyDecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)
        assert tree.get_depth() == max_depth
        assert tree.get_n_leaves() <= 2**max_depth

    def test_min_samples_leaf(self):
        X_train, _, y_train, _ = split_dataset("breast_cancer")
        tree = MyDecisionTreeClassifier(min_samples_leaf=20).fit(X_train, y_train)
        leaves = [node for node in all_nodes(tree.root_) if node.is_leaf]
        assert len(leaves) > 2
        assert min(node.n_samples for node in leaves) >= 20
        assert sum(node.n_samples for node in leaves) == len(y_train)

    def test_min_samples_split(self):
        X_train, _, y_train, _ = split_dataset("breast_cancer")
        tree = MyDecisionTreeClassifier(min_samples_split=50).fit(X_train, y_train)
        internal = [node for node in all_nodes(tree.root_) if not node.is_leaf]
        assert len(internal) > 1
        assert min(node.n_samples for node in internal) >= 50

    def test_children_partition_parent(self):
        X_train, _, y_train, _ = split_dataset("wine")
        tree = MyDecisionTreeClassifier().fit(X_train, y_train)
        for node in all_nodes(tree.root_):
            assert node.value.sum() == pytest.approx(1.0)
            if not node.is_leaf:
                assert node.left.n_samples + node.right.n_samples == node.n_samples


class TestPrediction:
    def test_shapes_and_consistency(self):
        X_train, X_test, y_train, _ = split_dataset("wine")
        tree = MyDecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
        probabilities = tree.predict_proba(X_test)
        assert probabilities.shape == (len(X_test), 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert (tree.predict(X_test) == tree.classes_[probabilities.argmax(axis=1)]).all()

    def test_string_labels(self):
        labels = np.where(CLIENTS_Y == 1, "ушёл", "остался")
        tree = MyDecisionTreeClassifier().fit(CLIENTS_X, labels)
        assert tree.classes_.tolist() == ["остался", "ушёл"]
        assert tree.predict(CLIENTS_X).tolist() == labels.tolist()

    def test_ties_predict_first_class(self):
        X = np.ones((4, 1))
        tree = MyDecisionTreeClassifier().fit(X, np.array([5, 5, 7, 7]))
        assert tree.predict(X).tolist() == [5, 5, 5, 5]

    @pytest.mark.parametrize("name", list(DATASETS))
    def test_full_tree_memorizes_training_data(self, name):
        X_train, _, y_train, _ = split_dataset(name)
        tree = MyDecisionTreeClassifier().fit(X_train, y_train)
        assert (tree.predict(X_train) == y_train).all()

    def test_xor_needs_zero_gain_splits(self):
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 5)
        y = np.array([0, 1, 1, 0] * 5)
        tree = MyDecisionTreeClassifier().fit(X, y)
        assert (tree.predict(X) == y).all()
        assert tree.get_depth() == 2


class TestAgainstSklearn:
    @pytest.mark.parametrize("name", list(DATASETS))
    @pytest.mark.parametrize("criterion", ["gini", "entropy"])
    def test_quality_is_close(self, name, criterion):
        X_train, X_test, y_train, y_test = split_dataset(name)
        mine = MyDecisionTreeClassifier(max_depth=4, min_samples_leaf=3, criterion=criterion).fit(X_train, y_train)
        reference = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, criterion=criterion, random_state=0)
        reference.fit(X_train, y_train)

        assert (mine.predict(X_train) == y_train).mean() == pytest.approx(reference.score(X_train, y_train), abs=0.02)
        assert (mine.predict(X_test) == y_test).mean() == pytest.approx(reference.score(X_test, y_test), abs=0.06)
        assert (mine.predict(X_test) == reference.predict(X_test)).mean() >= 0.9

    def test_criteria_build_different_trees(self):
        X_train, _, y_train, _ = split_dataset("wine")
        gini_tree = MyDecisionTreeClassifier(max_depth=3, criterion="gini").fit(X_train, y_train)
        entropy_tree = MyDecisionTreeClassifier(max_depth=3, criterion="entropy").fit(X_train, y_train)
        assert gini_tree.root_.impurity == pytest.approx(impurity(np.bincount(y_train), "gini"))
        assert entropy_tree.root_.impurity == pytest.approx(impurity(np.bincount(y_train), "entropy"))


class TestEngineering:
    def test_fit_speed(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(20_000, 10))
        y = (X[:, 0] * X[:, 1] + 0.5 * X[:, 2] + rng.normal(scale=0.5, size=20_000) > 0).astype(int)
        start = time.perf_counter()
        tree = MyDecisionTreeClassifier(max_depth=6).fit(X, y)
        elapsed = time.perf_counter() - start
        assert tree.get_depth() == 6
        assert elapsed < 10.0, f"обучение заняло {elapsed:.1f} с"

    def test_predict_speed(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(1_000_000, 5))
        y = (X[:, 0] * X[:, 1] + X[:, 2] + rng.normal(scale=0.5, size=len(X)) > 0).astype(int)
        tree = MyDecisionTreeClassifier(max_depth=8).fit(X[:5000], y[:5000])
        reference = DecisionTreeClassifier(max_depth=8, random_state=0).fit(X[:5000], y[:5000])

        start = time.perf_counter()
        reference.predict(X)
        sklearn_time = time.perf_counter() - start

        start = time.perf_counter()
        tree.predict(X)
        elapsed = time.perf_counter() - start

        # Порог привязан к скорости машины: обход дерева отдельно для каждого объекта в него не укладывается
        limit = 12 * sklearn_time + 0.3
        assert elapsed < limit, f"предсказание заняло {elapsed:.1f} с при пороге {limit:.1f} с"

    def test_no_sklearn_in_solution(self):
        source = (PRACTICE_DIR / "my_tree.py").read_text(encoding="utf-8")
        assert "sklearn" not in source
