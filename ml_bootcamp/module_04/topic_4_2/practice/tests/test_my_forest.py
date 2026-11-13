import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn import inspection
from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score as sklearn_r2

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))

from my_forest import MyRandomForestRegressor, partial_dependence, permutation_importance, r2_score  # noqa: E402


class LinearStub:
    """Модель с известным ответом: 2 * x0 + x1, остальные признаки не используются."""

    def predict(self, X):
        X = np.asarray(X)
        return 2 * X[:, 0] + X[:, 1]


class FirstColumnStub:
    """Модель, которая возвращает первый признак и запоминает, что ей передавали."""

    def __init__(self):
        self.seen = []

    def predict(self, X):
        self.seen.append(np.array(X, dtype=float))
        return np.asarray(X, dtype=float)[:, 0]


@pytest.fixture(scope="module")
def friedman():
    # y = 10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4 + шум; признаки 5-9 — шумовые
    X, y = make_friedman1(n_samples=2000, n_features=10, noise=1.0, random_state=0)
    return X[:1000], X[1000:], y[:1000], y[1000:]


@pytest.fixture(scope="module")
def forest(friedman):
    X_train, _, y_train, _ = friedman
    return MyRandomForestRegressor(n_estimators=60, random_state=0).fit(X_train, y_train)


@pytest.fixture(scope="module")
def sklearn_forest(friedman):
    X_train, _, y_train, _ = friedman
    return RandomForestRegressor(n_estimators=60, oob_score=True, random_state=0).fit(X_train, y_train)


class TestR2:
    def test_perfect(self):
        assert r2_score(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)

    def test_mean_prediction_gives_zero(self):
        assert r2_score(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0])) == pytest.approx(0.0)

    def test_hand_example(self):
        # ошибки 0, 0, 1 -> сумма квадратов 1; отклонения от среднего -1, 0, 1 -> сумма квадратов 2
        assert r2_score(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0])) == pytest.approx(0.5)

    def test_can_be_negative(self):
        assert r2_score(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == pytest.approx(-3.0)

    def test_constant_target(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert r2_score(np.array([5.0, 5.0]), np.array([5.0, 5.0])) == 1.0
            assert r2_score(np.array([5.0, 5.0]), np.array([4.0, 6.0])) == 0.0

    def test_matches_sklearn(self):
        rng = np.random.default_rng(0)
        y_true, y_pred = rng.normal(size=200), rng.normal(size=200)
        assert r2_score(y_true, y_pred) == pytest.approx(sklearn_r2(y_true, y_pred))

    def test_returns_float(self):
        assert isinstance(r2_score(np.array([1.0, 2.0]), np.array([1.0, 3.0])), float)


class TestFit:
    def test_returns_self_and_attributes(self, friedman):
        X_train, _, y_train, _ = friedman
        model = MyRandomForestRegressor(n_estimators=7, random_state=0)
        assert model.fit(X_train, y_train) is model
        assert len(model.estimators_) == 7
        assert len(model.estimators_samples_) == 7
        assert model.n_features_in_ == 10

    def test_bootstrap_samples(self, forest):
        n = 1000
        for sample in forest.estimators_samples_:
            assert sample.shape == (n,)
            assert sample.min() >= 0
            assert sample.max() < n
            assert len(np.unique(sample)) < n  # выборка с возвращением содержит повторы

    def test_out_of_bag_share(self, forest):
        shares = [1 - len(np.unique(sample)) / 1000 for sample in forest.estimators_samples_]
        assert np.mean(shares) == pytest.approx(0.368, abs=0.01)

    def test_trees_differ(self, forest):
        assert not np.array_equal(forest.estimators_samples_[0], forest.estimators_samples_[1])

    def test_tree_is_trained_on_its_sample(self, friedman, forest):
        # Дерево без ограничений запоминает свою выборку, но не объекты, которых в ней не было
        X_train, _, y_train, _ = friedman
        for tree, sample in zip(forest.estimators_[:5], forest.estimators_samples_[:5], strict=True):
            assert np.allclose(tree.predict(X_train[sample]), y_train[sample])
            oob = np.setdiff1d(np.arange(len(X_train)), sample)
            assert not np.allclose(tree.predict(X_train[oob]), y_train[oob])

    def test_tree_parameters_are_passed(self, friedman):
        X_train, _, y_train, _ = friedman
        model = MyRandomForestRegressor(
            n_estimators=3, max_features=0.5, max_depth=4, min_samples_leaf=7, random_state=0
        )
        model.fit(X_train, y_train)
        for tree in model.estimators_:
            assert tree.max_features_ == 5
            assert tree.get_depth() <= 4
            assert tree.tree_.n_node_samples[tree.tree_.children_left == -1].min() >= 7

    def test_without_bootstrap(self, friedman):
        X_train, _, y_train, _ = friedman
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = MyRandomForestRegressor(n_estimators=3, bootstrap=False, random_state=0).fit(X_train, y_train)
        for sample in model.estimators_samples_:
            assert np.array_equal(sample, np.arange(len(X_train)))
        assert np.isnan(model.oob_prediction_).all()
        assert np.isnan(model.oob_score_)

    def test_accepts_dataframe(self, friedman):
        X_train, X_test, y_train, _ = friedman
        columns = [f"x{i}" for i in range(10)]
        from_frame = MyRandomForestRegressor(n_estimators=5, random_state=3)
        from_frame.fit(pd.DataFrame(X_train, columns=columns), pd.Series(y_train))
        from_array = MyRandomForestRegressor(n_estimators=5, random_state=3).fit(X_train, y_train)
        assert np.allclose(from_frame.predict(pd.DataFrame(X_test, columns=columns)), from_array.predict(X_test))

    def test_input_is_not_modified(self, friedman):
        X_train, _, y_train, _ = friedman
        X_copy, y_copy = X_train.copy(), y_train.copy()
        MyRandomForestRegressor(n_estimators=3, random_state=0).fit(X_train, y_train)
        assert np.array_equal(X_train, X_copy)
        assert np.array_equal(y_train, y_copy)


class TestRandomness:
    def test_same_seed_same_forest(self, friedman):
        X_train, X_test, y_train, _ = friedman
        first = MyRandomForestRegressor(n_estimators=10, max_features=0.5, random_state=42).fit(X_train, y_train)
        second = MyRandomForestRegressor(n_estimators=10, max_features=0.5, random_state=42).fit(X_train, y_train)
        assert np.array_equal(first.predict(X_test), second.predict(X_test))
        assert first.oob_score_ == second.oob_score_

    def test_different_seed_different_forest(self, friedman):
        X_train, X_test, y_train, _ = friedman
        first = MyRandomForestRegressor(n_estimators=10, random_state=1).fit(X_train, y_train)
        second = MyRandomForestRegressor(n_estimators=10, random_state=2).fit(X_train, y_train)
        assert not np.allclose(first.predict(X_test), second.predict(X_test))

    def test_tree_does_not_depend_on_forest_size(self, friedman):
        # Независимые генераторы: первые пять деревьев леса из двадцати — это лес из пяти деревьев
        X_train, X_test, y_train, _ = friedman
        small = MyRandomForestRegressor(n_estimators=5, max_features=0.5, random_state=7).fit(X_train, y_train)
        large = MyRandomForestRegressor(n_estimators=20, max_features=0.5, random_state=7).fit(X_train, y_train)
        for b in range(5):
            assert np.array_equal(small.estimators_samples_[b], large.estimators_samples_[b])
        assert np.array_equal(small.tree_predictions(X_test), large.tree_predictions(X_test)[:5])

    def test_random_state_none_works(self, friedman):
        X_train, _, y_train, _ = friedman
        model = MyRandomForestRegressor(n_estimators=2).fit(X_train, y_train)
        assert len(model.estimators_) == 2


class TestPredict:
    def test_tree_predictions_shape(self, friedman, forest):
        _, X_test, _, _ = friedman
        assert forest.tree_predictions(X_test).shape == (60, 1000)

    def test_tree_predictions_rows_are_trees(self, friedman, forest):
        _, X_test, _, _ = friedman
        per_tree = forest.tree_predictions(X_test[:50])
        for b in (0, 17, 59):
            assert np.allclose(per_tree[b], forest.estimators_[b].predict(X_test[:50]))

    def test_forest_is_mean_of_trees(self, friedman, forest):
        _, X_test, _, _ = friedman
        prediction = forest.predict(X_test)
        assert prediction.shape == (1000,)
        assert np.allclose(prediction, forest.tree_predictions(X_test).mean(axis=0))

    def test_single_tree_forest(self, friedman):
        X_train, X_test, y_train, _ = friedman
        model = MyRandomForestRegressor(n_estimators=1, random_state=0).fit(X_train, y_train)
        assert np.allclose(model.predict(X_test), model.estimators_[0].predict(X_test))

    def test_prediction_within_target_range(self, friedman, forest):
        # Лес, как и дерево, не экстраполирует
        _, X_test, y_train, _ = friedman
        prediction = forest.predict(X_test * 10)
        assert prediction.min() >= y_train.min()
        assert prediction.max() <= y_train.max()

    def test_forest_beats_single_tree(self, friedman, forest):
        _, X_test, _, y_test = friedman
        tree_scores = [r2_score(y_test, row) for row in forest.tree_predictions(X_test)]
        assert r2_score(y_test, forest.predict(X_test)) > max(tree_scores)


class TestOob:
    def test_matches_definition(self, friedman, forest):
        X_train, _, _, _ = friedman
        n = len(X_train)
        total, count = np.zeros(n), np.zeros(n)
        for tree, sample in zip(forest.estimators_, forest.estimators_samples_, strict=True):
            oob = np.setdiff1d(np.arange(n), sample)
            total[oob] += tree.predict(X_train[oob])
            count[oob] += 1
        assert (count > 0).all()
        assert forest.oob_prediction_.shape == (n,)
        assert np.allclose(forest.oob_prediction_, total / count)

    def test_object_is_excluded_from_its_own_prediction(self):
        # Выброс запоминают только деревья, которые его видели; в OOB-предсказании их быть не должно
        X = np.arange(60, dtype=float).reshape(-1, 1)
        y = np.arange(60, dtype=float)
        y[30] = 1000.0
        model = MyRandomForestRegressor(n_estimators=100, random_state=0).fit(X, y)
        assert model.predict(X)[30] > 300
        assert model.oob_prediction_[30] < 60

    def test_score_is_r2_of_oob_predictions(self, friedman, forest):
        _, _, y_train, _ = friedman
        assert forest.oob_score_ == pytest.approx(sklearn_r2(y_train, forest.oob_prediction_))

    def test_oob_is_honest(self, friedman, forest):
        X_train, X_test, y_train, y_test = friedman
        train_score = r2_score(y_train, forest.predict(X_train))
        test_score = r2_score(y_test, forest.predict(X_test))
        assert train_score > 0.95
        assert forest.oob_score_ < train_score - 0.05
        assert forest.oob_score_ == pytest.approx(test_score, abs=0.04)

    def test_few_trees_leave_gaps(self, friedman):
        X_train, _, y_train, _ = friedman
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = MyRandomForestRegressor(n_estimators=1, random_state=0).fit(X_train, y_train)
        in_bag = np.zeros(len(X_train), dtype=bool)
        in_bag[model.estimators_samples_[0]] = True
        assert np.isnan(model.oob_prediction_[in_bag]).all()
        assert not np.isnan(model.oob_prediction_[~in_bag]).any()
        expected = sklearn_r2(y_train[~in_bag], model.oob_prediction_[~in_bag])
        assert model.oob_score_ == pytest.approx(expected)


class TestAgainstSklearn:
    def test_test_quality(self, friedman, forest, sklearn_forest):
        _, X_test, _, y_test = friedman
        mine = r2_score(y_test, forest.predict(X_test))
        reference = sklearn_forest.score(X_test, y_test)
        assert mine == pytest.approx(reference, abs=0.02)

    def test_oob_quality(self, forest, sklearn_forest):
        assert forest.oob_score_ == pytest.approx(sklearn_forest.oob_score_, abs=0.03)

    def test_max_features_quality(self, friedman):
        X_train, X_test, y_train, y_test = friedman
        mine = MyRandomForestRegressor(n_estimators=60, max_features=0.3, random_state=0).fit(X_train, y_train)
        reference = RandomForestRegressor(n_estimators=60, max_features=0.3, random_state=0).fit(X_train, y_train)
        assert r2_score(y_test, mine.predict(X_test)) == pytest.approx(reference.score(X_test, y_test), abs=0.02)


class TestPermutationImportance:
    def test_shape(self):
        X = np.random.default_rng(0).normal(size=(50, 4))
        result = permutation_importance(LinearStub(), X, LinearStub().predict(X), n_repeats=3, random_state=0)
        assert result.shape == (4, 3)

    def test_unused_feature_has_exactly_zero_importance(self):
        X = np.random.default_rng(0).normal(size=(200, 4))
        result = permutation_importance(LinearStub(), X, LinearStub().predict(X), n_repeats=4, random_state=0)
        assert np.allclose(result[2:], 0.0)
        assert (result[:2] > 0).all()
        assert result[0].mean() > result[1].mean()

    def test_known_value(self):
        # Модель и ответ равны x0. После перестановки x0 предсказание не связано с ответом:
        # сумма квадратов ошибок вдвое больше суммы квадратов отклонений, R² = -1, падение равно 2
        X = np.random.default_rng(1).normal(size=(5000, 2))
        result = permutation_importance(FirstColumnStub(), X, X[:, 0], n_repeats=10, random_state=0)
        assert result[0].mean() == pytest.approx(2.0, abs=0.1)
        assert np.allclose(result[1], 0.0)

    def test_only_one_column_is_permuted(self):
        X = np.random.default_rng(2).normal(size=(100, 3))
        stub = FirstColumnStub()
        permutation_importance(stub, X, X[:, 0], n_repeats=2, random_state=0)
        modified = [seen for seen in stub.seen if not np.array_equal(seen, X)]
        assert len(modified) == 6
        for k, seen in enumerate(modified):
            j = k // 2
            others = [c for c in range(3) if c != j]
            assert np.array_equal(seen[:, others], X[:, others])
            assert not np.array_equal(seen[:, j], X[:, j])
            assert np.array_equal(np.sort(seen[:, j]), np.sort(X[:, j]))  # перестановка, а не новые значения

    def test_new_permutation_for_each_repeat(self):
        X = np.random.default_rng(3).normal(size=(100, 2))
        result = permutation_importance(FirstColumnStub(), X, X[:, 0], n_repeats=5, random_state=0)
        assert len(np.unique(result[0])) == 5

    def test_input_is_not_modified(self):
        X = np.random.default_rng(4).normal(size=(100, 3))
        X_copy = X.copy()
        permutation_importance(LinearStub(), X, LinearStub().predict(X), n_repeats=2, random_state=0)
        assert np.array_equal(X, X_copy)

    def test_reproducible(self):
        X = np.random.default_rng(5).normal(size=(100, 3))
        y = LinearStub().predict(X)
        first = permutation_importance(LinearStub(), X, y, n_repeats=3, random_state=11)
        second = permutation_importance(LinearStub(), X, y, n_repeats=3, random_state=11)
        other = permutation_importance(LinearStub(), X, y, n_repeats=3, random_state=12)
        assert np.array_equal(first, second)
        assert not np.array_equal(first, other)

    def test_noise_features_of_forest(self, friedman, forest):
        _, X_test, _, y_test = friedman
        result = permutation_importance(forest, X_test, y_test, n_repeats=5, random_state=0)
        mean = result.mean(axis=1)
        assert np.abs(mean[5:]).max() < 0.01  # признаки 5-9 на ответ не влияют
        assert mean[:5].min() > 0.03
        assert mean[3] > 0.2  # слагаемое 10 * x3

    def test_accepts_dataframe(self, friedman, forest):
        _, X_test, _, y_test = friedman
        frame = pd.DataFrame(X_test[:200], columns=[f"x{i}" for i in range(10)])
        from_frame = permutation_importance(forest, frame, y_test[:200], n_repeats=2, random_state=0)
        from_array = permutation_importance(forest, X_test[:200], y_test[:200], n_repeats=2, random_state=0)
        assert np.allclose(from_frame, from_array)


class TestPartialDependence:
    def test_hand_example(self):
        X = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]])
        grid = np.array([0.0, 10.0])
        # признак 0: 2 * v + среднее x1 = 2 * v + 3
        assert np.allclose(partial_dependence(LinearStub(), X, 0, grid), [3.0, 23.0])
        # признак 1: 2 * среднее x0 + v = 2 + v
        assert np.allclose(partial_dependence(LinearStub(), X, 1, grid), [2.0, 12.0])

    def test_unused_feature_is_flat(self):
        X = np.random.default_rng(0).normal(size=(50, 3))
        values = partial_dependence(LinearStub(), X, 2, np.linspace(-5, 5, 7))
        assert values.shape == (7,)
        assert np.allclose(values, values[0])

    def test_input_is_not_modified(self):
        X = np.random.default_rng(1).normal(size=(50, 3))
        X_copy = X.copy()
        partial_dependence(LinearStub(), X, 0, np.array([0.0, 1.0]))
        assert np.array_equal(X, X_copy)

    def test_matches_sklearn(self, friedman, sklearn_forest):
        X_train, _, _, _ = friedman
        grid = np.linspace(0.05, 0.95, 10)
        for feature in (0, 3, 7):
            expected = inspection.partial_dependence(
                sklearn_forest, X_train, [feature], custom_values={feature: grid}, method="brute"
            )["average"][0]
            assert np.allclose(partial_dependence(sklearn_forest, X_train, feature, grid), expected)

    def test_recovers_known_dependence(self, friedman, forest):
        # В истинной зависимости слагаемое 10 * x3: при росте x3 от 0.1 до 0.9 ответ растёт на 8
        X_train, _, _, _ = friedman
        values = partial_dependence(forest, X_train, 3, np.array([0.1, 0.5, 0.9]))
        assert values[0] < values[1] < values[2]
        assert 5.0 < values[2] - values[0] < 9.0
        noise = partial_dependence(forest, X_train, 8, np.array([0.1, 0.5, 0.9]))
        assert np.ptp(noise) < 0.5

    def test_accepts_dataframe(self, friedman, forest):
        X_train, _, _, _ = friedman
        frame = pd.DataFrame(X_train[:100], columns=[f"x{i}" for i in range(10)])
        grid = np.array([0.2, 0.8])
        assert np.allclose(
            partial_dependence(forest, frame, 3, grid), partial_dependence(forest, X_train[:100], 3, grid)
        )


class TestEngineering:
    def test_only_tree_is_imported_from_sklearn(self):
        source = (PRACTICE_DIR / "my_forest.py").read_text(encoding="utf-8")
        imports = [line.strip() for line in source.splitlines() if line.strip().startswith(("import ", "from "))]
        sklearn_imports = [line.split("#")[0].strip() for line in imports if "sklearn" in line]
        assert sklearn_imports == ["from sklearn.tree import DecisionTreeRegressor"]

    def test_fit_speed(self):
        # Эталон: около 2 секунд. Отдельный вызов predict для каждого объекта при расчёте OOB не уложится.
        X, y = make_friedman1(n_samples=10000, n_features=10, noise=1.0, random_state=1)
        start = time.perf_counter()
        model = MyRandomForestRegressor(n_estimators=40, max_features=0.5, random_state=0).fit(X, y)
        elapsed = time.perf_counter() - start
        assert model.oob_score_ > 0.8
        assert elapsed < 30

    def test_importance_and_dependence_speed(self, friedman, forest):
        _, X_test, _, y_test = friedman
        start = time.perf_counter()
        permutation_importance(forest, X_test, y_test, n_repeats=3, random_state=0)
        partial_dependence(forest, X_test, 0, np.linspace(0, 1, 20))
        assert time.perf_counter() - start < 20
