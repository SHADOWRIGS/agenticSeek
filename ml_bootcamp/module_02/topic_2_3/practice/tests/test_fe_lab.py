import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))

from fe_lab import (  # noqa: E402
    JOURNAL_COLUMNS,
    compare_feature_sets,
    count_encode,
    evaluate_features,
    load_experiments,
    log_experiment,
    out_of_fold_target_encode,
)
from make_demo_data import TARGET, make_demo  # noqa: E402


def reference_encoding(fit_categories: list, fit_target: list, categories: list, smooth: float) -> list[float]:
    """Target encoding «в лоб», на словарях: независимая от pandas проверка формулы."""
    prior = sum(fit_target) / len(fit_target)
    totals: dict = {}
    counts: dict = {}
    for category, value in zip(fit_categories, fit_target, strict=True):
        totals[category] = totals.get(category, 0.0) + value
        counts[category] = counts.get(category, 0) + 1
    return [(totals[c] + smooth * prior) / (counts[c] + smooth) if c in counts else prior for c in categories]


@pytest.fixture(scope="module")
def noisy() -> tuple[pd.Series, pd.Series, pd.Series]:
    """Категории высокой кардинальности, никак не связанные с целевой переменной."""
    rng = np.random.default_rng(0)
    train = pd.Series([f"id_{code}" for code in rng.integers(0, 1500, size=3000)], name="noise")
    test = pd.Series([f"id_{code}" for code in rng.integers(0, 1500, size=1000)], name="noise")
    target = pd.Series(rng.integers(0, 2, size=3000), name="y")
    return train, test, target


@pytest.fixture(scope="module")
def toy() -> pd.DataFrame:
    """Маленькая задача классификации: два информативных признака, один шумовой и один строковый."""
    rng = np.random.default_rng(1)
    n = 600
    signal_a, signal_b, noise = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    target = (signal_a + signal_b + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pd.DataFrame(
        {
            "signal_a": signal_a,
            "signal_b": signal_b,
            "noise": noise,
            "flag": signal_a > 0,
            "text": rng.choice(["x", "y", "z"], size=n),
            "y": target,
            "y_real": 2 * signal_a - signal_b + rng.normal(scale=0.1, size=n),
        }
    )


class TestCountEncode:
    def test_counts_from_train(self):
        train = pd.Series(["a", "b", "a", "c", "a"], name="city")
        test = pd.Series(["a", "c", "zzz"], name="city")
        encoded_train, encoded_test = count_encode(train, test)
        assert encoded_train.tolist() == [3, 1, 3, 1, 3]
        assert encoded_test.tolist() == [3, 1, 0]

    def test_missing_is_a_category(self):
        train = pd.Series(["a", None, "a", None, None])
        encoded_train, encoded_test = count_encode(train, pd.Series([None, "a"]))
        assert encoded_train.tolist() == [2, 3, 2, 3, 3]
        assert encoded_test.tolist() == [3, 2]

    def test_missing_only_in_test(self):
        _, encoded_test = count_encode(pd.Series(["a", "b"]), pd.Series([None, "b"]))
        assert encoded_test.tolist() == [0, 1]

    def test_dtype_index_and_name(self):
        train = pd.Series(["a", "b", "a"], index=[10, 20, 30], name="city")
        test = pd.Series(["b", "q"], index=[7, 8], name="city")
        encoded_train, encoded_test = count_encode(train, test)
        assert encoded_train.dtype == "int64" and encoded_test.dtype == "int64"
        assert encoded_train.index.tolist() == [10, 20, 30] and encoded_test.index.tolist() == [7, 8]
        assert encoded_train.name == "city" and encoded_test.name == "city"

    def test_works_with_categorical_dtype(self):
        train = pd.Series(["a", "b", "a"], dtype="category")
        encoded_train, encoded_test = count_encode(train, pd.Series(["b", "a"], dtype="category"))
        assert encoded_train.tolist() == [2, 1, 2]
        assert encoded_test.tolist() == [1, 2]

    def test_inputs_are_not_modified(self):
        train, test = pd.Series(["a", None, "a"]), pd.Series(["a", "b"])
        train_copy, test_copy = train.copy(), test.copy()
        count_encode(train, test)
        pd.testing.assert_series_equal(train, train_copy)
        pd.testing.assert_series_equal(test, test_copy)


class TestOutOfFoldTargetEncode:
    def test_matches_reference_on_train(self, noisy):
        train, test, target = noisy
        encoded_train, _ = out_of_fold_target_encode(train, test, target, n_splits=4, smooth=3.0, seed=7)
        expected = np.empty(len(train))
        for fit_idx, encode_idx in KFold(n_splits=4, shuffle=True, random_state=7).split(train):
            expected[encode_idx] = reference_encoding(
                train.iloc[fit_idx].tolist(), target.iloc[fit_idx].tolist(), train.iloc[encode_idx].tolist(), 3.0
            )
        np.testing.assert_allclose(encoded_train.to_numpy(), expected)

    def test_test_part_uses_whole_train(self, noisy):
        train, test, target = noisy
        _, encoded_test = out_of_fold_target_encode(train, test, target, smooth=5.0)
        expected = reference_encoding(train.tolist(), target.tolist(), test.tolist(), 5.0)
        np.testing.assert_allclose(encoded_test.to_numpy(), expected)

    def test_own_target_does_not_leak(self, noisy):
        # Главное свойство схемы: кодировка строки не зависит от её собственной целевой переменной
        train, test, target = noisy
        before, _ = out_of_fold_target_encode(train, test, target, smooth=1.0)
        for position in (0, 17, 1234, 2999):
            flipped = target.copy()
            flipped.iloc[position] = 1 - flipped.iloc[position]
            after, _ = out_of_fold_target_encode(train, test, flipped, smooth=1.0)
            assert after.iloc[position] == pytest.approx(before.iloc[position])
            assert not np.allclose(after.to_numpy(), before.to_numpy())

    def test_noise_feature_stays_noise(self, noisy):
        # Наивное кодирование даёт здесь AUC около 0.9; честная схема не должна находить связь, которой нет
        train, test, target = noisy
        encoded_train, encoded_test = out_of_fold_target_encode(train, test, target, smooth=0.0)
        assert 0.45 < roc_auc_score(target, encoded_train) < 0.55

    def test_signal_is_kept(self):
        rng = np.random.default_rng(3)
        codes = rng.integers(0, 40, size=4000)
        probability = np.linspace(0.05, 0.95, 40)[codes]
        target = pd.Series((rng.random(4000) < probability).astype(int))
        categories = pd.Series([f"c{code}" for code in codes])
        encoded_train, encoded_test = out_of_fold_target_encode(
            categories.iloc[:3000], categories.iloc[3000:], target.iloc[:3000]
        )
        assert roc_auc_score(target.iloc[:3000], encoded_train) > 0.75
        assert roc_auc_score(target.iloc[3000:], encoded_test) > 0.75

    def test_unseen_category_gets_prior(self):
        train = pd.Series(["a", "a", "b", "b", "a", "b"])
        target = pd.Series([1, 1, 0, 0, 1, 1])
        _, encoded_test = out_of_fold_target_encode(train, pd.Series(["new", "a"]), target, n_splits=2, smooth=2.0)
        assert encoded_test.iloc[0] == pytest.approx(4 / 6)
        assert encoded_test.iloc[1] == pytest.approx((3 + 2.0 * 4 / 6) / (3 + 2.0))

    def test_category_missing_in_other_folds_gets_their_prior(self):
        # Категория встречается один раз: в остальных фолдах её нет, значит, кодировка — их среднее
        train = pd.Series(["a"] * 9 + ["unique"])
        target = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 1])
        encoded_train, _ = out_of_fold_target_encode(train, train, target, n_splits=5, smooth=0.0, seed=0)
        for fit_idx, encode_idx in KFold(n_splits=5, shuffle=True, random_state=0).split(train):
            if 9 in encode_idx:
                assert encoded_train.iloc[9] == pytest.approx(target.iloc[fit_idx].mean())

    def test_missing_is_a_category(self):
        train = pd.Series(["a", None, "a", None, "a", None, "a", None])
        target = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        encoded_train, encoded_test = out_of_fold_target_encode(
            train, pd.Series([None, "a"]), target, n_splits=2, smooth=0.0
        )
        assert encoded_test.tolist() == [1.0, 0.0]
        assert encoded_train.tolist() == [0.0, 1.0] * 4

    def test_smoothing_pulls_to_prior(self, noisy):
        train, test, target = noisy
        _, weak = out_of_fold_target_encode(train, test, target, smooth=0.0)
        _, strong = out_of_fold_target_encode(train, test, target, smooth=1e9)
        assert weak.std() > 0.2
        np.testing.assert_allclose(strong.to_numpy(), target.mean(), atol=1e-6)

    def test_regression_target(self):
        train = pd.Series(["a", "b", "a", "b", "a", "b"])
        target = pd.Series([10.0, 1.0, 12.0, 3.0, 14.0, 5.0])
        _, encoded_test = out_of_fold_target_encode(train, pd.Series(["a", "b"]), target, n_splits=3, smooth=0.0)
        assert encoded_test.tolist() == pytest.approx([12.0, 3.0])

    def test_dtype_index_name_and_no_missing(self, noisy):
        train, test, target = noisy
        train = train.set_axis(np.arange(len(train))[::-1])
        test = test.set_axis(np.arange(len(test)) + 100_000)
        target = target.set_axis(train.index)
        encoded_train, encoded_test = out_of_fold_target_encode(train, test, target)
        assert encoded_train.dtype == "float64" and encoded_test.dtype == "float64"
        assert encoded_train.index.equals(train.index) and encoded_test.index.equals(test.index)
        assert encoded_train.name == "noise" and encoded_test.name == "noise"
        assert encoded_train.notna().all() and encoded_test.notna().all()

    def test_folds_follow_row_order_not_index(self, noisy):
        train, test, target = noisy
        plain, _ = out_of_fold_target_encode(train, test, target, seed=3)
        shuffled_index = np.random.default_rng(5).permutation(len(train))
        relabeled, _ = out_of_fold_target_encode(
            train.set_axis(shuffled_index), test, target.set_axis(shuffled_index), seed=3
        )
        np.testing.assert_allclose(relabeled.to_numpy(), plain.to_numpy())

    def test_inputs_are_not_modified(self, noisy):
        train, test, target = noisy
        copies = train.copy(), test.copy(), target.copy()
        out_of_fold_target_encode(train, test, target)
        for original, copy in zip((train, test, target), copies, strict=True):
            pd.testing.assert_series_equal(original, copy)


class TestEvaluateFeatures:
    def test_result_format(self, toy):
        scores = evaluate_features(toy, ["signal_a", "signal_b"], "y", model=LogisticRegression(), n_splits=4)
        assert isinstance(scores, pd.Series)
        assert scores.name == "score"
        assert scores.index.name == "fold"
        assert scores.index.tolist() == [0, 1, 2, 3]

    def test_classifier_uses_stratified_folds(self, toy):
        scores = evaluate_features(toy, ["signal_a", "noise"], "y", model=LogisticRegression(), n_splits=5, seed=11)
        expected = cross_val_score(
            LogisticRegression(),
            toy[["signal_a", "noise"]],
            toy["y"],
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=11),
            scoring="roc_auc",
        )
        np.testing.assert_allclose(scores.to_numpy(), expected)

    def test_regressor_uses_plain_folds_and_scoring(self, toy):
        scores = evaluate_features(
            toy,
            ["signal_a", "signal_b"],
            "y_real",
            model=LinearRegression(),
            scoring="neg_root_mean_squared_error",
            seed=5,
        )
        expected = cross_val_score(
            LinearRegression(),
            toy[["signal_a", "signal_b"]],
            toy["y_real"],
            cv=KFold(n_splits=5, shuffle=True, random_state=5),
            scoring="neg_root_mean_squared_error",
        )
        np.testing.assert_allclose(scores.to_numpy(), expected)
        assert (scores < 0).all()

    def test_default_model_is_gradient_boosting(self, toy):
        scores = evaluate_features(toy, ["signal_a", "signal_b", "flag"], "y", n_splits=3, seed=2)
        expected = cross_val_score(
            HistGradientBoostingClassifier(random_state=2),
            toy[["signal_a", "signal_b", "flag"]],
            toy["y"],
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=2),
            scoring="roc_auc",
        )
        np.testing.assert_allclose(scores.to_numpy(), expected)

    def test_signal_beats_noise(self, toy):
        good = evaluate_features(toy, ["signal_a", "signal_b"], "y", model=LogisticRegression())
        bad = evaluate_features(toy, ["noise"], "y", model=LogisticRegression())
        assert good.mean() > 0.9
        assert 0.35 < bad.mean() < 0.65

    def test_model_is_used_but_not_fitted(self, toy):
        model = DummyClassifier(strategy="prior")
        scores = evaluate_features(toy, ["signal_a"], "y", model=model)
        assert scores.tolist() == [0.5] * 5
        assert not hasattr(model, "classes_")

    def test_is_reproducible(self, toy):
        first = evaluate_features(toy, ["signal_a", "noise"], "y", n_splits=3)
        second = evaluate_features(toy, ["signal_a", "noise"], "y", n_splits=3)
        pd.testing.assert_series_equal(first, second)

    def test_non_numeric_column_is_rejected(self, toy):
        with pytest.raises(TypeError, match="text"):
            evaluate_features(toy, ["signal_a", "text"], "y", model=LogisticRegression())

    def test_empty_feature_list_is_rejected(self, toy):
        with pytest.raises(ValueError):
            evaluate_features(toy, [], "y", model=LogisticRegression())

    def test_frame_is_not_modified(self, toy):
        copy = toy.copy()
        evaluate_features(toy, ["signal_a", "flag"], "y", model=LogisticRegression())
        pd.testing.assert_frame_equal(toy, copy)


@pytest.fixture(scope="module")
def sets() -> dict[str, list[str]]:
    return {"одно": ["signal_a"], "шум": ["noise"], "оба": ["signal_a", "signal_b"]}


@pytest.fixture(scope="module")
def table(toy, sets) -> pd.DataFrame:
    return compare_feature_sets(toy, sets, "y", baseline="одно", model=LogisticRegression(), n_splits=4, seed=9)


class TestCompareFeatureSets:
    def test_format(self, table):
        assert table.index.name == "feature_set"
        assert table.index.tolist() == ["одно", "шум", "оба"]
        assert table.columns.tolist() == ["n_features", "mean", "std", "delta", "wins"]
        assert table["n_features"].tolist() == [1, 1, 2]

    def test_values_come_from_evaluate_features(self, toy, sets, table):
        for name, columns in sets.items():
            scores = evaluate_features(toy, columns, "y", model=LogisticRegression(), n_splits=4, seed=9)
            assert table.loc[name, "mean"] == pytest.approx(scores.mean())
            assert table.loc[name, "std"] == pytest.approx(scores.std(ddof=1))

    def test_delta_and_wins(self, toy, sets, table):
        assert table.loc["одно", "delta"] == 0.0
        assert table.loc["одно", "wins"] == 0
        assert table.loc["оба", "delta"] == pytest.approx(table.loc["оба", "mean"] - table.loc["одно", "mean"])
        assert table.loc["оба", "delta"] > 0.05
        assert table.loc["оба", "wins"] == 4
        assert table.loc["шум", "delta"] < 0
        assert table.loc["шум", "wins"] == 0

    def test_unknown_baseline(self, toy, sets):
        with pytest.raises(KeyError):
            compare_feature_sets(toy, sets, "y", baseline="нет такого", model=LogisticRegression())

    def test_target_encoding_helps_on_demo_data(self):
        train, test, _ = make_demo(n_train=3000, n_test=500)
        numeric = ["tenure_months", "monthly_fee", "support_calls"]
        data = train[[*numeric, TARGET]].copy()
        data["city_te"], _ = out_of_fold_target_encode(train["city"], test["city"], train[TARGET])
        data["referrer_te"], _ = out_of_fold_target_encode(train["referrer_id"], test["referrer_id"], train[TARGET])
        sets = {"числа": numeric, "+ город": [*numeric, "city_te"], "+ шум": [*numeric, "referrer_te"]}
        table = compare_feature_sets(data, sets, TARGET, baseline="числа", n_splits=3)
        assert table.loc["+ город", "delta"] > 0.02
        assert abs(table.loc["+ шум", "delta"]) < 0.02


class TestJournal:
    def test_record_content(self, tmp_path):
        scores = pd.Series([0.71, 0.73, 0.75], name="score")
        before = datetime.now().astimezone()
        record = log_experiment(
            tmp_path / "journal.jsonl", "базовый", ["age", "город_te"], scores, note="первый запуск"
        )
        assert set(record) == set(JOURNAL_COLUMNS)
        assert record["name"] == "базовый"
        assert record["n_features"] == 2
        assert record["features"] == ["age", "город_te"]
        assert record["scores"] == [0.71, 0.73, 0.75]
        assert all(type(value) is float for value in record["scores"])
        assert record["mean"] == pytest.approx(0.73)
        assert record["std"] == pytest.approx(0.02)
        assert record["note"] == "первый запуск"
        moment = datetime.fromisoformat(record["time"])
        assert moment.utcoffset() == timedelta(0)
        assert abs(moment - before) < timedelta(minutes=1)

    def test_file_is_json_lines_and_appends(self, tmp_path):
        path = tmp_path / "deep" / "folder" / "journal.jsonl"
        first = log_experiment(path, "один", ["a"], pd.Series([0.5, 0.6]))
        first_text = path.read_text(encoding="utf-8")
        second = log_experiment(path, "два", ["a", "b"], pd.Series([0.7, 0.8]), note="лучше")
        text = path.read_text(encoding="utf-8")
        assert text.startswith(first_text)
        lines = text.strip().split("\n")
        assert [json.loads(line) for line in lines] == [first, second]

    def test_cyrillic_is_readable(self, tmp_path):
        path = tmp_path / "journal.jsonl"
        log_experiment(path, "город", ["признак"], pd.Series([0.5, 0.6]), note="заметка")
        text = path.read_text(encoding="utf-8")
        assert "город" in text and "признак" in text and "заметка" in text

    def test_load_sorted_by_mean(self, tmp_path):
        path = tmp_path / "journal.jsonl"
        log_experiment(path, "средний", ["a"], pd.Series([0.6, 0.7]))
        log_experiment(path, "лучший", ["a", "b"], pd.Series([0.8, 0.9]))
        log_experiment(path, "худший", ["c"], pd.Series([0.5, 0.5]))
        table = load_experiments(path)
        assert table.columns.tolist() == JOURNAL_COLUMNS
        assert table["name"].tolist() == ["лучший", "средний", "худший"]
        assert table.index.tolist() == [0, 1, 2]
        assert table.loc[0, "features"] == ["a", "b"]
        assert table.loc[0, "scores"] == [0.8, 0.9]

    def test_load_missing_file(self, tmp_path):
        table = load_experiments(tmp_path / "нет_такого.jsonl")
        assert table.empty
        assert table.columns.tolist() == JOURNAL_COLUMNS

    def test_round_trip_with_evaluate_features(self, toy, tmp_path):
        columns = ["signal_a", "signal_b"]
        scores = evaluate_features(toy, columns, "y", model=LogisticRegression(), n_splits=3)
        log_experiment(tmp_path / "journal.jsonl", "проверка", columns, scores)
        table = load_experiments(tmp_path / "journal.jsonl")
        assert table.loc[0, "mean"] == pytest.approx(scores.mean())
        assert table.loc[0, "scores"] == pytest.approx(scores.tolist())
