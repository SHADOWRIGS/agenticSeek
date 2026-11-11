import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))
sys.path.insert(0, str(PRACTICE_DIR.parent))

# isort: off
from fraud_data import TARGET, make_features, make_transactions  # noqa: E402

from business_metrics import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    STRATEGIES,
    Costs,
    best_threshold,
    bootstrap_savings_interval,
    break_even_threshold,
    compare_strategies,
    expected_value_decision,
    savings,
    savings_curve,
)

# isort: on

COSTS = Costs(review=100, friction=400)

# Пример на четырёх операциях: две заблокированы, одна из них мошенническая
Y4 = [1, 0, 1, 0]
AMOUNTS4 = [5000, 1000, 20000, 300]
PROBA4 = [0.9, 0.6, 0.3, 0.1]

# Пример, в котором лучшие пороги по F1 и по экономии различаются
Y5 = [1, 0, 0, 0, 1]
PROBA5 = [0.9, 0.8, 0.7, 0.6, 0.2]
AMOUNTS5 = [200, 1000, 1000, 1000, 50000]


def scored_table(model, features: pd.DataFrame, transactions: pd.DataFrame, mask) -> pd.DataFrame:
    part = transactions[mask]
    proba = model.predict_proba(features[mask])[:, 1]
    return pd.DataFrame({"is_fraud": part[TARGET].to_numpy(), "proba": proba, "amount": part["amount"].to_numpy()})


@pytest.fixture(scope="module")
def data():
    """Разбиение по времени: обучение — январь–март, валидация — апрель, тест — май–июнь."""
    transactions = make_transactions()
    features = make_features(transactions)
    month = transactions["timestamp"].dt.month
    return transactions, features, month <= 3, month == 4, month >= 5


@pytest.fixture(scope="module")
def scored(data):
    transactions, features, train, valid, test = data
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features[train], transactions.loc[train, TARGET])
    return scored_table(model, features, transactions, valid), scored_table(model, features, transactions, test)


@pytest.fixture(scope="module")
def scored_balanced(data):
    """Модель с class_weight="balanced": ранжирует так же, но вероятности завышены в разы."""
    transactions, features, train, _, test = data
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(features[train], transactions.loc[train, TARGET])
    return scored_table(model, features, transactions, test)


@pytest.fixture(scope="module")
def comparison(scored) -> pd.DataFrame:
    valid, test = scored
    return compare_strategies(valid, test, COSTS)


class TestSavings:
    def test_manual_example(self):
        # (5000 − 100) за пойманное мошенничество и −(100 + 400) за ложную блокировку
        assert savings(Y4, [1, 1, 0, 0], AMOUNTS4, COSTS) == pytest.approx(4400)

    def test_nothing_flagged_is_zero(self):
        assert savings(Y4, [0, 0, 0, 0], AMOUNTS4, COSTS) == 0
        assert savings([], [], [], COSTS) == 0

    def test_everything_flagged(self):
        assert savings(Y4, [True] * 4, AMOUNTS4, COSTS) == pytest.approx(4900 - 500 + 19900 - 500)

    def test_accepts_series_with_arbitrary_index_and_bool_flags(self):
        index = [10, 20, 30, 40]
        result = savings(
            pd.Series(Y4, index=index),
            np.array([True, True, False, False]),
            pd.Series(AMOUNTS4, index=index[::-1]),
            COSTS,
        )
        assert result == pytest.approx(4400)
        assert isinstance(result, float)

    def test_costs_are_used(self):
        assert savings(Y4, [1, 1, 0, 0], AMOUNTS4, Costs(review=0, friction=0)) == pytest.approx(5000)
        assert savings(Y4, [1, 1, 0, 0], AMOUNTS4, Costs(review=50, friction=1000)) == pytest.approx(4950 - 1050)

    def test_small_fraud_is_not_worth_blocking(self):
        assert savings([1], [1], [60], COSTS) == pytest.approx(-40)


class TestSavingsCurve:
    def test_manual_example(self):
        curve = savings_curve(Y4, PROBA4, AMOUNTS4, COSTS, thresholds=[0.2, 0.5, 0.95])
        assert list(curve.columns) == ["threshold", "n_flagged", "precision", "recall", "f1", "savings"]
        assert curve["threshold"].tolist() == [0.2, 0.5, 0.95]
        assert curve["n_flagged"].tolist() == [3, 2, 0]
        assert curve["precision"].tolist() == pytest.approx([2 / 3, 0.5, 0.0])
        assert curve["recall"].tolist() == pytest.approx([1.0, 0.5, 0.0])
        assert curve["f1"].tolist() == pytest.approx([0.8, 0.5, 0.0])
        assert curve["savings"].tolist() == pytest.approx([24300, 4400, 0])

    def test_threshold_is_inclusive(self):
        curve = savings_curve(Y4, PROBA4, AMOUNTS4, COSTS, thresholds=[0.6])
        assert curve["n_flagged"].tolist() == [2]

    def test_default_thresholds(self, scored):
        valid, _ = scored
        curve = savings_curve(valid["is_fraud"], valid["proba"], valid["amount"], COSTS)
        assert curve["threshold"].tolist() == pytest.approx(DEFAULT_THRESHOLDS.tolist())
        assert list(curve.index) == list(range(len(DEFAULT_THRESHOLDS)))

    def test_properties_on_real_scores(self, scored):
        valid, _ = scored
        curve = savings_curve(valid["is_fraud"], valid["proba"], valid["amount"], COSTS)
        assert curve["n_flagged"].is_monotonic_decreasing
        assert curve["recall"].is_monotonic_decreasing
        assert curve[["precision", "recall", "f1"]].notna().all().all()
        row = curve.iloc[9]  # порог 0.10
        flagged = valid["proba"] >= row["threshold"]
        assert row["savings"] == pytest.approx(savings(valid["is_fraud"], flagged, valid["amount"], COSTS))
        assert row["n_flagged"] == flagged.sum()

    def test_curve_has_interior_maximum(self, scored):
        # Блокировать слишком много и слишком мало одинаково невыгодно: максимум где-то между
        valid, _ = scored
        curve = savings_curve(valid["is_fraud"], valid["proba"], valid["amount"], COSTS)
        best = curve["savings"].idxmax()
        assert 0 < best < len(curve) - 1
        assert curve["savings"].max() > 2 * max(curve["savings"].iloc[0], curve["savings"].iloc[-1], 0)


class TestBestThreshold:
    def test_savings_and_f1_can_disagree(self):
        grid = [0.1, 0.5, 0.85]
        assert best_threshold(Y5, PROBA5, AMOUNTS5, COSTS, criterion="savings", thresholds=grid) == pytest.approx(0.1)
        assert best_threshold(Y5, PROBA5, AMOUNTS5, COSTS, criterion="f1", thresholds=grid) == pytest.approx(0.85)

    def test_default_criterion_is_savings(self):
        assert best_threshold(Y5, PROBA5, AMOUNTS5, COSTS, thresholds=[0.1, 0.5, 0.85]) == pytest.approx(0.1)

    def test_default_grid_and_smallest_threshold_on_ties(self):
        # Экономия одинакова для всех порогов от 0.01 до 0.20; F1 — для порогов от 0.81 до 0.90
        assert best_threshold(Y5, PROBA5, AMOUNTS5, COSTS, criterion="savings") == pytest.approx(0.01)
        assert best_threshold(Y5, PROBA5, AMOUNTS5, COSTS, criterion="f1") == pytest.approx(0.81)

    def test_unknown_criterion(self):
        with pytest.raises(ValueError):
            best_threshold(Y5, PROBA5, AMOUNTS5, COSTS, criterion="accuracy")

    def test_matches_curve_maximum(self, scored):
        valid, _ = scored
        curve = savings_curve(valid["is_fraud"], valid["proba"], valid["amount"], COSTS)
        threshold = best_threshold(valid["is_fraud"], valid["proba"], valid["amount"], COSTS)
        assert threshold == pytest.approx(curve.loc[curve["savings"].idxmax(), "threshold"])
        assert isinstance(threshold, float)

    def test_savings_threshold_is_far_below_one_half(self, scored):
        valid, _ = scored
        assert best_threshold(valid["is_fraud"], valid["proba"], valid["amount"], COSTS) < 0.3


class TestBreakEvenThreshold:
    def test_manual_example(self):
        thresholds = break_even_threshold([100, 600, 4600, 99600], COSTS)
        assert isinstance(thresholds, np.ndarray)
        assert thresholds.tolist() == pytest.approx([1.0, 0.5, 0.1, 0.005])

    def test_decreases_with_amount_and_is_not_clipped(self):
        thresholds = break_even_threshold(np.array([50, 100, 1000, 10_000, 100_000]), COSTS)
        assert np.all(np.diff(thresholds) < 0)
        assert thresholds[0] > 1

    def test_depends_on_costs(self):
        assert break_even_threshold([1000], Costs(review=0, friction=0)).tolist() == pytest.approx([0.0])
        assert break_even_threshold([1000], Costs(review=200, friction=0)).tolist() == pytest.approx([0.2])


class TestExpectedValueDecision:
    def test_manual_example(self):
        decision = expected_value_decision([0.99, 0.5, 0.11, 0.004], [100, 600, 4600, 99600], COSTS)
        assert decision.dtype == bool
        assert decision.tolist() == [False, False, True, False]

    def test_equivalent_to_positive_expected_savings(self):
        rng = np.random.default_rng(0)
        proba = rng.random(2000)
        amounts = np.exp(rng.normal(7, 1.5, 2000))
        expected = proba * (amounts - COSTS.review) - (1 - proba) * (COSTS.review + COSTS.friction)
        assert np.array_equal(expected_value_decision(proba, amounts, COSTS), expected > 0)

    def test_large_amounts_are_flagged_at_lower_probability(self, scored):
        _, test = scored
        flagged = expected_value_decision(test["proba"], test["amount"], COSTS)
        assert 0 < flagged.sum() < len(test)
        assert test.loc[flagged, "amount"].median() > 2 * test.loc[~flagged, "amount"].median()
        assert test.loc[flagged, "proba"].min() < 0.05

    def test_beats_single_threshold_on_calibrated_model(self, scored):
        valid, test = scored
        threshold = best_threshold(valid["is_fraud"], valid["proba"], valid["amount"], COSTS)
        by_threshold = savings(test["is_fraud"], test["proba"] >= threshold, test["amount"], COSTS)
        by_value = savings(
            test["is_fraud"], expected_value_decision(test["proba"], test["amount"], COSTS), test["amount"], COSTS
        )
        assert by_value > by_threshold > 0

    def test_needs_calibrated_probabilities(self, scored, scored_balanced):
        _, test = scored
        calibrated = savings(
            test["is_fraud"], expected_value_decision(test["proba"], test["amount"], COSTS), test["amount"], COSTS
        )
        flagged = expected_value_decision(scored_balanced["proba"], scored_balanced["amount"], COSTS)
        inflated = savings(scored_balanced["is_fraud"], flagged, scored_balanced["amount"], COSTS)
        assert flagged.mean() > 0.2  # завышенные вероятности: блокируется больше пятой части операций
        assert inflated < 0 < calibrated


class TestBootstrap:
    def test_degenerate_case(self):
        # Все операции дают одинаковый вклад 1000 рублей: разброса нет
        lower, upper = bootstrap_savings_interval([1] * 20, [1] * 20, [1100] * 20, COSTS, n_boot=50)
        assert lower == pytest.approx(20_000)
        assert upper == pytest.approx(20_000)

    def test_reproducible_and_seed_dependent(self, scored):
        _, test = scored
        args = (test["is_fraud"], test["proba"] >= 0.1, test["amount"], COSTS)
        first = bootstrap_savings_interval(*args, n_boot=200, seed=1)
        assert first == bootstrap_savings_interval(*args, n_boot=200, seed=1)
        assert first != bootstrap_savings_interval(*args, n_boot=200, seed=2)

    def test_interval_covers_point_estimate(self, scored):
        _, test = scored
        flagged = test["proba"] >= 0.1
        point = savings(test["is_fraud"], flagged, test["amount"], COSTS)
        lower, upper = bootstrap_savings_interval(test["is_fraud"], flagged, test["amount"], COSTS, n_boot=300)
        assert lower < point < upper
        assert 0.1 * abs(point) < upper - lower < 2 * abs(point)

    def test_difference_with_itself_is_zero(self, scored):
        _, test = scored
        flagged = test["proba"] >= 0.1
        interval = bootstrap_savings_interval(
            test["is_fraud"], flagged, test["amount"], COSTS, baseline=flagged, n_boot=100
        )
        assert interval == pytest.approx((0.0, 0.0))

    def test_paired_difference(self, scored):
        _, test = scored
        nothing = np.zeros(len(test), dtype=bool)
        flagged = test["proba"] >= 0.1
        alone = bootstrap_savings_interval(test["is_fraud"], flagged, test["amount"], COSTS, n_boot=200, seed=3)
        against_nothing = bootstrap_savings_interval(
            test["is_fraud"], flagged, test["amount"], COSTS, baseline=nothing, n_boot=200, seed=3
        )
        assert against_nothing == pytest.approx(alone)

    def test_paired_interval_is_narrower_than_unpaired_spread(self, scored):
        # Две близкие стратегии: разность известна намного точнее, чем каждая экономия по отдельности
        _, test = scored
        first, second = test["proba"] >= 0.10, test["proba"] >= 0.11
        single = bootstrap_savings_interval(test["is_fraud"], first, test["amount"], COSTS, n_boot=300)
        paired = bootstrap_savings_interval(test["is_fraud"], first, test["amount"], COSTS, baseline=second, n_boot=300)
        assert paired[1] - paired[0] < 0.5 * (single[1] - single[0])


class TestCompareStrategies:
    def test_manual_example(self):
        table = pd.DataFrame({"is_fraud": Y5, "proba": PROBA5, "amount": AMOUNTS5})
        result = compare_strategies(table, table, COSTS)
        assert list(result.index) == STRATEGIES
        assert list(result.columns) == ["threshold", "n_flagged", "precision", "recall", "savings", "share_of_oracle"]
        assert result["threshold"].iloc[:3].tolist() == pytest.approx([0.5, 0.81, 0.01])
        assert result["threshold"].iloc[3:].isna().all()
        assert result["n_flagged"].tolist() == [4, 1, 5, 5, 2]
        assert result["savings"].tolist() == pytest.approx([-1400, 100, 48500, 48500, 50000])
        assert result["precision"].tolist() == pytest.approx([0.25, 1.0, 0.4, 0.4, 1.0])
        assert result["recall"].tolist() == pytest.approx([0.5, 0.5, 1.0, 1.0, 1.0])
        assert result["share_of_oracle"].tolist() == pytest.approx([-0.028, 0.002, 0.97, 0.97, 1.0])

    def test_oracle_skips_fraud_cheaper_than_review(self):
        table = pd.DataFrame({"is_fraud": [1, 1, 0], "proba": [0.9, 0.9, 0.1], "amount": [60, 5000, 700]})
        result = compare_strategies(table, table, COSTS)
        assert result.loc["oracle", "n_flagged"] == 1
        assert result.loc["oracle", "savings"] == pytest.approx(4900)

    def test_thresholds_come_from_validation_data(self, scored, comparison):
        valid, _ = scored
        assert comparison.loc["threshold_savings", "threshold"] == pytest.approx(
            best_threshold(valid["is_fraud"], valid["proba"], valid["amount"], COSTS)
        )
        assert comparison.loc["threshold_f1", "threshold"] == pytest.approx(
            best_threshold(valid["is_fraud"], valid["proba"], valid["amount"], COSTS, criterion="f1")
        )

    def test_results_are_computed_on_test_data(self, scored, comparison):
        _, test = scored
        assert comparison.loc["threshold_0.5", "n_flagged"] == (test["proba"] >= 0.5).sum()
        assert comparison.loc["threshold_0.5", "savings"] == pytest.approx(
            savings(test["is_fraud"], test["proba"] >= 0.5, test["amount"], COSTS)
        )

    def test_ordering_of_strategies(self, comparison):
        # Чем больше стратегия знает о деньгах, тем больше экономия
        ordered = comparison["savings"].tolist()
        assert ordered == sorted(ordered)
        assert comparison["share_of_oracle"].max() == pytest.approx(1.0)
        assert comparison.loc["oracle", "precision"] == pytest.approx(1.0)
        assert 0.25 < comparison.loc["expected_value", "share_of_oracle"] < 0.75

    def test_inputs_are_not_modified(self, scored):
        valid, test = scored
        valid_before, test_before = valid.copy(), test.copy()
        compare_strategies(valid, test, COSTS)
        pd.testing.assert_frame_equal(valid, valid_before)
        pd.testing.assert_frame_equal(test, test_before)
