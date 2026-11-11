import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))
sys.path.insert(0, str(PRACTICE_DIR.parent))

from churn_data import make_clients  # noqa: E402

from churn_report import (  # noqa: E402
    FEATURES,
    TARGET,
    average_marginal_effects,
    bootstrap_coef_intervals,
    evaluate,
    fit_logit,
    odds_ratio_table,
    predict_scenarios,
    prepare_features,
)


@pytest.fixture(scope="module")
def clients() -> pd.DataFrame:
    return make_clients()


@pytest.fixture(scope="module")
def features(clients) -> pd.DataFrame:
    return prepare_features(clients)


@pytest.fixture(scope="module")
def result(features, clients):
    return fit_logit(features, clients[TARGET])


@pytest.fixture(scope="module")
def intervals(features, clients) -> pd.DataFrame:
    return bootstrap_coef_intervals(features, clients[TARGET], n_boot=100, seed=0)


@pytest.fixture(scope="module")
def metrics(features, clients) -> dict[str, float]:
    return evaluate(features, clients[TARGET])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class TestPrepareFeatures:
    def test_columns_and_index(self, clients, features):
        assert list(features.columns) == FEATURES
        assert features.index.equals(clients.index)

    def test_all_numeric_without_missing_values(self, features):
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes)
        assert features.notna().all().all()

    def test_encoding(self, clients, features):
        assert features["fee_k"].tolist() == pytest.approx((clients["monthly_fee"] / 1000).tolist())
        assert set(features["annual"].unique()) == {0, 1}
        assert (features["annual"] == 1).sum() == (clients["contract"] == "annual").sum()
        assert set(features["autopay"].unique()) == {0, 1}
        assert features["autopay"].sum() == clients["autopay"].sum()

    def test_input_is_not_modified(self, clients):
        before = clients.copy()
        prepare_features(clients)
        pd.testing.assert_frame_equal(clients, before)


class TestFitLogit:
    def test_parameter_names(self, result):
        assert list(result.params.index) == ["const", *FEATURES]

    def test_recovers_known_coefficients(self):
        # Данные с известными коэффициентами и без скрытых переменных
        rng = np.random.default_rng(0)
        x = pd.DataFrame({"a": rng.normal(size=20000), "b": rng.integers(0, 2, 20000)})
        y = pd.Series((rng.random(20000) < sigmoid(-0.5 + 1.0 * x["a"] - 2.0 * x["b"])).astype(int))
        params = fit_logit(x, y).params
        assert params["const"] == pytest.approx(-0.5, abs=0.08)
        assert params["a"] == pytest.approx(1.0, abs=0.08)
        assert params["b"] == pytest.approx(-2.0, abs=0.1)

    def test_no_regularization(self, features, clients):
        # У регуляризованной модели на маленькой выборке коэффициенты заметно меньше по модулю
        small = features.sample(200, random_state=0)
        params = fit_logit(small, clients.loc[small.index, TARGET]).params
        assert params["fee_k"] == pytest.approx(1.34, abs=0.03)

    def test_signs_on_course_data(self, result):
        assert result.params["tenure_months"] < 0
        assert result.params["annual"] < 0
        assert result.params["support_calls"] > 0


class TestOddsRatioTable:
    def test_structure(self, result):
        table = odds_ratio_table(result)
        assert list(table.index) == FEATURES
        assert list(table.columns) == ["coef", "odds_ratio", "ci_lower", "ci_upper", "p_value"]

    def test_values(self, result):
        table = odds_ratio_table(result)
        assert np.allclose(table["odds_ratio"], np.exp(table["coef"]))
        assert (table["ci_lower"] < table["odds_ratio"]).all()
        assert (table["odds_ratio"] < table["ci_upper"]).all()
        assert table["p_value"].between(0, 1).all()

    def test_intervals_are_for_odds_ratios(self, result):
        table = odds_ratio_table(result)
        # Интервал отношения шансов несимметричен, а его логарифм симметричен относительно coef
        log_mid = (np.log(table["ci_lower"]) + np.log(table["ci_upper"])) / 2
        assert np.allclose(log_mid, table["coef"])
        assert table.loc["annual", "ci_upper"] < 1
        assert table.loc["support_calls", "ci_lower"] > 1


class TestAverageMarginalEffects:
    def test_structure(self, result, features):
        effects = average_marginal_effects(result, features)
        assert list(effects.index) == FEATURES

    def test_binary_feature_is_difference_of_probabilities(self):
        rng = np.random.default_rng(1)
        x = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.integers(0, 2, 5000)})
        y = pd.Series((rng.random(5000) < sigmoid(0.3 + 0.8 * x["a"] - 1.5 * x["b"])).astype(int))
        fitted = fit_logit(x, y)
        effects = average_marginal_effects(fitted, x)

        const, wa, wb = fitted.params["const"], fitted.params["a"], fitted.params["b"]
        with_b = sigmoid(const + wa * x["a"] + wb)
        without_b = sigmoid(const + wa * x["a"])
        assert effects["b"] == pytest.approx((with_b - without_b).mean())

        p = sigmoid(const + wa * x["a"] + wb * x["b"])
        assert effects["a"] == pytest.approx((wa * p * (1 - p)).mean())

    def test_on_course_data(self, result, features):
        effects = average_marginal_effects(result, features)
        assert effects["annual"] == pytest.approx(-0.26, abs=0.02)
        assert effects["support_calls"] == pytest.approx(0.07, abs=0.01)
        assert effects.abs().max() < 1


class TestPredictScenarios:
    def test_matches_manual_calculation(self, result):
        scenario = pd.DataFrame(
            {
                "tenure_months": [3, 36],
                "fee_k": [0.79, 0.79],
                "services": [2, 2],
                "annual": [0, 1],
                "autopay": [0, 1],
                "support_calls": [1, 0],
            },
            index=["новичок", "давний клиент"],
        )
        probabilities = predict_scenarios(result, scenario)
        expected = sigmoid(result.params["const"] + scenario[FEATURES].to_numpy() @ result.params[FEATURES].to_numpy())
        assert list(probabilities.index) == ["новичок", "давний клиент"]
        assert np.allclose(probabilities.to_numpy(), expected)
        assert probabilities["новичок"] > probabilities["давний клиент"]

    def test_column_order_does_not_matter(self, result):
        scenario = pd.DataFrame(
            {
                "tenure_months": [12],
                "fee_k": [1.19],
                "services": [3],
                "annual": [0],
                "autopay": [1],
                "support_calls": [2],
            }
        )
        shuffled = scenario[list(reversed(FEATURES))]
        assert predict_scenarios(result, shuffled).iloc[0] == pytest.approx(predict_scenarios(result, scenario).iloc[0])

    def test_single_row(self, result, features):
        probability = predict_scenarios(result, features.iloc[[0]])
        assert len(probability) == 1
        assert 0 < probability.iloc[0] < 1


class TestBootstrap:
    def test_structure(self, intervals):
        assert list(intervals.index) == ["const", *FEATURES]
        assert list(intervals.columns) == ["lower", "upper"]

    def test_intervals_contain_point_estimate(self, intervals, result):
        assert (intervals["lower"] < result.params).all()
        assert (result.params < intervals["upper"]).all()

    def test_width_is_close_to_analytic(self, intervals, result):
        analytic = result.conf_int()
        ratio = (intervals["upper"] - intervals["lower"]) / (analytic[1] - analytic[0])
        assert ratio.between(0.6, 1.5).all()

    def test_reproducible(self, features, clients):
        small = features.iloc[:500]
        target = clients[TARGET].iloc[:500]
        first = bootstrap_coef_intervals(small, target, n_boot=30, seed=7)
        second = bootstrap_coef_intervals(small, target, n_boot=30, seed=7)
        pd.testing.assert_frame_equal(first, second)
        other = bootstrap_coef_intervals(small, target, n_boot=30, seed=8)
        assert not np.allclose(first.to_numpy(), other.to_numpy())


class TestEvaluate:
    def test_keys(self, metrics):
        assert set(metrics) == {"roc_auc", "log_loss", "baseline_log_loss"}

    def test_model_is_better_than_baseline(self, metrics):
        assert 0.74 < metrics["roc_auc"] < 0.80
        assert metrics["log_loss"] < metrics["baseline_log_loss"]
        assert metrics["baseline_log_loss"] == pytest.approx(0.651, abs=0.005)

    def test_noise_features_give_no_skill(self, clients):
        rng = np.random.default_rng(0)
        noise = pd.DataFrame(rng.normal(size=(len(clients), 3)), columns=["n1", "n2", "n3"])
        metrics = evaluate(noise, clients[TARGET])
        assert metrics["roc_auc"] == pytest.approx(0.5, abs=0.03)
        assert metrics["log_loss"] >= metrics["baseline_log_loss"] - 0.001
