import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))

from ensemble_lab import (  # noqa: E402
    ENSEMBLES,
    EPS,
    average,
    error_correlation,
    evaluate_ensembles,
    fit_blend_weights,
    fit_stacker,
    out_of_fold_predictions,
    predict_stacker,
    rank_average,
)

N_TRAIN, N_TEST = 300, 100


def make_models() -> dict:
    return {
        "logreg": LogisticRegression(C=1.0, max_iter=1000),
        "tree": DecisionTreeClassifier(max_depth=3, random_state=0),
        "knn": KNeighborsClassifier(n_neighbors=15),
    }


def neg_mse(y_true, y_pred) -> float:
    return -mean_squared_error(y_true, y_pred)


@pytest.fixture(scope="module")
def data():
    """Маленькая задача бинарной классификации; индексы таблиц намеренно не совпадают с номерами строк."""
    rng = np.random.default_rng(0)
    n = N_TRAIN + N_TEST
    features = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    logit = 1.2 * features["x0"] - 0.8 * features["x1"] + 0.9 * features["x2"] * features["x3"]
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int).to_numpy()
    features.index = rng.permutation(n) + 1000
    return {
        "X": features.iloc[:N_TRAIN],
        "y": target[:N_TRAIN],
        "X_test": features.iloc[N_TRAIN:],
        "y_test": target[N_TRAIN:],
    }


@pytest.fixture(scope="module")
def folds():
    return KFold(5, shuffle=True, random_state=0)


@pytest.fixture(scope="module")
def predictions(data, folds):
    """Результат out_of_fold_predictions на общих данных: (oof, test)."""
    return out_of_fold_predictions(make_models(), data["X"], data["y"], data["X_test"], folds)


@pytest.fixture(scope="module")
def calibrated():
    """Откалиброванные вероятности p и ответы, сгенерированные с этими вероятностями."""
    rng = np.random.default_rng(1)
    proba = rng.uniform(0.05, 0.95, size=4000)
    target = (rng.random(4000) < proba).astype(int)
    return pd.DataFrame({"good": proba}), target


@pytest.fixture(scope="module")
def noise_models():
    """Мало строк, много моделей-шумов: на таких данных любая подгонка весов выглядит успехом."""
    rng = np.random.default_rng(2)
    oof = pd.DataFrame(rng.uniform(0.2, 0.8, size=(60, 15)), columns=[f"noise{i}" for i in range(15)])
    target = np.tile([0, 1], 30)
    return oof, target


class TestOutOfFoldPredictions:
    def test_structure(self, data, predictions):
        oof, test = predictions
        assert isinstance(oof, pd.DataFrame) and isinstance(test, pd.DataFrame)
        assert list(oof.columns) == ["logreg", "tree", "knn"]
        assert list(test.columns) == ["logreg", "tree", "knn"]
        assert oof.index.equals(data["X"].index)
        assert test.index.equals(data["X_test"].index)
        assert (oof.dtypes == "float64").all() and (test.dtypes == "float64").all()
        assert not oof.isna().any().any() and not test.isna().any().any()
        assert oof.to_numpy().min() >= 0 and oof.to_numpy().max() <= 1

    def test_known_answer_for_regression(self):
        X = pd.DataFrame({"x": np.arange(6.0)})
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        X_test = pd.DataFrame({"x": [10.0, 20.0]}, index=["a", "b"])
        oof, test = out_of_fold_predictions({"mean": DummyRegressor(strategy="mean")}, X, y, X_test, KFold(3))
        # Фолды без перемешивания: [0, 1], [2, 3], [4, 5]; модель предсказывает среднее обучающей части
        assert oof["mean"].tolist() == pytest.approx([3.5, 3.5, 2.5, 2.5, 1.5, 1.5])
        assert test["mean"].tolist() == pytest.approx([2.5, 2.5])
        assert list(test.index) == ["a", "b"]

    def test_classifier_uses_probability_of_second_class(self):
        X = pd.DataFrame({"x": np.arange(8.0)})
        y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
        oof, test = out_of_fold_predictions({"prior": DummyClassifier(strategy="prior")}, X, y, X.iloc[:2], KFold(2))
        # Фолды: [0..3] и [4..7]; доля единиц в обучающей части: 1/4 и 3/4
        assert oof["prior"].tolist() == pytest.approx([0.25] * 4 + [0.75] * 4)
        assert test["prior"].tolist() == pytest.approx([0.5, 0.5])

    def test_matches_manual_computation(self, data, folds, predictions):
        oof, test = predictions
        X, y, X_test = data["X"], data["y"], data["X_test"]
        expected_oof = np.zeros(len(X))
        expected_test = np.zeros(len(X_test))
        for train_idx, valid_idx in folds.split(X, y):
            model = LogisticRegression(C=1.0, max_iter=1000).fit(X.iloc[train_idx], y[train_idx])
            expected_oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
            expected_test += model.predict_proba(X_test)[:, 1] / folds.get_n_splits()
        np.testing.assert_allclose(oof["logreg"].to_numpy(), expected_oof, atol=1e-12)
        np.testing.assert_allclose(test["logreg"].to_numpy(), expected_test, atol=1e-12)

    @pytest.mark.parametrize("position", [0, 57, 299])
    def test_own_target_does_not_leak(self, data, folds, predictions, position):
        """Меняем ответ одного объекта: его собственное предсказание измениться не должно."""
        oof, _ = predictions
        changed = data["y"].copy()
        changed[position] = 1 - changed[position]
        oof_changed, _ = out_of_fold_predictions(make_models(), data["X"], changed, data["X_test"], folds)
        np.testing.assert_allclose(oof_changed.iloc[position].to_numpy(), oof.iloc[position].to_numpy(), atol=1e-12)
        # Ответ при этом используется: предсказания объектов из других фолдов сдвинулись
        assert not np.allclose(oof_changed["logreg"].to_numpy(), oof["logreg"].to_numpy(), atol=1e-12)

    def test_memorizing_model_gains_nothing(self):
        """1-NN на своих обучающих объектах повторяет ответы; на out-of-fold запоминание не помогает."""
        rng = np.random.default_rng(3)
        X = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
        y = rng.integers(0, 2, size=200)
        oof, _ = out_of_fold_predictions({"nn": KNeighborsClassifier(n_neighbors=1)}, X, y, X.iloc[:5], KFold(5))
        accuracy = ((oof["nn"] > 0.5).astype(int).to_numpy() == y).mean()
        assert 0.3 < accuracy < 0.7

    def test_models_are_not_fitted(self, data, folds):
        models = make_models()
        params_before = {name: model.get_params() for name, model in models.items()}
        out_of_fold_predictions(models, data["X"], data["y"], data["X_test"], folds)
        for name, model in models.items():
            with pytest.raises(NotFittedError):
                check_is_fitted(model)
            assert model.get_params() == params_before[name]

    def test_same_folds_for_all_models(self, data):
        """Разбиение без фиксированного random_state: фолды всё равно должны быть общими."""
        models = {"first": LogisticRegression(max_iter=1000), "second": LogisticRegression(max_iter=1000)}
        oof, test = out_of_fold_predictions(models, data["X"], data["y"], data["X_test"], KFold(5, shuffle=True))
        np.testing.assert_allclose(oof["first"].to_numpy(), oof["second"].to_numpy(), atol=1e-12)
        np.testing.assert_allclose(test["first"].to_numpy(), test["second"].to_numpy(), atol=1e-12)

    def test_target_is_aligned_by_position(self, data, folds, predictions):
        oof, test = predictions
        as_series = pd.Series(data["y"], index=np.arange(len(data["y"]))[::-1])
        oof_series, test_series = out_of_fold_predictions(make_models(), data["X"], as_series, data["X_test"], folds)
        pd.testing.assert_frame_equal(oof_series, oof)
        pd.testing.assert_frame_equal(test_series, test)

    def test_stratified_folds_are_supported(self, data):
        cv = StratifiedKFold(4, shuffle=True, random_state=1)
        oof, _ = out_of_fold_predictions(
            {"logreg": LogisticRegression(max_iter=1000)}, data["X"], data["y"], data["X_test"], cv
        )
        assert roc_auc_score(data["y"], oof["logreg"]) > 0.6

    def test_empty_models(self, data, folds):
        with pytest.raises(ValueError):
            out_of_fold_predictions({}, data["X"], data["y"], data["X_test"], folds)

    @pytest.mark.parametrize("cv", [ShuffleSplit(5, test_size=0.2, random_state=0), TimeSeriesSplit(4)])
    def test_incomplete_split_is_rejected(self, data, cv):
        with pytest.raises(ValueError):
            out_of_fold_predictions(
                {"logreg": LogisticRegression(max_iter=1000)}, data["X"], data["y"], data["X_test"], cv
            )

    def test_reproducible_and_inputs_unchanged(self, data, folds, predictions):
        X_before, y_before, test_before = data["X"].copy(), data["y"].copy(), data["X_test"].copy()
        oof, test = out_of_fold_predictions(make_models(), data["X"], data["y"], data["X_test"], folds)
        pd.testing.assert_frame_equal(oof, predictions[0])
        pd.testing.assert_frame_equal(test, predictions[1])
        pd.testing.assert_frame_equal(data["X"], X_before)
        pd.testing.assert_frame_equal(data["X_test"], test_before)
        np.testing.assert_array_equal(data["y"], y_before)


class TestErrorCorrelation:
    def test_known_answer(self):
        y = np.array([0.0, 0.0, 0.0, 0.0])
        shift = np.array([1.0, -1.0, 2.0, -2.0])
        oof = pd.DataFrame({"a": y + shift, "same": y + 3 * shift, "opposite": y - shift})
        corr = error_correlation(oof, y)
        assert corr.loc["a", "same"] == pytest.approx(1.0)
        assert corr.loc["a", "opposite"] == pytest.approx(-1.0)
        assert corr.loc["same", "opposite"] == pytest.approx(-1.0)

    def test_structure(self, data, predictions):
        corr = error_correlation(predictions[0], data["y"])
        assert list(corr.index) == ["logreg", "tree", "knn"]
        assert list(corr.columns) == ["logreg", "tree", "knn"]
        np.testing.assert_allclose(np.diag(corr), 1.0)
        np.testing.assert_allclose(corr.to_numpy(), corr.to_numpy().T)
        assert (corr.to_numpy() <= 1 + 1e-12).all()

    def test_errors_not_predictions(self):
        """Предсказания двух моделей почти совпадают, а ошибки независимы."""
        rng = np.random.default_rng(4)
        y = rng.normal(0, 10, size=2000)
        oof = pd.DataFrame({"a": y + rng.normal(size=2000), "b": y + rng.normal(size=2000)})
        assert oof["a"].corr(oof["b"]) > 0.95
        assert abs(error_correlation(oof, y).loc["a", "b"]) < 0.1

    def test_matches_definition(self, data, predictions):
        oof = predictions[0]
        expected = np.corrcoef(data["y"] - oof["logreg"].to_numpy(), data["y"] - oof["knn"].to_numpy())[0, 1]
        assert error_correlation(oof, data["y"]).loc["logreg", "knn"] == pytest.approx(expected)

    def test_target_is_aligned_by_position(self, data, predictions):
        oof = predictions[0]
        as_series = pd.Series(data["y"], index=np.arange(len(data["y"]))[::-1])
        pd.testing.assert_frame_equal(error_correlation(oof, as_series), error_correlation(oof, data["y"]))


class TestAverage:
    PREDS = pd.DataFrame({"a": [0.2, 0.4, 1.0], "b": [0.6, 0.0, 0.0]}, index=[7, 8, 9])

    def test_simple_mean(self):
        result = average(self.PREDS)
        assert result.tolist() == pytest.approx([0.4, 0.2, 0.5])
        assert result.name == "average"
        assert list(result.index) == [7, 8, 9]

    def test_weights(self):
        assert average(self.PREDS, {"a": 0.25, "b": 0.75}).tolist() == pytest.approx([0.5, 0.1, 0.25])

    def test_weights_are_normalized(self):
        assert average(self.PREDS, {"a": 1, "b": 3}).tolist() == pytest.approx([0.5, 0.1, 0.25])

    def test_weights_as_series_in_any_order(self):
        weights = pd.Series({"b": 0.75, "a": 0.25})
        assert average(self.PREDS, weights).tolist() == pytest.approx([0.5, 0.1, 0.25])

    def test_zero_weight_removes_model(self):
        assert average(self.PREDS, {"a": 1.0, "b": 0.0}).tolist() == pytest.approx([0.2, 0.4, 1.0])

    @pytest.mark.parametrize("weights", [{"a": 1.0}, {"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 0.5, "z": 0.5}])
    def test_wrong_names(self, weights):
        with pytest.raises(KeyError):
            average(self.PREDS, weights)

    @pytest.mark.parametrize("weights", [{"a": -0.5, "b": 1.5}, {"a": 0.0, "b": 0.0}])
    def test_invalid_weights(self, weights):
        with pytest.raises(ValueError):
            average(self.PREDS, weights)

    def test_input_unchanged(self):
        before = self.PREDS.copy()
        average(self.PREDS, {"a": 1, "b": 3})
        pd.testing.assert_frame_equal(self.PREDS, before)


class TestRankAverage:
    def test_known_answer(self):
        preds = pd.DataFrame({"a": [0.1, 0.9, 0.5], "b": [30.0, 20.0, 10.0]}, index=["x", "y", "z"])
        result = rank_average(preds)
        # Ранги: a — 1, 3, 2; b — 3, 2, 1; делим на 3 и усредняем
        assert result.tolist() == pytest.approx([2 / 3, 5 / 6, 1 / 2])
        assert result.name == "rank_average"
        assert list(result.index) == ["x", "y", "z"]

    def test_ties_get_average_rank(self):
        preds = pd.DataFrame({"a": [0.2, 0.2, 0.7, 0.9]})
        assert rank_average(preds).tolist() == pytest.approx([1.5 / 4, 1.5 / 4, 3 / 4, 1.0])

    def test_scale_does_not_matter(self, predictions):
        oof = predictions[0]
        transformed = oof.copy()
        transformed["logreg"] = transformed["logreg"] * 1000
        transformed["knn"] = np.exp(5 * transformed["knn"])
        pd.testing.assert_series_equal(rank_average(transformed), rank_average(oof))

    def test_range(self, predictions):
        result = rank_average(predictions[0])
        assert result.min() > 0 and result.max() <= 1
        assert result.index.equals(predictions[0].index)

    def test_keeps_ranking_quality(self, data, predictions):
        oof = predictions[0]
        assert roc_auc_score(data["y"], rank_average(oof)) > roc_auc_score(data["y"], oof["tree"])


class TestFitBlendWeights:
    def test_structure(self, data, predictions):
        weights = fit_blend_weights(predictions[0], data["y"], roc_auc_score)
        assert isinstance(weights, pd.Series)
        assert weights.name == "weight"
        assert list(weights.index) == ["logreg", "tree", "knn"]
        assert weights.dtype == "float64"
        assert (weights >= 0).all()
        assert weights.sum() == pytest.approx(1.0)

    def test_opposite_errors_get_equal_weights(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        shift = np.array([1.0, -1.0, 1.0, -1.0, 2.0, -2.0])
        oof = pd.DataFrame({"plus": y + shift, "minus": y - shift})
        weights = fit_blend_weights(oof, y, mean_squared_error, greater_is_better=False)
        assert weights.tolist() == pytest.approx([0.5, 0.5])

    def test_unequal_errors(self):
        y = np.zeros(8)
        shift = np.array([1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 3.0, -3.0])
        oof = pd.DataFrame({"far": y + 3 * shift, "near": y - shift})
        # Ошибки гасятся при весах 1/4 и 3/4
        weights = fit_blend_weights(oof, y, mean_squared_error, greater_is_better=False)
        assert weights.tolist() == pytest.approx([0.25, 0.75])

    def test_useless_model_gets_zero(self):
        rng = np.random.default_rng(5)
        y = rng.normal(size=200)
        oof = pd.DataFrame({"junk": rng.normal(0, 5, size=200), "exact": y, "junk2": np.zeros(200)})
        weights = fit_blend_weights(oof, y, mean_squared_error, greater_is_better=False)
        assert weights.tolist() == pytest.approx([0.0, 1.0, 0.0])

    def test_one_round_selects_best_single_model(self, data, predictions):
        oof = predictions[0]
        weights = fit_blend_weights(oof, data["y"], roc_auc_score, n_rounds=1)
        best = max(oof.columns, key=lambda name: roc_auc_score(data["y"], oof[name]))
        assert weights[best] == pytest.approx(1.0)
        assert weights.sum() == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("metric", "greater_is_better"), [(roc_auc_score, True), (log_loss, False), (neg_mse, True)]
    )
    def test_not_worse_than_best_single_model(self, data, predictions, metric, greater_is_better):
        oof, y = predictions[0], data["y"]
        weights = fit_blend_weights(oof, y, metric, greater_is_better=greater_is_better)
        blended = metric(y, average(oof, weights))
        singles = [metric(y, oof[name]) for name in oof.columns]
        if greater_is_better:
            assert blended >= max(singles) - 1e-12
        else:
            assert blended <= min(singles) + 1e-12

    def test_direction_of_metric(self, data, predictions):
        oof, y = predictions[0], data["y"]
        as_loss = fit_blend_weights(oof, y, mean_squared_error, greater_is_better=False)
        as_score = fit_blend_weights(oof, y, neg_mse, greater_is_better=True)
        pd.testing.assert_series_equal(as_loss, as_score)

    def test_weights_are_multiples_of_round_share(self, data, predictions):
        weights = fit_blend_weights(predictions[0], data["y"], log_loss, greater_is_better=False, n_rounds=10)
        positive = weights[weights > 0]
        # Набор не длиннее 10 моделей: каждый ненулевой вес не меньше 1/10
        assert (positive >= 0.1 - 1e-12).all()

    def test_invalid_rounds(self, data, predictions):
        with pytest.raises(ValueError):
            fit_blend_weights(predictions[0], data["y"], roc_auc_score, n_rounds=0)

    def test_reproducible_and_inputs_unchanged(self, data, predictions):
        oof, y = predictions[0], data["y"]
        oof_before, y_before = oof.copy(), y.copy()
        first = fit_blend_weights(oof, y, roc_auc_score)
        second = fit_blend_weights(oof, y, roc_auc_score)
        pd.testing.assert_series_equal(first, second)
        pd.testing.assert_frame_equal(oof, oof_before)
        np.testing.assert_array_equal(y, y_before)


class TestStacker:
    def test_structure(self, data, predictions):
        oof, test = predictions
        result = predict_stacker(fit_stacker(oof, data["y"]), test)
        assert isinstance(result, pd.Series)
        assert result.name == "stacking"
        assert result.index.equals(test.index)
        assert result.min() > 0 and result.max() < 1

    def test_calibrated_model_is_left_alone(self, calibrated):
        oof, y = calibrated
        result = predict_stacker(fit_stacker(oof, y, C=100.0), oof)
        assert np.abs(result.to_numpy() - oof["good"].to_numpy()).mean() < 0.02

    def test_distorted_scale_is_repaired(self, calibrated):
        """Модель занижает вероятности (p в кубе): метамодель возвращает их в правильную шкалу."""
        oof, y = calibrated
        distorted = pd.DataFrame({"model": oof["good"] ** 3})
        result = predict_stacker(fit_stacker(distorted, y, C=100.0), distorted)
        assert log_loss(y, distorted["model"]) - log_loss(y, result) > 0.1
        assert log_loss(y, result) < log_loss(y, oof["good"]) + 0.01

    def test_noise_model_is_ignored(self, calibrated):
        oof, y = calibrated
        rng = np.random.default_rng(6)
        with_noise = oof.assign(noise=rng.uniform(0.05, 0.95, size=len(oof)))
        alone = predict_stacker(fit_stacker(oof, y), oof)
        both = predict_stacker(fit_stacker(with_noise, y), with_noise)
        assert np.abs(alone.to_numpy() - both.to_numpy()).mean() < 0.02

    def test_matches_logistic_regression_on_logits(self, data, predictions):
        oof, test = predictions
        clipped = np.clip(oof.to_numpy(), EPS, 1 - EPS)
        clipped_test = np.clip(test.to_numpy(), EPS, 1 - EPS)
        meta = LogisticRegression(C=0.5, max_iter=1000).fit(np.log(clipped / (1 - clipped)), data["y"])
        expected = meta.predict_proba(np.log(clipped_test / (1 - clipped_test)))[:, 1]
        result = predict_stacker(fit_stacker(oof, data["y"], C=0.5), test)
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-8)

    def test_column_order_and_extra_columns(self, data, predictions):
        oof, test = predictions
        stacker = fit_stacker(oof, data["y"])
        expected = predict_stacker(stacker, test)
        shuffled = test[["knn", "logreg", "tree"]].assign(extra=0.5)
        pd.testing.assert_series_equal(predict_stacker(stacker, shuffled), expected)

    def test_missing_column(self, data, predictions):
        oof, test = predictions
        stacker = fit_stacker(oof, data["y"])
        with pytest.raises(KeyError):
            predict_stacker(stacker, test.drop(columns="tree"))

    def test_zero_and_one_are_clipped(self):
        oof = pd.DataFrame({"hard": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]})
        y = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        result = predict_stacker(fit_stacker(oof, y), oof)
        assert np.isfinite(result).all()
        assert result.min() > 0 and result.max() < 1

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_values_outside_unit_interval(self, data, predictions, bad):
        oof = predictions[0].copy()
        oof.iloc[0, 0] = bad
        with pytest.raises(ValueError):
            fit_stacker(oof, data["y"])
        stacker = fit_stacker(predictions[0], data["y"])
        with pytest.raises(ValueError):
            predict_stacker(stacker, oof)

    def test_regularization_strength(self, data, predictions):
        oof, test = predictions
        weak = predict_stacker(fit_stacker(oof, data["y"], C=100.0), test)
        strong = predict_stacker(fit_stacker(oof, data["y"], C=1e-4), test)
        assert strong.std() < 0.5 * weak.std()

    def test_inputs_unchanged(self, data, predictions):
        oof, test = predictions
        oof_before, test_before = oof.copy(), test.copy()
        predict_stacker(fit_stacker(oof, data["y"]), test)
        pd.testing.assert_frame_equal(oof, oof_before)
        pd.testing.assert_frame_equal(test, test_before)


class TestEvaluateEnsembles:
    def test_structure(self, data, predictions):
        table = evaluate_ensembles(predictions[0], data["y"], roc_auc_score)
        assert list(table.index) == ["logreg", "tree", "knn", *ENSEMBLES]
        assert table.index.name == "method"
        assert list(table.columns) == ["score", "gain"]
        assert not table.isna().any().any()

    def test_single_models_and_simple_ensembles(self, data, predictions):
        oof, y = predictions[0], data["y"]
        table = evaluate_ensembles(oof, y, roc_auc_score)
        for name in oof.columns:
            assert table.loc[name, "score"] == pytest.approx(roc_auc_score(y, oof[name]))
        assert table.loc["average", "score"] == pytest.approx(roc_auc_score(y, average(oof)))
        assert table.loc["rank_average", "score"] == pytest.approx(roc_auc_score(y, rank_average(oof)))

    def test_gain_when_greater_is_better(self, data, predictions):
        oof = predictions[0]
        table = evaluate_ensembles(oof, data["y"], roc_auc_score, greater_is_better=True)
        best = table.loc[list(oof.columns), "score"].max()
        np.testing.assert_allclose(table["gain"].to_numpy(), (table["score"] - best).to_numpy(), atol=1e-12)
        assert table.loc[list(oof.columns), "gain"].max() == pytest.approx(0.0)

    def test_gain_when_lower_is_better(self, data, predictions):
        oof = predictions[0]
        table = evaluate_ensembles(oof, data["y"], log_loss, greater_is_better=False)
        best = table.loc[list(oof.columns), "score"].min()
        np.testing.assert_allclose(table["gain"].to_numpy(), (best - table["score"]).to_numpy(), atol=1e-12)
        assert table.loc[list(oof.columns), "gain"].max() == pytest.approx(0.0)

    def test_weighted_matches_manual_cross_fitting(self, data, predictions):
        oof, y = predictions[0], data["y"]
        cv = StratifiedKFold(4, shuffle=True, random_state=7)
        expected = np.zeros(len(oof))
        for train_idx, valid_idx in cv.split(oof, y):
            weights = fit_blend_weights(oof.iloc[train_idx], y[train_idx], roc_auc_score, n_rounds=20)
            expected[valid_idx] = average(oof.iloc[valid_idx], weights).to_numpy()
        table = evaluate_ensembles(oof, y, roc_auc_score, cv=cv, n_rounds=20)
        assert table.loc["weighted", "score"] == pytest.approx(roc_auc_score(y, expected))

    def test_stacking_matches_manual_cross_fitting(self, data, predictions):
        oof, y = predictions[0], data["y"]
        expected = np.zeros(len(oof))
        for train_idx, valid_idx in StratifiedKFold(5, shuffle=True, random_state=0).split(oof, y):
            stacker = fit_stacker(oof.iloc[train_idx], y[train_idx], C=0.3)
            expected[valid_idx] = predict_stacker(stacker, oof.iloc[valid_idx]).to_numpy()
        table = evaluate_ensembles(oof, y, log_loss, greater_is_better=False, C=0.3)
        assert table.loc["stacking", "score"] == pytest.approx(log_loss(y, expected))

    def test_stacking_score_is_honest(self, noise_models):
        """Модели — чистый шум. Оценка метамодели на своих обучающих строках была бы «лучше случайной»."""
        oof, y = noise_models
        in_sample = log_loss(y, predict_stacker(fit_stacker(oof, y, C=100.0), oof))
        table = evaluate_ensembles(oof, y, log_loss, greater_is_better=False, C=100.0)
        assert in_sample < 0.6
        assert table.loc["stacking", "score"] > np.log(2)

    def test_weighted_score_is_honest(self, noise_models):
        oof, y = noise_models
        weights = fit_blend_weights(oof, y, roc_auc_score)
        in_sample = roc_auc_score(y, average(oof, weights))
        table = evaluate_ensembles(oof, y, roc_auc_score)
        assert in_sample > 0.65
        assert table.loc["weighted", "score"] < in_sample - 0.05

    def test_reproducible_and_inputs_unchanged(self, data, predictions):
        oof, y = predictions[0], data["y"]
        oof_before, y_before = oof.copy(), y.copy()
        first = evaluate_ensembles(oof, y, roc_auc_score)
        second = evaluate_ensembles(oof, y, roc_auc_score)
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(oof, oof_before)
        np.testing.assert_array_equal(y, y_before)
