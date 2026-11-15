import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine, make_classification, make_friedman1
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import log_loss
from sklearn.utils.validation import check_is_fitted

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))

from boosting_bench import (  # noqa: E402
    LIBRARIES,
    MISSING,
    benchmark,
    evaluate,
    fit_with_early_stopping,
    make_models,
    prepare_categoricals,
    split_train_valid_test,
    summarize,
)

# Маленькие шумные данные и большой шаг: ранняя остановка срабатывает за два-три десятка итераций
FAST = {"max_trees": 300, "learning_rate": 0.3}
ROUNDS = 10
COLUMNS = ["score", "valid_score", "n_trees", "fit_time", "predict_time"]
TREES_PARAM = {"xgboost": "n_estimators", "lightgbm": "n_estimators", "catboost": "iterations", "sklearn": "max_iter"}
SEED_PARAM = {
    "xgboost": "random_state",
    "lightgbm": "random_state",
    "catboost": "random_seed",
    "sklearn": "random_state",
}


def is_fitted(model) -> bool:
    if hasattr(model, "is_fitted"):  # catboost
        return bool(model.is_fitted())
    try:
        check_is_fitted(model)
    except NotFittedError:
        return False
    return True


class ConstantRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


class FixedProbabilities:
    def __init__(self, proba, classes):
        self.proba = np.asarray(proba)
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X):
        return self.proba


@pytest.fixture(scope="module")
def reg_data():
    X, y = make_friedman1(n_samples=500, n_features=8, noise=3.0, random_state=0)
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(8)])
    return split_train_valid_test(X, pd.Series(y, name="y"), random_state=0)


@pytest.fixture(scope="module")
def clf_data():
    X, y = make_classification(n_samples=500, n_features=8, n_informative=4, flip_y=0.2, random_state=0)
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(8)])
    return split_train_valid_test(X, pd.Series(y, name="y"), random_state=0, stratify=True)


@pytest.fixture(scope="module")
def cat_frames():
    train = pd.DataFrame(
        {"city": ["b", "a", "c", "a", np.nan], "code": [3, 1, 2, 1, 3], "x": [0.5, 1.5, 2.5, 3.5, 4.5]}
    )
    valid = pd.DataFrame({"city": ["a", "d"], "code": [2, 7], "x": [5.5, 6.5]})
    test = pd.DataFrame({"city": ["c", np.nan, "z"], "code": [1, 3, 9], "x": [7.5, np.nan, 9.5]})
    return train, valid, test


@pytest.fixture(scope="module")
def cat_data():
    # Вероятность класса 1 определяется городом; в конце таблицы появляются города, которых раньше не было
    rng = np.random.default_rng(0)
    n = 1500
    city = rng.integers(0, 20, n)
    city[-250:] = np.where(rng.random(250) < 0.1, rng.integers(20, 23, 250), city[-250:])
    effect = np.append(np.where(np.arange(20) % 2 == 0, 2.0, -2.0), [0.0, 0.0, 0.0])
    y = (rng.random(n) < 1 / (1 + np.exp(-effect[city]))).astype(int)
    X = pd.DataFrame({"city": [f"city_{code:02d}" for code in city], "noise": rng.normal(size=n)})
    X.loc[rng.random(n) < 0.05, "city"] = np.nan
    return split_train_valid_test(X, pd.Series(y, name="y"), time=np.arange(n))


@pytest.fixture(scope="module")
def reg_fitted(reg_data):
    X_train, X_valid, _, y_train, y_valid, _ = reg_data
    models = make_models("regression", **FAST)
    return {
        name: fit_with_early_stopping(model, X_train, y_train, X_valid, y_valid, ROUNDS)
        for name, model in models.items()
    }


@pytest.fixture(scope="module")
def clf_fitted(clf_data):
    X_train, X_valid, _, y_train, y_valid, _ = clf_data
    models = make_models("classification", **FAST)
    return {
        name: fit_with_early_stopping(model, X_train, y_train, X_valid, y_valid, ROUNDS)
        for name, model in models.items()
    }


@pytest.fixture(scope="module")
def reg_bench(reg_data):
    return benchmark(make_models("regression", **FAST), reg_data, "regression", rounds=ROUNDS)


@pytest.fixture(scope="module")
def clf_bench(clf_data):
    return benchmark(make_models("classification", **FAST), clf_data, "classification", rounds=ROUNDS)


@pytest.fixture(scope="module")
def cat_bench(cat_data):
    models = make_models("classification", **FAST)
    return benchmark(models, cat_data, "classification", rounds=ROUNDS, cat_features=["city"])


class TestMakeModels:
    def test_keys_and_order(self):
        assert tuple(make_models("regression")) == LIBRARIES == ("xgboost", "lightgbm", "catboost", "sklearn")

    @pytest.mark.parametrize("library", LIBRARIES)
    def test_model_comes_from_its_library(self, library):
        for task in ("regression", "classification"):
            model = make_models(task)[library]
            assert type(model).__module__.split(".")[0] == library
        assert type(make_models("regression")["sklearn"]).__name__ == "HistGradientBoostingRegressor"
        assert type(make_models("classification")["sklearn"]).__name__ == "HistGradientBoostingClassifier"

    @pytest.mark.parametrize("library", LIBRARIES)
    def test_task_defines_model_type(self, library):
        assert not hasattr(make_models("regression")[library], "predict_proba")
        assert hasattr(make_models("classification")[library], "predict_proba")

    @pytest.mark.parametrize("library", LIBRARIES)
    def test_common_parameters(self, library):
        params = make_models("regression", random_state=7, max_trees=123, learning_rate=0.07)[library].get_params()
        assert params[TREES_PARAM[library]] == 123
        assert params["learning_rate"] == pytest.approx(0.07)
        assert params[SEED_PARAM[library]] == 7

    def test_threads(self):
        models = make_models("regression", n_jobs=2)
        assert models["xgboost"].get_params()["n_jobs"] == 2
        assert models["lightgbm"].get_params()["n_jobs"] == 2
        assert models["catboost"].get_params()["thread_count"] == 2

    def test_service_parameters(self):
        models = make_models("classification")
        assert models["xgboost"].get_params()["enable_categorical"] is True
        assert models["lightgbm"].get_params()["verbose"] == -1
        assert models["catboost"].get_params()["verbose"] == 0
        assert models["catboost"].get_params()["allow_writing_files"] is False
        assert models["sklearn"].get_params()["early_stopping"] is False

    def test_overrides_have_priority(self):
        overrides = {"xgboost": {"max_depth": 2, "learning_rate": 0.5}, "catboost": {"depth": 3}}
        models = make_models("regression", learning_rate=0.1, overrides=overrides)
        assert models["xgboost"].get_params()["max_depth"] == 2
        assert models["xgboost"].get_params()["learning_rate"] == pytest.approx(0.5)
        assert models["catboost"].get_params()["depth"] == 3
        assert models["lightgbm"].get_params()["learning_rate"] == pytest.approx(0.1)

    def test_errors(self):
        with pytest.raises(ValueError):
            make_models("ranking")
        with pytest.raises(ValueError):
            make_models("regression", overrides={"h2o": {"max_depth": 3}})

    def test_new_objects_on_every_call(self):
        first, second = make_models("regression"), make_models("regression")
        assert all(first[name] is not second[name] for name in LIBRARIES)


class TestSplit:
    def make(self, n=100):
        X = pd.DataFrame({"a": np.arange(n), "b": np.arange(n) * 2.0}, index=np.arange(1000, 1000 + n))
        y = pd.Series(np.arange(n) % 2, index=X.index)
        return X, y

    def test_sizes(self):
        X, y = self.make()
        parts = split_train_valid_test(X, y)
        assert [len(part) for part in parts] == [60, 20, 20, 60, 20, 20]
        parts = split_train_valid_test(X, y, valid_size=0.1, test_size=0.3)
        assert [len(part) for part in parts[:3]] == [60, 10, 30]

    def test_parts_form_a_partition(self):
        X, y = self.make()
        X_train, X_valid, X_test, y_train, y_valid, y_test = split_train_valid_test(X, y)
        labels = np.concatenate([X_train.index, X_valid.index, X_test.index])
        assert sorted(labels) == list(X.index)
        for X_part, y_part in ((X_train, y_train), (X_valid, y_valid), (X_test, y_test)):
            assert list(X_part.index) == list(y_part.index)
            assert X_part.equals(X.loc[X_part.index])

    def test_types_are_preserved(self):
        X, y = self.make()
        parts = split_train_valid_test(X, y)
        assert all(isinstance(part, pd.DataFrame) for part in parts[:3])
        assert all(isinstance(part, pd.Series) for part in parts[3:])
        parts = split_train_valid_test(X.to_numpy(), y.to_numpy())
        assert all(isinstance(part, np.ndarray) for part in parts)
        assert parts[0].shape == (60, 2) and parts[5].shape == (20,)

    def test_random_split_is_reproducible(self):
        X, y = self.make()
        first = split_train_valid_test(X, y, random_state=3)
        second = split_train_valid_test(X, y, random_state=3)
        other = split_train_valid_test(X, y, random_state=4)
        assert all(a.equals(b) for a, b in zip(first, second, strict=True))
        assert list(first[2].index) != list(other[2].index)
        assert list(first[2].index) != sorted(first[2].index)  # разбиение случайное, а не «последние строки»

    def test_stratify(self):
        X, _ = self.make(200)
        y = pd.Series(np.r_[np.ones(20, dtype=int), np.zeros(180, dtype=int)], index=X.index)
        parts = split_train_valid_test(X, y, stratify=True)
        assert [int(part.sum()) for part in parts[3:]] == [12, 4, 4]

    def test_time_split_orders_by_time(self):
        X, y = self.make()
        moment = np.random.default_rng(0).permutation(100)
        X_train, X_valid, X_test, y_train, _, _ = split_train_valid_test(X, y, time=moment)
        when = pd.Series(moment, index=X.index)
        assert [len(X_train), len(X_valid), len(X_test)] == [60, 20, 20]
        assert when[X_train.index].max() < when[X_valid.index].min()
        assert when[X_valid.index].max() < when[X_test.index].min()
        assert when[X_test.index].is_monotonic_increasing
        assert list(y_train.index) == list(X_train.index)

    def test_time_split_ignores_random_state(self):
        X, y = self.make()
        moment = np.random.default_rng(1).permutation(100)
        first = split_train_valid_test(X, y, time=moment, random_state=0)
        second = split_train_valid_test(X, y, time=moment, random_state=99)
        assert all(a.equals(b) for a, b in zip(first, second, strict=True))


class TestPrepareCategoricals:
    def test_category_kind(self, cat_frames):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            train, valid, test = prepare_categoricals(*cat_frames, ["city", "code"], "category")
        for frame in (train, valid, test):
            assert isinstance(frame["city"].dtype, pd.CategoricalDtype)
            assert list(frame["city"].cat.categories) == ["a", "b", "c"]
            assert list(frame["code"].cat.categories) == [1, 2, 3]
        assert train["city"].tolist()[:4] == ["b", "a", "c", "a"] and pd.isna(train["city"].iloc[4])
        assert valid["city"].tolist()[0] == "a" and pd.isna(valid["city"].iloc[1])  # "d" не было в train
        assert test["city"].isna().tolist() == [False, True, True]
        assert test["code"].isna().tolist() == [False, False, True]

    def test_string_kind(self, cat_frames):
        train, valid, test = prepare_categoricals(*cat_frames, ["city", "code"], "string")
        for frame in (train, valid, test):
            for column in ("city", "code"):
                assert frame[column].map(type).eq(str).all()
        assert train["city"].tolist() == ["b", "a", "c", "a", MISSING]
        assert test["city"].tolist() == ["c", MISSING, "z"]  # новое значение остаётся как есть
        assert valid["code"].tolist() == ["2", "7"]

    def test_ordinal_kind(self, cat_frames):
        train, valid, test = prepare_categoricals(*cat_frames, ["city", "code"], "ordinal")
        assert train["city"].tolist() == [1, 0, 2, 0, -1]
        assert valid["city"].tolist() == [0, -1]
        assert test["city"].tolist() == [2, -1, -1]
        assert test["code"].tolist() == [0, 2, -1]
        assert all(pd.api.types.is_integer_dtype(frame["city"]) for frame in (train, valid, test))

    @pytest.mark.parametrize("kind", ["category", "string", "ordinal"])
    def test_inputs_and_other_columns_are_untouched(self, cat_frames, kind):
        copies = [frame.copy() for frame in cat_frames]
        result = prepare_categoricals(*cat_frames, ["city"], kind)
        for frame, copy, prepared in zip(cat_frames, copies, result, strict=True):
            pd.testing.assert_frame_equal(frame, copy)
            assert prepared is not frame
            assert list(prepared.columns) == list(frame.columns)
            pd.testing.assert_series_equal(prepared["x"], frame["x"])
            pd.testing.assert_series_equal(prepared["code"], frame["code"])

    def test_unknown_kind(self, cat_frames):
        with pytest.raises(ValueError):
            prepare_categoricals(*cat_frames, ["city"], "onehot")


class TestEarlyStopping:
    @pytest.mark.parametrize("library", LIBRARIES)
    def test_stops_before_the_limit(self, library, reg_fitted, clf_fitted):
        for fitted in (reg_fitted, clf_fitted):
            _, n_trees = fitted[library]
            assert isinstance(n_trees, int)
            assert 1 <= n_trees < FAST["max_trees"]

    def test_n_trees_matches_library_attributes(self, reg_fitted, clf_fitted):
        for fitted in (reg_fitted, clf_fitted):
            assert fitted["xgboost"][1] == fitted["xgboost"][0].best_iteration + 1
            assert fitted["lightgbm"][1] == fitted["lightgbm"][0].best_iteration_
            assert fitted["catboost"][1] == fitted["catboost"][0].tree_count_
            assert fitted["sklearn"][1] == fitted["sklearn"][0].n_iter_

    @pytest.mark.parametrize("library", LIBRARIES)
    def test_same_as_model_trained_for_n_trees(self, library, reg_data, reg_fitted):
        X_train, _, X_test, y_train, _, _ = reg_data
        model, n_trees = reg_fitted[library]
        plain = make_models("regression", **(FAST | {"max_trees": n_trees}))[library].fit(X_train, y_train)
        np.testing.assert_allclose(model.predict(X_test), plain.predict(X_test), rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("library", ["xgboost", "lightgbm", "sklearn"])
    def test_same_as_model_trained_for_n_trees_classification(self, library, clf_data, clf_fitted):
        X_train, _, X_test, y_train, _, _ = clf_data
        model, n_trees = clf_fitted[library]
        plain = make_models("classification", **(FAST | {"max_trees": n_trees}))[library].fit(X_train, y_train)
        np.testing.assert_allclose(model.predict_proba(X_test), plain.predict_proba(X_test), rtol=1e-6, atol=1e-8)

    def test_catboost_classifier_is_prefix_of_full_model(self, clf_data, clf_fitted):
        # У CatBoost в классификации модель на n итераций не совпадает с первыми n деревьями более длинной,
        # поэтому сравниваем с моделью без остановки, обрезанной до n_trees при предсказании
        X_train, _, X_test, y_train, _, _ = clf_data
        model, n_trees = clf_fitted["catboost"]
        full = make_models("classification", **FAST)["catboost"].fit(X_train, y_train)
        np.testing.assert_allclose(model.predict_proba(X_test), full.predict_proba(X_test, ntree_end=n_trees))

    @pytest.mark.parametrize("library", LIBRARIES)
    def test_original_model_is_not_changed(self, library, reg_data):
        X_train, X_valid, _, y_train, y_valid, _ = reg_data
        model = make_models("regression", max_trees=30, learning_rate=0.3)[library]
        before = model.get_params()
        fitted, _ = fit_with_early_stopping(model, X_train, y_train, X_valid, y_valid, 5)
        assert fitted is not model
        assert model.get_params() == before
        assert not is_fitted(model)

    @pytest.mark.parametrize("library", LIBRARIES)
    def test_no_stop_when_loss_keeps_improving(self, library, reg_data):
        X_train, X_valid, _, y_train, y_valid, _ = reg_data
        model = make_models("regression", max_trees=20, learning_rate=0.05)[library]
        _, n_trees = fit_with_early_stopping(model, X_train, y_train, X_valid, y_valid, rounds=1000)
        assert n_trees == 20

    @pytest.mark.parametrize("library", LIBRARIES)
    def test_validation_data_define_the_stop(self, library, reg_data, reg_fitted):
        X_train, X_valid, _, y_train, y_valid, _ = reg_data
        shuffled = y_valid.sample(frac=1, random_state=0).to_numpy()  # валидация, не связанная с признаками
        model = make_models("regression", **FAST)[library]
        _, n_trees = fit_with_early_stopping(model, X_train, y_train, X_valid, shuffled, ROUNDS)
        assert n_trees < reg_fitted[library][1]
        assert n_trees <= ROUNDS + 1

    def test_unsupported_model(self, reg_data):
        X_train, X_valid, _, y_train, y_valid, _ = reg_data
        with pytest.raises(TypeError):
            fit_with_early_stopping(RandomForestRegressor(n_estimators=5), X_train, y_train, X_valid, y_valid)

    def test_no_warnings_and_no_files(self, reg_data, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        X_train, X_valid, _, y_train, y_valid, _ = reg_data
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for model in make_models("regression", max_trees=30, learning_rate=0.3).values():
                fit_with_early_stopping(model, X_train, y_train, X_valid, y_valid, 5)
        assert list(tmp_path.iterdir()) == []


class TestEvaluate:
    def test_regression_is_rmse(self):
        y = np.array([1.0, 2.0, 3.0, 6.0])
        # ошибки -1, 0, 1, 4 -> средний квадрат 18 / 4
        assert evaluate(ConstantRegressor(2.0), np.zeros((4, 1)), y, "regression") == pytest.approx(np.sqrt(4.5))

    def test_binary_log_loss(self):
        proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
        y = np.array([0, 1, 1])
        expected = -(np.log(0.9) + np.log(0.8) + np.log(0.4)) / 3
        assert evaluate(FixedProbabilities(proba, [0, 1]), np.zeros((3, 1)), y, "classification") == pytest.approx(
            expected
        )

    def test_multiclass_with_missing_class(self):
        proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        y = np.array([0, 1])  # класса 2 в этих данных нет
        value = evaluate(FixedProbabilities(proba, [0, 1, 2]), np.zeros((2, 1)), y, "classification")
        assert value == pytest.approx(log_loss(y, proba, labels=[0, 1, 2]))

    def test_returns_float_and_checks_task(self):
        value = evaluate(ConstantRegressor(0.0), np.zeros((2, 1)), np.array([1.0, 1.0]), "regression")
        assert isinstance(value, float)
        with pytest.raises(ValueError):
            evaluate(ConstantRegressor(0.0), np.zeros((2, 1)), np.array([1.0, 1.0]), "ranking")

    @pytest.mark.parametrize("library", LIBRARIES)
    def test_real_models(self, library, clf_data, clf_fitted):
        _, _, X_test, _, _, y_test = clf_data
        model = clf_fitted[library][0]
        expected = log_loss(y_test, model.predict_proba(X_test))
        assert evaluate(model, X_test, y_test, "classification") == pytest.approx(expected)


class TestBenchmark:
    def test_structure(self, reg_bench, clf_bench):
        for table in (reg_bench, clf_bench):
            assert list(table.index) == list(LIBRARIES)
            assert table.index.name == "model"
            assert list(table.columns) == COLUMNS
            assert pd.api.types.is_integer_dtype(table["n_trees"])
            assert np.isfinite(table.to_numpy(dtype=float)).all()
            assert (table[["fit_time", "predict_time"]] > 0).all().all()

    @pytest.mark.parametrize("library", LIBRARIES)
    def test_matches_separate_fit(self, library, reg_data, reg_bench, reg_fitted):
        _, X_valid, X_test, _, y_valid, y_test = reg_data
        model, n_trees = reg_fitted[library]
        assert reg_bench.loc[library, "n_trees"] == n_trees
        assert reg_bench.loc[library, "score"] == pytest.approx(evaluate(model, X_test, y_test, "regression"))
        assert reg_bench.loc[library, "valid_score"] == pytest.approx(evaluate(model, X_valid, y_valid, "regression"))

    def test_models_are_better_than_constant(self, reg_data, clf_data, reg_bench, clf_bench):
        y_train, y_test = reg_data[3], reg_data[5]
        assert (reg_bench["score"] < np.sqrt(np.mean((y_test - y_train.mean()) ** 2))).all()
        y_train, y_test = clf_data[3], clf_data[5]
        assert (clf_bench["score"] < log_loss(y_test, np.full(len(y_test), y_train.mean()))).all()

    def test_reproducible(self, reg_data, reg_bench):
        again = benchmark(make_models("regression", **FAST), reg_data, "regression", rounds=ROUNDS)
        pd.testing.assert_frame_equal(again[COLUMNS[:3]], reg_bench[COLUMNS[:3]])

    def test_test_part_does_not_affect_training(self, reg_data, reg_bench):
        # Тест с перепутанными ответами: если бы он участвовал в остановке, число деревьев изменилось бы
        X_train, X_valid, X_test, y_train, y_valid, y_test = reg_data
        garbage = y_test.sample(frac=1, random_state=1).to_numpy() * 10
        data = (X_train, X_valid, X_test, y_train, y_valid, garbage)
        table = benchmark(make_models("regression", **FAST), data, "regression", rounds=ROUNDS)
        pd.testing.assert_series_equal(table["n_trees"], reg_bench["n_trees"])
        pd.testing.assert_series_equal(table["valid_score"], reg_bench["valid_score"])
        assert (table["score"] > 3 * reg_bench["score"]).all()

    def test_validation_part_affects_training(self, reg_data, reg_bench):
        X_train, X_valid, X_test, y_train, y_valid, y_test = reg_data
        shuffled = y_valid.sample(frac=1, random_state=0).to_numpy()
        data = (X_train, X_valid, X_test, y_train, shuffled, y_test)
        table = benchmark(make_models("regression", **FAST), data, "regression", rounds=ROUNDS)
        assert (table["n_trees"] < reg_bench["n_trees"]).all()
        assert (table["valid_score"] > reg_bench["valid_score"]).all()

    def test_any_subset_and_names(self, clf_data, clf_bench):
        models = make_models("classification", **FAST)
        table = benchmark(
            {"lgb": models["lightgbm"], "hgb": models["sklearn"]}, clf_data, "classification", rounds=ROUNDS
        )
        assert list(table.index) == ["lgb", "hgb"]
        assert table.loc["lgb", "score"] == pytest.approx(clf_bench.loc["lightgbm", "score"])
        assert table.loc["hgb", "n_trees"] == clf_bench.loc["sklearn", "n_trees"]

    def test_multiclass(self):
        wine = load_wine(as_frame=True)
        data = split_train_valid_test(wine.data, wine.target, stratify=True)
        models = make_models("classification", max_trees=100, learning_rate=0.3)
        table = benchmark(models, data, "classification", rounds=ROUNDS)
        assert np.isfinite(table["score"]).all()
        assert (table["score"] < np.log(3)).all()  # лучше равных вероятностей трёх классов

    def test_categorical_features(self, cat_data, cat_bench):
        y_train, y_test = cat_data[3], cat_data[5]
        constant = log_loss(y_test, np.full(len(y_test), y_train.mean()))
        assert list(cat_bench.index) == list(LIBRARIES)
        assert np.isfinite(cat_bench["score"]).all()
        # вся информация — в категориальном признаке: модель, которая его не использует, останется у константы
        assert (cat_bench["score"] < 0.85 * constant).all()

    def test_categorical_inputs_are_untouched(self, cat_data):
        copies = [part.copy() for part in cat_data[:3]]
        models = make_models("classification", max_trees=20, learning_rate=0.3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            benchmark(models, cat_data, "classification", rounds=5, cat_features=["city"])
        for part, copy in zip(cat_data[:3], copies, strict=True):
            pd.testing.assert_frame_equal(part, copy)


class TestSummarize:
    def table(self, scores, times=(1.0, 2.0, 3.0)):
        index = pd.Index(["a", "b", "c"], name="model")
        return pd.DataFrame(
            {"score": scores, "valid_score": scores, "n_trees": 10, "fit_time": times, "predict_time": 0.1}, index=index
        )

    def test_hand_example(self):
        results = {
            "first": self.table([1.0, 2.0, 4.0]),  # ранги 1, 2, 3
            "second": self.table([3.0, 1.5, 6.0]),  # ранги 2, 1, 3
            "third": self.table([0.2, 0.4, 0.3]),  # ранги 1, 3, 2
        }
        summary = summarize(results)
        assert summary.index.name == "model"
        assert list(summary.columns) == ["mean_rank", "wins", "mean_ratio", "total_fit_time"]
        assert list(summary.index) == ["a", "b", "c"]
        assert summary["mean_rank"].tolist() == pytest.approx([4 / 3, 2.0, 8 / 3])
        assert summary["wins"].tolist() == [2, 1, 0]
        assert summary.loc["a", "mean_ratio"] == pytest.approx((1 + 2 + 1) / 3)
        assert summary.loc["c", "mean_ratio"] == pytest.approx((4 + 4 + 1.5) / 3)
        assert summary["total_fit_time"].tolist() == pytest.approx([3.0, 6.0, 9.0])

    def test_sorted_by_mean_rank(self):
        summary = summarize({"first": self.table([3.0, 2.0, 1.0]), "second": self.table([5.0, 4.0, 1.0])})
        assert list(summary.index) == ["c", "b", "a"]
        assert summary["mean_rank"].is_monotonic_increasing

    def test_ties(self):
        summary = summarize({"only": self.table([1.0, 1.0, 2.0])})
        assert summary.loc[["a", "b", "c"], "mean_rank"].tolist() == pytest.approx([1.5, 1.5, 3.0])
        assert summary.loc[["a", "b", "c"], "wins"].tolist() == [1, 1, 0]
        assert list(summary.index) == ["a", "b", "c"]  # при равном ранге — по названию

    def test_different_models(self):
        other = self.table([1.0, 2.0, 3.0]).rename(index={"c": "d"})
        with pytest.raises(ValueError):
            summarize({"first": self.table([1.0, 2.0, 3.0]), "second": other})

    def test_real_tables(self, reg_bench, clf_bench):
        summary = summarize({"regression": reg_bench, "classification": clf_bench})
        assert sorted(summary.index) == sorted(LIBRARIES)
        assert summary["mean_rank"].sum() == pytest.approx(10.0)
        assert summary["mean_ratio"].min() >= 1.0
