"""Тесты практики 7.4.4: исполняемая спецификация модуля sales_drift.

Запуск из папки practice:  uv run pytest -q
Данные тестов синтетические и строятся с фиксированным seed; сеть и файлы не используются.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))

import sales_drift as sd  # noqa: E402

LN2 = math.log(2)
# PSI для долей (0.5, 0.5) против (0.25, 0.75)
HALVES_PSI = 0.25 * math.log(2) + 0.25 * math.log(1.5)


def slow_psi(reference_shares, current_shares, eps=sd.EPS) -> float:
    """PSI по определению, без NumPy."""
    total = 0.0
    for expected, actual in zip(reference_shares, current_shares, strict=True):
        expected, actual = max(expected, eps), max(actual, eps)
        total += (actual - expected) * math.log(actual / expected)
    return total


def slow_numeric_psi(reference, current, bins=sd.DEFAULT_BINS) -> float:
    """Числовой PSI по определению: корзина значения — число границ строго меньше него."""
    reference = [float(value) for value in reference if not pd.isna(value)]
    current = [float(value) for value in current if not pd.isna(value)]
    edges = sorted({float(np.quantile(reference, step / bins)) for step in range(1, bins)})

    def shares(values):
        counts = [0] * (len(edges) + 1)
        for value in values:
            counts[sum(edge < value for edge in edges)] += 1
        return [count / len(values) for count in counts]

    return slow_psi(shares(reference), shares(current))


# Эталон для decile_sample: 20 000 чисел от 0 до 9999.5, по 10 % в каждой тысяче
DECILE_REFERENCE = np.arange(20_000) / 2


def decile_sample(shares, size=10_000) -> np.ndarray:
    """Выборка с заданными долями по корзинам эталона DECILE_REFERENCE (десять корзин по тысяче)."""
    parts = [
        np.linspace(1000 * index + 1, 1000 * index + 999, round(share * size)) for index, share in enumerate(shares)
    ]
    return np.concatenate(parts)


@pytest.fixture
def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Эталон и текущая выборка из одного распределения, по 5000 строк."""
    rng = np.random.default_rng(1)

    def make(size):
        return pd.DataFrame(
            {
                "price": rng.lognormal(7, 0.5, size).round(2),
                "units": rng.poisson(5, size).astype("float64"),
                "store_id": rng.choice(["S01", "S02", "S03", "S04"], size),
                "promo": rng.random(size) < 0.1,
            }
        )

    return make(5000), make(5000)


# --- psi_from_shares ---


def test_psi_of_equal_shares_is_zero():
    assert sd.psi_from_shares([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0.0


def test_psi_known_value():
    assert sd.psi_from_shares([0.5, 0.5], [0.25, 0.75]) == pytest.approx(HALVES_PSI)
    assert sd.psi_from_shares([0.1] * 10, [0.02, 0.18] + [0.1] * 8) == pytest.approx(
        0.08 * math.log(5) + 0.08 * math.log(1.8)
    )


def test_psi_is_symmetric_and_non_negative():
    first, second = [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]
    assert sd.psi_from_shares(first, second) == pytest.approx(sd.psi_from_shares(second, first))
    assert sd.psi_from_shares(first, second) > 0


def test_psi_replaces_small_shares_with_eps():
    value = sd.psi_from_shares([0.5, 0.5, 0.0], [0.4, 0.4, 0.2])
    assert math.isfinite(value)
    expected = 2 * (0.4 - 0.5) * math.log(0.4 / 0.5) + (0.2 - sd.EPS) * math.log(0.2 / sd.EPS)
    assert value == pytest.approx(expected)  # доли после замены заново не нормируются
    assert sd.psi_from_shares([0.5, 0.5, 0.0], [0.5, 0.5, 0.0]) == 0.0


def test_psi_eps_parameter():
    assert sd.psi_from_shares([1.0, 0.0], [0.5, 0.5], eps=0.01) == pytest.approx(
        slow_psi([1.0, 0.0], [0.5, 0.5], eps=0.01)
    )
    assert sd.psi_from_shares([1.0, 0.0], [0.5, 0.5], eps=0.01) < sd.psi_from_shares([1.0, 0.0], [0.5, 0.5])


def test_psi_accepts_arrays_and_returns_float():
    value = sd.psi_from_shares(np.array([0.5, 0.5]), pd.Series([0.25, 0.75]))
    assert type(value) is float
    assert value == pytest.approx(HALVES_PSI)


@pytest.mark.parametrize("pair", [([0.5, 0.5], [1.0]), ([], [])])
def test_psi_rejects_bad_shares(pair):
    with pytest.raises(ValueError):
        sd.psi_from_shares(*pair)


# --- numeric_psi ---


def test_numeric_psi_of_same_sample_is_zero():
    values = np.random.default_rng(0).normal(size=500)
    assert sd.numeric_psi(values, values) == 0.0
    assert sd.numeric_psi(values, values[::-1]) == 0.0


def test_numeric_psi_known_value():
    reference = DECILE_REFERENCE
    assert sd.numeric_psi(reference, decile_sample([0.02, 0.18] + [0.1] * 8)) == pytest.approx(
        0.08 * math.log(5) + 0.08 * math.log(1.8)
    )
    everything_on_top = (0.1 - sd.EPS) * math.log(0.1 / sd.EPS) * 9 + 0.9 * math.log(10)
    assert sd.numeric_psi(reference, np.full(50, 9999.5)) == pytest.approx(everything_on_top)


def test_numeric_psi_matches_definition_on_random_data():
    rng = np.random.default_rng(3)
    reference, current = rng.lognormal(0, 1, 800), rng.lognormal(0.3, 1.2, 300)
    assert sd.numeric_psi(reference, current) == pytest.approx(slow_numeric_psi(reference, current), abs=1e-12)
    assert sd.numeric_psi(current, reference) == pytest.approx(slow_numeric_psi(current, reference), abs=1e-12)
    assert sd.numeric_psi(reference, current) != pytest.approx(
        sd.numeric_psi(current, reference)
    )  # границы — по эталону


def test_numeric_psi_value_equal_to_edge_goes_to_left_bin():
    reference = np.arange(1, 101, dtype="float64")
    edge = float(np.quantile(reference, 0.1))  # первая граница: 10.9
    current = [edge, 15.0, 15.0, 15.0]  # 15 лежит во второй корзине при любом правиле
    assert sd.numeric_psi(reference, current) == pytest.approx(slow_psi([0.1] * 10, [0.25, 0.75] + [0.0] * 8))
    assert sd.numeric_psi(reference, current) == pytest.approx(slow_numeric_psi(reference, current), abs=1e-12)


def test_numeric_psi_merges_repeated_edges():
    reference = [0.0] * 90 + [1.0] * 10  # границы после удаления повторов: 0 и 0.1
    current = [0.0] * 50 + [1.0] * 50
    expected = slow_psi([0.9, 0.0, 0.1], [0.5, 0.0, 0.5])
    assert sd.numeric_psi(reference, current) == pytest.approx(expected)
    assert sd.numeric_psi([5.0] * 20, [5.0] * 7) == 0.0  # все значения в одной корзине


def test_numeric_psi_bins_parameter():
    rng = np.random.default_rng(4)
    reference, current = rng.normal(0, 1, 2000), rng.normal(0.5, 1, 2000)
    for bins in [2, 4, 20]:
        assert sd.numeric_psi(reference, current, bins=bins) == pytest.approx(
            slow_numeric_psi(reference, current, bins), abs=1e-12
        )
    assert sd.numeric_psi(reference, current, bins=4) != pytest.approx(sd.numeric_psi(reference, current))


def test_numeric_psi_values_outside_reference_range():
    reference = np.arange(1000, dtype="float64")
    value = sd.numeric_psi(reference, reference + 5000)
    assert math.isfinite(value)
    assert value == pytest.approx(sd.numeric_psi(reference, np.full(10, 1e12)))  # всё в верхней корзине
    assert value > sd.PSI_DRIFT


def test_numeric_psi_ignores_missing_values():
    rng = np.random.default_rng(5)
    reference, current = rng.normal(size=400), rng.normal(0.4, 1, size=300)
    with_gaps_reference = pd.Series([None, *reference, np.nan])
    with_gaps_current = [np.nan, *current, None, None]
    assert sd.numeric_psi(with_gaps_reference, with_gaps_current) == pytest.approx(sd.numeric_psi(reference, current))


def test_numeric_psi_accepts_any_sequence_and_returns_float():
    reference, current = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 1, 1, 9, 10]
    value = sd.numeric_psi(reference, current)
    assert type(value) is float
    assert value == pytest.approx(sd.numeric_psi(np.array(reference), pd.Series(current)))


@pytest.mark.parametrize("pair", [([], [1.0]), ([1.0], []), ([np.nan, None], [1.0]), ([1.0], [np.nan])])
def test_numeric_psi_rejects_empty_samples(pair):
    with pytest.raises(ValueError):
        sd.numeric_psi(*pair)


def test_numeric_psi_rejects_single_bin():
    with pytest.raises(ValueError):
        sd.numeric_psi([1.0, 2.0], [1.0, 2.0], bins=1)


# --- categorical_psi ---


def test_categorical_psi_of_same_distribution_is_zero():
    assert sd.categorical_psi(["a", "b", "b", "c"], ["b", "c", "a", "b"]) == 0.0
    assert sd.categorical_psi(["a", "b"], ["a", "a", "b", "b"]) == pytest.approx(0.0)


def test_categorical_psi_known_value():
    assert sd.categorical_psi(["a", "b"], ["a", "b", "b", "b"]) == pytest.approx(HALVES_PSI)
    assert sd.categorical_psi([True, False], [True, False, False, False]) == pytest.approx(HALVES_PSI)


def test_categorical_psi_counts_new_and_vanished_categories():
    value = sd.categorical_psi(["a", "b"] * 50, ["a", "c"] * 50)
    assert value == pytest.approx(slow_psi([0.5, 0.5, 0.0], [0.5, 0.0, 0.5]))
    assert value > sd.PSI_DRIFT


def test_categorical_psi_ignores_missing_values():
    assert sd.categorical_psi(["a", None, "b", np.nan], ["a", "b", "b", "b", None]) == pytest.approx(HALVES_PSI)


def test_categorical_psi_compares_values_as_strings():
    assert sd.categorical_psi([1, 2], ["1", "2", "2", "2"]) == pytest.approx(HALVES_PSI)


def test_categorical_psi_accepts_series_and_returns_float():
    value = sd.categorical_psi(pd.Series(["S01", "S02"]), np.array(["S01", "S02", "S02", "S02"]))
    assert type(value) is float
    assert value == pytest.approx(HALVES_PSI)


@pytest.mark.parametrize("pair", [([], ["a"]), (["a"], []), ([None], ["a"])])
def test_categorical_psi_rejects_empty_samples(pair):
    with pytest.raises(ValueError):
        sd.categorical_psi(*pair)


# --- psi_noise ---


def test_psi_noise_formula():
    assert sd.psi_noise(100, 100, 10) == pytest.approx(0.18)
    assert sd.psi_noise(1000, 500, 5) == pytest.approx(0.012)
    assert sd.psi_noise(10, 10, 1) == 0.0
    assert type(sd.psi_noise(100, 100, 10)) is float


@pytest.mark.parametrize("args", [(0, 100, 10), (100, 0, 10), (100, 100, 0), (-5, 100, 10)])
def test_psi_noise_rejects_non_positive_arguments(args):
    with pytest.raises(ValueError):
        sd.psi_noise(*args)


def test_psi_noise_predicts_psi_of_samples_without_drift():
    rng = np.random.default_rng(6)
    population = rng.normal(size=20_000)
    values = [sd.numeric_psi(rng.choice(population, 300), rng.choice(population, 300)) for _ in range(200)]
    assert np.mean(values) == pytest.approx(sd.psi_noise(300, 300, 10), rel=0.25)


# --- ks_test ---


def test_ks_of_same_sample():
    values = np.random.default_rng(0).normal(size=200)
    statistic, pvalue = sd.ks_test(values, values)
    assert statistic == 0.0
    assert pvalue == pytest.approx(1.0)


def test_ks_of_disjoint_samples():
    statistic, pvalue = sd.ks_test(np.arange(50), np.arange(100, 150))
    assert statistic == 1.0
    assert pvalue < 1e-6


def test_ks_known_statistic():
    statistic, pvalue = sd.ks_test([1, 2, 3, 4], [3, 4, 5, 6])
    assert statistic == pytest.approx(0.5)
    assert 0 < pvalue < 1


def test_ks_matches_scipy_and_returns_floats():
    rng = np.random.default_rng(7)
    reference, current = rng.normal(0, 1, 500), rng.normal(0.2, 1, 400)
    statistic, pvalue = sd.ks_test(reference, current)
    expected = stats.ks_2samp(reference, current)
    assert type(statistic) is float
    assert type(pvalue) is float
    assert statistic == pytest.approx(float(expected.statistic))
    assert pvalue == pytest.approx(float(expected.pvalue))


def test_ks_ignores_missing_values():
    reference, current = [1.0, 2.0, None, 3.0, 4.0], pd.Series([3.0, np.nan, 4.0, 5.0, 6.0])
    assert sd.ks_test(reference, current)[0] == pytest.approx(0.5)


@pytest.mark.parametrize("pair", [([], [1.0]), ([1.0], [None])])
def test_ks_rejects_empty_samples(pair):
    with pytest.raises(ValueError):
        sd.ks_test(*pair)


# --- missing_share и new_category_share ---


def test_missing_share():
    assert sd.missing_share([1, None, np.nan, "a"]) == 0.5
    assert sd.missing_share(pd.Series([1.0, 2.0, np.nan, 4.0])) == 0.25
    assert sd.missing_share(["a", "b"]) == 0.0
    assert sd.missing_share([None, None]) == 1.0
    assert sd.missing_share([]) == 0.0
    assert type(sd.missing_share([1, None])) is float


def test_new_category_share():
    assert sd.new_category_share(["a", "b"], ["a", "c", "c", "b"]) == 0.5
    assert sd.new_category_share(["a", "b"], ["b", "a", "a"]) == 0.0
    assert sd.new_category_share(["a"], ["x", "y"]) == 1.0
    assert type(sd.new_category_share(["a"], ["a"])) is float


def test_new_category_share_ignores_missing_and_compares_strings():
    assert sd.new_category_share(["a", None], ["a", None, "c", np.nan]) == 0.5
    assert sd.new_category_share([1, 2], ["1", "3"]) == 0.5
    assert sd.new_category_share(["a"], []) == 0.0
    assert sd.new_category_share(["a"], [None]) == 0.0
    assert sd.new_category_share([], ["a", "b"]) == 1.0


# --- drift_report ---


def test_report_structure(frames):
    reference, current = frames
    report = sd.drift_report(reference, current, numeric=["price", "units"], categorical=["store_id", "promo"])
    assert list(report) == ["price", "units", "store_id", "promo"]
    for name in ["price", "units"]:
        assert tuple(report[name]) == sd.NUMERIC_KEYS
        assert report[name]["kind"] == "numeric"
    for name in ["store_id", "promo"]:
        assert tuple(report[name]) == sd.CATEGORICAL_KEYS
        assert report[name]["kind"] == "categorical"


def test_report_values_come_from_metric_functions(frames):
    reference, current = frames
    report = sd.drift_report(reference, current, numeric=["price"], categorical=["store_id"])
    price, store = report["price"], report["store_id"]
    assert price["psi"] == pytest.approx(sd.numeric_psi(reference["price"], current["price"]))
    assert (price["ks_statistic"], price["ks_pvalue"]) == pytest.approx(
        sd.ks_test(reference["price"], current["price"])
    )
    assert price["psi_noise"] == pytest.approx(sd.psi_noise(5000, 5000, 10))
    assert store["psi"] == pytest.approx(sd.categorical_psi(reference["store_id"], current["store_id"]))
    assert store["psi_noise"] == pytest.approx(sd.psi_noise(5000, 5000, 4))
    assert store["new_category_share"] == 0.0
    assert store["new_categories"] == []
    for column in [price, store]:
        assert column["missing_reference"] == 0.0
        assert column["missing_current"] == 0.0


def test_report_without_drift_is_ok(frames):
    reference, current = frames
    report = sd.drift_report(reference, current, numeric=["price", "units"], categorical=["store_id", "promo"])
    assert {name: column["status"] for name, column in report.items()} == dict.fromkeys(report, "ok")


@pytest.mark.parametrize(
    ("shares", "status"),
    [
        ([0.1] * 10, "ok"),
        ([0.04, 0.16] + [0.1] * 8, "ok"),  # PSI 0.083
        ([0.02, 0.18] + [0.1] * 8, "warning"),  # PSI 0.176
        ([0.0, 0.2] + [0.1] * 8, "drift"),  # PSI 0.83
        ([0.0] * 9 + [1.0], "drift"),
    ],
)
def test_report_status_follows_psi_thresholds(shares, status):
    reference = pd.DataFrame({"x": DECILE_REFERENCE})
    current = pd.DataFrame({"x": decile_sample(shares)})
    column = sd.drift_report(reference, current, numeric=["x"], categorical=[])["x"]
    assert column["status"] == status
    assert column["psi"] == pytest.approx(slow_numeric_psi(reference["x"], current["x"]), abs=1e-9)


def test_report_status_thresholds_are_inclusive(monkeypatch):
    reference = pd.DataFrame({"x": DECILE_REFERENCE})
    current = pd.DataFrame({"x": decile_sample([0.02, 0.18] + [0.1] * 8)})
    psi = sd.numeric_psi(reference["x"], current["x"])
    monkeypatch.setattr(sd, "PSI_WARNING", psi)
    assert sd.drift_report(reference, current, ["x"], [])["x"]["status"] == "warning"
    monkeypatch.setattr(sd, "PSI_DRIFT", psi)
    assert sd.drift_report(reference, current, ["x"], [])["x"]["status"] == "drift"


def test_report_pvalue_does_not_affect_status():
    rng = np.random.default_rng(8)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 50_000)})
    current = pd.DataFrame({"x": rng.normal(0.05, 1, 50_000)})
    column = sd.drift_report(reference, current, numeric=["x"], categorical=[])["x"]
    assert column["ks_pvalue"] < 0.001  # критерий «видит» сдвиг в 0.05 стандартного отклонения
    assert column["psi"] < 0.01
    assert column["status"] == "ok"


@pytest.mark.parametrize(("missing", "status"), [(0.0, "ok"), (0.04, "ok"), (0.05, "drift"), (0.2, "drift")])
def test_report_missing_share_growth_is_drift(frames, missing, status):
    reference, current = frames
    current = current.copy()
    gaps = round(missing * len(current))
    current.loc[current.index[:gaps], "price"] = np.nan
    current.loc[current.index[:gaps], "store_id"] = None
    report = sd.drift_report(reference, current, numeric=["price"], categorical=["store_id"])
    for name in ["price", "store_id"]:
        assert report[name]["missing_reference"] == 0.0
        assert report[name]["missing_current"] == pytest.approx(missing)
        assert report[name]["status"] == status
        assert report[name]["psi"] < sd.PSI_WARNING  # пропуски в PSI не участвуют


def test_report_missing_share_is_compared_with_reference(frames):
    reference, current = frames
    reference, current = reference.copy(), current.copy()
    for name in ["price", "store_id"]:
        reference.loc[reference.index[:1000], name] = None  # 20 % пропусков в эталоне
        current.loc[current.index[:1100], name] = None  # 22 % в текущей выборке
    report = sd.drift_report(reference, current, numeric=["price"], categorical=["store_id"])
    for name, groups in [("price", 10), ("store_id", 4)]:
        assert report[name]["missing_reference"] == pytest.approx(0.2)
        assert report[name]["missing_current"] == pytest.approx(0.22)
        assert report[name]["status"] == "ok"
        assert report[name]["psi_noise"] == pytest.approx(sd.psi_noise(4000, 3900, groups))


@pytest.mark.parametrize(("count", "status"), [(0, "ok"), (25, "ok"), (50, "drift"), (600, "drift")])
def test_report_new_categories(frames, count, status):
    reference, current = frames
    current = current.copy()
    current.loc[current.index[: count // 2], "store_id"] = "S10"
    current.loc[current.index[count // 2 : count], "store_id"] = "S09"
    column = sd.drift_report(reference, current, numeric=[], categorical=["store_id"])["store_id"]
    assert column["new_category_share"] == pytest.approx(count / 5000)
    assert column["new_categories"] == (["S09", "S10"] if count else [])
    assert column["status"] == status
    assert column["psi_noise"] == pytest.approx(sd.psi_noise(5000, 5000, 6 if count else 4))


def test_report_small_sample_is_marked(frames):
    reference, current = frames
    small = current.head(80)
    report = sd.drift_report(reference, small, numeric=["price"], categorical=["store_id"])
    assert report["price"]["psi_noise"] == pytest.approx(sd.psi_noise(5000, 80, 10))
    assert report["price"]["psi_noise"] >= sd.PSI_WARNING
    assert report["price"]["status"] == "low_sample"  # какой бы ни был PSI, судить по нему нельзя
    assert report["store_id"]["psi_noise"] < sd.PSI_WARNING
    assert report["store_id"]["status"] == "ok"


def test_report_small_sample_still_reports_missing_and_new_categories(frames):
    reference, current = frames
    reference = reference.assign(cost=reference["price"], promo=reference["promo"].astype(str))
    small = current.head(80).assign(cost=current["price"].head(80) + 1e6, promo=current["promo"].head(80).astype(str))
    small.loc[small.index[:8], "price"] = np.nan
    small.loc[small.index[:2], "promo"] = "unknown"
    report = sd.drift_report(reference, small, numeric=["price", "cost"], categorical=["promo"], bins=20)
    assert report["price"]["status"] == "drift"  # рост доли пропусков виден и на малой выборке
    assert report["cost"]["status"] == "low_sample"
    assert report["cost"]["psi"] > sd.PSI_DRIFT
    assert report["promo"]["status"] == "drift"
    assert report["promo"]["new_categories"] == ["unknown"]


def test_report_noise_uses_actual_number_of_groups():
    rng = np.random.default_rng(9)
    reference = pd.DataFrame({"flag": [0.0, 1.0] * 1000})  # границы после удаления повторов: 0, 0.5, 1
    current = pd.DataFrame({"flag": rng.integers(0, 2, 1000).astype("float64")})
    column = sd.drift_report(reference, current, numeric=["flag"], categorical=[])["flag"]
    assert column["psi_noise"] == pytest.approx(sd.psi_noise(2000, 1000, 4))


def test_report_bins_parameter(frames):
    reference, current = frames
    column = sd.drift_report(reference, current, numeric=["price"], categorical=[], bins=4)["price"]
    assert column["psi"] == pytest.approx(sd.numeric_psi(reference["price"], current["price"], bins=4))
    assert column["psi_noise"] == pytest.approx(sd.psi_noise(5000, 5000, 4))


def test_report_rejects_absent_columns(frames):
    reference, current = frames
    with pytest.raises(ValueError, match="revenue"):
        sd.drift_report(reference, current, numeric=["price", "revenue"], categorical=[])
    with pytest.raises(ValueError, match="store_id"):
        sd.drift_report(reference, current.drop(columns="store_id"), numeric=[], categorical=["store_id"])


def test_report_does_not_change_frames(frames):
    reference, current = frames
    reference_copy, current_copy = reference.copy(), current.copy()
    sd.drift_report(reference, current, numeric=["price", "units"], categorical=["store_id", "promo"])
    pd.testing.assert_frame_equal(reference, reference_copy)
    pd.testing.assert_frame_equal(current, current_copy)
