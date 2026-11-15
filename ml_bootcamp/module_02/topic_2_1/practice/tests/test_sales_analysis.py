import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))

from make_raw_data import make_raw  # noqa: E402
from sales_analysis import add_features, evaluate_baselines, fill_calendar, load_raw  # noqa: E402

from sales_data import make_sales  # noqa: E402

N_DAYS = 731
N_PAIRS = 96


@pytest.fixture(scope="module")
def raw_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data") / "raw_sales.csv"
    make_raw().to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def loaded(raw_path) -> pd.DataFrame:
    return load_raw(raw_path)


@pytest.fixture(scope="module")
def full(loaded) -> pd.DataFrame:
    return fill_calendar(loaded)


def tiny(units: list[float], store: str = "S01", start: str = "2024-01-01") -> pd.DataFrame:
    """Полный календарь одной пары с заданными продажами."""
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=len(units), freq="D"),
            "store_id": store,
            "city": "Москва",
            "category": "Аксессуары",
            "product": "Мышь",
            "price": 1490,
            "promo": False,
            "units": units,
            "revenue": [u * 1490 for u in units],
        }
    )


class TestLoadRaw:
    def test_dates_are_parsed(self, loaded):
        assert pd.api.types.is_datetime64_any_dtype(loaded["date"])
        assert loaded["date"].dt.tz is None
        assert loaded["date"].notna().all()
        assert loaded["date"].min() == pd.Timestamp("2023-01-01")
        assert loaded["date"].max() == pd.Timestamp("2024-12-31")

    def test_both_date_formats_are_kept(self, loaded):
        assert loaded["store_id"].nunique() == 8
        per_store = loaded.groupby("store_id", observed=True)["date"].nunique()
        assert (per_store == N_DAYS).all()

    def test_matches_source_data(self, loaded):
        # Эталон: исходные данные без дней с нулевыми продажами. Ловит перепутанные день и месяц.
        source = make_sales()
        source = source[source["units"].isna() | (source["units"] > 0)]
        keys = ["store_id", "product", "date"]
        columns = [*keys, "units", "revenue", "price", "promo"]
        expected = source[columns].sort_values(keys, ignore_index=True)
        actual = loaded[columns].astype({"store_id": str, "product": str}).sort_values(keys, ignore_index=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    def test_dtypes(self, loaded):
        for column in ["store_id", "city", "category", "product"]:
            assert isinstance(loaded[column].dtype, pd.CategoricalDtype), column
        assert loaded["promo"].dtype == bool
        assert pd.api.types.is_integer_dtype(loaded["price"])
        assert pd.api.types.is_float_dtype(loaded["units"])

    def test_no_duplicates(self, loaded):
        assert not loaded.duplicated().any()
        assert not loaded.duplicated(["date", "store_id", "product"]).any()

    def test_missing_values_are_kept(self, loaded):
        assert loaded["units"].isna().sum() == 335

    def test_sorted_with_clean_index(self, loaded):
        expected = loaded.sort_values(["store_id", "product", "date"], ignore_index=True)
        pd.testing.assert_frame_equal(loaded, expected)


class TestFillCalendar:
    def test_every_pair_has_every_date(self, full):
        assert len(full) == N_DAYS * N_PAIRS
        sizes = full.groupby(["store_id", "product"], observed=True).size()
        assert len(sizes) == N_PAIRS
        assert (sizes == N_DAYS).all()

    def test_added_rows_are_zero_sales(self, loaded, full):
        added = len(full) - len(loaded)
        assert added > 0
        assert (full["units"] == 0).sum() == added
        assert (full.loc[full["units"] == 0, "revenue"] == 0).all()
        assert not full.loc[full["units"] == 0, "promo"].any()

    def test_original_missing_values_are_not_filled(self, full):
        assert full["units"].isna().sum() == 335
        assert full["revenue"].isna().sum() == 335

    def test_descriptive_columns_are_filled(self, full):
        assert full[["city", "category", "price"]].notna().all().all()
        assert (full.groupby(["store_id", "product"], observed=True)["city"].nunique() == 1).all()

    def test_totals_are_preserved(self, loaded, full):
        assert full["units"].sum() == loaded["units"].sum()
        assert full["revenue"].sum() == loaded["revenue"].sum()

    def test_price_is_taken_from_the_same_pair(self):
        first = tiny([1, 2, 3]).drop(index=1)
        second = tiny([5, 5, 5], store="S02").assign(price=999)
        result = fill_calendar(pd.concat([first, second], ignore_index=True))
        added = result[(result["store_id"] == "S01") & (result["date"] == "2024-01-02")]
        assert len(added) == 1
        assert added["price"].item() == 1490
        assert added["units"].item() == 0

    def test_sorted_with_clean_index(self, full):
        expected = full.sort_values(["store_id", "product", "date"], ignore_index=True)
        pd.testing.assert_frame_equal(full, expected)


class TestAddFeatures:
    def test_input_is_not_modified(self):
        data = tiny([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        before = data.copy()
        add_features(data, lags=(1,), windows=(3,))
        pd.testing.assert_frame_equal(data, before)

    def test_lags(self):
        result = add_features(tiny([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), lags=(1, 7), windows=(3,))
        assert np.isnan(result["units_lag_1"].iloc[0])
        assert result["units_lag_1"].iloc[1:].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert result["units_lag_7"].isna().sum() == 7
        assert result["units_lag_7"].iloc[7:].tolist() == [1, 2, 3]

    def test_window_uses_only_previous_days(self):
        result = add_features(tiny([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), lags=(1,), windows=(3,))
        assert result["units_mean_3"].isna().sum() == 3
        assert result["units_mean_3"].iloc[3] == pytest.approx(2.0)  # среднее 1, 2, 3
        assert result["units_mean_3"].iloc[9] == pytest.approx(8.0)  # среднее 7, 8, 9
        assert result["units_std_3"].iloc[3] == pytest.approx(1.0)

    def test_no_leakage_from_current_day(self):
        base = tiny([float(v) for v in range(1, 41)])
        changed = base.copy()
        changed.loc[30, "units"] = 1_000_000.0
        feature_columns = [c for c in add_features(base).columns if c.startswith("units_")]
        before = add_features(base)[feature_columns]
        after = add_features(changed)[feature_columns]
        pd.testing.assert_series_equal(before.iloc[30], after.iloc[30])
        assert after["units_lag_1"].iloc[31] == 1_000_000.0
        assert after["units_mean_7"].iloc[31] > before["units_mean_7"].iloc[31]

    def test_groups_do_not_mix(self):
        data = pd.concat([tiny([1.0] * 10), tiny([100.0] * 10, store="S02")], ignore_index=True)
        result = add_features(data, lags=(1,), windows=(3,))
        second = result[result["store_id"] == "S02"]
        assert np.isnan(second["units_lag_1"].iloc[0])
        assert second["units_mean_3"].isna().sum() == 3
        assert second["units_mean_3"].dropna().eq(100.0).all()

    def test_calendar_features(self):
        result = add_features(tiny([1.0] * 10, start="2024-12-28"), lags=(1,), windows=(3,))
        assert result["day_of_week"].tolist() == [5, 6, 0, 1, 2, 3, 4, 5, 6, 0]
        assert result["is_weekend"].tolist() == [True, True, False, False, False, False, False, True, True, False]
        december, january = result.iloc[0], result.iloc[-1]
        assert december["month_sin"] == pytest.approx(0.0, abs=1e-9)
        assert december["month_cos"] == pytest.approx(1.0)
        assert january["month_sin"] == pytest.approx(0.5)
        assert january["month_cos"] == pytest.approx(np.sqrt(3) / 2)

    def test_default_feature_set(self, full):
        result = add_features(full)
        expected = {
            "units_lag_1",
            "units_lag_7",
            "units_lag_28",
            "units_mean_7",
            "units_std_7",
            "units_mean_28",
            "units_std_28",
            "day_of_week",
            "is_weekend",
            "month_sin",
            "month_cos",
        }
        assert expected <= set(result.columns)
        assert len(result) == len(full)
        first_rows = result.groupby(["store_id", "product"], observed=True).head(1)
        assert first_rows["units_lag_1"].isna().all()


class TestEvaluateBaselines:
    def test_known_values(self):
        units = [2.0] * 28 + [4.0, 4.0]
        features = add_features(tiny(units))
        result = evaluate_baselines(features, test_start="2024-01-29")
        assert set(result.index) == {"naive_last", "seasonal_naive_7", "mean_28"}
        assert list(result.columns) == ["mae", "wape"]
        # Факт: 4 и 4. naive_last: 2 и 4; seasonal_naive_7: 2 и 2; mean_28: 2 и 58/28
        assert result.loc["naive_last", "mae"] == pytest.approx(1.0)
        assert result.loc["naive_last", "wape"] == pytest.approx(0.25)
        assert result.loc["seasonal_naive_7", "mae"] == pytest.approx(2.0)
        assert result.loc["seasonal_naive_7", "wape"] == pytest.approx(0.5)
        assert result.loc["mean_28", "mae"] == pytest.approx((2 + (4 - 58 / 28)) / 2)

    def test_missing_values_are_skipped(self):
        units = [2.0] * 28 + [np.nan] + [4.0] * 30
        result = evaluate_baselines(add_features(tiny(units)), test_start="2024-01-29")
        # naive_last: 29 января нет факта, 30 января нет прогноза, дальше прогноз 4 при факте 4
        assert result.loc["naive_last", "mae"] == pytest.approx(0.0)
        # seasonal_naive_7: 6 дней с прогнозом 2 при факте 4, один день без прогноза, 23 дня без ошибки
        assert result.loc["seasonal_naive_7", "mae"] == pytest.approx(12 / 29)
        assert result.loc["seasonal_naive_7", "wape"] == pytest.approx(12 / 116)
        # mean_28: пока пропуск находится в окне, прогноза нет; последние два дня — без ошибки
        assert result.loc["mean_28", "mae"] == pytest.approx(0.0)

    def test_on_real_data(self, full):
        result = evaluate_baselines(add_features(full), test_start="2024-11-01")
        assert (result["mae"] > 0).all()
        assert result["wape"].between(0, 2).all()
        # Усреднение за 28 дней сглаживает шум и должно быть точнее вчерашнего значения
        assert result.loc["mean_28", "wape"] < result.loc["naive_last", "wape"]
