import numpy as np
import pandas as pd
import pytest

from dashboard_data import (
    FREQS,
    WEEKDAYS,
    filter_sales,
    kpi_summary,
    promo_effect,
    time_series,
    top_products,
    totals_by_category,
    weekday_profile,
)

N_PAIRS = 96  # 8 магазинов x 12 товаров


class TestSamples:
    """Функции-образцы: эти тесты проходят и на заготовке."""

    def test_load_sales(self, sales):
        assert len(sales) == 731 * N_PAIRS
        expected = ["date", "store_id", "city", "category", "product", "price", "promo", "units", "revenue"]
        assert list(sales.columns) == expected

    def test_filter_period_is_inclusive(self, sales):
        out = filter_sales(sales, "2024-03-01", "2024-03-31")
        assert out["date"].min() == pd.Timestamp("2024-03-01")
        assert out["date"].max() == pd.Timestamp("2024-03-31")
        assert len(out) == 31 * N_PAIRS
        assert list(out.index) == list(range(len(out)))

    def test_filter_by_lists(self, sales):
        out = filter_sales(sales, cities=["Казань"], categories=["Электроника", "Для дома"])
        assert set(out["city"]) == {"Казань"}
        assert set(out["category"]) == {"Электроника", "Для дома"}

    def test_filter_none_and_empty_list_mean_no_restriction(self, sales):
        assert len(filter_sales(sales)) == len(sales)
        assert len(filter_sales(sales, None, None, [], [])) == len(sales)

    def test_filter_can_return_empty_table(self, sales):
        out = filter_sales(sales, start="2030-01-01")
        assert len(out) == 0
        assert list(out.columns) == list(sales.columns)

    def test_totals_by_category(self, january):
        out = totals_by_category(january, "units")
        expected = january.groupby("category")["units"].sum().sort_values(ascending=False)
        assert list(out.columns) == ["category", "units"]
        assert list(out["category"]) == list(expected.index)
        assert np.allclose(out["units"], expected.to_numpy())

    def test_totals_on_empty_table(self, empty):
        out = totals_by_category(empty, "revenue")
        assert len(out) == 0
        assert list(out.columns) == ["category", "revenue"]

    def test_totals_unknown_metric(self, january):
        with pytest.raises(ValueError):
            totals_by_category(january, "price")


class TestKpiSummary:
    KEYS = {"revenue", "units", "avg_daily_revenue", "promo_share", "missing_share"}

    def test_keys_and_types(self, january):
        kpi = kpi_summary(january)
        assert set(kpi) == self.KEYS
        assert all(isinstance(value, float) for value in kpi.values())

    def test_matches_direct_calculation(self, january):
        kpi = kpi_summary(january)
        assert kpi["revenue"] == pytest.approx(january["revenue"].sum())
        assert kpi["units"] == pytest.approx(january["units"].sum())
        assert kpi["avg_daily_revenue"] == pytest.approx(january["revenue"].sum() / 31)
        assert kpi["promo_share"] == pytest.approx(january["promo"].mean())
        assert kpi["missing_share"] == pytest.approx(january["units"].isna().mean())

    def test_average_uses_dates_present_in_selection(self, sales):
        weekends = sales[sales["date"].dt.day_of_week >= 5]
        kpi = kpi_summary(weekends)
        assert kpi["avg_daily_revenue"] == pytest.approx(weekends["revenue"].sum() / weekends["date"].nunique())

    def test_missing_share(self, january):
        damaged = january.copy()
        damaged.loc[damaged.index[: len(damaged) // 4], "units"] = np.nan
        assert kpi_summary(damaged)["missing_share"] == pytest.approx(damaged["units"].isna().mean())
        assert kpi_summary(damaged)["missing_share"] >= 0.25

    def test_empty_table_gives_zeros(self, empty):
        kpi = kpi_summary(empty)
        assert set(kpi) == self.KEYS
        assert all(value == 0.0 for value in kpi.values())


class TestTimeSeries:
    def test_columns(self, january):
        assert list(time_series(january).columns) == ["date", "city", "revenue"]
        assert list(time_series(january, "units", "D", "category").columns) == ["date", "category", "units"]

    def test_daily_matches_groupby(self, sales):
        out = time_series(sales, "revenue", "D", "city")
        expected = sales.groupby(["date", "city"], as_index=False)["revenue"].sum()
        pd.testing.assert_frame_equal(out, expected, check_dtype=False)

    def test_week_starts_on_monday(self, sales):
        out = time_series(sales, "revenue", "W", "city")
        assert (out["date"].dt.day_of_week == 0).all()
        assert out["date"].min() == pd.Timestamp("2022-12-26")  # 1 января 2023 года — воскресенье
        week = sales[(sales["date"] >= "2024-01-01") & (sales["date"] <= "2024-01-07") & (sales["city"] == "Москва")]
        row = out[(out["date"] == "2024-01-01") & (out["city"] == "Москва")]
        assert len(row) == 1
        assert row["revenue"].iloc[0] == pytest.approx(week["revenue"].sum())

    def test_month_starts_on_first_day(self, sales):
        out = time_series(sales, "units", "M", "category")
        assert (out["date"].dt.day == 1).all()
        assert len(out) == 24 * 4
        february = sales[(sales["date"].dt.year == 2024) & (sales["date"].dt.month == 2)]
        row = out[(out["date"] == "2024-02-01") & (out["category"] == "Для дома")]
        assert row["units"].iloc[0] == pytest.approx(february.loc[february["category"] == "Для дома", "units"].sum())

    @pytest.mark.parametrize("freq", list(FREQS))
    def test_total_does_not_depend_on_step(self, sales, freq):
        out = time_series(sales, "revenue", freq, "city")
        assert out["revenue"].sum() == pytest.approx(sales["revenue"].sum())
        assert out["revenue"].notna().all()

    def test_sorted_with_clean_index(self, sales):
        out = time_series(sales, "revenue", "W", "category")
        assert list(out.index) == list(range(len(out)))
        pd.testing.assert_frame_equal(out, out.sort_values(["date", "category"], ignore_index=True))

    def test_source_is_not_modified(self, january):
        before = january.copy()
        time_series(january, "revenue", "W", "city")
        pd.testing.assert_frame_equal(january, before)

    def test_empty_table(self, empty):
        out = time_series(empty, "units", "M", "category")
        assert len(out) == 0
        assert list(out.columns) == ["date", "category", "units"]

    @pytest.mark.parametrize(
        "kwargs", [{"metric": "price"}, {"freq": "ME"}, {"freq": "Y"}, {"by": "store_id"}, {"by": "product"}]
    )
    def test_invalid_arguments(self, january, kwargs):
        with pytest.raises(ValueError):
            time_series(january, **kwargs)


class TestTopProducts:
    def test_matches_direct_calculation(self, january):
        out = top_products(january, "revenue", n=3)
        totals = january.groupby(["product", "category"], as_index=False)["revenue"].sum()
        expected = totals.sort_values("revenue", ascending=False, ignore_index=True).head(3)
        assert list(out.columns) == ["product", "category", "revenue"]
        pd.testing.assert_frame_equal(out, expected, check_dtype=False)

    def test_default_is_five_rows(self, sales):
        out = top_products(sales)
        assert len(out) == 5
        assert list(out.index) == [0, 1, 2, 3, 4]
        assert out["revenue"].is_monotonic_decreasing

    def test_leader_depends_on_metric(self, sales):
        assert top_products(sales, "revenue", n=1)["product"].iloc[0] == "Смартфон"
        assert top_products(sales, "units", n=1)["product"].iloc[0] == "Мышь"

    def test_n_larger_than_number_of_products(self, sales):
        assert len(top_products(sales, "units", n=100)) == 12

    def test_source_is_not_modified(self, january):
        before = january.copy()
        top_products(january)
        pd.testing.assert_frame_equal(january, before)

    def test_empty_table(self, empty):
        out = top_products(empty, "units")
        assert len(out) == 0
        assert list(out.columns) == ["product", "category", "units"]

    def test_unknown_metric(self, january):
        with pytest.raises(ValueError):
            top_products(january, "promo")


class TestWeekdayProfile:
    def test_shape_and_labels(self, sales):
        out = weekday_profile(sales)
        assert list(out.columns) == WEEKDAYS
        assert list(out.index) == sorted(sales["category"].unique())
        assert out.index.name == "category"
        assert out.notna().all().all()

    def test_value_is_mean_of_daily_sums(self, january):
        out = weekday_profile(january, "revenue")
        daily = january.groupby(["date", "category"], as_index=False)["revenue"].sum()
        saturdays = daily[(daily["date"].dt.day_of_week == 5) & (daily["category"] == "Электроника")]
        assert out.loc["Электроника", "Сб"] == pytest.approx(saturdays["revenue"].mean())
        # среднее по строкам исходной таблицы — другая величина, в разы меньше
        rows = january[(january["date"].dt.day_of_week == 5) & (january["category"] == "Электроника")]
        assert out.loc["Электроника", "Сб"] > 2 * rows["revenue"].mean()

    def test_weekend_is_stronger_than_midweek(self, sales):
        out = weekday_profile(sales, "units")
        assert (out["Сб"] > out["Ср"]).all()
        assert (out["Вс"] > out["Ср"]).all()

    def test_absent_weekdays_are_nan(self, sales):
        three_days = sales[(sales["date"] >= "2024-01-01") & (sales["date"] <= "2024-01-03")]  # понедельник — среда
        out = weekday_profile(three_days)
        assert list(out.columns) == WEEKDAYS
        assert out[["Пн", "Вт", "Ср"]].notna().all().all()
        assert out[["Чт", "Пт", "Сб", "Вс"]].isna().all().all()

    def test_source_is_not_modified(self, january):
        before = january.copy()
        weekday_profile(january)
        pd.testing.assert_frame_equal(january, before)

    def test_empty_table(self, empty):
        out = weekday_profile(empty)
        assert out.shape == (0, 7)
        assert list(out.columns) == WEEKDAYS

    def test_unknown_metric(self, january):
        with pytest.raises(ValueError):
            weekday_profile(january, "price")


class TestPromoEffect:
    COLUMNS = ["product", "units_regular", "units_promo", "lift"]

    def test_one_row_per_product(self, sales):
        out = promo_effect(sales)
        assert list(out.columns) == self.COLUMNS
        assert sorted(out["product"]) == sorted(sales["product"].unique())
        assert list(out.index) == list(range(12))

    def test_matches_direct_calculation(self, sales):
        out = promo_effect(sales).set_index("product")
        mouse = sales[sales["product"] == "Мышь"]
        regular = mouse.loc[~mouse["promo"], "units"].mean()
        promo = mouse.loc[mouse["promo"], "units"].mean()
        assert out.loc["Мышь", "units_regular"] == pytest.approx(regular)
        assert out.loc["Мышь", "units_promo"] == pytest.approx(promo)
        assert out.loc["Мышь", "lift"] == pytest.approx(promo / regular)

    def test_lift_is_close_to_generator_value(self, sales):
        # в генератор данных заложен эффект промо 1.8
        out = promo_effect(sales)
        assert out["lift"].between(1.6, 2.0).all()
        assert out["lift"].is_monotonic_decreasing

    def test_product_without_promo_goes_last(self, sales):
        selection = sales[(sales["product"] != "Зонт") | ~sales["promo"]]
        out = promo_effect(selection)
        last = out.iloc[-1]
        assert last["product"] == "Зонт"
        assert np.isnan(last["units_promo"])
        assert np.isnan(last["lift"])
        assert last["units_regular"] > 0
        assert out["lift"].iloc[:-1].notna().all()

    def test_source_is_not_modified(self, january):
        before = january.copy()
        promo_effect(january)
        pd.testing.assert_frame_equal(january, before)

    def test_empty_table(self, empty):
        out = promo_effect(empty)
        assert len(out) == 0
        assert list(out.columns) == self.COLUMNS
