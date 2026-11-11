"""Тесты графиков работают на маленьких таблицах, собранных вручную: функции dashboard_figures
можно писать и проверять независимо от функций dashboard_data."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from dashboard_data import GROUPS, METRICS, WEEKDAYS
from dashboard_figures import (
    NO_DATA_TEXT,
    PROMO_Y_TITLE,
    category_bar,
    empty_figure,
    promo_figure,
    time_series_figure,
    top_products_figure,
    weekday_heatmap,
)

TOTALS = pd.DataFrame({"category": ["Электроника", "Аксессуары"], "revenue": [900.0, 100.0]})

TS = pd.DataFrame(
    {
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-08", "2024-01-08", "2024-01-15"]),
        "city": ["Казань", "Москва", "Казань", "Москва", "Москва"],
        "revenue": [10.0, 30.0, 12.0, 28.0, 35.0],
    }
)

TOP = pd.DataFrame(
    {
        "product": ["Смартфон", "Ноутбук", "Планшет"],
        "category": ["Электроника", "Электроника", "Электроника"],
        "units": [300.0, 200.0, 100.0],
    }
)

PROFILE = pd.DataFrame(
    [[1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 8.0], [10.0, 20.0, 30.0, 40.0, 50.0, np.nan, 80.0]],
    index=pd.Index(["Аксессуары", "Электроника"], name="category"),
    columns=WEEKDAYS,
)

EFFECT = pd.DataFrame(
    {
        "product": ["Мышь", "Зонт", "Чайник"],
        "units_regular": [7.0, 2.0, 3.5],
        "units_promo": [12.6, 3.2, np.nan],
        "lift": [1.8, 1.6, np.nan],
    }
)


def values(array) -> np.ndarray:
    return np.asarray(array, dtype=float)


def assert_has_title(fig: go.Figure) -> None:
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text


def assert_no_data(fig: go.Figure) -> None:
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert NO_DATA_TEXT in [annotation.text for annotation in fig.layout.annotations]


class TestSamples:
    """Функции-образцы: эти тесты проходят и на заготовке."""

    def test_empty_figure(self):
        assert_no_data(empty_figure())
        assert [annotation.text for annotation in empty_figure("Текст").layout.annotations] == ["Текст"]

    def test_category_bar(self):
        fig = category_bar(TOTALS, "revenue")
        assert_has_title(fig)
        assert len(fig.data) == 1
        assert fig.data[0].type == "bar"
        assert list(fig.data[0].x) == ["Электроника", "Аксессуары"]
        assert np.allclose(values(fig.data[0].y), [900.0, 100.0])
        assert fig.layout.yaxis.title.text == METRICS["revenue"]

    def test_category_bar_on_empty_table(self):
        assert_no_data(category_bar(TOTALS.iloc[0:0], "revenue"))


class TestTimeSeriesFigure:
    def test_one_line_per_group(self):
        fig = time_series_figure(TS, "revenue", "city")
        assert_has_title(fig)
        assert sorted(trace.name for trace in fig.data) == ["Казань", "Москва"]
        for trace in fig.data:
            assert trace.type in ("scatter", "scattergl")
            assert "lines" in trace.mode

    def test_trace_data(self):
        fig = time_series_figure(TS, "revenue", "city")
        moscow = next(trace for trace in fig.data if trace.name == "Москва")
        assert np.allclose(values(moscow.y), [30.0, 28.0, 35.0])
        assert list(pd.to_datetime(list(moscow.x))) == list(pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]))
        kazan = next(trace for trace in fig.data if trace.name == "Казань")
        assert np.allclose(values(kazan.y), [10.0, 12.0])

    def test_axis_and_legend_titles(self):
        fig = time_series_figure(TS, "revenue", "city")
        assert fig.layout.xaxis.title.text == "Дата"
        assert fig.layout.yaxis.title.text == METRICS["revenue"]
        assert fig.layout.legend.title.text == GROUPS["city"]

    def test_other_metric_and_group(self):
        table = TS.rename(columns={"city": "category", "revenue": "units"})
        fig = time_series_figure(table, "units", "category")
        assert fig.layout.yaxis.title.text == METRICS["units"]
        assert fig.layout.legend.title.text == GROUPS["category"]

    def test_empty_table(self):
        assert_no_data(time_series_figure(TS.iloc[0:0], "revenue", "city"))


class TestTopProductsFigure:
    def test_horizontal_bars(self):
        fig = top_products_figure(TOP, "units")
        assert_has_title(fig)
        assert len(fig.data) == 1
        trace = fig.data[0]
        assert trace.type == "bar"
        assert trace.orientation == "h"
        assert list(trace.y) == ["Смартфон", "Ноутбук", "Планшет"]
        assert np.allclose(values(trace.x), [300.0, 200.0, 100.0])

    def test_leader_is_on_top(self):
        assert top_products_figure(TOP, "units").layout.yaxis.autorange == "reversed"

    def test_axis_title(self):
        assert top_products_figure(TOP, "units").layout.xaxis.title.text == METRICS["units"]

    def test_empty_table(self):
        assert_no_data(top_products_figure(TOP.iloc[0:0], "units"))


class TestWeekdayHeatmap:
    def test_heatmap_trace(self):
        fig = weekday_heatmap(PROFILE, "revenue")
        assert_has_title(fig)
        assert len(fig.data) == 1
        trace = fig.data[0]
        assert trace.type == "heatmap"
        assert list(trace.x) == WEEKDAYS
        assert list(trace.y) == ["Аксессуары", "Электроника"]
        assert values(trace.z).shape == (2, 7)
        assert np.allclose(values(trace.z), PROFILE.to_numpy(), equal_nan=True)

    def test_axis_title(self):
        assert weekday_heatmap(PROFILE, "revenue").layout.xaxis.title.text == "День недели"

    def test_empty_table(self):
        assert_no_data(weekday_heatmap(PROFILE.iloc[0:0], "revenue"))


class TestPromoFigure:
    def test_two_bar_traces(self):
        fig = promo_figure(EFFECT)
        assert_has_title(fig)
        assert [trace.name for trace in fig.data] == ["Без промо", "Промо"]
        assert all(trace.type == "bar" for trace in fig.data)
        assert fig.layout.barmode == "group"

    def test_trace_data(self):
        regular, promo = promo_figure(EFFECT).data
        assert list(regular.x) == ["Мышь", "Зонт", "Чайник"]
        assert list(promo.x) == ["Мышь", "Зонт", "Чайник"]
        assert np.allclose(values(regular.y), [7.0, 2.0, 3.5])
        assert np.allclose(values(promo.y), [12.6, 3.2, np.nan], equal_nan=True)

    def test_axis_title(self):
        assert promo_figure(EFFECT).layout.yaxis.title.text == PROMO_Y_TITLE

    def test_empty_table(self):
        assert_no_data(promo_figure(EFFECT.iloc[0:0]))


@pytest.mark.parametrize("metric", list(METRICS))
def test_figures_accept_real_tables(sales, metric):
    """Связка с dashboard_data: проходит, когда готовы оба модуля."""
    from dashboard_data import promo_effect, time_series, top_products, weekday_profile

    assert len(time_series_figure(time_series(sales, metric, "M", "category"), metric, "category").data) == 4
    assert len(top_products_figure(top_products(sales, metric), metric).data[0].y) == 5
    assert values(weekday_heatmap(weekday_profile(sales, metric), metric).data[0].z).shape == (4, 7)
    assert len(promo_figure(promo_effect(sales)).data[0].x) == 12
