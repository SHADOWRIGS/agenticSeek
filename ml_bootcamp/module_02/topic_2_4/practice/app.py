"""Практика 2.4.4: дашборд разведочного анализа продаж.

Запуск из папки practice:  uv run python app.py
Адрес в браузере:          http://127.0.0.1:8050/

Каркас рабочий: макет собран, фильтры действуют на график «по категориям». Остальные графики и карточки
показателей нужно подключить — см. раздел «Callback-функции» в конце файла.
"""

from dash import Dash, Input, Output, callback, dcc, html

from dashboard_data import FREQS, GROUPS, METRICS, filter_sales, load_sales, totals_by_category
from dashboard_figures import category_bar, empty_figure

# Данные загружаются один раз при старте. Callback-функции их только читают и никогда не изменяют.
SALES = load_sales()
CITIES = sorted(SALES["city"].unique())
CATEGORIES = sorted(SALES["category"].unique())
MIN_DATE = SALES["date"].min().date()
MAX_DATE = SALES["date"].max().date()

KPI_TITLES = {
    "revenue": "Выручка, руб.",
    "units": "Продано, шт.",
    "avg_daily_revenue": "Выручка в день, руб.",
    "promo_share": "Доля промо",
    "missing_share": "Пропуски в данных",
}
TODO_TEXT = "Здесь будет график: напишите для него callback"

PAGE_STYLE = {"maxWidth": "1200px", "margin": "0 auto", "padding": "0 16px 32px", "fontFamily": "Arial, sans-serif"}
FILTERS_STYLE = {"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px 24px"}
KPI_ROW_STYLE = {"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "12px", "margin": "16px 0"}
KPI_CARD_STYLE = {"border": "1px solid #d0d7de", "borderRadius": "8px", "padding": "10px 14px"}
GRID_STYLE = {"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"}
LABEL_STYLE = {"fontSize": "13px", "color": "#57606a", "marginBottom": "4px"}


def format_number(value: float) -> str:
    """12345678.9 -> '12 345 679'."""
    return f"{value:,.0f}".replace(",", " ")


def format_share(value: float) -> str:
    """0.0812 -> '8.1 %'."""
    return f"{value * 100:.1f} %"


def kpi_card(key: str, text: str) -> html.Div:
    """Карточка одного показателя. Значение лежит в элементе с id вида kpi-revenue."""
    return html.Div(
        [html.Div(KPI_TITLES[key], style=LABEL_STYLE), html.Div(text, id=f"kpi-{key}", style={"fontSize": "22px"})],
        style=KPI_CARD_STYLE,
    )


def labeled(label: str, component) -> html.Div:
    return html.Div([html.Div(label, style=LABEL_STYLE), component])


app = Dash(__name__, title="Продажи: разведочный анализ")
server = app.server  # объект Flask: нужен тестам и серверу приложений при развёртывании

app.layout = html.Div(
    [
        html.H2("Продажи сети: разведочный анализ"),
        html.Div(
            [
                labeled(
                    "Период",
                    dcc.DatePickerRange(
                        id="filter-dates",
                        start_date=MIN_DATE,
                        end_date=MAX_DATE,
                        min_date_allowed=MIN_DATE,
                        max_date_allowed=MAX_DATE,
                        display_format="DD.MM.YYYY",
                        first_day_of_week=1,
                    ),
                ),
                labeled("Города", dcc.Dropdown(id="filter-cities", options=CITIES, multi=True, placeholder="все")),
                labeled(
                    "Категории",
                    dcc.Dropdown(id="filter-categories", options=CATEGORIES, multi=True, placeholder="все"),
                ),
                labeled(
                    "Показатель", dcc.RadioItems(id="filter-metric", options=METRICS, value="revenue", inline=True)
                ),
                labeled("Шаг по времени", dcc.RadioItems(id="filter-freq", options=FREQS, value="W", inline=True)),
                labeled("Разрез динамики", dcc.RadioItems(id="filter-by", options=GROUPS, value="city", inline=True)),
            ],
            style=FILTERS_STYLE,
        ),
        html.Div([kpi_card(key, "—") for key in KPI_TITLES], id="kpi-row", style=KPI_ROW_STYLE),
        dcc.Graph(id="graph-timeseries", figure=empty_figure(TODO_TEXT)),
        html.Div(
            [
                dcc.Graph(id="graph-category"),
                dcc.Graph(id="graph-top-products", figure=empty_figure(TODO_TEXT)),
                dcc.Graph(id="graph-weekday", figure=empty_figure(TODO_TEXT)),
                dcc.Graph(id="graph-promo", figure=empty_figure(TODO_TEXT)),
            ],
            style=GRID_STYLE,
        ),
    ],
    style=PAGE_STYLE,
)

# --- Callback-функции -------------------------------------------------------------------------------------
# Четыре фильтра выборки нужны каждому графику, поэтому их описание вынесено в общий список.
FILTER_INPUTS = [
    Input("filter-dates", "start_date"),
    Input("filter-dates", "end_date"),
    Input("filter-cities", "value"),
    Input("filter-categories", "value"),
]


@callback(Output("graph-category", "figure"), *FILTER_INPUTS, Input("filter-metric", "value"))
def update_category(start, end, cities, categories, metric):
    """Образец: фильтры -> таблица -> фигура. Функция ничего не хранит и ничего не изменяет."""
    selected = filter_sales(SALES, start, end, cities, categories)
    return category_bar(totals_by_category(selected, metric), metric)


# Допишите callback-функции по этому образцу:
#   kpi-row.children           — пять карточек kpi_card(...) по результату kpi_summary;
#                                revenue, units, avg_daily_revenue — через format_number, доли — через format_share
#   graph-timeseries.figure    — time_series + time_series_figure (зависит ещё от filter-freq и filter-by)
#   graph-top-products.figure  — top_products + top_products_figure
#   graph-weekday.figure       — weekday_profile + weekday_heatmap
#   graph-promo.figure         — promo_effect + promo_figure (от показателя не зависит)


if __name__ == "__main__":
    app.run(debug=True)
