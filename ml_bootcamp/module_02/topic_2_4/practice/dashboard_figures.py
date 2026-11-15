"""Практика 2.4.4: графики дашборда.

Каждая функция получает готовую таблицу из dashboard_data и возвращает plotly.graph_objects.Figure.
Здесь нет ни фильтрации, ни агрегации — только превращение таблицы в картинку. Функции empty_figure
и category_bar реализованы как образец, остальные нужно написать: uv run pytest -q

Общие требования ко всем графикам:

- у фигуры есть непустой заголовок (layout.title.text);
- если входная таблица пуста, функция возвращает empty_figure(): фигуру без трасс с текстом NO_DATA_TEXT.

Строить графики можно и через plotly.express, и через graph_objects: тесты проверяют результат, а не способ.
"""

import pandas as pd
import plotly.graph_objects as go

from dashboard_data import GROUPS, METRICS, WEEKDAYS  # noqa: F401

NO_DATA_TEXT = "Нет данных для выбранных фильтров"
TEMPLATE = "plotly_white"
PROMO_Y_TITLE = "Продажи товара в магазине за день, шт."


def empty_figure(message: str = NO_DATA_TEXT) -> go.Figure:
    """Образец. Фигура без трасс с сообщением по центру."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"size": 16})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template=TEMPLATE)
    return fig


def category_bar(totals: pd.DataFrame, metric: str) -> go.Figure:
    """Образец. Столбчатая диаграмма по результату totals_by_category.

    - одна трасса типа bar: x — категории, y — значения показателя, в порядке строк таблицы;
    - подпись оси Y — METRICS[metric].
    """
    if totals.empty:
        return empty_figure()
    fig = go.Figure(go.Bar(x=totals["category"], y=totals[metric], hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
    fig.update_layout(
        title=f"{METRICS[metric].split(',')[0]} по категориям",
        yaxis_title=METRICS[metric],
        template=TEMPLATE,
        margin={"t": 50, "r": 10},
    )
    return fig


def time_series_figure(ts: pd.DataFrame, metric: str, by: str) -> go.Figure:
    """Линейный график по результату time_series.

    - по одной трассе на каждое значение разреза (столбец by), имя трассы — это значение;
    - трассы типа scatter или scattergl, в режиме (mode) есть линии;
    - x — даты, y — значения показателя своей группы;
    - подпись оси X — "Дата", подпись оси Y — METRICS[metric];
    - заголовок легенды — GROUPS[by].
    """
    raise NotImplementedError


def top_products_figure(top: pd.DataFrame, metric: str) -> go.Figure:
    """Горизонтальная столбчатая диаграмма по результату top_products.

    - одна трасса типа bar с orientation="h": y — товары, x — значения показателя, в порядке строк таблицы;
    - самый крупный товар — наверху: layout.yaxis.autorange равен "reversed";
    - подпись оси X — METRICS[metric].
    """
    raise NotImplementedError


def weekday_heatmap(profile: pd.DataFrame, metric: str) -> go.Figure:
    """Тепловая карта по результату weekday_profile.

    - одна трасса типа heatmap: x — дни недели (столбцы таблицы), y — категории (индекс таблицы),
      z — значения таблицы той же формы;
    - подпись оси X — "День недели".
    """
    raise NotImplementedError


def promo_figure(effect: pd.DataFrame) -> go.Figure:
    """Сгруппированные столбцы по результату promo_effect.

    - две трассы типа bar с именами "Без промо" и "Промо" — именно в этом порядке;
    - x — товары в порядке строк таблицы, y — units_regular и units_promo соответственно;
    - layout.barmode равен "group";
    - подпись оси Y — PROMO_Y_TITLE.
    """
    raise NotImplementedError
