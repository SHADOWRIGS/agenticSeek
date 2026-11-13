"""Практика 2.4.4: подготовка данных для дашборда продаж.

Функции этого модуля не знают ни о Dash, ни о Plotly: на входе и на выходе — таблицы pandas.
Три функции реализованы как образец (load_sales, filter_sales, totals_by_category), остальные нужно написать.
Ожидаемое поведение описано в docstring каждой функции и проверяется тестами: uv run pytest -q

Ни одна функция не изменяет переданную ей таблицу.
"""

import sys
from pathlib import Path

import pandas as pd

# sales_data.py лежит на уровень выше папки practice, а в репозитории курса — в папке темы 2.1
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE.parent, _HERE.parent.parent / "topic_2_1"):
    if (_candidate / "sales_data.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from sales_data import make_sales  # noqa: E402

METRICS = {"revenue": "Выручка, руб.", "units": "Продажи, шт."}
FREQS = {"D": "день", "W": "неделя", "M": "месяц"}
GROUPS = {"city": "Город", "category": "Категория"}
WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


def _check_option(value: str, allowed: dict[str, str], name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name}: ожидается одно из {sorted(allowed)}, получено {value!r}")


def load_sales() -> pd.DataFrame:
    """Образец. Данные дашборда: одна строка — продажи одного товара в одном магазине за один день."""
    return make_sales()


def filter_sales(
    sales: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    cities: list[str] | None = None,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """Образец. Строки, попадающие под фильтры.

    - start и end — границы периода, обе включительно; None — граница не задана;
    - cities и categories — списки допустимых значений; None или пустой список — ограничения нет;
    - результат — новая таблица с теми же столбцами и индексом 0..n-1.
    """
    mask = pd.Series(True, index=sales.index)
    if start is not None:
        mask &= sales["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= sales["date"] <= pd.Timestamp(end)
    if cities:
        mask &= sales["city"].isin(cities)
    if categories:
        mask &= sales["category"].isin(categories)
    return sales[mask].reset_index(drop=True)


def totals_by_category(sales: pd.DataFrame, metric: str = "revenue") -> pd.DataFrame:
    """Образец. Сумма показателя по категориям.

    - столбцы: category и столбец с именем metric;
    - строки отсортированы по убыванию показателя, индекс — 0..n-1;
    - пропуски в показателе не участвуют в сумме;
    - для пустой таблицы — пустая таблица с теми же столбцами;
    - metric не из METRICS — ValueError.
    """
    _check_option(metric, METRICS, "metric")
    totals = sales.groupby("category", as_index=False)[metric].sum()
    return totals.sort_values(metric, ascending=False, ignore_index=True)


def kpi_summary(sales: pd.DataFrame) -> dict[str, float]:
    """Ключевые показатели выборки для карточек в верхней части дашборда.

    Ключи словаря:

    - revenue: суммарная выручка;
    - units: суммарное число проданных единиц;
    - avg_daily_revenue: выручка, делённая на число различных дат в выборке;
    - promo_share: доля строк с промо (от 0 до 1);
    - missing_share: доля строк с пропуском в units (от 0 до 1).

    Все значения — float. Для пустой таблицы все значения равны 0.0 (не NaN).
    """
    raise NotImplementedError


def time_series(sales: pd.DataFrame, metric: str = "revenue", freq: str = "W", by: str = "city") -> pd.DataFrame:
    """Динамика показателя с шагом freq в разрезе by.

    - столбцы: date, столбец с именем by, столбец с именем metric — в этом порядке;
    - date — начало периода: для "D" сам день, для "W" понедельник этой недели, для "M" первое число месяца;
    - одна строка на каждую пару (период, значение разреза), которая есть в данных; значение — сумма показателя;
    - строки отсортированы по date, затем по разрезу; индекс — 0..n-1;
    - для пустой таблицы — пустая таблица с теми же столбцами;
    - metric не из METRICS, freq не из FREQS, by не из GROUPS — ValueError.
    """
    raise NotImplementedError


def top_products(sales: pd.DataFrame, metric: str = "revenue", n: int = 5) -> pd.DataFrame:
    """Не более n товаров с наибольшей суммой показателя.

    - столбцы: product, category, столбец с именем metric — в этом порядке;
    - строки отсортированы по убыванию показателя, индекс — 0..k-1;
    - если товаров в выборке меньше n, возвращаются все;
    - для пустой таблицы — пустая таблица с теми же столбцами;
    - metric не из METRICS — ValueError.
    """
    raise NotImplementedError


def weekday_profile(sales: pd.DataFrame, metric: str = "revenue") -> pd.DataFrame:
    """Средний дневной показатель категории по дням недели — таблица для тепловой карты.

    - индекс — категории в алфавитном порядке, имя индекса — category;
    - столбцы — WEEKDAYS: всегда все семь, от понедельника до воскресенья;
    - значение — среднее по датам от дневной суммы показателя категории: сначала сумма по паре
      (дата, категория), затем среднее по всем датам выборки, приходящимся на этот день недели;
    - если дня недели в выборке нет, в его столбце NaN;
    - для пустой таблицы — таблица без строк с семью столбцами;
    - metric не из METRICS — ValueError.
    """
    raise NotImplementedError


def promo_effect(sales: pd.DataFrame) -> pd.DataFrame:
    """Сравнение продаж товаров в дни с промо и без него.

    - столбцы: product, units_regular, units_promo, lift — в этом порядке;
    - units_regular и units_promo — среднее значение units по строкам без промо и с промо
      (строка — один товар в одном магазине за один день; пропуски в расчёте не участвуют);
    - lift = units_promo / units_regular;
    - одна строка на каждый товар выборки; если у товара нет строк с промо, units_promo и lift равны NaN;
    - строки отсортированы по убыванию lift, строки с NaN — в конце; индекс — 0..n-1;
    - для пустой таблицы — пустая таблица с теми же столбцами.
    """
    raise NotImplementedError
