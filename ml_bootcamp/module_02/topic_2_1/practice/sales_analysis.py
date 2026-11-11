"""Практика 2.1.4: подготовка и анализ временных рядов продаж.

Реализуйте функции этого модуля. Ожидаемое поведение описано в docstring каждой функции
и проверяется тестами: uv run pytest -q

Функции fill_calendar, add_features и evaluate_baselines должны работать и с категориальными,
и со строковыми столбцами store_id и product.
"""

from pathlib import Path

import pandas as pd

KEYS = ["store_id", "product"]


def load_raw(path: Path) -> pd.DataFrame:
    """Читает сырую выгрузку и приводит типы.

    - столбец date: datetime64 без часового пояса; в файле встречаются форматы ГГГГ-ММ-ДД и ДД.ММ.ГГГГ;
    - store_id, city, category, product: тип category;
    - promo: bool; price: целое; units, revenue: float (пропуски сохраняются);
    - полные дубликаты строк удалены;
    - строки отсортированы по store_id, product, date, индекс — 0..n-1.
    """
    raise NotImplementedError


def fill_calendar(sales: pd.DataFrame) -> pd.DataFrame:
    """Восстанавливает полный календарь для каждой пары (store_id, product).

    В сырых данных нет строк за дни, когда товар не продавался. Результат содержит строку на каждую дату
    от минимальной до максимальной даты во всей таблице для каждой пары, встречающейся в данных:

    - в добавленных строках units = 0, revenue = 0, promo = False;
    - price в добавленных строках — последняя известная цена этой пары (если раньше цен не было — ближайшая
      следующая);
    - city и category заполнены значениями своей пары;
    - пропуски units и revenue, которые были в исходных строках, остаются пропусками;
    - строки отсортированы по store_id, product, date, индекс — 0..n-1.
    """
    raise NotImplementedError


def add_features(
    sales: pd.DataFrame, lags: tuple[int, ...] = (1, 7, 28), windows: tuple[int, ...] = (7, 28)
) -> pd.DataFrame:
    """Добавляет признаки для прогноза units. Исходная таблица не изменяется.

    На входе — результат fill_calendar. Для каждой пары (store_id, product) отдельно:

    - units_lag_{k}: значение units k дней назад;
    - units_mean_{w}, units_std_{w}: среднее и стандартное отклонение units за w предыдущих дней.
      Текущий день в окно не входит. Если в окне меньше w значений, признак равен NaN.

    Календарные признаки:

    - day_of_week: 0 — понедельник, 6 — воскресенье;
    - is_weekend: bool;
    - month_sin, month_cos: циклическое кодирование месяца с периодом 12.
    """
    raise NotImplementedError


def evaluate_baselines(features: pd.DataFrame, test_start: str) -> pd.DataFrame:
    """Сравнивает простые прогнозы на один день вперёд на периоде date >= test_start.

    На входе — результат add_features с параметрами по умолчанию. Прогнозы:

    - naive_last: units_lag_1;
    - seasonal_naive_7: units_lag_7;
    - mean_28: units_mean_28.

    Возвращает таблицу с индексом из названий прогнозов и столбцами:

    - mae: средняя абсолютная ошибка;
    - wape: сумма абсолютных ошибок, делённая на сумму фактических значений.

    Строки, где факт или прогноз равны NaN, в расчёте соответствующего прогноза не участвуют.
    """
    raise NotImplementedError
