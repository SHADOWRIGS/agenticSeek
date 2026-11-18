"""Обнаружение дрейфа данных: практика 7.4.4.

Функции сравнивают текущую выборку (окно запросов) с эталонной (обучающей): индекс стабильности PSI,
критерий Колмогорова — Смирнова, доля пропусков, доля новых категорий — и собирают отчёт по столбцам.

Термины:
- эталон (reference) — выборка, на которой модель обучена; текущая выборка (current) — то, что приходит
  в работе;
- доли — доли строк по корзинам (числовой признак) или по категориям (категориальный);
- пропуск — None или NaN; в расчёте PSI, критерия и новых категорий пропуски не участвуют.

Выборки передаются списком, массивом NumPy или pandas.Series. Все функции возвращают обычные float
Python, а не числа NumPy.

Готовые части менять не нужно: константы. Пороги читаются из констант модуля в момент вызова.
Реализуйте функции так, чтобы проходили тесты из tests/test_sales_drift.py.
"""

import pandas as pd

# Наименьшая доля корзины при расчёте PSI: меньшие значения заменяются на EPS
EPS = 1e-4
# Число корзин числового признака по умолчанию
DEFAULT_BINS = 10
# Пороги PSI: [PSI_WARNING, PSI_DRIFT) — "warning", от PSI_DRIFT — "drift"
PSI_WARNING = 0.1
PSI_DRIFT = 0.25
# Рост доли пропусков (текущая минус эталонная), начиная с которого столбец получает статус "drift"
MISSING_DELTA = 0.05
# Доля строк с новыми категориями, начиная с которой столбец получает статус "drift"
NEW_CATEGORY_SHARE = 0.01
# Ключи отчёта по столбцу
NUMERIC_KEYS = (
    "kind",
    "psi",
    "psi_noise",
    "ks_statistic",
    "ks_pvalue",
    "missing_reference",
    "missing_current",
    "status",
)
CATEGORICAL_KEYS = (
    "kind",
    "psi",
    "psi_noise",
    "new_category_share",
    "new_categories",
    "missing_reference",
    "missing_current",
    "status",
)


def psi_from_shares(reference_shares, current_shares, eps: float = EPS) -> float:
    """PSI по двум наборам долей.

    reference_shares и current_shares — доли по одним и тем же корзинам, в одном порядке.
    Каждая доля меньше eps заменяется на eps (после замены доли заново не нормируются), затем

        PSI = сумма по корзинам (current - reference) * ln(current / reference).

    ValueError, если наборы пусты или разной длины.
    """
    raise NotImplementedError


def numeric_psi(reference, current, bins: int = DEFAULT_BINS) -> float:
    """PSI числового признака: корзины — по квантилям эталона.

    Пропуски удаляются из обеих выборок. Границы корзин — квантили эталона уровней 1/bins, 2/bins, …,
    (bins-1)/bins (numpy.quantile с интерполяцией по умолчанию); повторяющиеся границы объединяются.
    При k различных границах корзин k + 1: значение попадает в корзину с номером, равным числу границ
    строго меньше него, — значение, равное границе, относится к корзине слева от неё; значения за пределами
    эталона попадают в крайние корзины. Доли обеих выборок по этим корзинам передаются в psi_from_shares.

    ValueError, если после удаления пропусков одна из выборок пуста или bins < 2.
    """
    raise NotImplementedError


def categorical_psi(reference, current) -> float:
    """PSI категориального признака: корзины — категории обеих выборок.

    Пропуски удаляются; значения сравниваются как строки (1 и "1" — одна категория). Корзины — все
    категории, встретившиеся хотя бы в одной выборке; доли передаются в psi_from_shares, так что
    категория, которой нет в одной из выборок, получает в ней долю EPS.

    ValueError, если после удаления пропусков одна из выборок пуста.
    """
    raise NotImplementedError


def psi_noise(n_reference: int, n_current: int, groups: int) -> float:
    """Ожидаемое значение PSI двух выборок из одного распределения (урок 7.4.3):

        (groups - 1) * (1 / n_reference + 1 / n_current),

    где groups — число корзин. ValueError, если размер выборки или число корзин не положительны.
    """
    raise NotImplementedError


def ks_test(reference, current) -> tuple[float, float]:
    """Двухвыборочный критерий Колмогорова — Смирнова: (статистика D, p-значение).

    Пропуски удаляются. Результат совпадает с scipy.stats.ks_2samp с параметрами по умолчанию.
    ValueError, если после удаления пропусков одна из выборок пуста.
    """
    raise NotImplementedError


def missing_share(values) -> float:
    """Доля пропусков (None и NaN) среди значений; для пустой выборки — 0.0."""
    raise NotImplementedError


def new_category_share(reference, current) -> float:
    """Доля строк текущей выборки с категорией, которой нет в эталоне.

    Пропуски удаляются из обеих выборок; значения сравниваются как строки. Доля считается среди
    оставшихся строк current; если их нет — 0.0. Если эталон пуст, новыми считаются все категории.
    """
    raise NotImplementedError


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    bins: int = DEFAULT_BINS,
) -> dict[str, dict]:
    """Отчёт о дрейфе по столбцам.

    numeric и categorical — имена столбцов, которые есть в обеих таблицах. Результат — словарь
    «имя столбца -> словарь»: сначала столбцы numeric, затем categorical, в заданном порядке.

    Для числового столбца — ключи NUMERIC_KEYS в этом порядке:
        "kind"              — "numeric";
        "psi"               — numeric_psi(reference[c], current[c], bins);
        "psi_noise"         — psi_noise(n_reference, n_current, groups): размеры выборок — без пропусков,
                              groups — действительное число корзин (число различных границ плюс 1);
        "ks_statistic", "ks_pvalue" — ks_test;
        "missing_reference", "missing_current" — missing_share;
        "status".
    Для категориального — ключи CATEGORICAL_KEYS в этом порядке:
        "kind"               — "categorical";
        "psi"                — categorical_psi;
        "psi_noise"          — то же, groups — число категорий в обеих выборках вместе;
        "new_category_share" — new_category_share;
        "new_categories"     — отсортированный список новых категорий (строки);
        "missing_reference", "missing_current", "status".

    Статус определяется по первому выполненному правилу:
        1. "drift"      — missing_current - missing_reference >= MISSING_DELTA; для категориального
                          столбца также new_category_share >= NEW_CATEGORY_SHARE;
        2. "low_sample" — psi_noise >= PSI_WARNING: выборка мала, по PSI судить нельзя;
        3. "drift"      — psi >= PSI_DRIFT;
        4. "warning"    — psi >= PSI_WARNING;
        5. "ok".
    p-значение критерия на статус не влияет (урок 7.4.3: на больших выборках оно срабатывает на любой
    сдвиг).

    ValueError, если столбца нет в одной из таблиц (имя есть в тексте ошибки). Таблицы не изменяются.
    """
    raise NotImplementedError
