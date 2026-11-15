"""Практика 3.2.4: решения о блокировке операций, оптимальные по деньгам.

Реализуйте функции этого модуля. Ожидаемое поведение описано в docstring каждой функции
и проверяется тестами: uv run pytest -q

Во всех функциях y_true — метки (1 — мошенничество), amounts — суммы операций в рублях,
proba — предсказанные вероятности мошенничества, flagged — решения (True или 1 — операция заблокирована).
Аргументы могут быть списками, массивами NumPy или Series; индекс Series значения не имеет.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    """Цена исходов в рублях."""

    review: float = 100.0  # проверка заблокированной операции: работа аналитика и звонок клиенту
    friction: float = 400.0  # потери от блокировки честной операции: недовольство и уход клиентов


DEFAULT_THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)  # 0.01, 0.02, ..., 0.99
STRATEGIES = ["threshold_0.5", "threshold_f1", "threshold_savings", "expected_value", "oracle"]


def savings(y_true, flagged, amounts, costs: Costs) -> float:
    """Экономия в рублях по сравнению с работой без контроля, когда ни одна операция не блокируется.

    Вклад каждой операции:

    - заблокирована, мошенническая: amount − costs.review (сумма спасена, проверка оплачена);
    - заблокирована, честная: −(costs.review + costs.friction);
    - не заблокирована: 0 (убыток от пропущенного мошенничества уже входит в базовую линию).

    Возвращает сумму вкладов как float. Для пустых данных — 0.0.
    """
    raise NotImplementedError


def savings_curve(y_true, proba, amounts, costs: Costs, thresholds=None) -> pd.DataFrame:
    """Экономия и метрики классификации для каждого порога.

    Операция блокируется, если proba >= threshold. thresholds по умолчанию — DEFAULT_THRESHOLDS.
    Результат: по одной строке на порог в исходном порядке, индекс 0..k−1, столбцы:

    - threshold;
    - n_flagged: число заблокированных операций;
    - precision, recall, f1: если знаменатель равен нулю, значение метрики 0.0;
    - savings: экономия в рублях (см. savings).
    """
    raise NotImplementedError


def best_threshold(y_true, proba, amounts, costs: Costs, criterion: str = "savings", thresholds=None) -> float:
    """Порог из сетки thresholds, при котором критерий максимален.

    criterion: "savings" — экономия, "f1" — F1-мера. Любое другое значение — ValueError.
    Если максимум достигается на нескольких порогах, возвращается наименьший из них.
    Функцию вызывают на валидационных данных: подбирать порог на тесте нельзя.
    """
    raise NotImplementedError


def break_even_threshold(amounts, costs: Costs) -> np.ndarray:
    """Индивидуальный порог безубыточности для каждой операции.

    Ожидаемая экономия от блокировки операции с вероятностью мошенничества p равна
    p · (amount − review) − (1 − p) · (review + friction). Она положительна, когда

        p > (review + friction) / (amount + friction).

    Функция возвращает правую часть для каждой операции. Для сумм меньше цены проверки порог
    больше единицы — такие операции блокировать невыгодно при любой вероятности; обрезать его не нужно.
    """
    raise NotImplementedError


def expected_value_decision(proba, amounts, costs: Costs) -> np.ndarray:
    """Решение по ожидаемой выгоде: блокировать операцию, если ожидаемая экономия строго положительна.

    Возвращает булев массив: proba > break_even_threshold(amounts, costs).
    Решение осмысленно только для откалиброванных вероятностей.
    """
    raise NotImplementedError


def bootstrap_savings_interval(
    y_true, flagged, amounts, costs: Costs, baseline=None, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95 % доверительный интервал экономии методом бутстрепа.

    На каждом из n_boot повторов операции выбираются с возвращением (столько же, сколько в данных),
    и на выборке считается экономия решений flagged. Если передан baseline — решения другой стратегии
    на тех же операциях, — считается разность: экономия flagged минус экономия baseline на той же выборке
    (парное сравнение). Возвращает (2.5-й процентиль, 97.5-й процентиль) как пару float.
    Генератор случайных чисел — np.random.default_rng(seed); одинаковый seed даёт одинаковый результат.
    """
    raise NotImplementedError


def compare_strategies(valid: pd.DataFrame, test: pd.DataFrame, costs: Costs) -> pd.DataFrame:
    """Сравнение стратегий принятия решений. Пороги подбираются на valid, результат считается на test.

    valid и test — таблицы со столбцами is_fraud, proba, amount. Индекс результата — STRATEGIES:

    - threshold_0.5: порог 0.5;
    - threshold_f1: порог с лучшей F1 на valid (сетка DEFAULT_THRESHOLDS);
    - threshold_savings: порог с лучшей экономией на valid (сетка DEFAULT_THRESHOLDS);
    - expected_value: решение по ожидаемой выгоде для каждой операции test;
    - oracle: блокируются все мошеннические операции с суммой больше costs.review и только они —
      недостижимый максимум экономии.

    Столбцы: threshold (NaN для expected_value и oracle), n_flagged, precision, recall, savings,
    share_of_oracle — доля экономии стратегии от экономии oracle.
    """
    raise NotImplementedError
