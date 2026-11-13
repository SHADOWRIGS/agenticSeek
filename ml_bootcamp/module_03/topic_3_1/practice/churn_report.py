"""Практика 3.1.4: модель оттока клиентов и её интерпретация.

Реализуйте функции этого модуля. Ожидаемое поведение описано в docstring каждой функции
и проверяется тестами: uv run pytest -q
"""

import pandas as pd

FEATURES = ["tenure_months", "fee_k", "services", "annual", "autopay", "support_calls"]
TARGET = "churn"


def prepare_features(clients: pd.DataFrame) -> pd.DataFrame:
    """Строит таблицу признаков из данных make_clients(). Исходная таблица не изменяется.

    Результат содержит ровно столбцы FEATURES в этом порядке, все числовые, индекс как у clients:

    - tenure_months, services, support_calls — без изменений;
    - fee_k — абонентская плата в тысячах рублей (monthly_fee / 1000);
    - annual — 1 для годового контракта, 0 для помесячного;
    - autopay — 1 или 0.
    """
    raise NotImplementedError


def fit_logit(features: pd.DataFrame, target: pd.Series):
    """Обучает логистическую регрессию без регуляризации со свободным членом.

    Возвращает результат statsmodels (Logit). Свободный член должен называться "const",
    остальные параметры — как столбцы features.
    """
    raise NotImplementedError


def odds_ratio_table(result) -> pd.DataFrame:
    """Таблица интерпретации коэффициентов для отчёта.

    Индекс — названия признаков без свободного члена, в порядке параметров модели. Столбцы:

    - coef: коэффициент (логарифм отношения шансов);
    - odds_ratio: отношение шансов, exp(coef);
    - ci_lower, ci_upper: границы 95 % доверительного интервала ОТНОШЕНИЯ ШАНСОВ;
    - p_value.
    """
    raise NotImplementedError


def average_marginal_effects(result, features: pd.DataFrame) -> pd.Series:
    """Средний эффект каждого признака на вероятность ухода, в долях (не в процентных пунктах).

    Индекс — названия признаков без свободного члена.

    - Для бинарного признака (в данных только значения 0 и 1): среднее по клиентам от разности
      вероятностей при значении признака 1 и при значении 0, остальные признаки клиента не меняются.
    - Для остальных признаков: среднее по клиентам от производной вероятности по признаку,
      равной coef · p · (1 − p).
    """
    raise NotImplementedError


def predict_scenarios(result, scenarios: pd.DataFrame) -> pd.Series:
    """Вероятность ухода для описанных клиентов.

    scenarios содержит столбцы признаков модели (без свободного члена), возможно в другом порядке.
    Возвращает Series с тем же индексом, что у scenarios. Функция должна работать и для одной строки.
    """
    raise NotImplementedError


def bootstrap_coef_intervals(
    features: pd.DataFrame, target: pd.Series, n_boot: int = 500, seed: int = 0
) -> pd.DataFrame:
    """95 % интервалы коэффициентов логистической регрессии методом бутстрепа.

    На каждом из n_boot повторов клиенты выбираются с возвращением (столько же, сколько в данных),
    и модель без регуляризации обучается заново. Результат: индекс — "const" и названия признаков,
    столбцы lower и upper — 2.5-й и 97.5-й процентили коэффициентов по повторам.
    Одинаковый seed должен давать одинаковый результат.
    """
    raise NotImplementedError


def evaluate(features: pd.DataFrame, target: pd.Series, n_splits: int = 5, seed: int = 0) -> dict[str, float]:
    """Оценивает качество модели на отложенных данных стратифицированной кросс-валидацией.

    Разбиение: StratifiedKFold(n_splits, shuffle=True, random_state=seed). Модель обучается на обучающих
    фолдах и предсказывает вероятности для проверочного. Возвращает словарь средних по фолдам значений:

    - "roc_auc": площадь под ROC-кривой;
    - "log_loss": логистическая функция потерь;
    - "baseline_log_loss": log loss предсказания, одинакового для всех клиентов и равного доле ушедших
      в обучающих фолдах.
    """
    raise NotImplementedError
