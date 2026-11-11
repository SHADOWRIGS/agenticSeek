"""Практика 3.3.4: сегментация клиентов интернет-магазина.

Реализуйте функции этого модуля. Ожидаемое поведение описано в docstring каждой функции
и проверяется тестами: uv run pytest -q
"""

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_check",
    "promo_share",
    "n_categories",
    "mobile_share",
    "account_age_days",
]
# monetary = frequency · avg_check: в профилях сегментов он нужен, а в кластеризации дублирует два признака
CLUSTER_FEATURES = [name for name in FEATURES if name != "monetary"]
LOG_FEATURES = ["recency_days", "frequency", "avg_check", "account_age_days"]
SHARE_FEATURES = ["promo_share", "mobile_share"]


def build_rfm_features(transactions: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    """Строит признаки клиентов по журналу заказов на дату snapshot_date. Журнал не изменяется.

    transactions — таблица make_transactions(): order_id, customer_id, order_date, amount, promo,
    category, mobile. Строки могут идти в любом порядке. Учитываются только заказы СТРОГО раньше
    snapshot_date (строка или pd.Timestamp); клиенты без таких заказов в результат не попадают.

    Результат: индекс — customer_id по возрастанию (имя индекса "customer_id"), столбцы — ровно FEATURES:

    - recency_days: целое число дней от последнего заказа до snapshot_date;
    - frequency: число заказов;
    - monetary: сумма заказов;
    - avg_check: средняя сумма заказа;
    - promo_share: доля заказов по акции;
    - n_categories: число различных категорий;
    - mobile_share: доля заказов с телефона;
    - account_age_days: целое число дней от первого заказа до snapshot_date.
    """
    raise NotImplementedError


def smooth_shares(features: pd.DataFrame, share_columns: Sequence[str], prior_strength: float = 2.0) -> pd.DataFrame:
    """Сглаживает доли: у клиента с одним-двумя заказами доля 0 или 1 — шум, а не свойство клиента.

    Для каждого столбца из share_columns:

        сглаженная доля = (доля · frequency + prior_strength · p0) / (frequency + prior_strength),

    где p0 — доля по всем заказам всех клиентов: sum(доля · frequency) / sum(frequency).
    Смысл: каждому клиенту добавлено prior_strength «виртуальных» заказов со средним поведением.
    Возвращает копию features, в которой изменены только столбцы share_columns. Исходная таблица
    не изменяется. При prior_strength=0 доли остаются прежними.
    """
    raise NotImplementedError


def preprocess(features: pd.DataFrame, log_columns: Sequence[str]):
    """Готовит матрицу для кластеризации. Возвращает (X, transformer). Таблица features не изменяется.

    X — np.ndarray той же формы, что features; порядок столбцов как в features. К столбцам log_columns
    применён log1p, затем все столбцы стандартизованы: среднее 0, стандартное отклонение 1.

    transformer — обученный преобразователь с методом transform(таблица) -> np.ndarray, который применяет
    к новой таблице с теми же столбцами те же шаги со средними и отклонениями, запомненными на features:
    transformer.transform(features) совпадает с X.
    """
    raise NotImplementedError


def evaluate_k(X: np.ndarray, k_values: Iterable[int], seed: int = 0) -> pd.DataFrame:
    """Таблица внутренних метрик k-means для каждого числа кластеров из k_values.

    Модель: KMeans(n_clusters=k, n_init=10, random_state=seed). Результат: индекс — k (имя индекса "k")
    в порядке k_values, столбцы: inertia, silhouette, calinski_harabasz, davies_bouldin.
    """
    raise NotImplementedError


def stability_score(X: np.ndarray, k: int, n_runs: int = 10, sample_share: float = 0.8, seed: int = 0) -> float:
    """Устойчивость разбиения на k кластеров к составу выборки: число от -1 до 1, чем больше, тем лучше.

    Эталонное разбиение — KMeans(n_clusters=k, n_init=10, random_state=seed) на всех объектах.
    n_runs раз: выбрать без возвращения долю sample_share объектов, обучить на них такую же модель
    и посчитать ARI между её метками и эталонными метками тех же объектов. Результат — среднее ARI.
    Одинаковый seed должен давать одинаковый результат.
    """
    raise NotImplementedError


def cluster_profiles(features: pd.DataFrame, labels) -> pd.DataFrame:
    """Профили кластеров в исходных единицах измерения.

    labels — метки кластеров в порядке строк features (массив или Series, индекс Series игнорируется).
    Результат: индекс — метки кластеров по возрастанию (имя индекса "cluster"); столбцы:
    size (число клиентов), share (доля клиентов), затем медиана каждого столбца features в его порядке.
    """
    raise NotImplementedError


def assign_segment(new_features: pd.DataFrame, transformer, model, segment_names: dict | None = None) -> pd.Series:
    """Определяет сегмент новых клиентов по уже обученной сегментации — без переобучения.

    new_features — таблица с теми же столбцами, на которых обучен transformer (результат preprocess);
    model — обученная модель кластеризации с методом predict (например, KMeans).
    Возвращает Series с индексом new_features и именем "segment": номер кластера, а если передан
    словарь segment_names (номер кластера -> название) — название сегмента.
    Функция должна работать и для одной строки.
    """
    raise NotImplementedError
