"""Синтетические клиенты интернет-магазина со скрытыми сегментами для уроков темы 3.3.

Использование:  from customers_data import make_customers, make_transactions
"""

import numpy as np
import pandas as pd

FEATURES = [
    "recency_days",
    "frequency",
    "avg_check",
    "promo_share",
    "n_categories",
    "mobile_share",
    "account_age_days",
]
NOISE_FEATURES = ["survey_score", "profile_update_days"]
SNAPSHOT_DATE = pd.Timestamp("2025-01-01")
CATEGORIES = ["электроника", "одежда", "дом", "красота", "спорт", "книги", "детям", "продукты"]

# Параметры скрытых сегментов. Нужны только генератору и самопроверке: в настоящей задаче сегментации
# ни этой таблицы, ни истинной метки сегмента не существует.
SEGMENTS = pd.DataFrame(
    {
        "share": [0.22, 0.20, 0.18, 0.25, 0.05, 0.10],  # доля клиентов
        "age": [800, 500, 45, 750, 650, 600],  # средний возраст аккаунта, дней
        "recency": [9, 30, 15, 280, 12, 70],  # средняя давность последней покупки, дней
        "rate": [14.0, 8.0, 12.0, 2.5, 24.0, 3.0],  # заказов в год, пока клиент активен
        "check": [3200, 1400, 2600, 2400, 28000, 11000],  # медиана среднего чека, руб.
        "check_sigma": [0.35, 0.35, 0.45, 0.45, 0.50, 0.40],  # разброс логарифма чека
        "promo": [0.25, 0.80, 0.45, 0.30, 0.08, 0.10],  # средняя доля заказов по акции
        "categories": [6.0, 4.0, 1.5, 2.5, 2.5, 3.0],  # среднее число категорий
        "mobile": [0.70, 0.60, 0.75, 0.40, 0.12, 0.50],  # средняя доля заказов с телефона
    },
    index=["loyal", "discount_hunters", "newcomers", "sleeping", "wholesale", "premium"],
)


def _draw_customers(n: int, seed: int, outlier_share: float) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    segment = rng.choice(SEGMENTS.index.to_numpy(), size=n, p=SEGMENTS["share"].to_numpy())
    p = SEGMENTS.loc[segment].reset_index(drop=True)

    age = np.clip(np.round(rng.gamma(4.0, p["age"] / 4.0)), 3, 1095)
    # Спящие не покупали минимум 120 дней; у остальных давность распределена экспоненциально
    recency = np.where(segment == "sleeping", 120 + rng.gamma(2.0, 80.0, n), rng.exponential(p["recency"]))
    recency = np.clip(np.round(recency), 1, age - 1)
    frequency = 1 + rng.poisson(p["rate"] * (age - recency) / 365)
    recency = np.where(frequency == 1, age, recency)  # единственная покупка — она же первая
    avg_check = np.exp(rng.normal(np.log(p["check"]), p["check_sigma"]))
    promo_share = rng.beta(8 * p["promo"], 8 * (1 - p["promo"]))
    mobile_share = rng.beta(8 * p["mobile"], 8 * (1 - p["mobile"]))
    n_categories = np.minimum(1 + rng.binomial(7, (p["categories"] - 1) / 7), frequency)

    # Выбросы: единичные клиенты с чеком в 5–15 раз больше обычного для своего сегмента
    is_outlier = rng.random(n) < outlier_share
    avg_check = np.where(is_outlier, avg_check * rng.uniform(5, 15, n), avg_check)

    customers = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "recency_days": recency.astype(int),
            "frequency": frequency.astype(int),
            "avg_check": np.round(avg_check, 2),
            "promo_share": np.round(promo_share, 3),
            "n_categories": n_categories.astype(int),
            "mobile_share": np.round(mobile_share, 3),
            "account_age_days": age.astype(int),
            "survey_score": np.clip(np.round(rng.normal(7, 1.5, n)), 1, 10).astype(int),
            "profile_update_days": rng.integers(0, 366, n),
        }
    )
    return customers, pd.Series(segment, index=customers.index, name="segment")


def make_customers(n: int = 3000, seed: int = 42, outlier_share: float = 0.01) -> tuple[pd.DataFrame, pd.Series]:
    """Клиенты магазина: одна строка на клиента. Возвращает (таблица признаков, истинный сегмент).

    Признаки FEATURES зависят от сегмента: давность последней покупки и возраст аккаунта (дней),
    число заказов, средний чек (руб., логнормальный), доли заказов по акции и с телефона, число категорий.
    Признаки NOISE_FEATURES (оценка в опросе, давность обновления профиля) от сегмента не зависят.
    У доли outlier_share клиентов средний чек завышен в 5–15 раз.

    Истинный сегмент возвращается отдельно и нужен только для проверки экспериментов.
    """
    return _draw_customers(n, seed, outlier_share)


def make_transactions(n: int = 3000, seed: int = 42, outlier_share: float = 0.01) -> tuple[pd.DataFrame, pd.Series]:
    """Журнал заказов тех же клиентов, что в make_customers(n, seed). Возвращает (журнал, истинный сегмент).

    Столбцы журнала: order_id, customer_id, order_date, amount (руб.), promo, category, mobile.
    История заканчивается накануне SNAPSHOT_DATE. Давность, число заказов, возраст аккаунта и число
    категорий восстанавливаются из журнала точно, средний чек и доли — приближённо: у клиента с двумя
    заказами доля заказов по акции может быть только 0, 0.5 или 1.
    Индекс истинного сегмента — customer_id.
    """
    customers, segment = _draw_customers(n, seed, outlier_share)
    rng = np.random.default_rng(seed + 1)
    frequency = customers["frequency"].to_numpy()
    row = np.repeat(np.arange(n), frequency)  # номер клиента для каждого заказа
    order_no = np.arange(len(row)) - np.repeat(np.cumsum(frequency) - frequency, frequency)

    # Первый заказ — в день регистрации, последний — recency дней назад, остальные между ними
    age = customers["account_age_days"].to_numpy()[row]
    recency = customers["recency_days"].to_numpy()[row]
    days_ago = np.round(rng.uniform(recency, age)).astype(int)
    days_ago = np.where(order_no == 0, age, days_ago)
    days_ago = np.where(order_no == frequency[row] - 1, recency, days_ago)

    # Первые n_categories заказов покрывают все категории клиента, дальше — случайная из его набора
    n_cat = customers["n_categories"].to_numpy()[row]
    preference = rng.random((n, len(CATEGORIES))).argsort(axis=1)
    slot = np.where(order_no < n_cat, order_no, rng.integers(0, n_cat))

    sigma = 0.3
    amount = customers["avg_check"].to_numpy()[row] * np.exp(rng.normal(-(sigma**2) / 2, sigma, len(row)))
    transactions = pd.DataFrame(
        {
            "customer_id": customers["customer_id"].to_numpy()[row],
            "order_date": SNAPSHOT_DATE - pd.to_timedelta(days_ago, unit="D"),
            "amount": np.round(amount, 2),
            "promo": rng.random(len(row)) < customers["promo_share"].to_numpy()[row],
            "category": np.array(CATEGORIES)[preference[row, slot]],
            "mobile": rng.random(len(row)) < customers["mobile_share"].to_numpy()[row],
        }
    )
    transactions = transactions.sort_values(["order_date", "customer_id"], kind="stable", ignore_index=True)
    transactions.insert(0, "order_id", np.arange(1, len(transactions) + 1))
    return transactions, segment.set_axis(customers["customer_id"]).rename("segment")
