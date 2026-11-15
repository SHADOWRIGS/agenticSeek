"""Синтетические заказы интернет-магазина для уроков темы 4.3: задача «вернёт ли покупатель заказ».

Использование:  from orders_data import CAT_FEATURES, FEATURES, NUM_FEATURES, TARGET, make_orders, split_by_month
"""

import numpy as np
import pandas as pd

TARGET = "is_returned"
CAT_FEATURES = ["payment", "category", "city", "promo_code", "seller_id"]
NUM_FEATURES = ["price", "discount", "n_items", "delivery_days", "customer_orders", "is_app", "weekday", "month"]
FEATURES = CAT_FEATURES + NUM_FEATURES

PAYMENTS = ["карта", "при получении", "рассрочка"]
CATEGORIES = [
    "одежда",
    "обувь",
    "электроника",
    "бытовая техника",
    "косметика",
    "детские товары",
    "спорт",
    "книги",
    "мебель",
    "продукты",
    "зоотовары",
    "украшения",
]
N_CITIES = 300
N_PROMO = 400
N_SELLERS = 3000
N_MONTHS = 24

_PAYMENT_SHARE = [0.62, 0.28, 0.10]
_PAYMENT_EFFECT = np.array([0.0, 0.45, -0.2])
_CATEGORY_EFFECT = np.array([0.9, 1.1, 0.1, -0.2, -0.5, 0.0, 0.2, -1.0, 0.3, -1.4, -0.7, 0.5])
_CATEGORY_LOG_PRICE = np.array([7.8, 8.3, 9.6, 9.9, 7.2, 7.5, 8.2, 6.4, 9.8, 6.9, 7.0, 8.6])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _zipf_weights(n: int, shift: float, rng: np.random.Generator) -> np.ndarray:
    """Вероятности значений признака: немного частых значений и длинный хвост редких."""
    weights = 1 / (rng.permutation(n) + shift)
    return weights / weights.sum()


def make_orders(n: int = 40_000, seed: int = 42, include_hidden: bool = False) -> pd.DataFrame:
    """Заказы интернет-магазина за 24 месяца: одна строка на заказ, отсортированы по времени.

    Категориальные признаки (строки):
        payment — способ оплаты, 3 значения;
        category — товарная категория, 12 значений;
        city — город доставки, 300 значений; у 2 % заказов пропущен;
        promo_code — промокод, 400 значений; у 70 % заказов промокода нет (пропуск);
        seller_id — продавец, 3000 значений. Продавцы приходят на площадку постепенно: четверть из них
                    начинает работать после первого месяца, поэтому в поздних месяцах есть продавцы,
                    которых не было в ранних.
    Частоты городов, промокодов и продавцов распределены неравномерно: несколько крупных значений
    и длинный хвост редких.

    Числовые признаки:
        price — сумма заказа, руб.; discount — доля скидки от 0 до 0.7; n_items — число товаров;
        delivery_days — срок доставки, дней (у 5 % заказов пропущен); customer_orders — сколько заказов
        покупатель сделал раньше; is_app — заказ из приложения (0/1); weekday — день недели (0–6);
        month — номер месяца заказа, от 0 до 23.

    Целевая переменная is_returned: 1, если заказ возвращён. Логарифм шансов возврата — сумма:
        эффекта способа оплаты и эффекта категории;
        эффекта пары «категория × способ оплаты» (случайный, со стандартным отклонением 0.5);
        случайных эффектов города (0.35), продавца (0.7) и промокода (0.4);
        0.35 · ln(price / 3000), ограниченного снизу нулём;
        1.2 · discount; 0.6, если доставка дольше 7 дней; −0.25 · ln(1 + customer_orders);
        0.15 · (n_items − 1), но не больше 0.6.
    Признаки is_app и weekday на возврат не влияют; месяц влияет только через состав продавцов.
    При include_hidden=True добавляется столбец true_proba — истинная вероятность возврата.
    """
    rng = np.random.default_rng(seed)
    city_effect = rng.normal(0, 0.35, N_CITIES)
    seller_effect = rng.normal(0, 0.7, N_SELLERS)
    promo_effect = rng.normal(0, 0.4, N_PROMO)
    pair_effect = rng.normal(0, 0.5, (len(CATEGORIES), len(PAYMENTS)))
    city_p = _zipf_weights(N_CITIES, 3, rng)
    promo_p = _zipf_weights(N_PROMO, 5, rng)
    seller_p = _zipf_weights(N_SELLERS, 10, rng)
    seller_start = np.where(rng.random(N_SELLERS) < 0.75, 0, rng.integers(1, N_MONTHS, N_SELLERS))
    # Продавец тяготеет к одной категории: 70 % его заказов приходится на неё
    seller_category = rng.integers(0, len(CATEGORIES), N_SELLERS)

    month = np.sort(rng.integers(0, N_MONTHS, n))
    seller = np.empty(n, dtype=int)
    for m in range(N_MONTHS):
        rows = np.flatnonzero(month == m)
        p = np.where(seller_start <= m, seller_p, 0.0)
        seller[rows] = rng.choice(N_SELLERS, size=len(rows), p=p / p.sum())
    category = np.where(rng.random(n) < 0.7, seller_category[seller], rng.integers(0, len(CATEGORIES), n))
    payment = rng.choice(len(PAYMENTS), size=n, p=_PAYMENT_SHARE)
    city = rng.choice(N_CITIES, size=n, p=city_p)
    promo = rng.choice(N_PROMO, size=n, p=promo_p)
    has_promo = rng.random(n) < 0.3

    price = np.exp(_CATEGORY_LOG_PRICE[category] + rng.normal(0, 0.6, n)).round(0)
    discount = np.where(has_promo, rng.beta(2, 5, n), rng.beta(1, 12, n)).clip(0, 0.7).round(2)
    n_items = np.minimum(1 + rng.poisson(0.8, n), 8)
    delivery_days = np.clip(np.round(rng.gamma(2.5, 1.6, n)), 1, 21)
    customer_orders = rng.negative_binomial(1, 0.18, n)
    is_app = (rng.random(n) < 0.55).astype(int)
    weekday = rng.integers(0, 7, n)

    logit = (
        -2.3
        + _PAYMENT_EFFECT[payment]
        + _CATEGORY_EFFECT[category]
        + pair_effect[category, payment]
        + city_effect[city]
        + seller_effect[seller]
        + np.where(has_promo, promo_effect[promo], 0.0)
        + 0.35 * np.maximum(np.log(price / 3000), 0)
        + 1.2 * discount
        + 0.6 * (delivery_days > 7)
        - 0.25 * np.log1p(customer_orders)
        + np.minimum(0.15 * (n_items - 1), 0.6)
    )
    proba = _sigmoid(logit)

    orders = pd.DataFrame(
        {
            "payment": np.array(PAYMENTS)[payment],
            "category": np.array(CATEGORIES)[category],
            "city": [f"city_{code:03d}" for code in city],
            "promo_code": [f"promo_{code:03d}" for code in promo],
            "seller_id": [f"s{code:04d}" for code in seller],
            "price": price,
            "discount": discount,
            "n_items": n_items,
            "delivery_days": delivery_days,
            "customer_orders": customer_orders,
            "is_app": is_app,
            "weekday": weekday,
            "month": month,
        }
    )
    orders.loc[~has_promo, "promo_code"] = np.nan
    orders.loc[rng.random(n) < 0.02, "city"] = np.nan
    orders.loc[rng.random(n) < 0.05, "delivery_days"] = np.nan
    orders[TARGET] = (rng.random(n) < proba).astype(int)
    if include_hidden:
        orders["true_proba"] = proba
    return orders


def split_by_month(
    orders: pd.DataFrame, valid_from: int = 16, test_from: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разбиение по времени: обучение — месяцы до valid_from, валидация — до test_from, тест — остальные."""
    month = orders["month"]
    train = orders[month < valid_from]
    valid = orders[(month >= valid_from) & (month < test_from)]
    test = orders[month >= test_from]
    return train, valid, test
