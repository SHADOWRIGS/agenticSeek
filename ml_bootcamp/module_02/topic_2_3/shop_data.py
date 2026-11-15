"""Синтетические данные интернет-магазина для урока 2.3.3: клиенты, заказы, позиции заказов.

Использование в ноутбуке:  from shop_data import make_shop
                           customers, orders, items = make_shop()

Работает и на pandas 2.x, и на pandas 3.x.
"""

import numpy as np
import pandas as pd

CATALOG = pd.DataFrame(
    {
        "product": ["Ноутбук", "Смартфон", "Наушники", "Клавиатура", "Мышь", "Чайник", "Блендер", "Рюкзак", "Термос"],
        "category": [
            "Электроника",
            "Электроника",
            "Аксессуары",
            "Аксессуары",
            "Аксессуары",
            "Для дома",
            "Для дома",
            "Товары в дорогу",
            "Товары в дорогу",
        ],
        "price": [74990, 49990, 5990, 3490, 1490, 2990, 4990, 3990, 1990],
        "weight": [1, 2, 6, 4, 7, 4, 2, 3, 3],
    }
)

FIRST_DAY = pd.Timestamp("2023-01-01")
LAST_DAY = pd.Timestamp("2024-12-31")


def make_shop(n_customers: int = 1500, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Возвращает таблицы customers, orders, items.

    Модель поведения: у каждого клиента своя частота заказов и свой срок «жизни». После его окончания
    клиент перестаёт покупать, и в данных это никак не отмечено — как и в настоящих магазинах.
    """
    rng = np.random.default_rng(seed)

    signup = FIRST_DAY + pd.to_timedelta(rng.integers(0, 547, size=n_customers), unit="D")
    customers = pd.DataFrame(
        {
            "customer_id": np.arange(1, n_customers + 1),
            "signup_date": signup,
            "city": rng.choice(
                ["Москва", "Казань", "Новосибирск", "Владивосток"], size=n_customers, p=[0.5, 0.2, 0.2, 0.1]
            ),
            "channel": rng.choice(["реклама", "поиск", "рекомендация"], size=n_customers, p=[0.5, 0.3, 0.2]),
        }
    )

    orders_per_day = rng.gamma(shape=2.0, scale=0.02, size=n_customers)
    lifetime_days = rng.exponential(scale=300, size=n_customers)
    premium_share = rng.beta(2, 5, size=n_customers)

    order_frames = []
    for row, rate, lifetime in zip(customers.itertuples(), orders_per_day, lifetime_days, strict=True):
        active_days = min(lifetime, (LAST_DAY - row.signup_date).days)
        n_orders = rng.poisson(rate * active_days)
        if n_orders == 0:
            continue
        moments = np.sort(rng.uniform(0, active_days, size=n_orders))
        order_frames.append(
            pd.DataFrame(
                {
                    "customer_id": row.customer_id,
                    "created_at": row.signup_date + pd.to_timedelta((moments * 86400).astype(int), unit="s"),
                }
            )
        )
    orders = pd.concat(order_frames).sort_values("created_at", ignore_index=True)
    orders.insert(0, "order_id", np.arange(1, len(orders) + 1))
    orders["device"] = rng.choice(["телефон", "компьютер"], size=len(orders), p=[0.65, 0.35])

    n_items = rng.integers(1, 5, size=len(orders))
    order_index = np.repeat(np.arange(len(orders)), n_items)
    premium = rng.random(len(order_index)) < premium_share[orders["customer_id"].to_numpy()[order_index] - 1]
    cheap = CATALOG["price"] < 10_000
    weights = np.where(
        premium[:, None], (CATALOG["weight"] * ~cheap).to_numpy(), (CATALOG["weight"] * cheap).to_numpy()
    )
    cumulative = (weights / weights.sum(axis=1, keepdims=True)).cumsum(axis=1)
    product_index = np.minimum((rng.random((len(order_index), 1)) > cumulative).sum(axis=1), len(CATALOG) - 1)
    items = pd.DataFrame(
        {
            "item_id": np.arange(1, len(order_index) + 1),
            "order_id": orders["order_id"].to_numpy()[order_index],
            "created_at": orders["created_at"].to_numpy()[order_index],
            "product": CATALOG["product"].to_numpy()[product_index],
            "category": CATALOG["category"].to_numpy()[product_index],
            "price": CATALOG["price"].to_numpy()[product_index],
            "quantity": rng.choice([1, 2, 3], size=len(order_index), p=[0.8, 0.15, 0.05]),
        }
    )
    return customers, orders, items


def make_labels(customers: pd.DataFrame, orders: pd.DataFrame, cutoff: str, horizon_days: int = 30) -> pd.DataFrame:
    """Метки для задачи «сделает ли клиент заказ в ближайшие horizon_days дней после cutoff».

    В выборку входят клиенты, зарегистрированные до cutoff. Столбцы: customer_id, time, label.
    """
    moment = pd.Timestamp(cutoff)
    known = customers.loc[customers["signup_date"] < moment, "customer_id"]
    window = orders[
        (orders["created_at"] >= moment) & (orders["created_at"] < moment + pd.Timedelta(days=horizon_days))
    ]
    return pd.DataFrame(
        {"customer_id": known.to_numpy(), "time": moment, "label": known.isin(window["customer_id"]).to_numpy()}
    )
