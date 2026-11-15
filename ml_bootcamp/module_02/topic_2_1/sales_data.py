"""Синтетические данные о продажах для уроков темы 2.1.

Использование в ноутбуке:  from sales_data import make_sales, make_orders
Сохранить в файлы:         uv run python sales_data.py --out data/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STORES = pd.DataFrame(
    {
        "store_id": ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08"],
        "city": ["Москва", "Москва", "Москва", "Казань", "Казань", "Новосибирск", "Новосибирск", "Владивосток"],
        "timezone": [
            "Europe/Moscow",
            "Europe/Moscow",
            "Europe/Moscow",
            "Europe/Moscow",
            "Europe/Moscow",
            "Asia/Novosibirsk",
            "Asia/Novosibirsk",
            "Asia/Vladivostok",
        ],
        "size": [1.6, 1.2, 0.9, 1.0, 0.7, 1.1, 0.8, 0.6],
    }
)

PRODUCTS = pd.DataFrame(
    {
        "product": [
            "Ноутбук",
            "Смартфон",
            "Планшет",
            "Наушники",
            "Клавиатура",
            "Мышь",
            "Кофеварка",
            "Чайник",
            "Блендер",
            "Рюкзак",
            "Термос",
            "Зонт",
        ],
        "category": [
            "Электроника",
            "Электроника",
            "Электроника",
            "Аксессуары",
            "Аксессуары",
            "Аксессуары",
            "Для дома",
            "Для дома",
            "Для дома",
            "Товары в дорогу",
            "Товары в дорогу",
            "Товары в дорогу",
        ],
        "price": [74990, 49990, 32990, 5990, 3490, 1490, 12990, 2990, 4990, 3990, 1990, 1290],
        "base_demand": [1.5, 3.0, 1.2, 6.0, 4.0, 7.0, 1.0, 3.5, 1.8, 2.5, 3.0, 2.0],
    }
)


def make_sales(start: str = "2023-01-01", end: str = "2024-12-31", seed: int = 42) -> pd.DataFrame:
    """Дневные продажи: одна строка на (дата, магазин, товар)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    grid = (
        pd.MultiIndex.from_product(
            [dates, STORES["store_id"], PRODUCTS["product"]], names=["date", "store_id", "product"]
        )
        .to_frame(index=False)
        .merge(STORES[["store_id", "city", "size"]], on="store_id")
        .merge(PRODUCTS, on="product")
    )

    day_number = (grid["date"] - dates[0]).dt.days
    trend = 1 + 0.25 * day_number / 365
    weekly = np.where(grid["date"].dt.day_of_week >= 5, 1.35, 1.0)
    yearly = 1 + 0.3 * np.cos(2 * np.pi * (grid["date"].dt.day_of_year - 355) / 365)
    grid["promo"] = rng.random(len(grid)) < 0.08
    promo_effect = np.where(grid["promo"], 1.8, 1.0)

    expected = grid["base_demand"] * grid["size"] * trend * weekly * yearly * promo_effect
    grid["units"] = rng.poisson(expected)
    grid["price"] = np.where(grid["promo"], (grid["price"] * 0.85).round(-1), grid["price"]).astype(int)
    grid["revenue"] = grid["units"] * grid["price"]

    # Немного «грязи», как в настоящих выгрузках: пропуски в количестве проданного
    missing = rng.random(len(grid)) < 0.005
    grid["units"] = grid["units"].astype("float64").mask(missing)
    grid["revenue"] = grid["revenue"].astype("float64").mask(missing)

    columns = ["date", "store_id", "city", "category", "product", "price", "promo", "units", "revenue"]
    return grid[columns].sort_values(["date", "store_id", "product"], ignore_index=True)


def make_orders(start: str = "2024-01-01", end: str = "2024-12-31", seed: int = 42) -> pd.DataFrame:
    """Отдельные заказы с точным временем в UTC — для урока о датах и часовых поясах."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    frames = []
    for store in STORES.itertuples():
        weekend_effect = np.where(days.day_of_week >= 5, 1.35, 1.0)
        orders_per_day = rng.poisson(40 * store.size * weekend_effect)
        local_days = np.repeat(days, orders_per_day)
        # Магазины работают с 9 до 21 по местному времени, пик продаж — около 17 часов
        seconds = (9 + 12 * rng.beta(5, 3, size=len(local_days))) * 3600
        local_time = local_days + pd.to_timedelta(seconds.astype(int), unit="s")
        product_index = rng.choice(
            len(PRODUCTS), size=len(local_days), p=PRODUCTS["base_demand"] / PRODUCTS["base_demand"].sum()
        )
        frames.append(
            pd.DataFrame(
                {
                    "created_at": local_time.tz_localize(store.timezone).tz_convert("UTC"),
                    "store_id": store.store_id,
                    "product": PRODUCTS["product"].to_numpy()[product_index],
                    "amount": PRODUCTS["price"].to_numpy()[product_index],
                }
            )
        )
    orders = pd.concat(frames).sort_values("created_at", ignore_index=True)
    orders.insert(0, "order_id", np.arange(1, len(orders) + 1))
    return orders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сохранить учебные данные в CSV")
    parser.add_argument("--out", type=Path, default=Path("data"), help="папка для файлов")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sales, orders = make_sales(), make_orders()
    sales.to_csv(args.out / "sales.csv", index=False)
    orders.to_csv(args.out / "orders.csv", index=False)
    STORES.drop(columns="size").to_csv(args.out / "stores.csv", index=False)
    print(f"sales: {len(sales):,} строк, orders: {len(orders):,} строк, папка: {args.out}")
