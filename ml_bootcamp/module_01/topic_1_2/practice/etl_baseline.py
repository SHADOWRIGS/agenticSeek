"""Наивная реализация ETL-пайплайна для практики 1.2.4.

Результат этого кода корректен и служит точкой отсчёта для тестов. Сам код намеренно
написан неэффективно: он медленный и держит в памяти все данные сразу. Задача практики —
написать собственную версию (например, etl_fast.py) с такой же функцией
run_pipeline(raw_dir, out_dir), которая проходит те же тесты, но работает во много раз
быстрее и почти не зависит по памяти от размера входа.

Запуск:
    uv run python etl_baseline.py --raw data/raw --out data/out
"""

import argparse
import resource
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

import rates_service

RAW_COLUMNS = [
    "event_id",
    "user_id",
    "event_time",
    "event_type",
    "product_id",
    "price",
    "currency",
    "quantity",
    "country",
]
CLEAN_COLUMNS = [
    "event_id",
    "user_id",
    "event_time",
    "event_type",
    "product_id",
    "category",
    "price",
    "currency",
    "quantity",
    "country",
    "amount_usd",
]
DAILY_COLUMNS = ["day", "country", "category", "orders", "units", "revenue_usd"]

EVENT_TYPES = ["view", "cart", "purchase"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY"]
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def load_products(raw_dir: Path) -> list[dict]:
    f = open(raw_dir / "products.csv", encoding="utf-8")
    lines = f.read().split("\n")
    f.close()
    products = []
    for line in lines[1:]:
        if line != "":
            product_id, category, list_price_usd = line.split(",")
            products.append(
                {"product_id": int(product_id), "category": category, "list_price_usd": float(list_price_usd)}
            )
    return products


def find_product(products: list[dict], product_id: int) -> dict | None:
    for product in products:
        if product["product_id"] == product_id:
            return product
    return None


def read_events(path: Path) -> list[list[str]]:
    f = open(path, encoding="utf-8")
    text = f.read()
    f.close()
    rows = []
    for line in text.split("\n")[1:]:
        if line != "":
            rows.append(line.split(","))
    return rows


def is_valid(fields: list[str], products: list[dict]) -> bool:
    if len(fields) != len(RAW_COLUMNS):
        return False
    row = dict(zip(RAW_COLUMNS, fields, strict=True))
    try:
        int(row["event_id"])
        int(row["user_id"])
        datetime.strptime(row["event_time"], TIME_FORMAT)
        product_id = int(row["product_id"])
        quantity = int(row["quantity"])
    except ValueError:
        return False
    if quantity < 1:
        return False
    if row["event_type"].strip().lower() not in EVENT_TYPES:
        return False
    if find_product(products, product_id) is None:
        return False
    if row["currency"].strip().upper() not in CURRENCIES:
        return False
    if row["price"] != "":
        try:
            price = float(row["price"])
        except ValueError:
            return False
        if not price > 0:
            return False
    return True


def clean_row(fields: list[str], products: list[dict]) -> dict:
    row = dict(zip(RAW_COLUMNS, fields, strict=True))
    product = find_product(products, int(row["product_id"]))
    if row["price"] == "":
        price = product["list_price_usd"]
        currency = "USD"
    else:
        price = float(row["price"])
        currency = row["currency"].strip().upper()
    country = row["country"].strip().upper()
    return {
        "event_id": int(row["event_id"]),
        "user_id": int(row["user_id"]),
        "event_time": datetime.strptime(row["event_time"], TIME_FORMAT),
        "event_type": row["event_type"].strip().lower(),
        "product_id": int(row["product_id"]),
        "category": product["category"],
        "price": price,
        "currency": currency,
        "quantity": int(row["quantity"]),
        "country": country if country != "" else "UNKNOWN",
    }


def fetch_rate(currency: str, day: date) -> float:
    while True:
        try:
            return rates_service.get_rate(currency, day)
        except ConnectionError:
            time.sleep(0.01)


def process_file(path: Path, products: list[dict]) -> tuple[list[dict], int]:
    raw_rows = read_events(path)
    valid_rows = [fields for fields in raw_rows if is_valid(fields, products)]
    clean_rows = [clean_row(fields, products) for fields in valid_rows]

    rates = {}
    for row in clean_rows:
        key = (row["currency"], row["event_time"].date())
        if key not in rates:
            rates[key] = fetch_rate(row["currency"], row["event_time"].date())

    enriched_rows = []
    for row in clean_rows:
        rate = rates[(row["currency"], row["event_time"].date())]
        enriched_rows.append({**row, "amount_usd": row["price"] * row["quantity"] * rate})
    return enriched_rows, len(raw_rows)


def drop_duplicates(rows: list[dict]) -> list[dict]:
    seen_ids = []
    unique_rows = []
    for row in rows:
        if row["event_id"] not in seen_ids:
            seen_ids.append(row["event_id"])
            unique_rows.append(row)
    return unique_rows


def aggregate_daily(rows: list[dict]) -> list[dict]:
    totals = {}
    for row in rows:
        if row["event_type"] != "purchase":
            continue
        key = (row["event_time"].date(), row["country"], row["category"])
        if key not in totals:
            totals[key] = {"orders": 0, "units": 0, "revenue_usd": 0.0}
        totals[key]["orders"] += 1
        totals[key]["units"] += row["quantity"]
        totals[key]["revenue_usd"] += row["amount_usd"]
    return [
        {"day": day, "country": country, "category": category, **totals[(day, country, category)]}
        for day, country, category in sorted(totals)
    ]


def run_pipeline(raw_dir: Path, out_dir: Path) -> dict[str, int]:
    """Читает raw_dir, пишет в out_dir файлы events_clean.parquet и daily_revenue.parquet, возвращает статистику."""
    products = load_products(raw_dir)

    all_rows = []
    rows_read = 0
    for path in sorted(raw_dir.glob("events_*.csv")):
        file_rows, n_read = process_file(path, products)
        all_rows = all_rows + file_rows
        rows_read += n_read

    unique_rows = drop_duplicates(all_rows)
    daily_rows = aggregate_daily(unique_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(unique_rows, columns=CLEAN_COLUMNS).to_parquet(out_dir / "events_clean.parquet", index=False)
    pd.DataFrame(daily_rows, columns=DAILY_COLUMNS).to_parquet(out_dir / "daily_revenue.parquet", index=False)

    return {
        "rows_read": rows_read,
        "rows_invalid": rows_read - len(all_rows),
        "rows_duplicate": len(all_rows) - len(unique_rows),
        "rows_clean": len(unique_rows),
    }


def peak_memory_mb() -> float:
    """Пиковое потребление памяти текущим процессом (без дочерних процессов)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux возвращает килобайты, macOS — байты.
    return peak / 1024**2 if sys.platform == "darwin" else peak / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description="Наивный ETL-пайплайн (точка отсчёта)")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"), help="папка с сырыми данными")
    parser.add_argument("--out", type=Path, default=Path("data/out"), help="папка для результата")
    args = parser.parse_args()

    start = time.perf_counter()
    stats = run_pipeline(args.raw, args.out)
    elapsed = time.perf_counter() - start

    for name, value in stats.items():
        print(f"{name}: {value}")
    print(f"время: {elapsed:.2f} с")
    print(f"пик памяти: {peak_memory_mb():.0f} МБ")


if __name__ == "__main__":
    main()
