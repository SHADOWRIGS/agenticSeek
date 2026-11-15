"""Генератор синтетических «сырых» данных для практики 1.2.4.

Создаёт в выходной папке справочник products.csv и несколько файлов events_NNN.csv
с событиями интернет-магазина. В данные намеренно добавлена «грязь»: битые строки,
пропуски, разный регистр и пробелы, дубликаты (в том числе между файлами).

Запуск:
    uv run python generate_data.py --rows 50000 --files 8 --out data/raw
"""

import argparse
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

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
PRODUCT_COLUMNS = ["product_id", "category", "list_price_usd"]

CATEGORIES = ["books", "electronics", "toys", "clothing", "sports", "home", "beauty", "grocery"]
COUNTRIES = ["US", "DE", "FR", "GB", "JP", "BR", "IN", "CA", "ES", "IT"]
EVENT_TYPES = ["view", "cart", "purchase"]
EVENT_WEIGHTS = [0.60, 0.25, 0.15]
# Сколько единиц валюты дают за один доллар: нужно только для правдоподобных цен.
UNITS_PER_USD = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 150.0}
CURRENCY_WEIGHTS = [0.50, 0.25, 0.15, 0.10]

START_TIME = datetime(2025, 1, 1)
PERIOD_DAYS = 30
N_PRODUCTS = 500

INVALID_SHARE = 0.04
DUPLICATE_SHARE = 0.03
RESERVOIR_SIZE = 2000


@dataclass(frozen=True, slots=True)
class DataSummary:
    """Что именно сгенерировано: по этим числам тесты сверяют статистику пайплайна."""

    files: int
    rows_total: int
    rows_invalid: int
    rows_duplicate: int

    @property
    def rows_clean(self) -> int:
        return self.rows_total - self.rows_invalid - self.rows_duplicate


def write_products(path: Path, rng: random.Random) -> list[float]:
    """Пишет справочник товаров и возвращает список цен (индекс = product_id - 1)."""
    prices = [round(rng.uniform(2, 300), 2) for _ in range(N_PRODUCTS)]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(PRODUCT_COLUMNS) + "\n")
        for product_id, price in enumerate(prices, start=1):
            f.write(f"{product_id},{rng.choice(CATEGORIES)},{price:.2f}\n")
    return prices


def make_event(event_id: int, n_users: int, prices: list[float], rng: random.Random) -> list[str]:
    """Корректное событие в виде списка строковых полей."""
    product_id = rng.randrange(1, N_PRODUCTS + 1)
    currency = rng.choices(list(UNITS_PER_USD), weights=CURRENCY_WEIGHTS)[0]
    price = prices[product_id - 1] * UNITS_PER_USD[currency] * rng.uniform(0.9, 1.1)
    event_time = START_TIME + timedelta(seconds=rng.randrange(PERIOD_DAYS * 24 * 3600))
    return [
        str(event_id),
        str(rng.randrange(1, n_users + 1)),
        event_time.strftime("%Y-%m-%d %H:%M:%S"),
        rng.choices(EVENT_TYPES, weights=EVENT_WEIGHTS)[0],
        str(product_id),
        f"{price:.2f}",
        currency,
        str(rng.randrange(1, 6)),
        rng.choice(COUNTRIES),
    ]


def add_noise(fields: list[str], rng: random.Random) -> list[str]:
    """«Грязь», которую пайплайн обязан исправить: строка остаётся корректной."""
    if rng.random() < 0.03:
        event_type = fields[3]
        fields[3] = rng.choice([f" {event_type} ", event_type.upper(), event_type.capitalize()])
    if rng.random() < 0.03:
        country = fields[8]
        fields[8] = rng.choice([country.lower(), f" {country} ", f"{country.lower()} "])
    elif rng.random() < 0.02:
        fields[8] = ""
    if rng.random() < 0.02:
        fields[6] = fields[6].lower()
    if rng.random() < 0.03:
        fields[5] = ""
    return fields


def break_row(fields: list[str], rng: random.Random) -> list[str]:
    """Дефект, из-за которого строку нужно отбросить."""
    match rng.randrange(10):
        case 0:
            return fields[: rng.randrange(3, 8)]  # строка оборвана
        case 1:
            return [*fields, "extra"]  # лишнее поле
        case 2:
            fields[0] = f"ID-{fields[0]}"
        case 3:
            fields[1] = ""
        case 4:
            fields[2] = rng.choice(["not_a_date", "2025-02-30 12:00:00", ""])
        case 5:
            fields[3] = "refund"
        case 6:
            fields[4] = str(N_PRODUCTS + rng.randrange(1, 100))  # товара нет в справочнике
        case 7:
            fields[5] = rng.choice(["abc", "-5.00", "0"])
        case 8:
            fields[6] = "XXX"
        case _:
            fields[7] = rng.choice(["0", "-1", "two", ""])
    return fields


def generate(out_dir: Path, rows: int = 50_000, files: int = 8, seed: int = 42) -> DataSummary:
    """Генерирует данные и возвращает сводку. Результат полностью определяется аргументами."""
    if rows < files or files < 1:
        raise ValueError("нужно files >= 1 и rows >= files")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("events_*.csv"):
        old_file.unlink()

    rng = random.Random(seed)
    prices = write_products(out_dir / "products.csv", rng)
    n_users = max(100, rows // 20)

    next_event_id = 1
    rows_invalid = rows_duplicate = 0
    reservoir: list[str] = []  # корректные строки, из которых берутся дубликаты

    for file_index in range(files):
        rows_in_file = rows // files + (1 if file_index < rows % files else 0)
        with (out_dir / f"events_{file_index:03d}.csv").open("w", encoding="utf-8", newline="\n") as f:
            f.write(",".join(RAW_COLUMNS) + "\n")
            for _ in range(rows_in_file):
                if reservoir and rng.random() < DUPLICATE_SHARE:
                    f.write(rng.choice(reservoir) + "\n")
                    rows_duplicate += 1
                    continue

                fields = make_event(next_event_id, n_users, prices, rng)
                next_event_id += 1
                if rng.random() < INVALID_SHARE:
                    f.write(",".join(break_row(fields, rng)) + "\n")
                    rows_invalid += 1
                    continue

                line = ",".join(add_noise(fields, rng))
                f.write(line + "\n")
                if len(reservoir) < RESERVOIR_SIZE:
                    reservoir.append(line)
                else:
                    reservoir[rng.randrange(RESERVOIR_SIZE)] = line

    return DataSummary(files=files, rows_total=rows, rows_invalid=rows_invalid, rows_duplicate=rows_duplicate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Генератор сырых данных для практики 1.2.4")
    parser.add_argument("--rows", type=int, default=50_000, help="сколько строк с данными создать во всех файлах")
    parser.add_argument("--files", type=int, default=8, help="на сколько файлов events_NNN.csv их разложить")
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="папка для результата")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = generate(args.out, rows=args.rows, files=args.files, seed=args.seed)
    size_mb = sum(p.stat().st_size for p in args.out.glob("*.csv")) / 1e6
    print(f"Папка: {args.out} ({size_mb:.1f} МБ, файлов с событиями: {summary.files})")
    print(f"Строк всего: {summary.rows_total}")
    print(f"  из них битых: {summary.rows_invalid}")
    print(f"  дубликатов:   {summary.rows_duplicate}")
    print(f"  должно остаться после очистки: {summary.rows_clean}")


if __name__ == "__main__":
    main()
