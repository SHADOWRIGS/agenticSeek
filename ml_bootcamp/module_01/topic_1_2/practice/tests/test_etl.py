"""Тесты корректности ETL-пайплайна.

По умолчанию проверяется etl_baseline. Чтобы проверить свою версию, укажите имя модуля
в переменной окружения:

    ETL_MODULE=etl_fast uv run pytest

Модуль должен содержать функцию run_pipeline(raw_dir: Path, out_dir: Path) -> dict[str, int].
"""

import importlib
import os
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import etl_baseline
import generate_data
import rates_service

etl = importlib.import_module(os.environ.get("ETL_MODULE", "etl_baseline"))

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
STAT_KEYS = {"rows_read", "rows_invalid", "rows_duplicate", "rows_clean"}

HEADER = "event_id,user_id,event_time,event_type,product_id,price,currency,quantity,country"
GOLDEN_PRODUCTS = """\
product_id,category,list_price_usd
1,books,12.50
2,electronics,199.99
3,toys,8.00
"""
GOLDEN_EVENTS_0 = f"""\
{HEADER}
1,10,2025-01-01 10:00:00,purchase,1,12.50,USD,2,US
2,11,2025-01-01 11:30:00, Purchase ,2,180.00,eur,1, de
3,12,2025-01-02 09:15:00,view,3,,EUR,1,
4,13,2025-01-02 09:20:00,cart,1,11.00,GBP,3,GB
1,10,2025-01-01 10:00:00,purchase,1,12.50,USD,2,US
5,14,2025-01-02 10:00:00,purchase,2
6,,2025-01-02 10:05:00,purchase,2,199.99,USD,1,US
7,15,2025-02-30 10:00:00,purchase,2,199.99,USD,1,US
8,16,2025-01-03 10:00:00,refund,2,199.99,USD,1,US
"""
GOLDEN_EVENTS_1 = f"""\
{HEADER}
4,13,2025-01-02 09:20:00,cart,1,11.00,GBP,3,GB
9,17,2025-01-03 12:00:00,purchase,99,5.00,USD,1,US
10,18,2025-01-03 12:05:00,purchase,3,abc,USD,1,US
11,19,2025-01-03 12:10:00,purchase,3,-8.00,USD,1,US
12,20,2025-01-03 12:15:00,purchase,3,8.00,XXX,1,US
13,21,2025-01-03 12:20:00,purchase,3,8.00,USD,0,US
14,22,2025-01-03 12:25:00,PURCHASE,3,1200.00,jpy,2,jp
15,23,2025-01-03 12:30:00,purchase,1,12.50,USD,1,US,extra
ID-16,24,2025-01-03 12:35:00,purchase,1,12.50,USD,1,US
17,25,2025-01-01 23:59:59,purchase,1,12.00,USD,1,US
"""


@pytest.fixture(autouse=True)
def fast_rates_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Отключает искусственную задержку сервиса курсов, чтобы тесты шли быстро."""
    monkeypatch.setenv("RATES_LATENCY_MS", "0")


def rate(currency: str, day: date) -> float:
    for _ in range(3):
        try:
            return rates_service.get_rate(currency, day)
        except ConnectionError:
            continue
    raise AssertionError("сервис курсов не ответил за три попытки")


def read_events(out_dir: Path) -> pd.DataFrame:
    """Читает events_clean.parquet и приводит его к каноническому виду: порядок строк и точные типы не важны."""
    df = pd.read_parquet(out_dir / "events_clean.parquet")
    assert list(df.columns) == CLEAN_COLUMNS
    df = df.sort_values("event_id").reset_index(drop=True)
    for column in ["event_id", "user_id", "product_id", "quantity"]:
        df[column] = df[column].astype("int64")
    for column in ["price", "amount_usd"]:
        df[column] = df[column].astype("float64")
    for column in ["event_type", "category", "currency", "country"]:
        df[column] = df[column].astype(str)
    df["event_time"] = pd.to_datetime(df["event_time"]).astype("datetime64[ns]")
    return df


def read_daily(out_dir: Path) -> pd.DataFrame:
    """Читает daily_revenue.parquet и приводит его к каноническому виду."""
    df = pd.read_parquet(out_dir / "daily_revenue.parquet")
    assert list(df.columns) == DAILY_COLUMNS
    df["day"] = pd.to_datetime(df["day"].astype(str)).astype("datetime64[ns]")
    for column in ["country", "category"]:
        df[column] = df[column].astype(str)
    for column in ["orders", "units"]:
        df[column] = df[column].astype("int64")
    df["revenue_usd"] = df["revenue_usd"].astype("float64")
    return df.sort_values(["day", "country", "category"]).reset_index(drop=True)


@pytest.fixture
def golden_raw(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "products.csv").write_text(GOLDEN_PRODUCTS, encoding="utf-8")
    (raw_dir / "events_000.csv").write_text(GOLDEN_EVENTS_0, encoding="utf-8")
    (raw_dir / "events_001.csv").write_text(GOLDEN_EVENTS_1, encoding="utf-8")
    return raw_dir


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, generate_data.DataSummary]:
    raw_dir = tmp_path_factory.mktemp("generated_raw")
    summary = generate_data.generate(raw_dir, rows=6000, files=4, seed=7)
    return raw_dir, summary


def test_golden_events(golden_raw: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    stats = etl.run_pipeline(golden_raw, out_dir)

    assert {key: stats[key] for key in STAT_KEYS} == {
        "rows_read": 19,
        "rows_invalid": 11,
        "rows_duplicate": 2,
        "rows_clean": 6,
    }

    eur = rate("EUR", date(2025, 1, 1))
    gbp = rate("GBP", date(2025, 1, 2))
    jpy = rate("JPY", date(2025, 1, 3))
    expected = pd.DataFrame(
        [
            (1, 10, "2025-01-01 10:00:00", "purchase", 1, "books", 12.50, "USD", 2, "US", 25.0),
            (2, 11, "2025-01-01 11:30:00", "purchase", 2, "electronics", 180.0, "EUR", 1, "DE", 180.0 * eur),
            (3, 12, "2025-01-02 09:15:00", "view", 3, "toys", 8.0, "USD", 1, "UNKNOWN", 8.0),
            (4, 13, "2025-01-02 09:20:00", "cart", 1, "books", 11.0, "GBP", 3, "GB", 33.0 * gbp),
            (14, 22, "2025-01-03 12:25:00", "purchase", 3, "toys", 1200.0, "JPY", 2, "JP", 2400.0 * jpy),
            (17, 25, "2025-01-01 23:59:59", "purchase", 1, "books", 12.0, "USD", 1, "US", 12.0),
        ],
        columns=CLEAN_COLUMNS,
    )
    expected["event_time"] = pd.to_datetime(expected["event_time"]).astype("datetime64[ns]")
    for column in ["event_type", "category", "currency", "country"]:
        expected[column] = expected[column].astype(str)

    pd.testing.assert_frame_equal(read_events(out_dir), expected, rtol=1e-9)


def test_golden_daily_revenue(golden_raw: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    etl.run_pipeline(golden_raw, out_dir)

    eur = rate("EUR", date(2025, 1, 1))
    jpy = rate("JPY", date(2025, 1, 3))
    expected = pd.DataFrame(
        [
            ("2025-01-01", "DE", "electronics", 1, 1, 180.0 * eur),
            ("2025-01-01", "US", "books", 2, 3, 37.0),
            ("2025-01-03", "JP", "toys", 1, 2, 2400.0 * jpy),
        ],
        columns=DAILY_COLUMNS,
    )
    expected["day"] = pd.to_datetime(expected["day"]).astype("datetime64[ns]")
    for column in ["country", "category"]:
        expected[column] = expected[column].astype(str)

    pd.testing.assert_frame_equal(read_daily(out_dir), expected, rtol=1e-9)


def test_stats_match_generated_data(generated: tuple[Path, generate_data.DataSummary], tmp_path: Path) -> None:
    raw_dir, summary = generated
    stats = etl.run_pipeline(raw_dir, tmp_path / "out")

    assert stats["rows_read"] == summary.rows_total
    assert stats["rows_invalid"] == summary.rows_invalid
    assert stats["rows_duplicate"] == summary.rows_duplicate
    assert stats["rows_clean"] == summary.rows_clean


def test_clean_events_invariants(generated: tuple[Path, generate_data.DataSummary], tmp_path: Path) -> None:
    raw_dir, summary = generated
    out_dir = tmp_path / "out"
    etl.run_pipeline(raw_dir, out_dir)
    events = read_events(out_dir)

    assert len(events) == summary.rows_clean
    assert events["event_id"].is_unique
    assert not events.isna().any().any()
    assert set(events["event_type"]) <= {"view", "cart", "purchase"}
    assert set(events["currency"]) <= {"USD", "EUR", "GBP", "JPY"}
    assert set(events["country"]) <= {*generate_data.COUNTRIES, "UNKNOWN"}
    assert "UNKNOWN" in set(events["country"])
    assert set(events["category"]) <= set(generate_data.CATEGORIES)
    assert (events["price"] > 0).all()
    assert (events["quantity"] >= 1).all()
    assert (events["amount_usd"] > 0).all()
    usd = events[events["currency"] == "USD"]
    assert ((usd["amount_usd"] - usd["price"] * usd["quantity"]).abs() < 1e-6).all()


def test_daily_revenue_consistent_with_events(
    generated: tuple[Path, generate_data.DataSummary], tmp_path: Path
) -> None:
    raw_dir, _ = generated
    out_dir = tmp_path / "out"
    etl.run_pipeline(raw_dir, out_dir)
    events = read_events(out_dir)
    daily = read_daily(out_dir)

    purchases = events[events["event_type"] == "purchase"]
    expected = (
        purchases.assign(day=purchases["event_time"].dt.normalize())
        .groupby(["day", "country", "category"], as_index=False)
        .agg(orders=("event_id", "count"), units=("quantity", "sum"), revenue_usd=("amount_usd", "sum"))
        .sort_values(["day", "country", "category"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(daily, expected, rtol=1e-9)


def test_same_result_as_baseline(generated: tuple[Path, generate_data.DataSummary], tmp_path: Path) -> None:
    """Главная проверка для вашей версии: результат совпадает с результатом etl_baseline."""
    raw_dir, _ = generated
    expected_stats = etl_baseline.run_pipeline(raw_dir, tmp_path / "expected")
    stats = etl.run_pipeline(raw_dir, tmp_path / "actual")

    assert {key: stats[key] for key in STAT_KEYS} == expected_stats
    pd.testing.assert_frame_equal(read_events(tmp_path / "actual"), read_events(tmp_path / "expected"), rtol=1e-9)
    pd.testing.assert_frame_equal(read_daily(tmp_path / "actual"), read_daily(tmp_path / "expected"), rtol=1e-9)


def test_rerun_overwrites_output(golden_raw: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    etl.run_pipeline(golden_raw, out_dir)
    etl.run_pipeline(golden_raw, out_dir)

    assert len(read_events(out_dir)) == 6
    assert len(read_daily(out_dir)) == 3


def test_generator_is_deterministic(tmp_path: Path) -> None:
    first = generate_data.generate(tmp_path / "a", rows=500, files=2, seed=1)
    second = generate_data.generate(tmp_path / "b", rows=500, files=2, seed=1)

    assert first == second
    for name in ["products.csv", "events_000.csv", "events_001.csv"]:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
