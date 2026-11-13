"""Создаёт «сырую» выгрузку продаж для практики 2.1.4.

Запуск из папки practice:  uv run python make_raw_data.py
Результат:                 data/raw_sales.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sales_data import make_sales  # noqa: E402

OTHER_FORMAT_STORES = ["S07", "S08"]


def make_raw(seed: int = 7) -> pd.DataFrame:
    """Продажи в том виде, в каком их обычно отдаёт учётная система."""
    sales = make_sales()

    # В журнале продаж нет строк о том, что товар не продавался
    raw = sales[sales["units"].isna() | (sales["units"] > 0)].copy()

    # Два магазина выгружают даты в другом формате
    iso = raw["date"].dt.strftime("%Y-%m-%d")
    dotted = raw["date"].dt.strftime("%d.%m.%Y")
    raw["date"] = iso.where(~raw["store_id"].isin(OTHER_FORMAT_STORES), dotted)

    # Часть строк выгружена дважды, порядок строк случайный
    duplicates = raw.sample(frac=0.02, random_state=seed)
    return pd.concat([raw, duplicates]).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    out = Path("data")
    out.mkdir(exist_ok=True)
    raw = make_raw()
    raw.to_csv(out / "raw_sales.csv", index=False)
    print(f"{out / 'raw_sales.csv'}: {len(raw):,} строк")
