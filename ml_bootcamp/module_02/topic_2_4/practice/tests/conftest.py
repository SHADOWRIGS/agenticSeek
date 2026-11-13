import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard_data import load_sales  # noqa: E402


@pytest.fixture(scope="session")
def sales() -> pd.DataFrame:
    return load_sales()


@pytest.fixture(scope="session")
def january(sales) -> pd.DataFrame:
    """Январь 2024 года в Москве: небольшая выборка для сверки с прямым расчётом."""
    mask = (sales["date"] >= "2024-01-01") & (sales["date"] <= "2024-01-31") & (sales["city"] == "Москва")
    return sales[mask].reset_index(drop=True)


@pytest.fixture(scope="session")
def empty(sales) -> pd.DataFrame:
    return sales.iloc[0:0]
