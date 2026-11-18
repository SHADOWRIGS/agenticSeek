"""Обучает модель прогноза продаж и сохраняет каталог модели: model.joblib и metadata.json.

Готовый скрипт практики 7.4.4, менять не нужно. Файл sales_data.py из module_02/topic_2_1 должен лежать рядом.

    uv run python train_model.py --out model --version 3
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["store_id", "product", "price", "promo", "dow", "month"]
CATEGORICAL = ["store_id", "product"]


def load_sales() -> pd.DataFrame:
    """Данные темы: генератор урока 2.1.1."""
    from sales_data import make_sales

    return make_sales()


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["dow"] = frame["date"].dt.day_of_week
    frame["month"] = frame["date"].dt.month
    frame["promo"] = frame["promo"].astype(int)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучить модель и сохранить каталог модели")
    parser.add_argument("--out", type=Path, default=Path("model"), help="каталог модели")
    parser.add_argument("--version", default="3", help="версия модели для metadata.json")
    parser.add_argument("--train-end", default="2024-06-30", help="последний день обучающей выборки")
    args = parser.parse_args()

    sales = add_date_features(load_sales().dropna(subset=["units"]))
    train = sales[sales["date"] <= args.train_end]
    valid = sales[sales["date"] > args.train_end]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    pipeline = Pipeline(
        [
            ("prepare", ColumnTransformer([("category", encoder, CATEGORICAL)], remainder="passthrough")),
            ("model", HistGradientBoostingRegressor(loss="poisson", random_state=0)),
        ]
    )
    pipeline.fit(train[FEATURES], train["units"])
    valid_mae = float(mean_absolute_error(valid["units"], pipeline.predict(valid[FEATURES])))

    metadata = {
        "name": "sales-forecast",
        "version": args.version,
        "trained_on": [str(train["date"].min().date()), str(train["date"].max().date())],
        "valid_mae": round(valid_mae, 4),
        "features": FEATURES,
        "stores": sorted(train["store_id"].unique()),
        "products": sorted(train["product"].unique()),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.out / "model.joblib")
    (args.out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
