"""Демо-данные «в духе конкурса» для практики 2.3.4: уйдёт ли клиент оператора связи.

Создать файлы:          uv run python make_demo_data.py
Оценить свою отправку:  uv run python make_demo_data.py --score submission.csv

Структура повторяет конкурс Kaggle: train.csv с целевой переменной, test.csv без неё, sample_submission.csv
с образцом ответа. Файл solution.csv на Kaggle скрыт от участников; здесь он нужен, чтобы посчитать
«публичный» и «приватный» результат без сети.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGET = "churn"
BRANDS = ["Samsung", "Xiaomi", "Apple", "Huawei", "Realme", "Honor", "Nokia", "Tecno"]
BRAND_EFFECT = [0.0, 0.25, -0.5, 0.1, 0.35, 0.15, 0.6, 0.5]
PLAN_FEE = {"Базовый": 350, "Стандарт": 600, "Плюс": 900, "Безлимит": 1400}
CONTRACT_EFFECT = {"помесячно": 0.8, "год": -0.2, "два года": -0.9}


def make_demo(
    n_train: int = 8000, n_test: int = 4000, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Возвращает (train, test, solution).

    train содержит столбец churn, test — нет. solution: id, churn и usage («public» для 30 % строк test,
    «private» для остальных) — как скрытый файл ответов на Kaggle.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test

    n_cities = 400
    city_effect = rng.normal(0, 0.7, n_cities)
    popularity = rng.pareto(1.1, n_cities) + 0.05
    city = rng.choice(n_cities, size=n, p=popularity / popularity.sum())

    brand = rng.choice(len(BRANDS), size=n, p=[0.24, 0.22, 0.2, 0.1, 0.08, 0.08, 0.04, 0.04])
    model_number = rng.integers(1, 180, size=n)
    plan = rng.choice(list(PLAN_FEE), size=n, p=[0.3, 0.35, 0.25, 0.1])
    contract = rng.choice(list(CONTRACT_EFFECT), size=n, p=[0.55, 0.3, 0.15])

    age = rng.integers(18, 75, size=n)
    tenure = np.minimum(rng.geometric(0.04, size=n), 120)
    list_fee = pd.Series(plan).map(PLAN_FEE).to_numpy()
    fee_ratio = np.exp(rng.normal(0, 0.18, size=n))
    monthly_fee = (list_fee * fee_ratio).round(-1)
    call_rate = rng.gamma(shape=1.2, scale=0.12, size=n)
    support_calls = rng.poisson(call_rate * tenure)
    data_gb = rng.lognormal(2.3, 0.8, size=n).round(1)
    data_missing = rng.random(n) < 0.08
    signup = pd.Timestamp("2025-01-01") - pd.to_timedelta(tenure * 30 + rng.integers(0, 30, size=n), unit="D")

    logit = (
        -1.9
        + city_effect[city]
        + np.array(BRAND_EFFECT)[brand]
        + pd.Series(contract).map(CONTRACT_EFFECT).to_numpy()
        + 3.0 * (fee_ratio - 1)
        + 4.0 * (call_rate - 0.144)
        - 0.012 * tenure
        + 0.9 * data_missing
        + 0.9 * ((contract == "помесячно") & (tenure <= 3))
        + 0.004 * (age - 45)
    )
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    data = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "age": age,
            "city": [f"Город-{code:03d}" for code in city],
            "plan": plan,
            "contract": contract,
            "signup_date": signup.strftime("%Y-%m-%d"),
            "tenure_months": tenure,
            "monthly_fee": monthly_fee.astype(int),
            "support_calls": support_calls,
            "data_gb": np.where(data_missing, np.nan, data_gb),
            "device_model": [f"{BRANDS[b]} {'ASXM'[m % 4]}{m}" for b, m in zip(brand, model_number, strict=True)],
            "referrer_id": [f"R{code:05d}" for code in rng.integers(0, 6000, size=n)],
            TARGET: churn,
        }
    )
    # Пропуски без закономерности: возраст и город иногда не заполнены
    data["age"] = data["age"].astype("float64").mask(rng.random(n) < 0.03)
    data["city"] = data["city"].mask(rng.random(n) < 0.02)

    train = data.iloc[:n_train].reset_index(drop=True)
    test_full = data.iloc[n_train:].reset_index(drop=True)
    usage = np.where(rng.random(n_test) < 0.3, "public", "private")
    solution = pd.DataFrame({"id": test_full["id"], TARGET: test_full[TARGET], "usage": usage})
    return train, test_full.drop(columns=TARGET), solution


def score_submission(submission: pd.DataFrame, solution: pd.DataFrame) -> dict[str, float]:
    """ROC AUC отправки на публичной и приватной частях тестовых данных."""
    if list(submission.columns) != ["id", TARGET]:
        raise ValueError(f"в отправке должны быть столбцы id и {TARGET}")
    merged = solution.merge(submission, on="id", how="left", suffixes=("", "_predicted"), validate="one_to_one")
    if merged[f"{TARGET}_predicted"].isna().any():
        raise ValueError("в отправке нет прогноза для части строк test или есть пропуски")
    return {
        part: float(roc_auc_score(rows[TARGET], rows[f"{TARGET}_predicted"])) for part, rows in merged.groupby("usage")
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Демо-данные для практики 2.3.4")
    parser.add_argument("--out", type=Path, default=Path("data"), help="папка для файлов")
    parser.add_argument("--score", type=Path, help="файл отправки, который нужно оценить")
    args = parser.parse_args()

    if args.score:
        result = score_submission(pd.read_csv(args.score), pd.read_csv(args.out / "solution.csv"))
        print(f"public: {result['public']:.4f}   private: {result['private']:.4f}")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        train, test, solution = make_demo()
        train.to_csv(args.out / "train.csv", index=False)
        test.to_csv(args.out / "test.csv", index=False)
        solution.to_csv(args.out / "solution.csv", index=False)
        test[["id"]].assign(**{TARGET: 0.5}).to_csv(args.out / "sample_submission.csv", index=False)
        print(f"train: {train.shape}, test: {test.shape}, доля ушедших: {train[TARGET].mean():.3f}, папка: {args.out}")
