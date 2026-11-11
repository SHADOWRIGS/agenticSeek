"""Синтетические данные о мошеннических операциях по банковским картам для уроков темы 3.2.

Использование:  from fraud_data import make_transactions, make_features
"""

import numpy as np
import pandas as pd

TARGET = "is_fraud"
CATEGORIES = ["grocery", "restaurants", "transport", "electronics", "travel", "online_services", "cash"]

_CATEGORY_SHARE = [0.34, 0.16, 0.14, 0.08, 0.05, 0.17, 0.06]
_CATEGORY_EFFECT = np.array([-0.7, -0.3, -0.5, 1.0, 0.5, 0.6, 0.9])
# Ночью операций мало, днём и вечером — много
_HOUR_WEIGHT = np.array([2, 1, 1, 1, 1, 2, 4, 7, 9, 10, 11, 12, 13, 12, 11, 11, 12, 13, 14, 13, 11, 8, 5, 3])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def make_transactions(n: int = 60_000, seed: int = 42, include_hidden: bool = False) -> pd.DataFrame:
    """Операции по картам за 180 дней: одна строка на операцию, отсортированы по времени.

    Мошеннических операций около 2 %. Вероятность мошенничества зависит от признаков операции,
    растёт со временем и зависит от скрытой уязвимости клиента: у одних клиентов мошенничество
    встречается намного чаще, чем у других с такими же операциями.
    При include_hidden=True в таблицу добавляются скрытые столбцы client_risk и true_proba.
    """
    rng = np.random.default_rng(seed)
    n_clients = max(n // 12, 1)

    # У 3 % клиентов данные карты утекли: мошенники совершают по ней серию операций
    client_compromised = rng.random(n_clients) < 0.03
    client_risk = 0.4 * rng.normal(0, 1, n_clients) + 3.0 * client_compromised
    client_account_age = rng.integers(10, 3000, n_clients)
    client_avg_amount = np.round(np.exp(rng.normal(7.3, 0.5, n_clients)), -1)

    client = rng.integers(0, n_clients, n)
    day = rng.integers(0, 180, n)
    hour = rng.choice(24, size=n, p=_HOUR_WEIGHT / _HOUR_WEIGHT.sum())
    second = rng.integers(0, 3600, n)
    timestamp = pd.Timestamp("2025-01-01") + pd.to_timedelta(day * 86400 + hour * 3600 + second, unit="s")

    amount_ratio = np.exp(rng.normal(0, 0.9, n))
    amount = np.maximum(np.round(client_avg_amount[client] * amount_ratio), 10)
    category = rng.choice(len(CATEGORIES), size=n, p=_CATEGORY_SHARE)
    is_online = rng.random(n) < np.where(np.isin(category, [3, 4, 5]), 0.7, 0.1)
    is_foreign = rng.random(n) < 0.04
    new_device = rng.random(n) < 0.08
    tx_last_24h = rng.poisson(1.2, n)

    night = hour < 6
    log_ratio = np.log(amount_ratio)
    account_age = client_account_age[client]
    logit = (
        -7.5
        + 0.3 * log_ratio
        + 1.5 * np.maximum(log_ratio, 0)  # крупные для клиента суммы подозрительнее
        + 2.5 * (amount_ratio < 0.15)  # «проверочные» списания на маленькую сумму
        + _CATEGORY_EFFECT[category]
        + 0.6 * night
        + 0.4 * is_online
        + 1.6 * (night & is_online)
        + 2.0 * is_foreign
        + 1.4 * new_device
        + 1.8 * (new_device & is_online)
        + 0.6 * np.minimum(tx_last_24h, 6)
        + 1.0 * (account_age < 90)
        - 0.2 * np.log(account_age / 365)
        + 0.6 * day / 180
        + client_risk[client]
    )
    proba = _sigmoid(logit)
    is_fraud = rng.random(n) < proba

    transactions = pd.DataFrame(
        {
            "client_id": client + 1,
            "timestamp": timestamp,
            "amount": amount,
            "client_avg_amount": client_avg_amount[client],
            "account_age_days": client_account_age[client],
            "merchant_category": np.array(CATEGORIES)[category],
            "is_online": is_online,
            "is_foreign": is_foreign,
            "new_device": new_device,
            "tx_last_24h": tx_last_24h,
            TARGET: is_fraud.astype(int),
        }
    )
    if include_hidden:
        transactions["client_risk"] = client_risk[client]
        transactions["true_proba"] = proba
    transactions = transactions.sort_values("timestamp", kind="stable").reset_index(drop=True)
    transactions.insert(0, "transaction_id", np.arange(1, n + 1))
    return transactions


def make_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Числовые признаки операции для моделей. Индекс — как у transactions.

    Сумма операции входит в виде логарифма и в виде логарифма отношения к обычной сумме клиента;
    время — признаком «ночь» (с 0 до 6 часов); категория продавца — индикаторами, базовая категория grocery.
    """
    features = pd.DataFrame(index=transactions.index)
    features["log_amount"] = np.log(transactions["amount"])
    features["log_amount_ratio"] = np.log(transactions["amount"] / transactions["client_avg_amount"])
    features["log_account_age"] = np.log(transactions["account_age_days"])
    features["tx_last_24h"] = transactions["tx_last_24h"]
    features["is_night"] = (transactions["timestamp"].dt.hour < 6).astype(int)
    for column in ["is_online", "is_foreign", "new_device"]:
        features[column] = transactions[column].astype(int)
    for name in CATEGORIES[1:]:
        features[f"cat_{name}"] = (transactions["merchant_category"] == name).astype(int)
    return features
