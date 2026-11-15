"""Синтетические данные об оттоке клиентов интернет-провайдера для уроков 3.1.3 и 3.1.4.

Использование:  from churn_data import make_clients
"""

import numpy as np
import pandas as pd

# Истинные коэффициенты модели оттока в логарифмах шансов. Нужны только для самопроверки:
# в настоящей задаче они неизвестны, а признака loyalty в данных нет.
TRUE_CHURN_EFFECTS = {
    "tenure_months": -0.035,
    "monthly_fee_thousands": 0.5,
    "annual_contract": -1.2,
    "autopay": -0.2,
    "support_calls": 0.35,
    "loyalty (скрытый)": -0.7,
}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def make_clients(n: int = 5000, seed: int = 42, include_hidden: bool = False) -> pd.DataFrame:
    """Клиенты провайдера: одна строка на клиента.

    Скрытая лояльность клиента влияет и на отток, и на то, подключает ли он автоплатёж и годовой контракт.
    Поэтому наблюдаемая связь автоплатежа с оттоком сильнее его настоящего эффекта.
    При include_hidden=True скрытый признак loyalty добавляется в таблицу — для экспериментов в уроке.
    """
    rng = np.random.default_rng(seed)
    loyalty = rng.normal(0, 1, n)

    tenure_months = np.clip(np.round(rng.gamma(shape=1.6, scale=14, size=n)), 1, 72).astype(int)
    monthly_fee = rng.choice([490, 790, 1190, 1790], size=n, p=[0.3, 0.35, 0.25, 0.1])
    services = rng.integers(1, 6, n)
    contract = np.where(rng.random(n) < _sigmoid(-0.5 + 0.8 * loyalty), "annual", "monthly")
    autopay = rng.random(n) < _sigmoid(-0.3 + 1.0 * loyalty)

    # Обращения в поддержку за последний год; период наблюдения короче года у новых клиентов
    exposure_years = np.minimum(tenure_months, 12) / 12
    calls_rate = np.exp(-0.2 + 0.25 * services - 0.3 * loyalty)
    support_calls = rng.poisson(calls_rate * exposure_years)

    logit = (
        -0.6
        - 0.035 * tenure_months
        + 0.5 * monthly_fee / 1000
        - 1.2 * (contract == "annual")
        - 0.2 * autopay
        + 0.35 * support_calls
        - 0.7 * loyalty
    )
    churn = rng.random(n) < _sigmoid(logit)

    clients = pd.DataFrame(
        {
            "client_id": np.arange(1, n + 1),
            "tenure_months": tenure_months,
            "monthly_fee": monthly_fee,
            "services": services,
            "contract": contract,
            "autopay": autopay,
            "exposure_years": exposure_years,
            "support_calls": support_calls,
            "churn": churn.astype(int),
        }
    )
    if include_hidden:
        clients["loyalty"] = loyalty
    return clients
