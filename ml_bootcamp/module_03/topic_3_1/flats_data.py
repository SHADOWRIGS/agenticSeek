"""Синтетические данные о ценах квартир для уроков темы 3.1.

Использование:  from flats_data import make_flats
"""

import numpy as np
import pandas as pd

FEATURES = ["area", "rooms", "age", "distance"]
TARGET = "price"


def make_flats(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Квартиры: площадь (м²), число комнат, возраст дома (лет), расстояние до центра (км), цена (млн руб.).

    Истинная зависимость: price = 2.0 + 0.11·area + 0.3·rooms − 0.04·age − 0.15·distance + шум,
    шум — нормальный со стандартным отклонением 0.8. Число комнат сильно связано с площадью.
    """
    rng = np.random.default_rng(seed)
    area = rng.uniform(30, 120, n)
    rooms = np.clip(np.round(area / 25 + rng.normal(0, 0.5, n)), 1, 5)
    age = rng.uniform(0, 50, n)
    distance = rng.uniform(1, 25, n)
    price = 2.0 + 0.11 * area + 0.3 * rooms - 0.04 * age - 0.15 * distance + rng.normal(0, 0.8, n)
    return pd.DataFrame({"area": area, "rooms": rooms, "age": age, "distance": distance, "price": price})
