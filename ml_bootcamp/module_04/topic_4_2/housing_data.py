"""Синтетические объявления о продаже квартир для уроков темы 4.2.

Использование:  from housing_data import FEATURES, TARGET, make_housing, true_price
"""

import numpy as np
import pandas as pd

TARGET = "price"
FEATURES = [
    "area",
    "living_area",
    "rooms",
    "floor",
    "floors_total",
    "age",
    "district",
    "metro_min",
    "renovation",
    "balcony",
    "month",
    "weekday",
    "agent_id",
    "desc_len",
]
# Признаки, которые не входят в формулу цены
NOISE_FEATURES = ["weekday", "agent_id", "desc_len"]
PROXY_FEATURES = ["living_area", "rooms"]

# Коэффициент района к цене квадратного метра; первые два района — исторические
DISTRICT_COEF = {
    "Центр": 1.45,
    "Старый город": 1.20,
    "Заречье": 1.00,
    "Северный": 0.90,
    "Южный": 0.85,
    "Заводской": 0.72,
}
DISTRICT_SHARE = [0.12, 0.18, 0.20, 0.18, 0.17, 0.15]
HISTORIC_DISTRICTS = ["Центр", "Старый город"]
RENOVATION_COEF = [1.00, 1.08, 1.20]
BASE_PRICE_M2 = 120.0  # тыс. руб. за м²


def true_price(flats: pd.DataFrame) -> np.ndarray:
    """Ожидаемая цена квартиры (млн руб.) без шума — истинная зависимость, известная только в учебных данных.

    price = area · price_m2 / 1000, где цена метра (тыс. руб.) — произведение множителей:

        120
        · коэффициент района (DISTRICT_COEF)
        · (area / 50) ** -0.15                      маленькие квартиры дороже в пересчёте на метр
        · возраст дома: в исторических районах 0.95 + 0.002 · age (старый фонд ценится),
                        в остальных 0.75 + 0.25 · exp(-age / 20) (дом дешевеет и выходит на плато)
        · метро: 1 - 0.2 · min(metro_min, 20) / 20  (после 20 минут пешком цена не меняется)
        · этаж: 0.92 для первого, 0.96 для последнего (если он не первый), иначе 1
        · ремонт: 1.00 / 1.08 / 1.20 для renovation = 0 / 1 / 2
        · балкон: 1.03
        · время: 1 + 0.006 · month                  цены растут на 0.6 % базового уровня в месяц

    Признаки living_area, rooms, weekday, agent_id, desc_len в формулу не входят.
    """
    historic = flats["district"].isin(HISTORIC_DISTRICTS).to_numpy()
    age = flats["age"].to_numpy()
    floor = flats["floor"].to_numpy()
    age_coef = np.where(historic, 0.95 + 0.002 * age, 0.75 + 0.25 * np.exp(-age / 20))
    floor_coef = np.where(floor == 1, 0.92, np.where(floor == flats["floors_total"].to_numpy(), 0.96, 1.0))
    price_m2 = (
        BASE_PRICE_M2
        * flats["district"].map(DISTRICT_COEF).to_numpy()
        * (flats["area"].to_numpy() / 50) ** -0.15
        * age_coef
        * (1 - 0.2 * np.minimum(flats["metro_min"].to_numpy(), 20) / 20)
        * floor_coef
        * np.array(RENOVATION_COEF)[flats["renovation"].to_numpy()]
        * (1 + 0.03 * flats["balcony"].to_numpy())
        * (1 + 0.006 * flats["month"].to_numpy())
    )
    return flats["area"].to_numpy() * price_m2 / 1000


def make_housing(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Объявления о продаже квартир: одна строка на объявление, цена price — в млн руб.

    Признаки:
        area — общая площадь, м²; living_area — жилая площадь, м² (почти точно 0.62 · area);
        rooms — число комнат (определяется площадью); floor, floors_total — этаж и этажность дома;
        age — возраст дома, лет; district — район (строка, 6 значений);
        metro_min — минут пешком до метро; renovation — ремонт: 0 нет, 1 косметический, 2 капитальный;
        balcony — наличие балкона (0/1); month — номер месяца публикации, от 0 до 35;
        weekday — день недели публикации (0–6); agent_id — идентификатор риелтора (1–400);
        desc_len — длина текста объявления в символах.

    Цена: true_price(flats) · множитель шума. Шум логнормальный со средним 1; его разброс растёт
    с возрастом дома: стандартное отклонение логарифма равно 0.04 + 0.001 · age.
    Признаки weekday, agent_id и desc_len на цену не влияют; living_area и rooms связаны с ценой
    только через area.
    """
    rng = np.random.default_rng(seed)
    area = np.clip(rng.lognormal(mean=np.log(52), sigma=0.38, size=n), 22, 160).round(1)
    living_area = np.clip(0.62 * area + rng.normal(0, 2.5, n), 10, None).round(1)
    rooms = np.clip(np.round(area / 24 + rng.normal(0, 0.45, n)), 1, 5).astype(int)
    floors_total = rng.choice([5, 9, 12, 16, 25], size=n, p=[0.30, 0.30, 0.15, 0.15, 0.10])
    floor = rng.integers(1, floors_total + 1)
    age = np.clip(np.round(rng.gamma(shape=1.5, scale=22, size=n)), 0, 90).astype(int)
    district = rng.choice(list(DISTRICT_COEF), size=n, p=DISTRICT_SHARE)
    metro_min = np.clip(np.round(rng.gamma(shape=2.0, scale=7, size=n)), 1, 45).astype(int)
    renovation = rng.choice([0, 1, 2], size=n, p=[0.3, 0.5, 0.2])
    balcony = (rng.random(n) < 0.6).astype(int)
    month = rng.integers(0, 36, n)
    weekday = rng.integers(0, 7, n)
    agent_id = rng.integers(1, 401, n)
    desc_len = np.round(rng.gamma(shape=3.0, scale=250, size=n)).astype(int)

    flats = pd.DataFrame(
        {
            "area": area,
            "living_area": living_area,
            "rooms": rooms,
            "floor": floor,
            "floors_total": floors_total,
            "age": age,
            "district": district,
            "metro_min": metro_min,
            "renovation": renovation,
            "balcony": balcony,
            "month": month,
            "weekday": weekday,
            "agent_id": agent_id,
            "desc_len": desc_len,
        }
    )
    sigma = 0.04 + 0.001 * age
    noise = np.exp(rng.normal(0, 1, n) * sigma - sigma**2 / 2)
    flats[TARGET] = (true_price(flats) * noise).round(3)
    return flats
