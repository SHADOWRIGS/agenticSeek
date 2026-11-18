"""Данные итогового проекта: почасовой прокат велосипедов по станциям города.

Набор синтетический и полностью воспроизводимый: при одном и том же seed таблицы совпадают до последней
строки. Закономерности заложены в генератор явно (см. README.md, раздел «Паспорт данных»), поэтому
решение можно проверить, а не только сравнить с чужим.

Использование:

    from rentals_data import make_rentals
    train, test = make_rentals()
    # train — история с целевой переменной и «грязью» реальной выгрузки
    # test  — три месяца, на которые нужен прогноз: без rentals и returns

Истинные значения за тестовый период возвращает make_test_target(); их использует evaluate.py.
"""

import numpy as np
import pandas as pd

START = pd.Timestamp("2023-01-01 00:00")
TRAIN_END = pd.Timestamp("2024-09-30 23:00")
TEST_START = pd.Timestamp("2024-10-01 00:00")
END = pd.Timestamp("2024-12-31 23:00")

# Станции: район, средний спрос в час при «идеальных» условиях, число мест, дата открытия
STATIONS = pd.DataFrame(
    [
        ("S01", "center", 9.0, 40, "2023-01-01"),
        ("S02", "center", 7.5, 32, "2023-01-01"),
        ("S03", "center", 6.0, 28, "2023-01-01"),
        ("S04", "residential", 4.0, 24, "2023-01-01"),
        ("S05", "residential", 3.5, 20, "2023-01-01"),
        ("S06", "residential", 3.0, 20, "2023-01-01"),
        ("S07", "park", 4.5, 30, "2023-01-01"),
        ("S08", "park", 3.0, 24, "2023-01-01"),
        ("S09", "university", 5.0, 30, "2023-01-01"),
        ("S10", "university", 4.0, 24, "2023-06-01"),
        ("S11", "residential", 3.2, 20, "2024-08-15"),
        ("S12", "center", 5.5, 28, "2023-01-01"),
    ],
    columns=["station_id", "district", "base_demand", "capacity", "opened"],
).astype({"opened": "datetime64[us]"})

# Нерабочие праздничные дни: спрос в эти дни ведёт себя как в выходные
HOLIDAYS = pd.to_datetime(
    [f"{year}-01-0{day}" for year in (2023, 2024) for day in range(1, 9)]
    + [f"{year}-{md}" for year in (2023, 2024) for md in ("02-23", "03-08", "05-01", "05-09", "06-12", "11-04")]
)

# Станции, закрытые на обслуживание: в эти дни строк в выгрузке нет
MAINTENANCE = {
    "S05": ("2024-03-11", "2024-03-24"),
    "S07": ("2023-11-01", "2023-11-20"),
}

ERROR_CODE = -1  # так счётчик записывает сбой: это не «минус один прокат»

_SEMESTER = {1: 0.35, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.6, 7: 0.25, 8: 0.25, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0}


def _bump(hours: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-((hours - center) ** 2) / (2 * width**2))


def _hour_profiles() -> dict[str, np.ndarray]:
    """Суточные профили спроса, каждый нормирован к среднему 1."""
    h = np.arange(24, dtype=float)
    profiles = {
        "commute": _bump(h, 8.5, 1.2) + 1.1 * _bump(h, 18.0, 1.5) + 0.25 * _bump(h, 13.0, 2.5) + 0.03,
        "leisure": _bump(h, 15.0, 3.2) + 0.03,
        "campus": 0.8 * _bump(h, 9.0, 1.3) + _bump(h, 14.5, 2.5) + 0.5 * _bump(h, 19.0, 1.5) + 0.03,
    }
    return {name: values / values.mean() for name, values in profiles.items()}


def _weather(hours: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Погода одна на весь город: температура, осадки, ветер."""
    n_days = len(hours) // 24
    day_of_year = hours.day_of_year.to_numpy()
    hour = hours.hour.to_numpy()

    daily_noise = np.zeros(n_days)
    shocks = rng.normal(0.0, 2.5, n_days)
    for day in range(1, n_days):
        daily_noise[day] = 0.8 * daily_noise[day - 1] + shocks[day]
    temp = (
        7.0
        + 15.0 * np.sin(2 * np.pi * (day_of_year - 109) / 365.25)
        + 4.0 * np.sin(2 * np.pi * (hour - 9) / 24)
        + np.repeat(daily_noise, 24)
        + rng.normal(0.0, 0.7, len(hours))
    )

    wet = np.zeros(n_days, dtype=bool)
    draws = rng.random(n_days)
    for day in range(1, n_days):
        wet[day] = draws[day] < (0.55 if wet[day - 1] else 0.2)
    raining = np.repeat(wet, 24) & (rng.random(len(hours)) < 0.35)
    precip = np.where(raining, rng.gamma(1.5, 1.2, len(hours)), 0.0)

    wind = rng.gamma(4.0, 1.0, len(hours))
    return pd.DataFrame(
        {"timestamp": hours, "temp": temp.round(1), "precip_mm": precip.round(1), "wind": wind.round(1)}
    )


def _full_table(seed: int, with_expected: bool = False) -> pd.DataFrame:
    """Полная чистая таблица за весь период: все станции, все часы после открытия.

    with_expected=True добавляет столбец expected — ожидаемый спрос без случайных «эффектов дня».
    Это предел того, что можно предсказать по календарю, погоде и станции.
    """
    rng = np.random.default_rng(seed)
    hours = pd.date_range(START, END, freq="h")
    weather = _weather(hours, rng)
    profiles = _hour_profiles()

    hour = hours.hour.to_numpy()
    date = hours.normalize()
    day_off = (hours.day_of_week.to_numpy() >= 5) | date.isin(HOLIDAYS)
    semester = hours.month.map(_SEMESTER).to_numpy(dtype=float)
    years_passed = (hours - START).total_seconds().to_numpy() / (365.25 * 24 * 3600)

    temp_effect = 0.05 + 0.95 * np.exp(-(((weather["temp"].to_numpy() - 22.0) / 14.0) ** 2))
    wind_effect = 1.0 - 0.03 * np.clip(weather["wind"].to_numpy() - 6.0, 0.0, None)
    trend = 1.0 + 0.25 * years_passed
    raining = weather["precip_mm"].to_numpy() > 0

    n_days = len(hours) // 24
    city_day = np.repeat(rng.lognormal(0.0, 0.18, n_days), 24)

    shape_by_district = {
        "center": np.where(day_off, 0.6 * profiles["leisure"][hour], profiles["commute"][hour]),
        "residential": np.where(day_off, 0.8 * profiles["leisure"][hour], profiles["commute"][hour]),
        "park": np.where(day_off, 1.6 * profiles["leisure"][hour], 0.5 * profiles["leisure"][hour]),
        "university": semester * np.where(day_off, 0.35 * profiles["leisure"][hour], profiles["campus"][hour]),
    }

    parts = []
    for station in STATIONS.itertuples(index=False):
        rain_effect = np.where(raining, 0.2 if station.district == "park" else 0.45, 1.0)
        station_day = np.repeat(rng.lognormal(0.0, 0.10, n_days), 24)
        expected = (
            station.base_demand * shape_by_district[station.district] * temp_effect * rain_effect * wind_effect * trend
        )
        intensity = expected * city_day * station_day
        rentals = rng.poisson(intensity)
        returns = rng.poisson(0.5 * rentals + 0.5 * intensity)
        part = weather.copy()
        part.insert(0, "station_id", station.station_id)
        part["rentals"] = rentals
        part["returns"] = returns
        if with_expected:
            part["expected"] = expected
        parts.append(part[part["timestamp"] >= station.opened])

    table = pd.concat(parts, ignore_index=True)
    return table.sort_values(["timestamp", "station_id"], ignore_index=True)


def make_rentals(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (train, test).

    train: station_id, timestamp, temp, precip_mm, wind, rentals, returns — история по 30 сентября 2024 года
        в том виде, в каком её выгрузили: с пропусками в температуре, дубликатами строк, кодом ошибки
        счётчика в rentals и без строк за дни обслуживания станций.
    test: station_id, timestamp, temp, precip_mm, wind — с 1 октября по 31 декабря 2024 года. Погода в test —
        прогноз, который известен заранее (в учебных данных он совпадает с фактом).
    """
    table = _full_table(seed)
    rng = np.random.default_rng(seed + 1)

    test = table.loc[table["timestamp"] >= TEST_START, ["station_id", "timestamp", "temp", "precip_mm", "wind"]]
    train = table[table["timestamp"] <= TRAIN_END].copy()

    for station_id, (first_day, last_day) in MAINTENANCE.items():
        closed = (
            (train["station_id"] == station_id)
            & (train["timestamp"] >= pd.Timestamp(first_day))
            & (train["timestamp"] < pd.Timestamp(last_day) + pd.Timedelta(days=1))
        )
        train = train[~closed]

    train = train.reset_index(drop=True)
    n = len(train)
    train.loc[rng.random(n) < 0.008, "temp"] = np.nan
    broken = rng.random(n) < 0.0005
    train.loc[broken, "rentals"] = ERROR_CODE
    duplicates = train.sample(frac=0.003, random_state=seed)
    train = pd.concat([train, duplicates]).sort_values(["timestamp", "station_id"], kind="stable", ignore_index=True)

    return train, test.reset_index(drop=True)


def make_test_target(seed: int = 42) -> pd.DataFrame:
    """Истинное число прокатов за тестовый период: station_id, timestamp, rentals."""
    table = _full_table(seed)
    target = table.loc[table["timestamp"] >= TEST_START, ["station_id", "timestamp", "rentals"]]
    return target.reset_index(drop=True)


if __name__ == "__main__":
    train_table, test_table = make_rentals()
    print(f"train: {train_table.shape}, {train_table['timestamp'].min()} — {train_table['timestamp'].max()}")
    print(f"test:  {test_table.shape}, {test_table['timestamp'].min()} — {test_table['timestamp'].max()}")
    print(train_table.head())
