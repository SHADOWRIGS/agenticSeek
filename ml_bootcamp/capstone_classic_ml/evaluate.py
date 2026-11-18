"""Проверка прогноза итогового проекта.

Запуск:

    uv run python evaluate.py predictions.csv

Файл прогноза — CSV со столбцами station_id, timestamp, prediction: по одной строке на каждую пару
«станция — час» из тестовой таблицы. Скрипт проверяет формат, считает метрики и показывает их рядом
с опорными прогнозами.

Истинные значения скрипт получает из того же генератора, что и данные. Подсмотреть их нетрудно —
но тогда проект не покажет вам ничего о вашей модели.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rentals_data import ERROR_CODE, STATIONS, make_rentals, make_test_target

REQUIRED_COLUMNS = ("station_id", "timestamp", "prediction")
KEYS = ["station_id", "timestamp"]


def check_predictions(predictions: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Проверяет таблицу прогноза и возвращает её в порядке строк test.

    Бросает ValueError с понятным сообщением, если нет нужных столбцов, если строки не совпадают
    со строками test (лишние, пропущенные, повторяющиеся) или если прогноз не число, пропуск,
    бесконечность либо отрицательное значение.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in predictions.columns]
    if missing:
        raise ValueError(f"в прогнозе нет столбцов: {', '.join(missing)}")

    table = predictions[list(REQUIRED_COLUMNS)].copy()
    table["station_id"] = table["station_id"].astype("str")
    table["timestamp"] = pd.to_datetime(table["timestamp"])
    if table.duplicated(KEYS).any():
        raise ValueError(f"повторяющиеся пары «станция — час»: {int(table.duplicated(KEYS).sum())}")

    expected = test[KEYS].copy()
    expected["station_id"] = expected["station_id"].astype("str")
    merged = expected.merge(table, on=KEYS, how="left", indicator=True)
    absent = int((merged["_merge"] == "left_only").sum())
    if absent:
        raise ValueError(f"нет прогноза для {absent} строк тестовой таблицы")
    if len(table) != len(expected):
        raise ValueError(f"в прогнозе {len(table) - len(expected)} строк, которых нет в тестовой таблице")

    values = pd.to_numeric(merged["prediction"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("в прогнозе есть пропуски, бесконечности или нечисловые значения")
    if (values < 0).any():
        raise ValueError("в прогнозе есть отрицательные значения: число прокатов не бывает меньше нуля")

    merged["prediction"] = values
    return merged[list(REQUIRED_COLUMNS)]


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.abs(predicted - actual).mean()),
        "bias_pct": float(100.0 * (predicted.sum() - actual.sum()) / actual.sum()),
    }


def score(predictions: pd.DataFrame, seed: int = 42) -> dict:
    """Считает метрики прогноза.

    Возвращает словарь:
        mae — средняя абсолютная ошибка в прокатах на станцию в час;
        bias_pct — смещение суммарного объёма в процентах: 100 * (сумма прогноза − сумма факта) / сумма факта;
        mae_by_district — MAE по районам, словарь «район — значение».
    """
    _, test = make_rentals(seed)
    actual = make_test_target(seed)["rentals"].to_numpy(dtype=float)
    checked = check_predictions(predictions, test)
    predicted = checked["prediction"].to_numpy(dtype=float)

    result = _metrics(actual, predicted)
    district = checked["station_id"].map(STATIONS.set_index("station_id")["district"])
    errors = pd.Series(np.abs(predicted - actual))
    by_district = errors.groupby(district.to_numpy()).mean()
    result["mae_by_district"] = {name: float(value) for name, value in by_district.items()}
    return result


def reference_scores(seed: int = 42) -> dict[str, dict[str, float]]:
    """Метрики двух опорных прогнозов: «всегда ноль» и «среднее по станции и часу недели»."""
    train, test = make_rentals(seed)
    actual = make_test_target(seed)["rentals"].to_numpy(dtype=float)

    clean = train.drop_duplicates()
    clean = clean[clean["rentals"] != ERROR_CODE]
    hour_of_week = clean["timestamp"].dt.day_of_week * 24 + clean["timestamp"].dt.hour
    means = clean["rentals"].groupby([clean["station_id"], hour_of_week.rename("hour_of_week")]).mean()
    test_hour_of_week = (test["timestamp"].dt.day_of_week * 24 + test["timestamp"].dt.hour).rename("hour_of_week")
    index = pd.MultiIndex.from_arrays([test["station_id"], test_hour_of_week])
    by_hour_of_week = means.reindex(index).to_numpy(dtype=float)

    return {
        "всегда ноль": _metrics(actual, np.zeros_like(actual)),
        "среднее по станции и часу недели": _metrics(actual, by_hour_of_week),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("Использование: python evaluate.py predictions.csv")
        return 2
    path = Path(argv[0])
    if not path.exists():
        print(f"Файл не найден: {path}")
        return 2
    try:
        result = score(pd.read_csv(path))
    except ValueError as error:
        print(f"Прогноз не принят: {error}")
        return 1

    print(f"{'Прогноз':<36}{'MAE':>8}{'Смещение, %':>14}")
    for name, metrics in reference_scores().items():
        print(f"{name:<36}{metrics['mae']:>8.3f}{metrics['bias_pct']:>14.1f}")
    print(f"{'ваш прогноз':<36}{result['mae']:>8.3f}{result['bias_pct']:>14.1f}")
    print()
    print("MAE по районам:")
    for name, value in result["mae_by_district"].items():
        print(f"  {name:<14}{value:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
