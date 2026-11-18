"""Тесты материалов итогового проекта: генератора данных и скрипта проверки.

Эти тесты проходят сразу: они проверяют не ваше решение, а то, что данные у вас те же, что в задании,
и что evaluate.py считает метрики так, как описано в README.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

KIT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT_DIR))

import evaluate  # noqa: E402
import rentals_data as rd  # noqa: E402


@pytest.fixture(scope="module")
def tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    return rd.make_rentals()


@pytest.fixture(scope="module")
def target() -> pd.DataFrame:
    return rd.make_test_target()


@pytest.fixture()
def perfect(target: pd.DataFrame) -> pd.DataFrame:
    return target.rename(columns={"rentals": "prediction"})


def test_tables_are_reproducible(tables):
    train, test = tables
    train_again, test_again = rd.make_rentals()
    pd.testing.assert_frame_equal(train, train_again)
    pd.testing.assert_frame_equal(test, test_again)


def test_shapes_and_columns(tables):
    train, test = tables
    assert list(train.columns) == ["station_id", "timestamp", "temp", "precip_mm", "wind", "rentals", "returns"]
    assert list(test.columns) == ["station_id", "timestamp", "temp", "precip_mm", "wind"]
    assert train.shape == (165880, 7)
    assert test.shape == (26496, 5)


def test_periods_do_not_overlap(tables):
    train, test = tables
    assert train["timestamp"].min() == rd.START
    assert train["timestamp"].max() == rd.TRAIN_END
    assert test["timestamp"].min() == rd.TEST_START
    assert test["timestamp"].max() == rd.END


def test_train_has_documented_mess(tables):
    train, _ = tables
    assert train.duplicated().sum() == 496
    assert train["temp"].isna().sum() == 1375
    assert (train["rentals"] == rd.ERROR_CODE).sum() == 81
    assert train["precip_mm"].isna().sum() == 0


def test_test_table_is_clean(tables):
    _, test = tables
    assert not test.duplicated(["station_id", "timestamp"]).any()
    assert not test.isna().any().any()
    assert test.groupby("station_id").size().eq(92 * 24).all()


def test_maintenance_days_are_absent(tables):
    train, _ = tables
    for station_id, (first_day, last_day) in rd.MAINTENANCE.items():
        rows = train[
            (train["station_id"] == station_id)
            & (train["timestamp"] >= pd.Timestamp(first_day))
            & (train["timestamp"] < pd.Timestamp(last_day) + pd.Timedelta(days=1))
        ]
        assert rows.empty


def test_stations_start_at_opening_date(tables):
    train, _ = tables
    first_rows = train.groupby("station_id")["timestamp"].min()
    opened = rd.STATIONS.set_index("station_id")["opened"]
    assert (first_rows == opened.reindex(first_rows.index)).all()
    assert first_rows["S11"] == pd.Timestamp("2024-08-15")


def test_target_matches_test_rows(tables, target):
    _, test = tables
    pd.testing.assert_frame_equal(target[["station_id", "timestamp"]], test[["station_id", "timestamp"]])
    assert (target["rentals"] >= 0).all()


def test_perfect_forecast_scores_zero(perfect):
    result = evaluate.score(perfect)
    assert result["mae"] == 0.0
    assert result["bias_pct"] == 0.0
    assert set(result["mae_by_district"]) == {"center", "residential", "park", "university"}


def test_row_order_does_not_matter(perfect):
    shuffled = perfect.sample(frac=1.0, random_state=0)
    assert evaluate.score(shuffled)["mae"] == 0.0


def test_zero_forecast_scores_mean_target(target):
    zeros = target.assign(prediction=0.0).drop(columns="rentals")
    result = evaluate.score(zeros)
    assert result["mae"] == pytest.approx(target["rentals"].mean())
    assert result["bias_pct"] == pytest.approx(-100.0)


def test_doubled_forecast_has_plus_100_percent_bias(target):
    doubled = target.assign(prediction=2.0 * target["rentals"]).drop(columns="rentals")
    assert evaluate.score(doubled)["bias_pct"] == pytest.approx(100.0)


def test_reference_scores_match_readme():
    reference = evaluate.reference_scores()
    assert reference["всегда ноль"]["mae"] == pytest.approx(1.428, abs=0.001)
    assert reference["среднее по станции и часу недели"]["mae"] == pytest.approx(1.863, abs=0.001)
    assert reference["среднее по станции и часу недели"]["bias_pct"] > 80.0


def test_csv_round_trip(perfect, tmp_path):
    path = tmp_path / "predictions.csv"
    perfect.to_csv(path, index=False)
    assert evaluate.score(pd.read_csv(path))["mae"] == 0.0


def test_missing_column_is_rejected(perfect):
    with pytest.raises(ValueError, match="нет столбцов"):
        evaluate.score(perfect.rename(columns={"prediction": "rentals"}))


def test_missing_rows_are_rejected(perfect):
    with pytest.raises(ValueError, match="нет прогноза для 10 строк"):
        evaluate.score(perfect.iloc[10:])


def test_extra_rows_are_rejected(perfect):
    extra = perfect.iloc[[0]].assign(timestamp=pd.Timestamp("2025-01-01"))
    with pytest.raises(ValueError, match="которых нет в тестовой таблице"):
        evaluate.score(pd.concat([perfect, extra], ignore_index=True))


def test_duplicate_rows_are_rejected(perfect):
    with pytest.raises(ValueError, match="повторяющиеся"):
        evaluate.score(pd.concat([perfect, perfect.iloc[[0]]], ignore_index=True))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, "много"])
def test_non_numeric_forecast_is_rejected(perfect, bad_value):
    broken = perfect.astype({"prediction": "object"})
    broken.loc[5, "prediction"] = bad_value
    with pytest.raises(ValueError, match="пропуски, бесконечности или нечисловые"):
        evaluate.score(broken)


def test_negative_forecast_is_rejected(perfect):
    broken = perfect.astype({"prediction": "float64"})
    broken.loc[5, "prediction"] = -0.5
    with pytest.raises(ValueError, match="отрицательные"):
        evaluate.score(broken)


def test_command_line_report(perfect, tmp_path, capsys):
    path = tmp_path / "predictions.csv"
    perfect.to_csv(path, index=False)
    assert evaluate.main([str(path)]) == 0
    output = capsys.readouterr().out
    assert "ваш прогноз" in output
    assert "всегда ноль" in output
    assert "MAE по районам" in output


def test_command_line_rejects_bad_file(perfect, tmp_path, capsys):
    path = tmp_path / "predictions.csv"
    perfect.iloc[100:].to_csv(path, index=False)
    assert evaluate.main([str(path)]) == 1
    assert "Прогноз не принят" in capsys.readouterr().out
    assert evaluate.main([str(tmp_path / "нет_такого.csv")]) == 2
    assert evaluate.main([]) == 2
