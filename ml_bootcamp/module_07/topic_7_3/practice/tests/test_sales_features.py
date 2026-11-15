"""Тесты практики 7.3.4: исполняемая спецификация модуля sales_features.

Запуск из папки practice:  uv run pytest -q
Тесты не используют сеть: хранилище Feast (реестр, офлайн-таблица, SQLite) создаётся во временных
каталогах pytest. Все даты отсчитываются от текущего дня, поэтому результат не зависит от даты запуска.
"""

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))

import sales_features as sf  # noqa: E402

# Предупреждения сторонних библиотек, не зависящие от решения: см. README
pytestmark = [
    pytest.mark.filterwarnings("ignore:fetch_arrow_table.. is deprecated:DeprecationWarning"),
]

TODAY = pd.Timestamp.now(tz="UTC").floor("D")  # сегодня, 00:00 UTC
NOW = (TODAY + timedelta(hours=6)).to_pydatetime()  # момент публикации и запросов
DAY = timedelta(days=1)
TTL = sf.DEFAULT_TTL
PAIRS = [("S01", "Мышь"), ("S01", "Чайник"), ("S02", "Мышь"), ("S02", "Зонт")]
ROWS = [{"store_id": store_id, "product": product} for store_id, product in PAIRS]
VIEW_REFS = [f"{sf.FEATURE_VIEW}:{name}" for name in sf.FEATURE_NAMES]


# ---------------------------------------------------------------------------------------------
# Данные и помощники
# ---------------------------------------------------------------------------------------------


def make_sales(days=40, pairs=PAIRS, seed=0):
    """Дневные продажи: по строке на пару и день, последний день — вчера."""
    rng = np.random.default_rng(seed)
    last = TODAY.tz_localize(None) - DAY
    dates = [last - timedelta(days=offset) for offset in range(days - 1, -1, -1)]
    frame = pd.DataFrame(
        [(date, store_id, product) for date in dates for store_id, product in pairs],
        columns=["date", "store_id", "product"],
    )
    frame["price"] = 990
    frame["promo"] = rng.random(len(frame)) < 0.2
    frame["units"] = rng.poisson(6, len(frame)).astype("float64")
    return frame


def one_pair_sales(units, promo=None):
    """Продажи одной пары: последний день — вчера, значения units по дням слева направо."""
    last = TODAY.tz_localize(None) - DAY
    dates = [last - timedelta(days=offset) for offset in range(len(units) - 1, -1, -1)]
    return pd.DataFrame(
        {
            "date": dates,
            "store_id": "S01",
            "product": "Мышь",
            "units": np.array(units, dtype="float64"),
            "promo": promo if promo is not None else [False] * len(units),
        }
    )


def make_features(days=10, pairs=PAIRS, last=TODAY, seed=0):
    """Таблица признаков с произвольными значениями: по строке на пару и день, последняя отметка — last."""
    rng = np.random.default_rng(seed)
    stamps = [last - timedelta(days=offset) for offset in range(days - 1, -1, -1)]
    frame = pd.DataFrame(
        [(store_id, product, stamp) for store_id, product in pairs for stamp in stamps],
        columns=["store_id", "product", "event_timestamp"],
    )
    frame["event_timestamp"] = frame["event_timestamp"].astype(sf.TIMESTAMP_DTYPE)
    frame["created"] = (frame["event_timestamp"] + sf.LOAD_DELAY).astype(sf.TIMESTAMP_DTYPE)
    frame["units_mean_7d"] = rng.uniform(1, 20, len(frame)).round(3)
    frame["units_mean_28d"] = rng.uniform(1, 20, len(frame)).round(3)
    frame["promo_share_28d"] = rng.uniform(0, 1, len(frame)).round(3)
    return frame[sf.FEATURE_COLUMNS]


def entity(rows):
    """Таблица запросов из троек (store_id, product, момент запроса)."""
    frame = pd.DataFrame(rows, columns=["store_id", "product", "event_timestamp"])
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"], utc=True)
    return frame


def value_at(features, pair, stamp, name="units_mean_7d"):
    """Значение признака пары в строке с отметкой времени stamp."""
    mask = (features["store_id"] == pair[0]) & (features["product"] == pair[1]) & (features["event_timestamp"] == stamp)
    assert mask.sum() == 1
    return float(features.loc[mask, name].iloc[0])


def latest_rows(features):
    """Последняя строка каждой пары."""
    ordered = features.sort_values("event_timestamp")
    return ordered.drop_duplicates(sf.KEYS, keep="last").reset_index(drop=True)[sf.FEATURE_COLUMNS]


def repo_dir(tmp_path):
    return tmp_path / "repo"


def fill_offline(tmp_path, features):
    """Кладёт таблицу в офлайн-хранилище в обход publish_features."""
    features.to_parquet(repo_dir(tmp_path) / sf.OFFLINE_FILE, index=False)


def fill_online(store, features):
    """Кладёт последние значения пар в онлайн-хранилище в обход publish_features."""
    store.write_to_online_store(sf.FEATURE_VIEW, latest_rows(features))


def raw_online(store, rows):
    """Чтение онлайн-хранилища средствами Feast, без online_features."""
    return store.get_online_features(features=VIEW_REFS, entity_rows=rows).to_dict()


def published(tmp_path, features):
    """Хранилище, в которое таблица опубликована функциями решения."""
    store = sf.create_store(repo_dir(tmp_path))
    sf.publish_features(store, features, NOW)
    return store


@pytest.fixture
def sales():
    return make_sales()


@pytest.fixture
def features():
    return make_features()


# ---------------------------------------------------------------------------------------------
# build_pair_features
# ---------------------------------------------------------------------------------------------


def test_features_shape_columns_and_types(sales):
    result = sf.build_pair_features(sales)
    assert list(result.columns) == sf.FEATURE_COLUMNS
    assert len(result) == len(sales)
    assert isinstance(result.index, pd.RangeIndex)
    assert str(result["event_timestamp"].dtype) == sf.TIMESTAMP_DTYPE
    assert str(result["created"].dtype) == sf.TIMESTAMP_DTYPE
    for name in sf.FEATURE_NAMES:
        assert result[name].dtype == np.float64
    expected_order = result.sort_values(["store_id", "product", "event_timestamp"], ignore_index=True)
    pd.testing.assert_frame_equal(result, expected_order)


def test_features_timestamp_is_start_of_next_day():
    result = sf.build_pair_features(one_pair_sales([1, 2, 3]))
    assert result["event_timestamp"].tolist() == [TODAY - 2 * DAY, TODAY - DAY, TODAY]
    assert (result["created"] - result["event_timestamp"]).tolist() == [sf.LOAD_DELAY] * 3


def test_features_rolling_means():
    promo = [True, False, False, True, False, False, False, False, False, False]
    result = sf.build_pair_features(one_pair_sales([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], promo))
    assert result["units_mean_7d"].tolist() == pytest.approx([1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7])
    assert result["units_mean_28d"].tolist() == pytest.approx([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])
    assert result["promo_share_28d"].tolist() == pytest.approx(
        [1, 1 / 2, 1 / 3, 2 / 4, 2 / 5, 2 / 6, 2 / 7, 2 / 8, 2 / 9, 0.2]
    )


def test_features_28_day_window_forgets_old_days():
    result = sf.build_pair_features(one_pair_sales([100] + [2] * 28, [True] + [False] * 28))
    assert result["units_mean_28d"].iloc[-2] == pytest.approx((100 + 2 * 27) / 28)
    assert result["units_mean_28d"].iloc[-1] == pytest.approx(2)
    assert result["promo_share_28d"].iloc[-2] == pytest.approx(1 / 28)
    assert result["promo_share_28d"].iloc[-1] == pytest.approx(0)


def test_features_missing_units_are_skipped():
    result = sf.build_pair_features(one_pair_sales([np.nan, np.nan, 4, np.nan, 8]))
    values = result["units_mean_7d"].tolist()
    assert np.isnan(values[0]) and np.isnan(values[1])
    assert values[2:] == pytest.approx([4, 4, 6])
    assert result["promo_share_28d"].tolist() == pytest.approx([0, 0, 0, 0, 0])


def test_features_window_is_calendar_days():
    sales = one_pair_sales([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    with_gap = sales.drop(index=[3, 4, 5, 6, 7]).reset_index(drop=True)  # остались дни 1-3 и 9-10
    result = sf.build_pair_features(with_gap)
    assert len(result) == 5
    # окно дня 9 — дни 3…9: в данных есть только 3-й и 9-й; окно дня 10 — дни 4…10: только 9-й и 10-й
    assert result["units_mean_7d"].tolist() == pytest.approx([10, 15, 20, 60, 95])
    assert result["units_mean_28d"].tolist() == pytest.approx([10, 15, 20, 37.5, 50])


def test_features_do_not_depend_on_later_days(sales):
    full = sf.build_pair_features(sales)
    cut = sales["date"].sort_values().unique()[-6]
    truncated = sf.build_pair_features(sales[sales["date"] <= cut])
    expected = full[full["event_timestamp"] <= truncated["event_timestamp"].max()].reset_index(drop=True)
    pd.testing.assert_frame_equal(truncated, expected)


def test_features_pairs_are_independent_and_row_order_is_irrelevant(sales):
    expected = sf.build_pair_features(sales)
    shuffled = sales.sample(frac=1, random_state=1)
    pd.testing.assert_frame_equal(sf.build_pair_features(shuffled), expected)
    only = sales[(sales["store_id"] == "S02") & (sales["product"] == "Зонт")]
    part = expected[(expected["store_id"] == "S02") & (expected["product"] == "Зонт")].reset_index(drop=True)
    pd.testing.assert_frame_equal(sf.build_pair_features(only), part)


def test_features_input_is_not_modified(sales):
    before = sales.copy()
    sf.build_pair_features(sales)
    pd.testing.assert_frame_equal(sales, before)


def test_features_duplicate_day_is_rejected(sales):
    doubled = pd.concat([sales, sales.tail(1)], ignore_index=True)
    with pytest.raises(ValueError):
        sf.build_pair_features(doubled)


# ---------------------------------------------------------------------------------------------
# point_in_time_join
# ---------------------------------------------------------------------------------------------


def test_join_takes_latest_value_known_at_request_time(features):
    pair = PAIRS[0]
    stamp = TODAY - 3 * DAY
    requests = entity(
        [
            (*pair, stamp - timedelta(seconds=1)),  # за секунду до обновления: вчерашнее значение
            (*pair, stamp),  # ровно в момент обновления: новое значение
            (*pair, stamp + timedelta(hours=12)),
        ]
    )
    result = sf.point_in_time_join(requests, features, TTL)
    assert result["units_mean_7d"].tolist() == pytest.approx(
        [value_at(features, pair, stamp - DAY), value_at(features, pair, stamp), value_at(features, pair, stamp)]
    )
    assert result["promo_share_28d"].iloc[2] == pytest.approx(value_at(features, pair, stamp, "promo_share_28d"))


def test_join_never_looks_into_the_future(features):
    first = features["event_timestamp"].min()
    requests = entity([(*PAIRS[0], first - timedelta(hours=1)), (*PAIRS[1], first - 5 * DAY)])
    result = sf.point_in_time_join(requests, features, TTL)
    assert result[sf.FEATURE_NAMES].isna().all().all()


def test_join_ttl_bounds_are_inclusive(features):
    pair = PAIRS[2]
    requests = entity([(*pair, TODAY + TTL), (*pair, TODAY + TTL + timedelta(seconds=1))])
    result = sf.point_in_time_join(requests, features, TTL)
    assert result["units_mean_7d"].iloc[0] == pytest.approx(value_at(features, pair, TODAY))
    assert np.isnan(result["units_mean_7d"].iloc[1])


def test_join_ttl_is_a_parameter(features):
    pair = PAIRS[2]
    requests = entity([(*pair, TODAY + 5 * DAY)])
    assert sf.point_in_time_join(requests, features, TTL)[sf.FEATURE_NAMES].isna().all().all()
    wide = sf.point_in_time_join(requests, features, timedelta(days=7))
    assert wide["units_mean_28d"].iloc[0] == pytest.approx(value_at(features, pair, TODAY, "units_mean_28d"))


def test_join_does_not_mix_pairs(features):
    stamp = TODAY - DAY
    requests = entity([(*pair, stamp + timedelta(hours=1)) for pair in PAIRS])
    result = sf.point_in_time_join(requests, features, TTL)
    assert result["units_mean_7d"].tolist() == pytest.approx([value_at(features, pair, stamp) for pair in PAIRS])


def test_join_unknown_pair_gets_missing_values(features):
    requests = entity([("S99", "Мышь", TODAY), (*PAIRS[0], TODAY)])
    result = sf.point_in_time_join(requests, features, TTL)
    assert result.loc[0, sf.FEATURE_NAMES].isna().all()
    assert result.loc[1, "units_mean_7d"] == pytest.approx(value_at(features, PAIRS[0], TODAY))


def test_join_keeps_rows_order_index_and_columns(features):
    stamps = [TODAY - 4 * DAY, TODAY, TODAY - 2 * DAY, TODAY - 6 * DAY]
    requests = entity([(*pair, stamp) for pair, stamp in zip(PAIRS, stamps, strict=True)])
    requests["units"] = [5.0, 7.0, 1.0, 3.0]
    requests["promo"] = [True, False, False, True]
    requests.index = [40, 10, 30, 20]
    before_requests, before_features = requests.copy(), features.copy()

    result = sf.point_in_time_join(requests, features, TTL)

    assert list(result.columns) == [*requests.columns, *sf.FEATURE_NAMES]
    pd.testing.assert_frame_equal(result[list(requests.columns)], before_requests)
    expected = [value_at(features, pair, stamp) for pair, stamp in zip(PAIRS, stamps, strict=True)]
    assert result["units_mean_7d"].tolist() == pytest.approx(expected)
    pd.testing.assert_frame_equal(requests, before_requests)
    pd.testing.assert_frame_equal(features, before_features)


def test_join_same_timestamp_takes_latest_created(features):
    pair = PAIRS[0]
    fixed = features[(features["store_id"] == pair[0]) & (features["product"] == pair[1])].tail(1).copy()
    fixed["created"] = fixed["created"] + timedelta(hours=5)
    fixed["units_mean_7d"] = 77.0
    for table in (pd.concat([features, fixed], ignore_index=True), pd.concat([fixed, features], ignore_index=True)):
        result = sf.point_in_time_join(entity([(*pair, TODAY + timedelta(hours=1))]), table, TTL)
        assert result["units_mean_7d"].iloc[0] == pytest.approx(77.0)


def test_join_does_not_leak_the_label_day():
    # продажи дня, для которого строится прогноз, в признак попасть не должны
    sales = one_pair_sales([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000])
    label_day = sales["date"].iloc[-1]
    requests = entity([("S01", "Мышь", label_day.tz_localize("UTC"))])
    result = sf.point_in_time_join(requests, sf.build_pair_features(sales), TTL)
    assert result["units_mean_7d"].iloc[0] == pytest.approx(np.mean([3, 4, 5, 6, 7, 8, 9]))
    assert result["units_mean_28d"].iloc[0] == pytest.approx(np.mean([1, 2, 3, 4, 5, 6, 7, 8, 9]))


# ---------------------------------------------------------------------------------------------
# validate_features
# ---------------------------------------------------------------------------------------------


def test_validate_good_table(features):
    assert sf.validate_features(features, NOW) == {"success": True, "failed": [], "rows": len(features)}


def test_validate_accepts_any_column_order_and_few_gaps(features):
    features.loc[3, "units_mean_7d"] = np.nan  # 1 из 40 — меньше MAX_NULL_SHARE
    report = sf.validate_features(features[sorted(features.columns)], NOW)
    assert report["success"] is True and report["failed"] == []


def drop_column(frame):
    return frame.drop(columns="units_mean_28d")


def extra_column(frame):
    return frame.assign(units_mean_14d=1.0)


def renamed_column(frame):
    return frame.rename(columns={"promo_share_28d": "promo_share"})


def feature_as_text(frame):
    return frame.assign(units_mean_7d=frame["units_mean_7d"].astype(str))


def naive_timestamps(frame):
    return frame.assign(event_timestamp=frame["event_timestamp"].dt.tz_localize(None))


def no_rows(frame):
    return frame.iloc[0:0]


def missing_key(frame):
    return frame.assign(store_id=frame["store_id"].where(frame.index != 5))


def duplicated_rows(frame):
    return pd.concat([frame, frame.tail(3)], ignore_index=True)


def negative_units(frame):
    return frame.assign(units_mean_7d=frame["units_mean_7d"].where(frame.index != 7, -1.0))


def units_in_wrong_scale(frame):
    return frame.assign(units_mean_28d=frame["units_mean_28d"] * 1000)


def share_in_percent(frame):
    return frame.assign(promo_share_28d=frame["promo_share_28d"] * 100)


def many_gaps(frame):
    return frame.assign(units_mean_28d=frame["units_mean_28d"].where(frame.index % 10 != 0))


def timestamp_from_future(frame):
    last_row = frame.index == frame.index[-1]
    return frame.assign(event_timestamp=frame["event_timestamp"].where(~last_row, TODAY + 2 * DAY))


def three_days_old(frame):
    return frame[frame["event_timestamp"] <= TODAY - 3 * DAY]


def old_and_duplicated(frame):
    return duplicated_rows(three_days_old(frame))


def gaps_and_negative(frame):
    return negative_units(many_gaps(frame))


@pytest.mark.parametrize(
    ("corrupt", "failed"),
    [
        (drop_column, ["schema"]),
        (extra_column, ["schema"]),
        (renamed_column, ["schema"]),
        (feature_as_text, ["schema"]),
        (naive_timestamps, ["schema"]),
        (no_rows, ["not_empty"]),
        (missing_key, ["keys_not_null"]),
        (duplicated_rows, ["unique_keys"]),
        (negative_units, ["ranges"]),
        (units_in_wrong_scale, ["ranges"]),
        (share_in_percent, ["ranges"]),
        (many_gaps, ["null_share"]),
        (timestamp_from_future, ["no_future"]),
        (three_days_old, ["freshness"]),
        (old_and_duplicated, ["unique_keys", "freshness"]),
        (gaps_and_negative, ["ranges", "null_share"]),
    ],
    ids=lambda value: value.__name__ if callable(value) else None,
)
def test_validate_reports_failed_checks(features, corrupt, failed):
    bad = corrupt(features)
    report = sf.validate_features(bad, NOW)
    assert report == {"success": False, "failed": failed, "rows": len(bad)}
    assert set(failed) <= set(sf.CHECKS)


def test_validate_freshness_depends_on_now_and_max_age(features):
    assert sf.validate_features(features, NOW + 3 * DAY)["failed"] == ["freshness"]
    assert sf.validate_features(features, NOW + 3 * DAY, max_age=timedelta(days=4))["success"] is True
    assert sf.validate_features(features, NOW - DAY)["failed"] == ["no_future"]


def test_validate_does_not_modify_table(features):
    bad = gaps_and_negative(features)
    before = bad.copy()
    sf.validate_features(bad, NOW)
    pd.testing.assert_frame_equal(bad, before)


# ---------------------------------------------------------------------------------------------
# create_store
# ---------------------------------------------------------------------------------------------


def test_store_registers_entities_view_and_service(tmp_path):
    store = sf.create_store(repo_dir(tmp_path))
    assert store.project == sf.PROJECT
    assert {item.name: item.join_key for item in store.list_entities()} == {"store": "store_id", "product": "product"}

    view = store.get_feature_view(sf.FEATURE_VIEW)
    assert [field.name for field in view.features] == sf.FEATURE_NAMES
    assert {str(field.dtype) for field in view.features} == {"Float64"}
    assert set(view.entities) == {"store", "product"}
    assert view.ttl == sf.DEFAULT_TTL
    assert view.online is True
    assert view.batch_source.timestamp_field == "event_timestamp"
    assert view.batch_source.created_timestamp_column == "created"
    assert Path(view.batch_source.path) == (repo_dir(tmp_path) / sf.OFFLINE_FILE).resolve()

    service = store.get_feature_service(sf.FEATURE_SERVICE)
    projections = {
        projection.name: [field.name for field in projection.features]
        for projection in service.feature_view_projections
    }
    assert projections == {sf.FEATURE_VIEW: sf.FEATURE_NAMES}


def test_store_keeps_everything_inside_repo_dir(tmp_path, monkeypatch, features):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    store = sf.create_store(repo_dir(tmp_path))
    sf.publish_features(store, features, NOW)
    sf.get_training_frame(store, entity([(*PAIRS[0], TODAY)]))
    sf.online_features(store, ROWS, NOW)
    assert list(workdir.iterdir()) == []
    assert {"registry.db", "online_store.db", sf.OFFLINE_FILE} <= {path.name for path in repo_dir(tmp_path).iterdir()}


def test_store_starts_with_empty_typed_offline_table(tmp_path):
    sf.create_store(repo_dir(tmp_path))
    table = pd.read_parquet(repo_dir(tmp_path) / sf.OFFLINE_FILE)
    assert len(table) == 0
    assert set(table.columns) == set(sf.FEATURE_COLUMNS)
    assert sf.validate_features(table, NOW)["failed"] == ["not_empty"]


def test_store_ttl_is_a_parameter(tmp_path):
    store = sf.create_store(repo_dir(tmp_path), ttl=timedelta(hours=36))
    assert store.get_feature_view(sf.FEATURE_VIEW).ttl == timedelta(hours=36)


def test_store_can_be_reopened_without_losing_data(tmp_path, features):
    first = sf.create_store(repo_dir(tmp_path))
    fill_offline(tmp_path, features)
    fill_online(first, features)
    offline_before = (repo_dir(tmp_path) / sf.OFFLINE_FILE).read_bytes()

    second = sf.create_store(repo_dir(tmp_path))

    assert (repo_dir(tmp_path) / sf.OFFLINE_FILE).read_bytes() == offline_before
    assert raw_online(second, ROWS[:1])["units_mean_7d"] == pytest.approx([value_at(features, PAIRS[0], TODAY)])
    assert len(second.list_feature_views()) == 1 and len(second.list_feature_services()) == 1


# ---------------------------------------------------------------------------------------------
# get_training_frame
# ---------------------------------------------------------------------------------------------


def training_requests(features, seed=3):
    """Запросы вразнобой: все пары, разные моменты, включая слишком ранние и слишком поздние."""
    rng = np.random.default_rng(seed)
    rows = [
        (*pair, TODAY - timedelta(days=int(days), hours=int(hours)))
        for pair in [*PAIRS, ("S99", "Мышь")]
        for days, hours in zip(rng.integers(-4, 13, 12), rng.integers(0, 24, 12), strict=True)
    ]
    frame = entity(rows).sample(frac=1, random_state=seed)
    frame["units"] = rng.poisson(6, len(frame)).astype("float64")
    return frame


def test_training_frame_matches_point_in_time_join(tmp_path, features):
    store = sf.create_store(repo_dir(tmp_path))
    fill_offline(tmp_path, features)
    requests = training_requests(features)
    before = requests.copy()

    result = sf.get_training_frame(store, requests)

    expected = sf.point_in_time_join(requests, features, TTL)
    assert 0 < expected["units_mean_7d"].isna().sum() < len(expected)
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(requests, before)


def test_training_frame_keeps_rows_order_index_and_columns(tmp_path, features):
    store = sf.create_store(repo_dir(tmp_path))
    fill_offline(tmp_path, features)
    requests = training_requests(features)
    result = sf.get_training_frame(store, requests)
    assert list(result.columns) == [*requests.columns, *sf.FEATURE_NAMES]
    pd.testing.assert_frame_equal(result[list(requests.columns)], requests)
    known = result[result["store_id"] != "S99"].iloc[0]
    assert result.loc[result["store_id"] == "S99", sf.FEATURE_NAMES].isna().all().all()
    stamp = known["event_timestamp"].floor("D")
    if TODAY - 9 * DAY <= stamp <= TODAY + TTL:
        expected = value_at(features, (known["store_id"], known["product"]), min(stamp, TODAY))
        assert known["units_mean_7d"] == pytest.approx(expected)


def test_training_frame_restores_order_of_large_request(tmp_path):
    # на больших запросах офлайн-хранилище возвращает строки в произвольном порядке
    features = make_features(days=300)
    store = sf.create_store(repo_dir(tmp_path))
    fill_offline(tmp_path, features)
    rng = np.random.default_rng(1)
    shifts = zip(rng.integers(-4, 303, 600), rng.integers(0, 24, 600), strict=True)
    requests = entity(
        [(*PAIRS[i % 4], TODAY - timedelta(days=int(days), hours=int(hours))) for i, (days, hours) in enumerate(shifts)]
    )
    requests["units"] = rng.poisson(6, len(requests)).astype("float64")
    result = sf.get_training_frame(store, requests)
    pd.testing.assert_frame_equal(result, sf.point_in_time_join(requests, features, TTL))


def test_training_frame_uses_ttl_of_the_view(tmp_path, features):
    store = sf.create_store(repo_dir(tmp_path), ttl=timedelta(days=4))
    fill_offline(tmp_path, features)
    requests = entity([(*PAIRS[1], TODAY + 3 * DAY), (*PAIRS[1], TODAY + 5 * DAY)])
    result = sf.get_training_frame(store, requests)
    assert result["units_mean_7d"].iloc[0] == pytest.approx(value_at(features, PAIRS[1], TODAY))
    assert np.isnan(result["units_mean_7d"].iloc[1])


def test_training_frame_does_not_leak_the_label_day(tmp_path):
    sales = one_pair_sales([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000])
    store = published(tmp_path, sf.build_pair_features(sales))
    requests = entity([("S01", "Мышь", sales["date"].iloc[-1].tz_localize("UTC"))])
    requests["units"] = [1000.0]
    result = sf.get_training_frame(store, requests)
    assert result["units_mean_7d"].iloc[0] == pytest.approx(np.mean([3, 4, 5, 6, 7, 8, 9]))


def test_training_frame_from_empty_store(tmp_path):
    store = sf.create_store(repo_dir(tmp_path))
    result = sf.get_training_frame(store, entity([(*PAIRS[0], TODAY)]))
    assert len(result) == 1 and result[sf.FEATURE_NAMES].isna().all().all()


# ---------------------------------------------------------------------------------------------
# publish_features
# ---------------------------------------------------------------------------------------------


def test_publish_fills_offline_and_online_stores(tmp_path, features):
    store = sf.create_store(repo_dir(tmp_path))
    report = sf.publish_features(store, features, NOW)
    assert report == {"success": True, "failed": [], "rows": len(features)}

    offline = pd.read_parquet(repo_dir(tmp_path) / sf.OFFLINE_FILE)
    assert str(offline["event_timestamp"].dtype) == sf.TIMESTAMP_DTYPE
    pd.testing.assert_frame_equal(offline[sf.FEATURE_COLUMNS], features)

    online = raw_online(store, ROWS)
    for name in sf.FEATURE_NAMES:
        assert online[name] == pytest.approx([value_at(features, pair, TODAY, name) for pair in PAIRS])


def test_publish_online_value_is_the_latest_known_at_now(tmp_path, features):
    # публикация «вчера утром»: строки с сегодняшней отметкой ещё не существуют
    yesterday = features[features["event_timestamp"] <= TODAY - DAY]
    store = sf.create_store(repo_dir(tmp_path))
    sf.publish_features(store, yesterday, NOW - DAY)
    assert raw_online(store, ROWS)["units_mean_7d"] == pytest.approx(
        [value_at(features, pair, TODAY - DAY) for pair in PAIRS]
    )


@pytest.mark.parametrize(
    ("corrupt", "check"),
    [
        (units_in_wrong_scale, "ranges"),
        (duplicated_rows, "unique_keys"),
        (three_days_old, "freshness"),
        (drop_column, "schema"),
    ],
    ids=lambda value: value.__name__ if callable(value) else None,
)
def test_publish_rejects_bad_table_and_changes_nothing(tmp_path, features, corrupt, check):
    yesterday = features[features["event_timestamp"] <= TODAY - DAY]
    store = sf.create_store(repo_dir(tmp_path))
    sf.publish_features(store, yesterday, NOW - DAY)
    offline_before = (repo_dir(tmp_path) / sf.OFFLINE_FILE).read_bytes()
    online_before = raw_online(store, ROWS)

    with pytest.raises(sf.DataValidationError, match=check):
        sf.publish_features(store, corrupt(features), NOW)

    assert (repo_dir(tmp_path) / sf.OFFLINE_FILE).read_bytes() == offline_before
    assert raw_online(store, ROWS) == online_before


def test_publish_rejected_first_table_leaves_store_empty(tmp_path, features):
    store = sf.create_store(repo_dir(tmp_path))
    with pytest.raises(sf.DataValidationError):
        sf.publish_features(store, share_in_percent(features), NOW)
    assert len(pd.read_parquet(repo_dir(tmp_path) / sf.OFFLINE_FILE)) == 0
    assert raw_online(store, ROWS)["units_mean_7d"] == [None] * len(ROWS)


def test_publish_next_day_replaces_values(tmp_path, features):
    yesterday = features[features["event_timestamp"] <= TODAY - DAY]
    store = sf.create_store(repo_dir(tmp_path))
    sf.publish_features(store, yesterday, NOW - DAY)
    sf.publish_features(store, features, NOW)
    assert len(pd.read_parquet(repo_dir(tmp_path) / sf.OFFLINE_FILE)) == len(features)
    assert raw_online(store, ROWS)["units_mean_7d"] == pytest.approx(
        [value_at(features, pair, TODAY) for pair in PAIRS]
    )


def test_publish_max_age_is_passed_to_validation(tmp_path, features):
    store = sf.create_store(repo_dir(tmp_path))
    with pytest.raises(sf.DataValidationError, match="freshness"):
        sf.publish_features(store, features, NOW + 36 * timedelta(hours=1))
    report = sf.publish_features(store, features, NOW + 36 * timedelta(hours=1), max_age=timedelta(days=2))
    assert report["success"] is True


# ---------------------------------------------------------------------------------------------
# online_features
# ---------------------------------------------------------------------------------------------


def test_online_returns_fresh_values_in_request_order(tmp_path, features):
    store = sf.create_store(repo_dir(tmp_path))
    fill_online(store, features)
    rows = [ROWS[2], ROWS[0], ROWS[3]]
    result = sf.online_features(store, rows, NOW)

    assert list(result.columns) == [*sf.KEYS, *sf.FEATURE_NAMES, "event_timestamp", "status"]
    assert isinstance(result.index, pd.RangeIndex) and len(result) == 3
    assert result[sf.KEYS].to_dict("records") == rows
    for name in sf.FEATURE_NAMES:
        assert result[name].dtype == np.float64
        assert result[name].tolist() == pytest.approx([value_at(features, PAIRS[i], TODAY, name) for i in (2, 0, 3)])
    assert result["status"].tolist() == ["fresh"] * 3
    assert result["event_timestamp"].tolist() == [TODAY] * 3
    assert str(result["event_timestamp"].dt.tz) == "UTC"


def test_online_unknown_pair_is_missing(tmp_path, features):
    store = sf.create_store(repo_dir(tmp_path))
    fill_online(store, features)
    result = sf.online_features(store, [{"store_id": "S99", "product": "Мышь"}, ROWS[1]], NOW)
    assert result["status"].tolist() == ["missing", "fresh"]
    assert result.loc[0, sf.FEATURE_NAMES].isna().all()
    assert pd.isna(result.loc[0, "event_timestamp"])
    assert result.loc[1, "units_mean_7d"] == pytest.approx(value_at(features, PAIRS[1], TODAY))


def test_online_empty_store_gives_missing(tmp_path):
    store = sf.create_store(repo_dir(tmp_path))
    result = sf.online_features(store, ROWS, NOW)
    assert result["status"].tolist() == ["missing"] * len(ROWS)
    assert result[sf.FEATURE_NAMES].isna().all().all()


def test_online_stale_values_are_hidden(tmp_path, features):
    store = sf.create_store(repo_dir(tmp_path))
    fill_online(store, features)
    # Feast отдаёт значение любого возраста: убеждаемся, что прячет его именно online_features
    assert raw_online(store, ROWS[:1])["units_mean_7d"][0] is not None

    result = sf.online_features(store, ROWS, NOW + 5 * DAY)

    assert result["status"].tolist() == ["stale"] * len(ROWS)
    assert result[sf.FEATURE_NAMES].isna().all().all()
    assert result["event_timestamp"].tolist() == [TODAY] * len(ROWS)


def test_online_ttl_bound_is_inclusive(tmp_path, features):
    store = sf.create_store(repo_dir(tmp_path), ttl=timedelta(hours=30))
    fill_online(store, features)
    edge = (TODAY + timedelta(hours=30)).to_pydatetime()
    assert sf.online_features(store, ROWS[:1], edge)["status"].tolist() == ["fresh"]
    assert sf.online_features(store, ROWS[:1], edge + timedelta(seconds=1))["status"].tolist() == ["stale"]


def test_online_statuses_are_set_per_pair(tmp_path, features):
    store = sf.create_store(repo_dir(tmp_path))
    old = PAIRS[1]
    is_old = (features["store_id"] == old[0]) & (features["product"] == old[1])
    mixed = features[~is_old | (features["event_timestamp"] <= TODAY - 4 * DAY)]  # у одной пары обновления прекратились
    fill_online(store, mixed)
    result = sf.online_features(store, [*ROWS, {"store_id": "S02", "product": "Чайник"}], NOW)
    assert result["status"].tolist() == ["fresh", "stale", "fresh", "fresh", "missing"]
    assert result["event_timestamp"].iloc[1] == TODAY - 4 * DAY
    assert result["units_mean_7d"].isna().tolist() == [False, True, False, False, True]


def test_online_missing_feature_value_is_not_missing_pair(tmp_path, features):
    store = sf.create_store(repo_dir(tmp_path))
    last = (
        (features["store_id"] == PAIRS[0][0])
        & (features["product"] == PAIRS[0][1])
        & (features["event_timestamp"] == TODAY)
    )
    features.loc[last, "units_mean_28d"] = np.nan
    fill_online(store, features)
    result = sf.online_features(store, ROWS[:1], NOW)
    assert result["status"].tolist() == ["fresh"]
    assert np.isnan(result["units_mean_28d"].iloc[0])
    assert result["units_mean_7d"].iloc[0] == pytest.approx(value_at(features, PAIRS[0], TODAY))


# ---------------------------------------------------------------------------------------------
# check_consistency
# ---------------------------------------------------------------------------------------------

MISMATCH_COLUMNS = [*sf.KEYS, "feature", "online", "offline"]


def test_consistency_after_publish_has_no_mismatches(tmp_path, features):
    store = published(tmp_path, features)
    result = sf.check_consistency(store, ROWS, NOW)
    assert list(result.columns) == MISMATCH_COLUMNS
    assert len(result) == 0


def test_consistency_finds_value_written_to_online_store_only(tmp_path, features):
    store = published(tmp_path, features)
    pair = PAIRS[2]
    patch = latest_rows(features)
    patch = patch[(patch["store_id"] == pair[0]) & (patch["product"] == pair[1])].assign(
        units_mean_7d=55.5, promo_share_28d=0.75
    )
    store.write_to_online_store(sf.FEATURE_VIEW, patch)

    result = sf.check_consistency(store, ROWS, NOW)

    assert list(result.columns) == MISMATCH_COLUMNS
    assert isinstance(result.index, pd.RangeIndex)
    assert result[[*sf.KEYS, "feature"]].to_dict("records") == [
        {"store_id": pair[0], "product": pair[1], "feature": "units_mean_7d"},
        {"store_id": pair[0], "product": pair[1], "feature": "promo_share_28d"},
    ]
    assert result["online"].tolist() == pytest.approx([55.5, 0.75])
    assert result["offline"].tolist() == pytest.approx(
        [value_at(features, pair, TODAY), value_at(features, pair, TODAY, "promo_share_28d")]
    )


def test_consistency_finds_offline_table_changed_without_publishing(tmp_path, features):
    store = published(tmp_path, features)
    pair = PAIRS[3]
    last = (features["store_id"] == pair[0]) & (features["product"] == pair[1]) & (features["event_timestamp"] == TODAY)
    corrected = features.copy()
    corrected.loc[last, "units_mean_28d"] = 3.25
    fill_offline(tmp_path, corrected)  # историю пересчитали, а онлайн-хранилище не обновили

    result = sf.check_consistency(store, [ROWS[0], ROWS[3]], NOW)

    assert result[[*sf.KEYS, "feature"]].to_dict("records") == [
        {"store_id": pair[0], "product": pair[1], "feature": "units_mean_28d"}
    ]
    assert result["online"].tolist() == pytest.approx([value_at(features, pair, TODAY, "units_mean_28d")])
    assert result["offline"].tolist() == pytest.approx([3.25])


def test_consistency_missing_on_one_side_is_a_mismatch(tmp_path, features):
    store = sf.create_store(repo_dir(tmp_path))
    fill_offline(tmp_path, features)  # в офлайн-таблице данные есть, материализации не было
    result = sf.check_consistency(store, ROWS[:1], NOW)
    assert result["feature"].tolist() == sf.FEATURE_NAMES
    assert result["online"].isna().all()
    assert result["offline"].tolist() == pytest.approx(
        [value_at(features, PAIRS[0], TODAY, name) for name in sf.FEATURE_NAMES]
    )


def test_consistency_stale_and_unknown_pairs_agree(tmp_path, features):
    store = published(tmp_path, features)
    rows = [*ROWS, {"store_id": "S99", "product": "Мышь"}]
    # через пять дней значение устарело и для обучения (ttl), и для прогноза (stale): расхождения нет
    assert len(sf.check_consistency(store, rows, NOW + 5 * DAY)) == 0
    assert len(sf.check_consistency(store, rows, NOW)) == 0
