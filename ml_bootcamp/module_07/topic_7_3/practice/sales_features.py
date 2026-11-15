"""Хранилище признаков для прогноза продаж: практика 7.3.4.

Восемь функций проводят признак от таблицы продаж до модели и сервиса:
построить таблицу признаков с отметками времени -> проверить её -> опубликовать в Feast
(офлайн-таблица и онлайн-хранилище) -> получить обучающую выборку без утечки -> получить
признаки для прогноза с контролем свежести -> сверить онлайн- и офлайн-значения.

Термины:
- пара — сочетание (store_id, product), ключ признаков;
- отметка времени признака (event_timestamp) — момент, начиная с которого значение известно:
  статистика «на конец дня D» получает отметку 00:00 UTC дня D+1;
- момент запроса — event_timestamp строки, для которой нужны признаки (обучающий пример или прогноз);
- значение актуально в момент t, если t - ttl <= event_timestamp <= t (обе границы включены).

Все отметки времени — с часовым поясом UTC. Параметр now везде передаётся явно: функции не читают
системные часы.

Готовые части менять не нужно: константы и DataValidationError. Реализуйте функции так, чтобы
проходили тесты из tests/test_sales_features.py.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from feast import FeatureStore

PROJECT = "sales"
KEYS = ["store_id", "product"]
# Признаки пары и порядок столбцов таблицы признаков
FEATURE_NAMES = ["units_mean_7d", "units_mean_28d", "promo_share_28d"]
FEATURE_COLUMNS = [*KEYS, "event_timestamp", "created", *FEATURE_NAMES]
# Тип отметок времени в таблице признаков (микросекунды: наносекунды DuckDB в Feast не читает)
TIMESTAMP_DTYPE = "datetime64[us, UTC]"
# Задержка загрузки: created = event_timestamp + LOAD_DELAY
LOAD_DELAY = timedelta(hours=3)
# Имена объектов Feast
FEATURE_VIEW = "pair_stats"
FEATURE_SERVICE = "sales_forecast_v1"
# Файл офлайн-таблицы внутри каталога хранилища
OFFLINE_FILE = "pair_stats.parquet"
DEFAULT_TTL = timedelta(days=2)
# Проверки validate_features в порядке выполнения
CHECKS = ("schema", "not_empty", "keys_not_null", "unique_keys", "ranges", "null_share", "no_future", "freshness")
# Наибольшее допустимое значение средних продаж и наибольшая доля пропусков в столбце признака
MAX_UNITS = 1000.0
MAX_NULL_SHARE = 0.05
# Допуск сравнения онлайн- и офлайн-значений
TOLERANCE = 1e-9


class DataValidationError(ValueError):
    """Таблица признаков не прошла проверку."""


def build_pair_features(sales: pd.DataFrame) -> pd.DataFrame:
    """Строит таблицу признаков пар из дневных продаж.

    sales — столбцы "date" (datetime64 без часового пояса, полночь), "store_id", "product",
    "units" (float, возможны пропуски), "promo" (bool); порядок строк произвольный, другие столбцы
    игнорируются. Дни могут отсутствовать.

    Результат — по одной строке на каждую строку sales, столбцы FEATURE_COLUMNS в этом порядке:
        "event_timestamp" — date + 1 день, тип TIMESTAMP_DTYPE: статистика за день D известна
                            с 00:00 UTC дня D+1;
        "created"         — event_timestamp + LOAD_DELAY, тип TIMESTAMP_DTYPE;
        "units_mean_7d"   — среднее units пары за календарные дни D-6 … D (D включён); пропуски
                            и отсутствующие дни не учитываются; если значений нет — NaN;
        "units_mean_28d"  — то же за дни D-27 … D;
        "promo_share_28d" — доля строк с promo за дни D-27 … D среди имеющихся строк пары.
    Все три признака — float64. Строки отсортированы по (store_id, product, event_timestamp),
    индекс — RangeIndex.

    Значение за день D не зависит от строк с датой позже D. Повтор (date, store_id, product)
    в sales — ValueError. Таблица sales не изменяется.
    """
    raise NotImplementedError


def point_in_time_join(entity_df: pd.DataFrame, features: pd.DataFrame, ttl: timedelta = DEFAULT_TTL) -> pd.DataFrame:
    """Присоединяет к строкам entity_df признаки, актуальные в момент запроса. Только pandas, без Feast.

    entity_df — столбцы "store_id", "product", "event_timestamp" (момент запроса, UTC) и любые
    другие. features — таблица со столбцами FEATURE_COLUMNS.

    Для каждой строки entity_df берётся строка features той же пары с наибольшей отметкой времени
    среди актуальных: t - ttl <= event_timestamp <= t, где t — момент запроса. Если у пары
    несколько строк с одной отметкой времени, берётся строка с наибольшим "created". Если
    актуальной строки нет — признаки равны NaN.

    Результат — копия entity_df (те же строки, порядок, индекс и столбцы) с добавленными справа
    столбцами FEATURE_NAMES. Входные таблицы не изменяются.
    """
    raise NotImplementedError


def validate_features(features: pd.DataFrame, now: datetime, max_age: timedelta = timedelta(days=1)) -> dict:
    """Проверяет таблицу признаков перед публикацией. Ничего не записывает и не изменяет.

    Возвращает словарь {"success": bool, "failed": [...], "rows": int}: "failed" — имена
    непройденных проверок в порядке CHECKS, "success" — True, если список пуст, "rows" — число строк.

    Проверки:
        "schema"        — набор столбцов равен FEATURE_COLUMNS (порядок не важен); столбцы FEATURE_NAMES
                          числовые с плавающей точкой; "event_timestamp" — отметки времени с часовым
                          поясом. Если проверка не пройдена, остальные не выполняются: failed == ["schema"];
        "not_empty"     — в таблице есть строки. Для пустой таблицы failed == ["not_empty"];
        "keys_not_null" — нет пропусков в "store_id", "product", "event_timestamp";
        "unique_keys"   — нет повторов (store_id, product, event_timestamp);
        "ranges"        — непустые значения "units_mean_7d" и "units_mean_28d" лежат в [0, MAX_UNITS],
                          "promo_share_28d" — в [0, 1];
        "null_share"    — доля пропусков в каждом столбце FEATURE_NAMES не больше MAX_NULL_SHARE;
        "no_future"     — нет отметок времени позже now;
        "freshness"     — наибольшая отметка времени не раньше now - max_age.

    Реализация свободная: pandas, pandera или Great Expectations — тесты проверяют только результат.
    """
    raise NotImplementedError


def create_store(repo_dir: str | Path, ttl: timedelta = DEFAULT_TTL) -> FeatureStore:
    """Создаёт (или открывает) локальное хранилище признаков Feast в каталоге repo_dir.

    Конфигурация — объектом RepoConfig, без файла feature_store.yaml: проект PROJECT, провайдер
    "local", реестр — файл repo_dir/registry.db, онлайн-хранилище — SQLite в файле
    repo_dir/online_store.db, офлайн-хранилище — {"type": "duckdb"},
    entity_key_serialization_version=3. Все пути — абсолютные.

    В хранилище зарегистрированы:
        сущности "store" (ключ соединения "store_id") и "product" (ключ "product"), значения — строки;
        источник — FileSource на файл repo_dir/OFFLINE_FILE: отметка времени "event_timestamp",
            время загрузки "created";
        представление признаков FEATURE_VIEW: обе сущности, признаки FEATURE_NAMES типа Float64,
            заданный ttl, online=True;
        сервис признаков FEATURE_SERVICE со всеми признаками представления.

    Каталог repo_dir создаётся при необходимости. Если файла офлайн-таблицы ещё нет, создаётся пустая
    таблица со столбцами FEATURE_COLUMNS нужных типов. Существующий файл не изменяется. Повторный
    вызов для того же каталога не приводит к ошибке и не теряет опубликованные данные. Вне repo_dir
    ничего не создаётся.
    """
    raise NotImplementedError


def get_training_frame(store: FeatureStore, entity_df: pd.DataFrame) -> pd.DataFrame:
    """Обучающая выборка: признаки сервиса FEATURE_SERVICE на момент каждой строки entity_df.

    entity_df — как в point_in_time_join. Признаки берутся из офлайн-хранилища Feast
    (get_historical_features); актуальность определяет ttl представления признаков.

    Результат — те же строки в том же порядке и с тем же индексом, столбцы entity_df без изменений
    (включая тип "event_timestamp") и справа столбцы FEATURE_NAMES. Для опубликованной таблицы
    результат совпадает с point_in_time_join(entity_df, таблица, ttl представления).
    Таблица entity_df не изменяется.
    """
    raise NotImplementedError


def publish_features(
    store: FeatureStore, features: pd.DataFrame, now: datetime, max_age: timedelta = timedelta(days=1)
) -> dict:
    """Публикует таблицу признаков: проверка -> офлайн-таблица -> онлайн-хранилище.

    features — полная таблица признаков (вся история), now — момент публикации (UTC).

    1. validate_features(features, now, max_age). Если проверка не пройдена — DataValidationError,
       в сообщении перечислены имена непройденных проверок; офлайн-таблица и онлайн-хранилище
       остаются прежними.
    2. Таблица записывается в файл источника представления FEATURE_VIEW (parquet, без индекса,
       отметки времени типа TIMESTAMP_DTYPE), заменяя прежнее содержимое.
    3. В онлайн-хранилище записываются последние значения каждой пары с отметкой времени
       в интервале [now - ttl, now], где ttl — ttl представления.

    Возвращает отчёт validate_features.
    """
    raise NotImplementedError


def online_features(store: FeatureStore, entity_rows: list[dict], now: datetime) -> pd.DataFrame:
    """Признаки для прогноза из онлайн-хранилища с контролем свежести.

    entity_rows — список словарей {"store_id": ..., "product": ...}; now — момент запроса (UTC).

    Результат — по одной строке на элемент entity_rows в том же порядке, индекс RangeIndex, столбцы:
        "store_id", "product";
        FEATURE_NAMES — float64;
        "event_timestamp" — отметка времени значения в хранилище (UTC) или NaT, если пары нет;
        "status" — "fresh": значение актуально в момент now (now - event_timestamp <= ttl);
                   "stale": значение есть, но старше ttl — признаки заменяются на NaN;
                   "missing": пары нет в онлайн-хранилище — признаки NaN.

    Feast 0.68 сам возраст значения при чтении не проверяет: устаревшее значение приходит
    со статусом PRESENT. Поэтому свежесть определяется здесь — по отметке времени и ttl
    представления FEATURE_VIEW.
    """
    raise NotImplementedError


def check_consistency(store: FeatureStore, entity_rows: list[dict], now: datetime) -> pd.DataFrame:
    """Сверяет онлайн- и офлайн-значения признаков в момент now.

    Онлайн-значения — online_features(store, entity_rows, now); офлайн-значения —
    get_training_frame для тех же пар с моментом запроса now.

    Результат — таблица расхождений со столбцами "store_id", "product", "feature", "online",
    "offline": по строке на каждую пару и признак, где значения различаются больше чем на TOLERANCE
    или пропущено ровно одно из двух. Порядок строк — как в entity_rows, внутри пары — как
    в FEATURE_NAMES; индекс RangeIndex. Если расхождений нет — пустая таблица с теми же столбцами.
    """
    raise NotImplementedError
