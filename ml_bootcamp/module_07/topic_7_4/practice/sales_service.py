"""Сервис прогноза продаж: практика 7.4.4.

Четыре функции превращают каталог модели в HTTP-сервис:
прочитать модель и её описание -> собрать таблицу признаков из строк запроса -> получить прогнозы
и предупреждения -> собрать приложение FastAPI с проверкой входа, готовностью и метриками.

Каталог модели содержит два файла: MODEL_FILE (Pipeline scikit-learn, сохранённый joblib) и METADATA_FILE
(JSON-объект с ключами METADATA_KEYS: версия модели, список признаков, известные магазины и товары).
Его создаёт готовый скрипт train_model.py.

Строка запроса — словарь с ключами "store_id", "product", "date" (datetime.date), "price", "promo".

Готовые части менять не нужно: константы, ModelLoadError и LoadedModel. Реализуйте функции так, чтобы
проходили тесты из tests/test_sales_service.py.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from prometheus_client import CollectorRegistry

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
# Обязательные ключи описания модели
METADATA_KEYS = ("version", "features", "stores", "products")
# Признаки, которые умеет строить build_frame
KNOWN_FEATURES = ("store_id", "product", "price", "promo", "dow", "month")
# Ограничения запроса
STORE_PATTERN = r"^S\d{2}$"
MAX_PRODUCT_LENGTH = 50
MAX_PRICE = 1_000_000
MAX_BATCH = 1000
# Границы корзин гистограмм: время обработки запроса (секунды) и значения прогноза (штуки)
DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
UNITS_BUCKETS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
# Тексты ответов об ошибках
NOT_READY_DETAIL = "модель не загружена"
ERROR_DETAIL = "внутренняя ошибка сервиса"
APP_TITLE = "Прогноз продаж"


class ModelLoadError(Exception):
    """Каталог модели нельзя прочитать: нет файла, испорчено описание, не хватает ключей."""


@dataclass
class LoadedModel:
    """Модель и её описание в памяти процесса."""

    pipeline: Any  # объект с методом predict(DataFrame) -> массив чисел
    version: str
    features: list[str]
    stores: frozenset[str]
    products: frozenset[str]


def load_model(model_dir: Path) -> LoadedModel:
    """Читает модель и её описание из каталога model_dir (Path или строка).

    METADATA_FILE — текст UTF-8 с JSON-объектом; из него берутся:
        "version"  -> LoadedModel.version, всегда строка (12 -> "12");
        "features" -> LoadedModel.features, список в том же порядке;
        "stores"   -> LoadedModel.stores, frozenset;
        "products" -> LoadedModel.products, frozenset.
    Другие ключи описания игнорируются. MODEL_FILE читается joblib.load -> LoadedModel.pipeline.

    ModelLoadError, если: каталога или любого из двух файлов нет; описание — не JSON или не объект
    (список, строка); в описании нет одного из METADATA_KEYS (имя ключа есть в тексте ошибки);
    файл модели не читается. Исключения других типов наружу не выходят.
    """
    raise NotImplementedError


def build_frame(rows: list[dict], features: list[str]) -> pd.DataFrame:
    """Собирает таблицу признаков модели из строк запроса.

    rows — непустой список строк запроса (см. описание модуля). features — имена столбцов результата;
    каждое должно входить в KNOWN_FEATURES.

    Результат — DataFrame: по строке на строку запроса в том же порядке, индекс — RangeIndex, столбцы —
    ровно features в заданном порядке. Значения столбцов:
        "store_id", "product" — как в запросе;
        "price" — float64;
        "promo" — int64: 1, если акция, иначе 0;
        "dow"   — int64: день недели даты, понедельник — 0;
        "month" — int64: месяц даты, 1…12.

    ValueError, если rows пуст или в features есть имя вне KNOWN_FEATURES (имя есть в тексте ошибки).
    Словари rows не изменяются.
    """
    raise NotImplementedError


def predict_rows(model: LoadedModel, rows: list[dict]) -> list[dict]:
    """Возвращает прогноз и предупреждения для каждой строки запроса.

    Таблица признаков строится build_frame(rows, model.features) и передаётся в model.pipeline.predict
    ОДНИМ вызовом для всех строк.

    Результат — список словарей в порядке rows, в каждом ровно два ключа:
        "units"    — float: прогноз, не меньше 0.0 (отрицательные заменяются нулём), округлён
                     до трёх знаков после запятой;
        "warnings" — список строк: "unknown store_id", если магазина нет в model.stores;
                     "unknown product", если товара нет в model.products; порядок именно такой;
                     пустой список, если всё известно.

    Исключения model.pipeline.predict не перехватываются.
    """
    raise NotImplementedError


def create_app(model_dir: Path, registry: CollectorRegistry | None = None) -> FastAPI:
    """Собирает приложение FastAPI для модели из каталога model_dir.

    Модель
        Создание приложения модель не читает. Она загружается один раз при старте (lifespan) вызовом
        load_model(model_dir) — именно через имя модуля, чтобы функцию можно было подменить, —
        и хранится в app.state.model. Если load_model бросил ModelLoadError, приложение всё равно
        стартует, а app.state.model равен None. Обработчики берут модель из app.state.model при
        каждом запросе: подмена объекта в app.state действует сразу. Заголовок приложения — APP_TITLE.

    Эндпоинты
        GET  /health         200 {"status": "ok"} — всегда, и без модели.
        GET  /ready          200 {"status": "ready", "model_version": <версия>};
                             без модели — 503 {"detail": NOT_READY_DETAIL}.
        POST /predict        тело — строка запроса (см. ниже);
                             200 {"units": <float>, "model_version": <версия>, "warnings": [...]}.
        POST /predict/batch  тело — {"items": [<строка запроса>, ...]}, от 1 до MAX_BATCH строк;
                             200 {"predictions": [{"units": ..., "warnings": [...]}, ...],
                                  "model_version": <версия>}; модель вызывается один раз на пакет.
        GET  /metrics        метрики в текстовом формате Prometheus; в описании OpenAPI отсутствует.
        Значения units и warnings — из predict_rows. Без модели верный запрос к /predict
        и /predict/batch получает 503 {"detail": NOT_READY_DETAIL}. Если модель бросила исключение —
        500 {"detail": ERROR_DETAIL}: ни текста исключения, ни трассировки в ответе нет, сервис
        продолжает работать. В описании OpenAPI у /ready, /predict и /predict/batch указан ответ 503.

    Строка запроса (JSON-объект)
        "store_id" — строка, соответствует STORE_PATTERN;
        "product"  — строка длиной от 1 до MAX_PRODUCT_LENGTH;
        "date"     — дата в формате ISO: "2025-01-15";
        "price"    — число больше 0 и не больше MAX_PRICE; строка с числом не принимается;
        "promo"    — true или false; строки и числа не принимаются; по умолчанию false.
        Первые четыре поля обязательны. Лишние поля — ошибка, и в строке запроса, и в теле пакета.
        Запрос, не прошедший проверку, получает стандартный ответ FastAPI: код 422 и список "detail",
        где "loc" каждой ошибки оканчивается именем поля (в пакете — "items", номер строки, имя поля).
        До модели такой запрос не доходит.

    Метрики
        Все метрики регистрируются в registry; если он не передан — в новом CollectorRegistry(),
        своём у каждого приложения (не в общем реестре prometheus_client.REGISTRY). /metrics отдаёт
        generate_latest этого реестра.
        sales_requests_total{endpoint, status}        Counter: каждый запрос, кроме запросов к /metrics.
            endpoint — шаблон маршрута ("/predict", "/predict/batch", "/health", "/ready"); для адресов,
            у которых маршрута нет, — "other". status — код ответа строкой, включая 404, 422, 500, 503.
        sales_request_duration_seconds{endpoint}      Histogram с границами DURATION_BUCKETS: время
            обработки тех же запросов.
        sales_predictions_total                       Counter: число строк, для которых выдан прогноз.
        sales_prediction_units                        Histogram с границами UNITS_BUCKETS: значения units
            выданных прогнозов.
        sales_promo_rows_total                        Counter: строки с promo = true среди выданных.
        sales_unknown_category_total{field}           Counter: строки с предупреждением; field —
            "store_id" или "product".
        sales_model_info{version}                     Gauge: 1 для версии загруженной модели; пока
            модель не загружена, ряда нет.
        Запрос, завершившийся 422, 503 или 500, метрики прогнозов не меняет.
    """
    raise NotImplementedError
