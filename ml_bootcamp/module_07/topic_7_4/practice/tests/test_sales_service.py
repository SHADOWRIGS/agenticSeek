"""Тесты практики 7.4.4: исполняемая спецификация модуля sales_service.

Запуск из папки practice:  uv run pytest -q
Тесты не используют сеть и Docker: приложение вызывается через TestClient, каталог модели создаётся
во временном каталоге pytest. Модель в тестах — маленький Pipeline на синтетических данных.
"""

import datetime as dt
import json
import sys
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import prometheus_client
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))

import sales_service as ss  # noqa: E402

FEATURES = ["store_id", "product", "price", "promo", "dow", "month"]
STORES = ["S01", "S02", "S03", "S04"]
PRODUCTS = ["Мышь", "Чайник", "Рюкзак"]
VERSION = "7"
ITEM = {"store_id": "S01", "product": "Мышь", "date": "2025-01-15", "price": 1490.0, "promo": False}
ROW = {**ITEM, "date": dt.date(2025, 1, 15)}


def training_frame() -> pd.DataFrame:
    """Синтетические продажи: 4 магазина, 3 товара, 30 дней."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-11-01", periods=30, freq="D")
    frame = pd.DataFrame(
        [(day, store, product) for day in days for store in STORES for product in PRODUCTS],
        columns=["date", "store_id", "product"],
    )
    frame["promo"] = (rng.random(len(frame)) < 0.2).astype("int64")
    frame["price"] = frame["product"].map({"Мышь": 1490.0, "Чайник": 2990.0, "Рюкзак": 3990.0})
    frame["price"] = np.where(frame["promo"] == 1, frame["price"] * 0.85, frame["price"])
    frame["dow"] = frame["date"].dt.day_of_week.astype("int64")
    frame["month"] = frame["date"].dt.month.astype("int64")
    store_effect = frame["store_id"].map(dict(zip(STORES, [8.0, 6.0, 4.0, 3.0], strict=True)))
    frame["units"] = store_effect * (1 + frame["promo"]) + (frame["dow"] >= 5) * 2.0 + rng.normal(0, 0.3, len(frame))
    return frame


def write_model_dir(path: Path, metadata: dict | None = None) -> Path:
    """Каталог модели: Pipeline, обученный на синтетических данных, и описание."""
    frame = training_frame()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    pipeline = Pipeline(
        [
            ("prepare", ColumnTransformer([("category", encoder, ["store_id", "product"])], remainder="passthrough")),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(frame[FEATURES], frame["units"])
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path / ss.MODEL_FILE)
    if metadata is None:
        metadata = {"name": "sales-forecast", "version": VERSION, "features": FEATURES}
        metadata |= {"stores": STORES, "products": PRODUCTS}
    (path / ss.METADATA_FILE).write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory) -> Path:
    return write_model_dir(tmp_path_factory.mktemp("model"))


@pytest.fixture(scope="session")
def pipeline(model_dir):
    return joblib.load(model_dir / ss.MODEL_FILE)


def expected_units(pipeline, items: list[dict]) -> list[float]:
    """Прогнозы, посчитанные без функций решения."""
    frame = pd.DataFrame(items)
    dates = pd.to_datetime(frame["date"])
    frame["dow"] = dates.dt.day_of_week.astype("int64")
    frame["month"] = dates.dt.month.astype("int64")
    frame["promo"] = frame["promo"].astype("int64")
    frame["price"] = frame["price"].astype("float64")
    return [round(max(float(value), 0.0), 3) for value in pipeline.predict(frame[FEATURES])]


class FakePipeline:
    """Подмена модели: запоминает полученные таблицы и возвращает заданные значения."""

    def __init__(self, values=None, error: Exception | None = None):
        self.values = values
        self.error = error
        self.frames = []

    def predict(self, frame):
        self.frames.append(frame.copy())
        if self.error is not None:
            raise self.error
        if self.values is None:
            return np.arange(len(frame), dtype="float64")
        return np.asarray(self.values, dtype="float64")


def fake_model(pipeline, version: str = VERSION) -> "ss.LoadedModel":
    return ss.LoadedModel(pipeline, version, list(FEATURES), frozenset(STORES), frozenset(PRODUCTS))


def open_client(model_dir: Path, registry: CollectorRegistry | None = None, **kwargs) -> TestClient:
    """Клиент приложения; модель загружается при входе в with."""
    return TestClient(ss.create_app(model_dir, registry), **kwargs)


def sample(registry: CollectorRegistry, name: str, **labels) -> float:
    """Значение метрики из реестра; отсутствующий ряд считается нулём."""
    return registry.get_sample_value(name, labels) or 0.0


# --- load_model ---


def test_load_model_reads_pipeline_and_metadata(model_dir, pipeline):
    model = ss.load_model(model_dir)
    assert isinstance(model, ss.LoadedModel)
    assert model.version == VERSION
    assert model.features == FEATURES
    assert model.stores == frozenset(STORES)
    assert model.products == frozenset(PRODUCTS)
    frame = training_frame()[FEATURES].head(5)
    np.testing.assert_allclose(model.pipeline.predict(frame), pipeline.predict(frame))


def test_load_model_version_is_always_a_string(tmp_path):
    metadata = {"version": 12, "features": FEATURES, "stores": STORES, "products": PRODUCTS}
    model = ss.load_model(write_model_dir(tmp_path / "m", metadata))
    assert model.version == "12"


def test_load_model_accepts_string_path(model_dir):
    assert ss.load_model(str(model_dir)).version == VERSION


def test_load_model_missing_directory(tmp_path):
    with pytest.raises(ss.ModelLoadError):
        ss.load_model(tmp_path / "nowhere")


def test_load_model_missing_model_file(tmp_path):
    path = write_model_dir(tmp_path / "m")
    (path / ss.MODEL_FILE).unlink()
    with pytest.raises(ss.ModelLoadError):
        ss.load_model(path)


def test_load_model_missing_metadata_file(tmp_path):
    path = write_model_dir(tmp_path / "m")
    (path / ss.METADATA_FILE).unlink()
    with pytest.raises(ss.ModelLoadError):
        ss.load_model(path)


@pytest.mark.parametrize("content", ["{not json", "[1, 2, 3]", '"text"'])
def test_load_model_metadata_must_be_json_object(tmp_path, content):
    path = write_model_dir(tmp_path / "m")
    (path / ss.METADATA_FILE).write_text(content, encoding="utf-8")
    with pytest.raises(ss.ModelLoadError):
        ss.load_model(path)


@pytest.mark.parametrize("key", ss.METADATA_KEYS)
def test_load_model_requires_metadata_keys(tmp_path, key):
    metadata = {"version": VERSION, "features": FEATURES, "stores": STORES, "products": PRODUCTS}
    del metadata[key]
    path = write_model_dir(tmp_path / "m", metadata)
    with pytest.raises(ss.ModelLoadError, match=key):
        ss.load_model(path)


# --- build_frame ---


def test_frame_has_feature_columns_in_order():
    frame = ss.build_frame([ROW, {**ROW, "store_id": "S02", "promo": True}], FEATURES)
    assert list(frame.columns) == FEATURES
    assert len(frame) == 2
    assert isinstance(frame.index, pd.RangeIndex)
    assert frame["store_id"].tolist() == ["S01", "S02"]
    assert frame["product"].tolist() == ["Мышь", "Мышь"]


def test_frame_column_types():
    frame = ss.build_frame([{**ROW, "price": 1490, "promo": True}, ROW], FEATURES)
    assert frame["price"].dtype == "float64"
    assert frame["promo"].dtype == "int64"
    assert frame["dow"].dtype == "int64"
    assert frame["month"].dtype == "int64"
    assert frame["promo"].tolist() == [1, 0]
    assert frame["price"].tolist() == [1490.0, 1490.0]


def test_frame_date_features():
    rows = [
        {**ROW, "date": dt.date(2024, 12, 30)},
        {**ROW, "date": dt.date(2025, 1, 5)},
        {**ROW, "date": dt.date(2024, 2, 29)},
    ]
    frame = ss.build_frame(rows, FEATURES)
    assert frame["dow"].tolist() == [0, 6, 3]  # понедельник — 0
    assert frame["month"].tolist() == [12, 1, 2]


def test_frame_follows_requested_feature_list():
    frame = ss.build_frame([ROW], ["month", "price", "store_id"])
    assert list(frame.columns) == ["month", "price", "store_id"]
    assert frame.iloc[0].tolist() == [1, 1490.0, "S01"]


def test_frame_rejects_unknown_feature():
    with pytest.raises(ValueError, match="units_mean_7d"):
        ss.build_frame([ROW], ["price", "units_mean_7d"])


def test_frame_rejects_empty_rows():
    with pytest.raises(ValueError):
        ss.build_frame([], FEATURES)


def test_frame_does_not_change_rows():
    rows = [dict(ROW), {**ROW, "promo": True}]
    snapshot = [dict(row) for row in rows]
    ss.build_frame(rows, FEATURES)
    assert rows == snapshot


# --- predict_rows ---


def test_predict_rows_matches_pipeline(model_dir, pipeline):
    items = [ITEM, {**ITEM, "store_id": "S03", "promo": True, "price": 1270.0}, {**ITEM, "date": "2025-01-18"}]
    rows = [{**item, "date": dt.date.fromisoformat(item["date"])} for item in items]
    result = ss.predict_rows(fake_model(pipeline), rows)
    assert [row["units"] for row in result] == expected_units(pipeline, items)
    assert len({row["units"] for row in result}) == 3


def test_predict_rows_result_shape():
    result = ss.predict_rows(fake_model(FakePipeline([2.5])), [ROW])
    assert result == [{"units": 2.5, "warnings": []}]
    assert type(result[0]["units"]) is float


def test_predict_rows_calls_model_once_for_all_rows():
    fake = FakePipeline()
    rows = [{**ROW, "store_id": store} for store in STORES] * 5
    result = ss.predict_rows(fake_model(fake), rows)
    assert len(fake.frames) == 1
    assert list(fake.frames[0].columns) == FEATURES
    assert fake.frames[0]["store_id"].tolist() == [row["store_id"] for row in rows]
    assert [row["units"] for row in result] == [float(number) for number in range(20)]


def test_predict_rows_clips_negative_and_rounds():
    result = ss.predict_rows(fake_model(FakePipeline([-3.2, 1.23456, 0.0004, 7.0005])), [ROW] * 4)
    assert [row["units"] for row in result] == [0.0, 1.235, 0.0, 7.0]


def test_predict_rows_warns_about_unknown_categories():
    rows = [
        ROW,
        {**ROW, "store_id": "S99"},
        {**ROW, "product": "Самокат"},
        {**ROW, "store_id": "S99", "product": "Самокат"},
    ]
    result = ss.predict_rows(fake_model(FakePipeline()), rows)
    assert [row["warnings"] for row in result] == [
        [],
        ["unknown store_id"],
        ["unknown product"],
        ["unknown store_id", "unknown product"],
    ]


def test_predict_rows_uses_model_feature_list():
    fake = FakePipeline()
    model = replace(fake_model(fake), features=["price", "dow"])
    ss.predict_rows(model, [ROW])
    assert list(fake.frames[0].columns) == ["price", "dow"]


# --- create_app: готовность и загрузка модели ---


def test_app_is_alive_without_model(tmp_path):
    with open_client(tmp_path) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_app_without_model_is_not_ready(tmp_path):
    with open_client(tmp_path) as client:
        for response in [
            client.get("/ready"),
            client.post("/predict", json=ITEM),
            client.post("/predict/batch", json={"items": [ITEM]}),
        ]:
            assert response.status_code == 503
            assert response.json() == {"detail": ss.NOT_READY_DETAIL}


def test_app_ready_reports_model_version(model_dir):
    with open_client(model_dir) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready", "model_version": VERSION}
        assert client.get("/health").json() == {"status": "ok"}


def test_app_loads_model_once_at_startup(model_dir, monkeypatch):
    calls = []
    original = ss.load_model

    def counting(path):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(ss, "load_model", counting)
    app = ss.create_app(model_dir)
    assert calls == []  # при создании приложения модель не читается
    with TestClient(app) as client:
        assert calls == [model_dir]
        for _ in range(3):
            assert client.post("/predict", json=ITEM).status_code == 200
        client.post("/predict/batch", json={"items": [ITEM, ITEM]})
        client.get("/ready")
    assert calls == [model_dir]


def test_app_keeps_model_in_state(model_dir):
    with open_client(model_dir) as client:
        assert isinstance(client.app.state.model, ss.LoadedModel)
        assert client.app.state.model.version == VERSION


def test_app_state_model_is_none_when_loading_failed(tmp_path):
    with open_client(tmp_path) as client:
        assert client.app.state.model is None


def test_app_reads_model_from_state_on_every_request(model_dir):
    with open_client(model_dir) as client:
        client.app.state.model = fake_model(FakePipeline([42.0]), version="8")
        assert client.post("/predict", json=ITEM).json() == {"units": 42.0, "model_version": "8", "warnings": []}
        assert client.get("/ready").json() == {"status": "ready", "model_version": "8"}
        client.app.state.model = None
        assert client.post("/predict", json=ITEM).status_code == 503


# --- create_app: предсказание ---


def test_app_predict_returns_units_version_and_warnings(model_dir, pipeline):
    with open_client(model_dir) as client:
        response = client.post("/predict", json=ITEM)
    assert response.status_code == 200
    assert response.json() == {"units": expected_units(pipeline, [ITEM])[0], "model_version": VERSION, "warnings": []}


def test_app_predict_depends_on_request(model_dir, pipeline):
    items = [{**ITEM, "store_id": "S04"}, {**ITEM, "promo": True, "price": 1270.0}, {**ITEM, "date": "2025-01-19"}]
    with open_client(model_dir) as client:
        units = [client.post("/predict", json=item).json()["units"] for item in items]
    assert units == expected_units(pipeline, items)
    assert len(set(units)) == 3


def test_app_predict_promo_defaults_to_false(model_dir):
    item = {key: value for key, value in ITEM.items() if key != "promo"}
    with open_client(model_dir) as client:
        assert client.post("/predict", json=item).json() == client.post("/predict", json=ITEM).json()


def test_app_predict_warns_about_unknown_categories(model_dir):
    with open_client(model_dir) as client:
        body = client.post("/predict", json={**ITEM, "store_id": "S99", "product": "Самокат"}).json()
    assert body["warnings"] == ["unknown store_id", "unknown product"]
    assert body["units"] >= 0


def test_app_predict_never_returns_negative_units(model_dir):
    with open_client(model_dir) as client:
        client.app.state.model = fake_model(FakePipeline([-5.0]))
        assert client.post("/predict", json=ITEM).json()["units"] == 0.0


INVALID = {
    "store_id без буквы": {"store_id": "01"},
    "store_id из одной цифры": {"store_id": "S1"},
    "store_id из трёх цифр": {"store_id": "S001"},
    "store_id строчной буквой": {"store_id": "s01"},
    "store_id числом": {"store_id": 1},
    "store_id пустой": {"store_id": ""},
    "product пустой": {"product": ""},
    "product длиннее предела": {"product": "я" * (ss.MAX_PRODUCT_LENGTH + 1)},
    "product null": {"product": None},
    "дата не в формате ISO": {"date": "15.01.2025"},
    "несуществующая дата": {"date": "2025-02-30"},
    "дата null": {"date": None},
    "цена ноль": {"price": 0},
    "цена отрицательная": {"price": -1490},
    "цена выше предела": {"price": ss.MAX_PRICE + 1},
    "цена строкой": {"price": "1490"},
    "цена null": {"price": None},
    "promo строкой": {"promo": "yes"},
    "promo числом": {"promo": 1},
    "promo null": {"promo": None},
    "лишнее поле": {"prise": 1490},
}


@pytest.mark.parametrize("patch", INVALID.values(), ids=INVALID.keys())
def test_app_predict_rejects_invalid_input(model_dir, patch):
    fake = FakePipeline([1.0])
    with open_client(model_dir) as client:
        client.app.state.model = fake_model(fake)
        response = client.post("/predict", json={**ITEM, **patch})
    assert response.status_code == 422
    assert fake.frames == []  # до модели запрос не дошёл


@pytest.mark.parametrize("field", ["store_id", "product", "date", "price"])
def test_app_predict_requires_fields(model_dir, field):
    item = {key: value for key, value in ITEM.items() if key != field}
    with open_client(model_dir) as client:
        response = client.post("/predict", json=item)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert [error["loc"][-1] for error in detail] == [field]


@pytest.mark.parametrize("body", [[], "text", 5, None, {"items": [ITEM]}])
def test_app_predict_rejects_wrong_body(model_dir, body):
    with open_client(model_dir) as client:
        assert client.post("/predict", json=body).status_code == 422


def test_app_predict_reports_every_invalid_field(model_dir):
    with open_client(model_dir) as client:
        response = client.post("/predict", json={**ITEM, "price": -5, "store_id": "магазин"})
    assert sorted(error["loc"][-1] for error in response.json()["detail"]) == ["price", "store_id"]


@pytest.mark.parametrize(
    "patch",
    [
        {"price": ss.MAX_PRICE},
        {"price": 1},
        {"price": 0.01},
        {"product": "я" * ss.MAX_PRODUCT_LENGTH},
        {"store_id": "S00"},
    ],
)
def test_app_predict_accepts_boundary_values(model_dir, patch):
    with open_client(model_dir) as client:
        assert client.post("/predict", json={**ITEM, **patch}).status_code == 200


# --- create_app: пакетное предсказание ---


def test_app_batch_matches_single_requests(model_dir, pipeline):
    items = [ITEM, {**ITEM, "store_id": "S04"}, {**ITEM, "promo": True, "price": 1270.0}, {**ITEM, "store_id": "S99"}]
    with open_client(model_dir) as client:
        single = [client.post("/predict", json=item).json() for item in items]
        response = client.post("/predict/batch", json={"items": items})
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == VERSION
    assert body["predictions"] == [{"units": row["units"], "warnings": row["warnings"]} for row in single]
    assert [row["units"] for row in body["predictions"]] == expected_units(pipeline, items)
    assert body["predictions"][3]["warnings"] == ["unknown store_id"]


def test_app_batch_calls_model_once(model_dir):
    fake = FakePipeline()
    with open_client(model_dir) as client:
        client.app.state.model = fake_model(fake)
        body = client.post("/predict/batch", json={"items": [ITEM] * 7}).json()
    assert len(fake.frames) == 1
    assert len(fake.frames[0]) == 7
    assert [row["units"] for row in body["predictions"]] == [float(number) for number in range(7)]


def test_app_batch_size_limits(model_dir):
    with open_client(model_dir) as client:
        client.app.state.model = fake_model(FakePipeline())
        assert client.post("/predict/batch", json={"items": []}).status_code == 422
        largest = client.post("/predict/batch", json={"items": [ITEM] * ss.MAX_BATCH})
        assert largest.status_code == 200
        assert len(largest.json()["predictions"]) == ss.MAX_BATCH
        assert client.post("/predict/batch", json={"items": [ITEM] * (ss.MAX_BATCH + 1)}).status_code == 422


def test_app_batch_rejects_invalid_item_and_names_it(model_dir):
    fake = FakePipeline()
    with open_client(model_dir) as client:
        client.app.state.model = fake_model(fake)
        response = client.post("/predict/batch", json={"items": [ITEM, {**ITEM, "price": -1}, ITEM]})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-3:] == ["items", 1, "price"]
    assert fake.frames == []


@pytest.mark.parametrize("body", [{}, {"items": ITEM}, [ITEM], {"items": [ITEM], "extra": 1}])
def test_app_batch_rejects_wrong_body(model_dir, body):
    with open_client(model_dir) as client:
        assert client.post("/predict/batch", json=body).status_code == 422


# --- create_app: ошибки и описание ---


def test_app_model_failure_is_reported_without_details(model_dir):
    with open_client(model_dir, raise_server_exceptions=False) as client:
        good = client.app.state.model
        client.app.state.model = fake_model(FakePipeline(error=ZeroDivisionError("secret-internal-text")))
        for response in [client.post("/predict", json=ITEM), client.post("/predict/batch", json={"items": [ITEM]})]:
            assert response.status_code == 500
            assert response.json() == {"detail": ss.ERROR_DETAIL}
            assert "secret-internal-text" not in response.text
            assert "Traceback" not in response.text
        client.app.state.model = good
        assert client.post("/predict", json=ITEM).status_code == 200  # сервис пережил ошибку


def test_app_openapi_describes_contract(model_dir):
    with open_client(model_dir) as client:
        spec = client.get("/openapi.json").json()
    assert spec["info"]["title"] == ss.APP_TITLE
    assert {"/health", "/ready", "/predict", "/predict/batch"} <= set(spec["paths"])
    assert "/metrics" not in spec["paths"]
    for path in ["/ready", "/predict", "/predict/batch"]:
        method = "get" if path == "/ready" else "post"
        assert "503" in spec["paths"][path][method]["responses"]
    assert "422" in spec["paths"]["/predict"]["post"]["responses"]
    reference = spec["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    schema = spec["components"]["schemas"][reference.rsplit("/", 1)[-1]]
    assert sorted(schema["required"]) == ["date", "price", "product", "store_id"]
    assert set(schema["properties"]) == {"store_id", "product", "date", "price", "promo"}
    assert schema["additionalProperties"] is False


# --- create_app: метрики ---


def test_metrics_endpoint_serves_prometheus_text(model_dir):
    with open_client(model_dir) as client:
        client.post("/predict", json=ITEM)
        response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    families = {family.name for family in text_string_to_metric_families(response.text)}
    assert {
        "sales_requests",
        "sales_request_duration_seconds",
        "sales_predictions",
        "sales_prediction_units",
        "sales_promo_rows",
        "sales_unknown_category",
        "sales_model_info",
    } <= families


def test_metrics_count_requests_by_endpoint_and_status(model_dir):
    registry = CollectorRegistry()
    with open_client(model_dir, registry) as client:
        for _ in range(3):
            client.post("/predict", json=ITEM)
        client.post("/predict", json={**ITEM, "price": -1})
        client.post("/predict/batch", json={"items": [ITEM, ITEM]})
        client.post("/predict/batch", json={"items": []})
        client.get("/health")
        client.get("/ready")
    assert sample(registry, "sales_requests_total", endpoint="/predict", status="200") == 3
    assert sample(registry, "sales_requests_total", endpoint="/predict", status="422") == 1
    assert sample(registry, "sales_requests_total", endpoint="/predict/batch", status="200") == 1
    assert sample(registry, "sales_requests_total", endpoint="/predict/batch", status="422") == 1
    assert sample(registry, "sales_requests_total", endpoint="/health", status="200") == 1
    assert sample(registry, "sales_requests_total", endpoint="/ready", status="200") == 1


def test_metrics_count_not_ready_responses(tmp_path):
    registry = CollectorRegistry()
    with open_client(tmp_path, registry) as client:
        client.post("/predict", json=ITEM)
        client.get("/ready")
    assert sample(registry, "sales_requests_total", endpoint="/predict", status="503") == 1
    assert sample(registry, "sales_requests_total", endpoint="/ready", status="503") == 1
    assert sample(registry, "sales_predictions_total") == 0


def test_metrics_unknown_paths_share_one_label(model_dir):
    registry = CollectorRegistry()
    with open_client(model_dir, registry) as client:
        for path in ["/nope/1", "/nope/2", "/admin", "/predict/S01"]:
            assert client.get(path).status_code == 404
        text = client.get("/metrics").text
    assert sample(registry, "sales_requests_total", endpoint="other", status="404") == 4
    assert "nope" not in text
    assert "/admin" not in text


def test_metrics_endpoint_is_not_counted(model_dir):
    registry = CollectorRegistry()
    with open_client(model_dir, registry) as client:
        for _ in range(3):
            text = client.get("/metrics").text
    assert 'endpoint="/metrics"' not in text
    assert sample(registry, "sales_requests_total", endpoint="/metrics", status="200") == 0


def test_metrics_duration_histogram(model_dir):
    registry = CollectorRegistry()
    with open_client(model_dir, registry) as client:
        for _ in range(4):
            client.post("/predict", json=ITEM)
        client.post("/predict", json={**ITEM, "price": -1})
        text = client.get("/metrics").text
    assert sample(registry, "sales_request_duration_seconds_count", endpoint="/predict") == 5
    assert sample(registry, "sales_request_duration_seconds_sum", endpoint="/predict") > 0
    assert sample(registry, "sales_request_duration_seconds_bucket", endpoint="/predict", le="+Inf") == 5
    family = next(f for f in text_string_to_metric_families(text) if f.name == "sales_request_duration_seconds")
    bounds = {
        s.labels["le"] for s in family.samples if s.name.endswith("_bucket") and s.labels["endpoint"] == "/predict"
    }
    assert {float(bound) for bound in bounds} == {*ss.DURATION_BUCKETS, float("inf")}


def test_metrics_count_predicted_rows(model_dir):
    registry = CollectorRegistry()
    with open_client(model_dir, registry) as client:
        client.post("/predict", json=ITEM)
        client.post("/predict", json=ITEM)
        client.post("/predict/batch", json={"items": [ITEM] * 5})
        client.post("/predict", json={**ITEM, "price": "дорого"})
    assert sample(registry, "sales_predictions_total") == 7


def test_metrics_prediction_histogram(model_dir):
    registry = CollectorRegistry()
    with open_client(model_dir, registry) as client:
        client.app.state.model = fake_model(FakePipeline([0.5, 3.0, 150.0, -2.0]))
        client.post("/predict/batch", json={"items": [ITEM] * 4})
        text = client.get("/metrics").text
    assert sample(registry, "sales_prediction_units_count") == 4
    assert sample(registry, "sales_prediction_units_sum") == pytest.approx(153.5)  # отрицательный прогноз учтён как 0
    assert sample(registry, "sales_prediction_units_bucket", le="1.0") == 2
    assert sample(registry, "sales_prediction_units_bucket", le="5.0") == 3
    assert sample(registry, "sales_prediction_units_bucket", le="100.0") == 3
    assert sample(registry, "sales_prediction_units_bucket", le="+Inf") == 4
    family = next(f for f in text_string_to_metric_families(text) if f.name == "sales_prediction_units")
    bounds = {float(s.labels["le"]) for s in family.samples if s.name.endswith("_bucket")}
    assert bounds == {*ss.UNITS_BUCKETS, float("inf")}


def test_metrics_count_promo_rows(model_dir):
    registry = CollectorRegistry()
    promo = {**ITEM, "promo": True, "price": 1270.0}
    with open_client(model_dir, registry) as client:
        client.post("/predict", json=promo)
        client.post("/predict", json=ITEM)
        client.post("/predict/batch", json={"items": [promo, ITEM, promo]})
    assert sample(registry, "sales_promo_rows_total") == 3
    assert sample(registry, "sales_predictions_total") == 5


def test_metrics_count_unknown_categories(model_dir):
    registry = CollectorRegistry()
    with open_client(model_dir, registry) as client:
        client.post("/predict", json=ITEM)
        client.post("/predict", json={**ITEM, "store_id": "S99"})
        client.post("/predict/batch", json={"items": [{**ITEM, "store_id": "S98", "product": "Самокат"}, ITEM]})
    assert sample(registry, "sales_unknown_category_total", field="store_id") == 2
    assert sample(registry, "sales_unknown_category_total", field="product") == 1


def test_metrics_model_info(model_dir, tmp_path):
    registry, empty = CollectorRegistry(), CollectorRegistry()
    with open_client(model_dir, registry), open_client(tmp_path, empty) as not_ready:
        assert sample(registry, "sales_model_info", version=VERSION) == 1
        assert "sales_model_info{" not in not_ready.get("/metrics").text  # модели нет — ряда нет
    assert empty.get_sample_value("sales_model_info", {"version": VERSION}) is None


def test_metrics_failed_prediction_is_counted_as_500(model_dir):
    registry = CollectorRegistry()
    with open_client(model_dir, registry, raise_server_exceptions=False) as client:
        client.app.state.model = fake_model(FakePipeline(error=RuntimeError("boom")))
        client.post("/predict", json=ITEM)
        client.post("/predict/batch", json={"items": [ITEM, ITEM]})
    assert sample(registry, "sales_requests_total", endpoint="/predict", status="500") == 1
    assert sample(registry, "sales_requests_total", endpoint="/predict/batch", status="500") == 1
    assert sample(registry, "sales_predictions_total") == 0
    assert sample(registry, "sales_prediction_units_count") == 0
    assert sample(registry, "sales_request_duration_seconds_count", endpoint="/predict") == 1


def test_metrics_each_app_has_its_own_registry(model_dir):
    first, second = ss.create_app(model_dir), ss.create_app(model_dir)  # без «Duplicated timeseries»
    with TestClient(first) as one, TestClient(second) as two:
        one.post("/predict", json=ITEM)
        one.post("/predict", json=ITEM)
        two.post("/predict", json=ITEM)
        assert "sales_predictions_total 2.0" in one.get("/metrics").text
        assert "sales_predictions_total 1.0" in two.get("/metrics").text
    assert prometheus_client.REGISTRY.get_sample_value("sales_predictions_total") is None


def test_metrics_use_given_registry(model_dir):
    registry = CollectorRegistry()
    with open_client(model_dir, registry) as client:
        client.post("/predict", json=ITEM)
        text = client.get("/metrics").text
    assert registry.get_sample_value("sales_predictions_total") == 1
    assert prometheus_client.generate_latest(registry).decode() == text
