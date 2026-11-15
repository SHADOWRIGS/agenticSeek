"""Тесты приложения: браузер не нужен, запросы идут через тестовый клиент Flask.

TestSkeleton проходит и на заготовке. TestCallbacks начнёт проходить, когда вы подключите все графики.
"""

import json

import pytest

import app as app_module

REQUIRED_IDS = {
    "filter-dates",
    "filter-cities",
    "filter-categories",
    "filter-metric",
    "filter-freq",
    "filter-by",
    "kpi-row",
    "graph-timeseries",
    "graph-category",
    "graph-top-products",
    "graph-weekday",
    "graph-promo",
}
STUDENT_OUTPUTS = [
    "kpi-row.children",
    "graph-timeseries.figure",
    "graph-top-products.figure",
    "graph-weekday.figure",
    "graph-promo.figure",
]
FAR_FUTURE = {("filter-dates", "start_date"): "2030-01-01", ("filter-dates", "end_date"): "2030-12-31"}


@pytest.fixture(scope="module")
def client():
    return app_module.app.server.test_client()


def collect_props(node, found: dict[str, dict]) -> dict[str, dict]:
    """Обходит дерево макета и собирает свойства всех компонентов, у которых есть id."""
    if isinstance(node, list):
        for item in node:
            collect_props(item, found)
    elif isinstance(node, dict):
        props = node.get("props", {})
        if "id" in props:
            found[props["id"]] = props
        collect_props(props.get("children"), found)
    return found


@pytest.fixture(scope="module")
def layout_props(client) -> dict[str, dict]:
    return collect_props(client.get("/_dash-layout").get_json(), {})


@pytest.fixture(scope="module")
def callbacks(client) -> list[dict]:
    return client.get("/_dash-dependencies").get_json()


def output_names(spec: dict) -> list[str]:
    """Имена выходов callback: 'a.figure' или, для нескольких выходов, '..a.figure...b.figure..'."""
    output = spec["output"]
    return output[2:-2].split("...") if output.startswith("..") else [output]


def find_callback(callbacks: list[dict], output: str) -> dict:
    for spec in callbacks:
        if output in output_names(spec):
            return spec
    raise AssertionError(f"нет callback с выходом {output}")


def fire(client, spec: dict, layout_props: dict[str, dict], overrides: dict | None = None):
    """Вызывает callback так же, как это делает браузер: значения входов берутся из макета."""
    overrides = overrides or {}

    def with_value(dependency: dict) -> dict:
        key = (dependency["id"], dependency["property"])
        value = overrides.get(key, layout_props[dependency["id"]].get(dependency["property"]))
        return {**dependency, "value": value}

    names = output_names(spec)
    outputs = [dict(zip(("id", "property"), name.rsplit(".", 1), strict=True)) for name in names]
    payload = {
        "output": spec["output"],
        "outputs": outputs if spec["output"].startswith("..") else outputs[0],
        "inputs": [with_value(dependency) for dependency in spec["inputs"]],
        "state": [with_value(dependency) for dependency in spec["state"]],
        "changedPropIds": [],
    }
    return client.post("/_dash-update-component", json=payload)


class TestSkeleton:
    def test_index_page(self, client):
        assert client.get("/").status_code == 200

    def test_layout_has_required_ids(self, layout_props):
        missing = REQUIRED_IDS - set(layout_props)
        assert not missing, f"в макете нет компонентов: {sorted(missing)}"

    def test_filters_have_initial_values(self, layout_props):
        assert layout_props["filter-dates"]["start_date"] == "2023-01-01"
        assert layout_props["filter-dates"]["end_date"] == "2024-12-31"
        assert layout_props["filter-metric"]["value"] == "revenue"
        assert layout_props["filter-freq"]["value"] == "W"
        assert layout_props["filter-by"]["value"] == "city"

    def test_sample_callback_returns_figure(self, client, callbacks, layout_props):
        spec = find_callback(callbacks, "graph-category.figure")
        response = fire(client, spec, layout_props)
        assert response.status_code == 200
        figure = response.get_json()["response"]["graph-category"]["figure"]
        assert [trace["type"] for trace in figure["data"]] == ["bar"]
        assert len(figure["data"][0]["x"]) == 4

    def test_sample_callback_reacts_to_filters(self, client, callbacks, layout_props):
        spec = find_callback(callbacks, "graph-category.figure")
        overrides = {("filter-categories", "value"): ["Электроника", "Для дома"]}
        figure = fire(client, spec, layout_props, overrides).get_json()["response"]["graph-category"]["figure"]
        assert sorted(figure["data"][0]["x"]) == ["Для дома", "Электроника"]

    def test_sample_callback_handles_empty_selection(self, client, callbacks, layout_props):
        spec = find_callback(callbacks, "graph-category.figure")
        response = fire(client, spec, layout_props, FAR_FUTURE)
        assert response.status_code == 200
        assert response.get_json()["response"]["graph-category"]["figure"]["data"] == []


class TestCallbacks:
    def test_every_output_has_callback(self, callbacks):
        connected = {name for spec in callbacks for name in output_names(spec)}
        missing = [output for output in STUDENT_OUTPUTS if output not in connected]
        assert not missing, f"нет callback для: {missing}"

    def test_all_callbacks_run_on_initial_state(self, client, callbacks, layout_props):
        for spec in callbacks:
            response = fire(client, spec, layout_props)
            assert response.status_code == 200, f"callback {spec['output']} завершился с ошибкой"

    def test_all_callbacks_handle_empty_selection(self, client, callbacks, layout_props):
        for spec in callbacks:
            response = fire(client, spec, layout_props, FAR_FUTURE)
            assert response.status_code == 200, f"callback {spec['output']} падает на пустой выборке"
            for component_id, props in response.get_json()["response"].items():
                if "figure" in props:
                    assert props["figure"]["data"] == [], f"{component_id}: на пустой выборке остались трассы"

    def test_graphs_have_traces_on_initial_state(self, client, callbacks, layout_props):
        expected_traces = {"graph-timeseries": 4, "graph-top-products": 1, "graph-weekday": 1, "graph-promo": 2}
        for component_id, n_traces in expected_traces.items():
            spec = find_callback(callbacks, f"{component_id}.figure")
            figure = fire(client, spec, layout_props).get_json()["response"][component_id]["figure"]
            assert len(figure["data"]) == n_traces, component_id

    def test_timeseries_reacts_to_split(self, client, callbacks, layout_props):
        spec = find_callback(callbacks, "graph-timeseries.figure")
        overrides = {("filter-by", "value"): "category", ("filter-cities", "value"): ["Казань"]}
        figure = fire(client, spec, layout_props, overrides).get_json()["response"]["graph-timeseries"]["figure"]
        names = sorted(trace["name"] for trace in figure["data"])
        assert names == ["Аксессуары", "Для дома", "Товары в дорогу", "Электроника"]

    def test_kpi_cards_show_totals(self, client, callbacks, layout_props):
        spec = find_callback(callbacks, "kpi-row.children")
        response = fire(client, spec, layout_props)
        assert response.status_code == 200
        text = json.dumps(response.get_json(), ensure_ascii=False)
        sales = app_module.SALES
        assert app_module.format_number(sales["revenue"].sum()) in text
        assert app_module.format_number(sales["units"].sum()) in text
        assert app_module.format_share(sales["promo"].mean()) in text
