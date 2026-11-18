# Практика 7.4.4: материалы

Постановка задачи: [lesson_7_4_4.md](../lesson_7_4_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `sales_service.py` | заготовка: четыре функции — `load_model`, `build_frame`, `predict_rows`, `create_app`. Готовые части: константы (`MODEL_FILE`, `METADATA_FILE`, `METADATA_KEYS`, `KNOWN_FEATURES`, `STORE_PATTERN`, `MAX_PRODUCT_LENGTH`, `MAX_PRICE`, `MAX_BATCH`, `DURATION_BUCKETS`, `UNITS_BUCKETS`, `NOT_READY_DETAIL`, `ERROR_DETAIL`, `APP_TITLE`), `ModelLoadError` и `LoadedModel` |
| `sales_drift.py` | заготовка: восемь функций — `psi_from_shares`, `numeric_psi`, `categorical_psi`, `psi_noise`, `ks_test`, `missing_share`, `new_category_share`, `drift_report`. Готовые части: константы порогов и ключей отчёта |
| `tests/test_sales_service.py` | 98 тестов: чтение каталога модели (13), таблица признаков (7), прогноз и предупреждения (6), приложение — готовность, проверка входа, пакет, ошибки, OpenAPI (58), метрики (14) |
| `tests/test_sales_drift.py` | 72 теста: PSI по долям (8), числовой PSI (14), категориальный PSI (9), ожидаемый шум PSI (6), критерий Колмогорова — Смирнова (7), пропуски и новые категории (3), отчёт (25) |
| `main.py` | готовая точка входа: `uvicorn main:app`, каталог модели — из переменной `MODEL_DIR` |
| `train_model.py` | готовый скрипт: обучает модель и создаёт каталог `model/` (`model.joblib`, `metadata.json`) |
| `Dockerfile`, `.dockerignore` | готовый образ сервиса (разбор — урок 7.4.2) |
| `compose.yaml`, `monitoring/` | готовый стенд: сервис, Prometheus, Pushgateway, Grafana; правила оповещений и панель (урок 7.4.3) |
| `k8s/` | готовые манифесты: Deployment, Service, ConfigMap, Secret, HorizontalPodAutoscaler (урок 7.4.2) |

Тесты не используют сеть, Docker и внешние сервисы: приложение вызывается через `TestClient`, каталог
модели создаётся во временном каталоге pytest. Модель в тестах — маленький `Pipeline` (`Ridge`)
на синтетических данных; `train_model.py` и `sales_data.py` тестам не нужны.

## Окружение

Основное окружение курса с добавлением пакетов сервиса:

```bash
uv add fastapi "uvicorn[standard]" prometheus-client scipy joblib scikit-learn pandas
uv add --dev pytest httpx2 httpx ruff
uv run pytest -q          # на заготовке: 170 failed
```

Версии, на которых проверены заготовка и тесты: Python 3.12.13, `fastapi 0.162.7`, `pydantic 2.14.3`,
`starlette 1.10.0`, `uvicorn 0.54.0` (`uvloop 0.23.0`, `httptools 0.9.0`), `prometheus-client 0.26.0`,
`httpx2 2.16.0`, `httpx 0.28.1`, `scikit-learn 1.9.1`, `pandas 3.1.1`, `numpy 2.5.4`, `scipy 1.18.1`,
`joblib 1.6.0`, `pytest 9.2.0`.

`httpx2` нужен `TestClient`: в `starlette 1.10.0` импорт `fastapi.testclient` с одним только `httpx` выдаёт
`StarletteDeprecationWarning`, и запуск с `-W error` падает на сборе тестов. `httpx` используют только
ваши скрипты нагрузки; тестам он не нужен.

Образы стенда: `python:3.12-slim`, `ghcr.io/astral-sh/uv:0.11.32`, `prom/prometheus:v3.16.0`,
`prom/pushgateway:v1.11.5`, `grafana/grafana:13.3.1`; Docker 29.9.2, Compose v5.6.1.

## Запуск тестов

На заготовке все 170 тестов падают с `NotImplementedError`.

По шагам:

```bash
uv run pytest -q tests/test_sales_service.py -k test_load_model
uv run pytest -q tests/test_sales_service.py -k test_frame
uv run pytest -q tests/test_sales_service.py -k test_predict_rows
uv run pytest -q tests/test_sales_service.py -k test_app
uv run pytest -q tests/test_sales_service.py -k test_metrics
uv run pytest -q tests/test_sales_drift.py -k "test_psi or test_numeric or test_categorical"
uv run pytest -q tests/test_sales_drift.py -k "test_ks or test_missing or test_new_category"
uv run pytest -q tests/test_sales_drift.py -k test_report
```

Эталонное решение проходит все тесты за 5 секунд. Строгая проверка — без предупреждений:

```bash
uv run python -W error -m pytest -q
```

Фильтров предупреждений в тестах нет: любое предупреждение при `-W error` — из вашего решения или из
несовпадения версий пакетов с перечисленными выше.

Проверка стиля:

```bash
uv run ruff check .
uv run ruff format --check sales_service.py sales_drift.py main.py train_model.py tests/test_sales_service.py tests/test_sales_drift.py
```

## Запуск сервиса и стенда

```bash
cp ../../../module_02/topic_2_1/sales_data.py .     # генератор данных темы — для train_model.py
uv run python train_model.py --out model --version 3
uv run uvicorn main:app --port 8000                  # сервис без контейнера

docker build -t sales-service:1.0.0 --build-arg MODEL_VERSION=3 .
docker compose up -d                                 # сервис, Prometheus :9090, Pushgateway :9091, Grafana :3000
docker compose down -v
```

`Dockerfile` ожидает рядом `pyproject.toml`, `uv.lock` и `.python-version` вашего проекта и каталог
`model/`. В образ попадают только `sales_service.py`, `main.py` и модель. Вход в Grafana — `admin` /
`admin`: это учебный стенд, порты открыты только на `127.0.0.1`.

Что проверено при подготовке: образ с эталонным решением собран из этого `Dockerfile` (в среде
подготовки — через прокси, с добавленным сертификатом), стенд поднят этим `compose.yaml`, обе цели
сбора в состоянии `up`, панель Grafana загружена, оповещение `SalesUnknownCategories` сработало на потоке
с незнакомым магазином. Манифесты `k8s/` проверены на одноузловом `k3s v1.34.1` с образом сервиса
из урока 7.4.2 (тот же `Dockerfile`, другой модуль приложения) и по схеме (`kubernetes-validate 1.37.0`).

## Замечания

- Фикстуры объявлены на уровне модуля: `model_dir` и `pipeline` (каталог модели и сама модель, одни
  на весь прогон) и `frames` (две выборки по 5000 строк из одного распределения). Функции решения
  в фикстурах не вызываются — поэтому на заготовке тесты падают, а не завершаются ошибкой подготовки.
- Тесты приложения создают его помощником `open_client` и входят в `with`: модель загружается в `lifespan`.
  Несколько тестов подменяют `client.app.state.model` объектом с поддельной моделью (`FakePipeline`):
  обработчики обязаны брать модель из `app.state.model` при каждом запросе, а не запоминать её.
- `test_app_loads_model_once_at_startup` подменяет `sales_service.load_model` и считает вызовы:
  приложение должно вызывать `load_model` через имя модуля, один раз и не раньше старта.
- Метрики регистрируйте в реестре приложения. Объявление `Counter(...)` на уровне модуля попадает
  в общий реестр, и второе приложение в том же процессе падает с `Duplicated timeseries in
  CollectorRegistry` — тесты создают десятки приложений.
- Значения метрик тесты читают двумя способами: из переданного реестра (`registry.get_sample_value`)
  и из текста `/metrics`. Имена и метки заданы в docstring `create_app`; им же соответствуют правила
  в `monitoring/alerts.yml` и панель Grafana.
- Ответ 500 проверяется клиентом с `raise_server_exceptions=False`: подойдёт и обработчик исключений
  приложения, и `try` в обработчике запроса. Текста исключения в ответе быть не должно.
- Пороги в `sales_drift.py` читаются из констант модуля в момент вызова:
  `test_report_status_thresholds_are_inclusive` меняет их через `monkeypatch`. Значение порога
  в параметре по умолчанию этот тест не пройдёт.
- `ks_test` должен совпадать с `scipy.stats.ks_2samp` с параметрами по умолчанию. Для двух выборок ровно
  по 10 000 строк с почти одинаковым распределением `scipy 1.18.1` выдаёт `RuntimeWarning: ks_2samp: Exact
  calculation unsuccessful`; тесты таких размеров избегают, в своих скриптах задайте `method="asymp"`.
- Статистический тест `test_psi_noise_predicts_psi_of_samples_without_drift` использует фиксированный
  seed и допуск 25 %: результат не зависит от запуска.
- После работы не оставляйте в репозитории `model/`, `__pycache__`, `.pytest_cache`, `.ruff_cache`
  и файл `.env`.
