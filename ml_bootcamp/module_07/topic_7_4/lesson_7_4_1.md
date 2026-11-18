# Урок 7.4.1: FastAPI для ML сервисов, OpenAPI документация

> **Модуль 7:** ML Engineering & MLOps · **Тема 7.4:** Деплой и мониторинг
> **Тип:** теория · **Время:** 120 минут + 90 минут на домашнее задание

## Цели урока

После урока вы сможете:

- описать вход и выход сервиса предсказаний схемами Pydantic и объяснить, что они проверяют, а что нет;
- загружать модель один раз при старте процесса и отличать «процесс жив» от «сервис готов»;
- написать эндпоинты одиночного и пакетного предсказания с версией модели в ответе;
- вернуть потребителю понятную ошибку и не показать ему трассировку;
- прочитать схему OpenAPI своего сервиса и сказать, что из неё получает потребитель;
- выбрать между `def` и `async def` для модели, которая считает на процессоре, — по замерам, а не по привычке;
- измерить задержку (p50, p95, p99) и пропускную способность при одном и нескольких воркерах;
- протестировать сервис через `TestClient` без запуска сервера.

## Зачем это нужно

К этому уроку у проекта прогноза продаж есть пайплайн (тема 7.1), запись о каждом обучении и реестр
моделей (тема 7.2), хранилище признаков (тема 7.3). Модель при этом остаётся файлом: чтобы получить
прогноз, нужен Python, `scikit-learn` той же версии и знание, какие столбцы и в каком порядке подать.
Кассовой системе, сайту и отчёту закупщика это недоступно. Им нужен адрес, на который отправляют запрос
и получают число.

Сервис — это договор. В нём записано: какие поля принимаются, в каких границах, что вернётся, чем ответ
на плохой запрос отличается от ответа сломанного сервиса и сколько придётся ждать. Модель в этом договоре
занимает одну строку; остальное — урок.

Версии урока: Python 3.12.13, `fastapi 0.162.7`, `pydantic 2.14.3`, `starlette 1.10.0`, `uvicorn 0.54.0`
(с `uvloop 0.23.0` и `httptools 0.9.0`), `httpx 0.28.1`, `httpx2 2.16.0`, `scikit-learn 1.9.1`,
`pandas 3.1.1`, `numpy 2.5.4`, `joblib 1.6.0`, `pytest 9.2.0`; машина — 4 ядра x86-64, Linux.

```bash
uv init --python 3.12
uv add fastapi "uvicorn[standard]" pandas scikit-learn joblib
uv add --dev pytest httpx2 httpx
```

## 1. Модель и её описание

Сервис получает модель как каталог из двух файлов: `model.joblib` и `metadata.json`. В проекте каталог
формирует шаг публикации — версия из реестра с псевдонимом `champion` (урок 7.2.2). Для урока его создаёт
скрипт `train_model.py`; файл `sales_data.py` из `module_02/topic_2_1/` лежит рядом.

```python
FEATURES = ["store_id", "product", "price", "promo", "dow", "month"]
CATEGORICAL = ["store_id", "product"]

sales = add_date_features(make_sales().dropna(subset=["units"]))      # dow, month, promo -> int
train = sales[sales["date"] <= "2024-06-30"]                          # проверка — второе полугодие 2024

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
pipeline = Pipeline([
    ("prepare", ColumnTransformer([("category", encoder, CATEGORICAL)], remainder="passthrough")),
    ("model", HistGradientBoostingRegressor(loss="poisson", random_state=0)),
])
pipeline.fit(train[FEATURES], train["units"])

joblib.dump(pipeline, out / "model.joblib")
(out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
```

Содержимое `metadata.json`:

```text
{"name": "sales-forecast", "version": "3", "trained_on": ["2023-01-01", "2024-06-30"], "valid_mae": 1.8294,
 "features": ["store_id", "product", "price", "promo", "dow", "month"], "stores": ["S01", …, "S08"],
 "products": ["Блендер", "Зонт", …, "Чайник"]}
```

Обучение заняло 3.6 секунды, файл модели — 302 КБ. Преобразования — внутри `Pipeline` (урок 7.1.1):
сервису остаётся собрать таблицу с шестью столбцами. Версия, список признаков и известные категории
читаются из `metadata.json`, а не зашиты в код: новая версия модели не требует новой версии сервиса.
`handle_unknown="ignore"` означает, что незнакомый магазин не вызовет исключения, — к этому вернёмся
в разделе 4.

Файл `joblib` — это pickle: его загрузка выполняет код. Загружайте только модели из своего реестра
и той версией `scikit-learn`, которой они сохранены (урок 7.2.3).

## 2. Схемы: договор о входе и выходе

```python
import datetime as dt

from pydantic import BaseModel, ConfigDict, Field, StrictBool

MAX_BATCH = 1000


class SaleRequest(BaseModel):
    """Одна строка запроса: что, где, когда и по какой цене продаётся."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"store_id": "S01", "product": "Мышь", "date": "2025-01-15", "price": 1490, "promo": False}]
        },
    )

    store_id: str = Field(pattern=r"^S\d{2}$", description="Код магазина, например S01")
    product: str = Field(min_length=1, max_length=50, description="Название товара")
    date: dt.date = Field(description="День, на который нужен прогноз")
    price: float = Field(gt=0, le=1_000_000, strict=True, description="Цена в этот день, рублей")
    promo: StrictBool = Field(default=False, description="Действует ли акция")


class PredictResponse(BaseModel):
    units: float = Field(ge=0, description="Прогноз продаж, штук")
    model_version: str
    warnings: list[str] = Field(default_factory=list, description="Что в запросе модель видит впервые")


class BatchRequest(BaseModel):
    items: list[SaleRequest] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    predictions: list[PredictResponse]
    model_version: str
```

Что здесь решено:

- **границы значений** — в схеме, а не в `if` внутри обработчика: `gt=0` для цены, шаблон для кода
  магазина, длина списка от 1 до 1000. Запрос, не прошедший схему, до модели не доходит;
- **`extra="forbid"`** — опечатка в имени поля (`prise`) становится ошибкой. По умолчанию лишние поля
  молча отбрасываются, и потребитель уверен, что цена учтена;
- **`dt.date`, а не `date`** — поле называется `date`, и аннотация `date: date` при значении
  по умолчанию ссылалась бы сама на себя;
- **ограничение размера пакета** — без него один запрос на миллион строк займёт воркер и память.

Ответы на запросы с ошибками (сервис целиком — в следующих разделах):

```text
{"store_id": "магазин 1", "price": -5, …}
422 {"detail": [{"type": "string_pattern_mismatch", "loc": ["body", "store_id"],
                 "msg": "String should match pattern '^S\\d{2}$'", "input": "магазин 1", …},
                {"type": "greater_than", "loc": ["body", "price"],
                 "msg": "Input should be greater than 0", "input": -5, "ctx": {"gt": 0.0}}]}
{…, "prise": 1490}
422 {"detail": [{"type": "extra_forbidden", "loc": ["body", "prise"], "msg": "Extra inputs are not permitted", …}]}
{…, "date": "15.01.2025"}
422 {"detail": [{"type": "date_from_datetime_parsing", "loc": ["body", "date"],
                 "msg": "Input should be a valid date or datetime, invalid character in year", …}]}
{"store_id": "S01"}
422 {"detail": [{"type": "missing", "loc": ["body", "product"], …}, {… "date" …}, {… "price" …}]}
```

Ошибки перечислены все сразу, с путём к полю (`loc`) и машинно-читаемым типом — потребитель может
показать их в форме, не разбирая текст.

**Приведение типов.** Без `strict=True` и `StrictBool` Pydantic работает в мягком режиме. Тот же сервис
с полями `price: float` и `promo: bool` на запрос `{"price": "1490", "promo": "yes"}` ответил:

```text
200 {"units": 20.578, "model_version": "3", "warnings": []}
```

Строка `"yes"` превратилась в `True`, строка `"1490"` — в число, и модель посчитала прогноз для акции
по полной цене. Со строгими полями тот же запрос отклонён: `float_type` и `bool_type`. Целое `1490`
для строгого `float` по-прежнему допустимо.

Включить строгий режим для всей модели (`ConfigDict(strict=True)`) нельзя: FastAPI сначала разбирает
JSON, затем проверяет получившийся словарь, а в нём дата — строка. Проверено: запрос с верной датой
получает `422 date_type: Input should be a valid date`. Строгость задаётся по полям.

**Чего схема не проверяет.** Запрос для магазина `S99` и сочетание «акция по полной цене» формально
верны. Схема проверяет форму данных, а не их правдоподобие; о втором — раздел 4 и урок 7.4.3.

## 3. Загрузка модели один раз

```python
class LoadedModel:
    """Модель и её описание: читается с диска один раз."""

    def __init__(self, model_dir: Path):
        self.metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.pipeline = joblib.load(model_dir / "model.joblib")
        self.version = str(self.metadata["version"])
        self.features = self.metadata["features"]
        self.known = {"store_id": set(self.metadata["stores"]), "product": set(self.metadata["products"])}

    def predict(self, items: list[SaleRequest]) -> list[PredictResponse]:
        frame = pd.DataFrame([item.model_dump() for item in items])
        frame["date"] = pd.to_datetime(frame["date"])
        frame["dow"] = frame["date"].dt.day_of_week
        frame["month"] = frame["date"].dt.month
        frame["promo"] = frame["promo"].astype(int)
        units = self.pipeline.predict(frame[self.features]).clip(min=0)
        result = []
        for item, value in zip(items, units, strict=True):
            unseen = [f"unknown {field}" for field, known in self.known.items() if getattr(item, field) not in known]
            result.append(PredictResponse(units=round(float(value), 3), model_version=self.version, warnings=unseen))
        return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    model_dir = Path(os.environ.get("MODEL_DIR", "model"))
    try:
        app.state.model = LoadedModel(model_dir)
        logger.info("модель %s загружена из %s", app.state.model.version, model_dir)
    except (OSError, KeyError, ValueError) as error:
        logger.error("модель не загружена: %r", error)
    yield
    app.state.model = None


app = FastAPI(title="Прогноз продаж", version="1.0.0", lifespan=lifespan)
```

`lifespan` — функция, код которой до `yield` выполняется один раз при старте процесса, а после `yield` —
при остановке. Модель кладётся в `app.state` и живёт столько же, сколько процесс.

Сколько стоит альтернатива — читать модель в каждом запросе? Первая загрузка в процессе (импорт
`scikit-learn` и чтение файла) — 1 684 мс; повторная — 5.0 мс (медиана из 30); предсказание для одной
строки — 3.4 мс (раздел 7). Загрузка в обработчике более чем удвоила бы время ответа — и это для модели
в 302 КБ. Для модели в сотни мегабайт счёт пойдёт на секунды и на память: каждый одновременный запрос
держал бы свою копию.

Путь к модели приходит из переменной окружения `MODEL_DIR`: один и тот же код работает на ноутбуке,
в контейнере и в кластере (урок 7.4.2), различаясь только окружением.

Деталь, которая видна только при запуске: сообщение `logger.info` о загрузке в выводе uvicorn
не появилось. Uvicorn настраивает только свои логгеры, а у вашего по умолчанию уровень `WARNING`;
сообщение `logger.error` появилось. Журнал сервиса настраивается явно: `logging.basicConfig(level="INFO")`.

## 4. Эндпоинты предсказания

```python
def get_model(request: Request) -> LoadedModel:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="модель не загружена")
    return model


@app.post("/predict", summary="Прогноз для одной строки", responses={503: {"description": "Модель не загружена"}})
def predict(item: SaleRequest, request: Request) -> PredictResponse:
    return get_model(request).predict([item])[0]


@app.post("/predict/batch", summary="Прогноз для списка строк", responses={503: {"description": "Модель не загружена"}})
def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
    model = get_model(request)
    return BatchResponse(predictions=model.predict(batch.items), model_version=model.version)
```

```bash
uv run uvicorn app:app --port 8000
curl -s -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  -d '{"store_id": "S01", "product": "Мышь", "date": "2025-01-15", "price": 1490}'
```

```text
{"units":16.524,"model_version":"3","warnings":[]}
```

| Запрос | Ответ |
|---|---|
| `S01`, «Мышь», 1490, без акции | `{"units": 16.524, "model_version": "3", "warnings": []}` |
| то же, акция, цена 1270 | `{"units": 29.695, …}` |
| магазин `S99` | `{"units": 10.162, "model_version": "3", "warnings": ["unknown store_id"]}` |
| товар «Самокат» | `{"units": 7.452, "model_version": "3", "warnings": ["unknown product"]}` |
| пакет из двух строк (`S01` и `S08`) | `{"predictions": [{"units": 16.524, …}, {"units": 5.973, …}], "model_version": "3"}` |
| пустой пакет | `422 too_short: List should have at least 1 item after validation, not 0` |
| пакет из 1001 строки | `422 too_long: List should have at most 1000 items after validation, not 1001` |

**Версия модели в ответе.** Прогноз без версии нельзя разобрать задним числом: когда закупщик через
месяц спросит, почему на 15 января было 16.5, отвечать придётся по журналу потребителя, а в нём должна
быть версия. Она же позволяет увидеть, что после выкладки часть экземпляров отвечает старой моделью.

**Незнакомая категория.** Для магазина `S99` модель вернула 10.2 — число без признаков неуверенности.
`OneHotEncoder(handle_unknown="ignore")` закодировал магазин нулями, и модель ответила «как для никакого
магазина». Варианта три: отклонить запрос (422), ответить и предупредить, ответить молча. Здесь выбран
второй: новый магазин — законное событие, прогноз для него лучше, чем отказ, но потребитель и мониторинг
должны об этом знать. Третий вариант — тот, что получается сам собой, если ничего не делать.

**Один вызов модели на пакет.** `predict_batch` не вызывает `predict` в цикле: вся таблица уходит
в модель одним вызовом. Что это даёт — раздел 7.

## 5. Готовность и ошибки

```python
@app.exception_handler(Exception)
async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
    logger.exception("ошибка при обработке %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "внутренняя ошибка сервиса"})


@app.get("/health", summary="Процесс жив")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", summary="Сервис готов принимать запросы", responses={503: {"description": "Модель не загружена"}})
def ready(request: Request) -> dict[str, str]:
    return {"status": "ready", "model_version": get_model(request).version}
```

Два эндпоинта отвечают на два разных вопроса:

| | `/health` | `/ready` |
|---|---|---|
| Вопрос | процесс жив и отвечает? | можно ли направлять сюда запросы? |
| Что проверяет | ничего, кроме способности ответить | модель загружена |
| Что делает оркестратор при отказе | перезапускает контейнер | перестаёт направлять запросы, не перезапуская |

Запуск с каталогом, в котором модели нет (`MODEL_DIR=/nonexistent`):

```text
модель не загружена: FileNotFoundError(2, 'No such file or directory')
INFO:     Application startup complete.
GET /health  -> 200 {"status":"ok"}
GET /ready   -> 503 {"detail":"модель не загружена"}
POST /predict -> 503 {"detail":"модель не загружена"}
```

Сервис стартовал, но не готов. Альтернатива — не перехватывать ошибку в `lifespan`: процесс завершится,
и оркестратор будет перезапускать его по кругу. Оба поведения допустимы. Недопустим третий вариант —
отвечать `200` на `/ready` без модели: запросы пойдут на экземпляр, который не может ответить.

`/health` не должен проверять зависимости. Если проверка живости обращается к базе, недоступность базы
перезапустит все экземпляры сервиса разом — и к отказу базы добавится отказ сервиса.

**Коды ответов — часть договора:**

| Код | Что значит | Что делать потребителю |
|---|---|---|
| 200 | прогноз получен | использовать; проверить `warnings` |
| 422 | запрос не соответствует схеме | исправить запрос; повтор не поможет |
| 503 | сервис не готов | повторить позже |
| 500 | ошибка внутри сервиса | повторить; сообщить владельцу сервиса |

Обработчик `Exception` нужен, чтобы 500 оставался частью договора: тело ответа — JSON того же вида, что
у остальных ошибок, трассировка — в журнале, а не у потребителя. Проверка — в разделе 9.

## 6. OpenAPI: что получает потребитель

FastAPI строит описание сервиса из сигнатур обработчиков и схем Pydantic. Отдельно писать и обновлять
его не нужно — и оно не может разойтись с кодом.

Документ по адресу `/openapi.json` (версия спецификации 3.1.0) перечисляет четыре эндпоинта с кодами
ответов — `/predict`: 200, 422, 503; `/ready`: 200, 503 — и шесть схем, включая `HTTPValidationError`
для ответа 422. Фрагмент схемы `SaleRequest` (весь документ — 4 448 байт):

```json
{"properties": {
   "store_id": {"type": "string", "pattern": "^S\\d{2}$", "description": "Код магазина, например S01"},
   "product": {"type": "string", "maxLength": 50, "minLength": 1, "description": "Название товара"},
   "date": {"type": "string", "format": "date", "description": "День, на который нужен прогноз"},
   "price": {"type": "number", "maximum": 1000000.0, "exclusiveMinimum": 0.0, "description": "Цена в этот день, рублей"},
   "promo": {"type": "boolean", "description": "Действует ли акция", "default": false}},
 "additionalProperties": false,
 "required": ["store_id", "product", "date", "price"],
 "examples": [{"date": "2025-01-15", "price": 1490, "product": "Мышь", "promo": false, "store_id": "S01"}]}
```

Что с этим делает потребитель:

- **читает.** `/docs` (Swagger UI) и `/redoc` показывают эндпоинты, поля, границы и пример; из `/docs`
  запрос можно отправить прямо в браузере. Оба адреса ответили `200`. Страницы загружают сценарии
  с внешнего CDN: в закрытой сети откроется пустая страница, а `/openapi.json` работает всегда;
- **генерирует клиента.** По `openapi.json` инструменты генерации кода создают типизированного клиента
  на языке потребителя; этот путь в уроке не проверялся запуском;
- **проверяет совместимость.** Сохранённый в репозитории `openapi.json` — снимок договора. Сравнение
  снимков двух версий показывает, что изменилось: добавленное необязательное поле совместимо, удалённое
  или ставшее обязательным — нет.

Всё в схеме взято из кода: `description`, `summary`, `examples`, `responses={503: …}`. Код 503 без явного
указания в документе не появился бы — FastAPI не знает, что `get_model` может его вернуть. Чего в OpenAPI
нет: времени ответа, допустимой нагрузки, смысла предупреждений. Это остаётся тексту — README сервиса.

## 7. Сколько стоит один прогноз

Сначала — без HTTP, прямой вызов `LoadedModel.predict` после прогрева:

| Строк в вызове | p50, мс | p95, мс | p99, мс | На строку |
|---|---|---|---|---|
| 1 | 3.41 | 4.69 | 6.25 | 3 408 мкс |
| 10 | 3.58 | 5.32 | 6.74 | 359 мкс |
| 100 | 4.55 | 6.67 | 7.14 | 45.5 мкс |
| 1000 | 7.89 | 12.71 | 16.81 | 7.9 мкс |

Одна строка и сто строк стоят почти одинаково. Из 3.4 мс на одну строку: сборка `DataFrame` и признаков
даты — 0.9 мс, `ColumnTransformer` — 1.7 мс, сам градиентный бустинг — 0.6 мс. Пять шестых времени —
постоянные расходы pandas и scikit-learn на вызов, не зависящие от числа строк.

Теперь через HTTP. Сервер — `uvicorn app:app`, один воркер. Клиент — скрипт на `httpx.AsyncClient`
на той же машине: заданное число сопрограмм отправляет запросы одну за другой по постоянным соединениям,
время каждого запроса измеряется `time.perf_counter()`, первые 20 запросов (прогрев) не учитываются,
перцентили считает `np.percentile`.

| Нагрузка | p50, мс | p95, мс | p99, мс | Запросов/с | Строк/с |
|---|---|---|---|---|---|
| `/predict`, 1 клиент, 2 000 запросов | 5.3 | 6.8 | 8.1 | 182 | 182 |
| `/predict/batch` по 100 строк, 1 клиент, 500 запросов | 6.5 | 8.0 | 8.9 | 152 | 15 243 |

(Замер с `OMP_NUM_THREADS=1`; почему — в разделе 8.) Разбор JSON, проверка схемой, сборка ответа и сеть
добавили к 3.4 мс ещё около двух. Сто строк одним запросом обрабатываются за 6.5 мс вместо 530 мс
по одной — в 84 раза больше строк в секунду. Потребителю, которому нужен прогноз по всем товарам
магазина, пакетный эндпоинт нужен не для удобства.

Почему три числа, а не среднее: p50 — типичный запрос, p99 — каждый сотый. Потребитель, делающий
десять запросов на страницу, попадает на p99 на каждой десятой странице. Обещание о задержке дают
по p95 или p99.

## 8. `def` или `async def`

FastAPI выполняет обработчики двух видов по-разному:

- `async def` — прямо в цикле событий. Пока обработчик не дошёл до `await`, цикл не делает больше ничего:
  не принимает соединения, не отвечает на `/health`;
- `def` — в отдельном потоке из пула (40 потоков по умолчанию); цикл событий свободен.

Модель считает на процессоре и ничего не ждёт: `await` внутри `pipeline.predict` нет. К обработчику
`def` из раздела 4 добавлены два варианта (`run_in_threadpool` — из `fastapi.concurrency`):

```python
@app.post("/predict-async")           # модель считает прямо в цикле событий
async def predict_async(item: SaleRequest, request: Request) -> PredictResponse:
    return get_model(request).predict([item])[0]


@app.post("/predict-thread")          # async def, расчёт вынесен в поток явно
async def predict_thread(item: SaleRequest, request: Request) -> PredictResponse:
    model = get_model(request)
    return (await run_in_threadpool(model.predict, [item]))[0]
```

Один воркер, 16 одновременных клиентов, 4 000 запросов; параллельно раз в 20 мс опрашивается `/health`.
`OMP_NUM_THREADS=1`:

| Обработчик | p50, мс | p95, мс | p99, мс | Запросов/с | `/health` p50, мс | `/health` p99, мс |
|---|---|---|---|---|---|---|
| `def` | 99.8 | 146.9 | 182.3 | 155 | 76.8 | 143.5 |
| `async def` | 59.7 | 117.0 | 125.4 | 250 | 518.2 | 791.2 |
| `async def` + `run_in_threadpool` | 95.6 | 144.6 | 176.6 | 160 | 114.3 | 205.5 |

Результат не совпал с расхожим «`async def` для модели — всегда плохо»:

- **пропускная способность у `async def` выше** — 250 запросов в секунду против 155. Процесс Python
  исполняет байт-код в одном потоке (GIL); 16 потоков пула не считают параллельно, а отбирают друг
  у друга право на исполнение. Строго последовательная обработка в цикле событий оказалась дешевле;
- **но всё остальное в это время стоит.** `/health` при `async def` отвечает за полсекунды в медиане
  и за 0.8 секунды в p99 — проверка живости с таймаутом в секунду на таком сервисе начнёт срабатывать
  под обычной нагрузкой, и оркестратор перезапустит здоровый процесс. При `def` — 77 мс;
- `async def` с `run_in_threadpool` ведёт себя как `def`: это он и есть, записанный явно;
- под нагрузкой 16 клиентов один воркер с `def` обслуживает **меньше** запросов, чем от одного клиента
  (155 против 182 в секунду), а p50 вырос с 5 до 100 мс. Запросы стоят в очереди; очередь не ускоряет.

Вывод для CPU-модели: `def`, один расчёт в один момент времени на процесс, параллельность — процессами.
`async def` уместен там, где обработчик ждёт — другой сервис, базу, хранилище признаков, — и ждёт через
`await`.

**Потоки внутри модели.** `HistGradientBoostingRegressor` считает прогноз в несколько потоков OpenMP —
по числу ядер. Для одной строки это накладные расходы без выигрыша. Те же замеры без `OMP_NUM_THREADS=1`:

| Обработчик, 1 воркер, 16 клиентов | p50, мс | p99, мс | Запросов/с | `/health` p50, мс |
|---|---|---|---|---|
| `def` | 134.0 | 203.2 | 118 | 16.2 |
| `async def` | 163.4 | 352.6 | 92 | 1 446.5 |

С потоками OpenMP оба варианта медленнее, а `async def` — медленнее `def` и отвечает на `/health`
полторы секунды. Пакет из 100 строк от одного клиента: 13.6 мс против 6.5 мс. Почему `async def`
страдает от потоков OpenMP сильнее, в уроке не разбиралось; воспроизводится устойчиво и исчезает при
`OMP_NUM_THREADS=1`. Для сервиса с маленькими запросами эта переменная окружения — первое, что стоит
проверить замером.

**Несколько воркеров.** `uvicorn app:app --workers 4` запускает четыре процесса, каждый со своей копией
модели. 16 клиентов, 4 000 запросов, `OMP_NUM_THREADS=1`:

| Конфигурация | p50, мс | p95, мс | p99, мс | Запросов/с |
|---|---|---|---|---|
| 1 воркер, `def` | 99.8 | 146.9 | 182.3 | 155 |
| 4 воркера, `def` | 28.5 | 75.7 | 105.5 | 470 |
| 4 воркера, `async def` | 23.2 | 66.0 | 108.1 | 563 |
| 4 воркера, `def`, без `OMP_NUM_THREADS=1` | 51.9 | 95.4 | 123.3 | 293 |
| 4 воркера, `/predict/batch` по 100 строк | 28.9 | 84.4 | 138.3 | 433 (43 295 строк/с) |

Четыре процесса дали втрое больше, а не вчетверо: клиент нагрузки работает на тех же четырёх ядрах.
Без ограничения потоков четыре воркера по четыре потока OpenMP делят четыре ядра — 293 запроса в секунду
вместо 470. Цена воркеров — память: модель и библиотеки загружены в каждом процессе.

**Ловушка, найденная замером.** Первый прогон с четырьмя воркерами был сделан с `uvicorn` без дополнений
(без `uvloop`). Результат для одного клиента:

| Сервер | `/health` p50, мс | `/predict` p50, мс |
|---|---|---|
| 1 воркер, цикл `asyncio` | 0.99 | 5.08 |
| 2 или 4 воркера, цикл `asyncio` (`--loop asyncio`) | 44.0 | 48.0 |
| 4 воркера, `uvloop` | 0.64 | 5.49 |

Сорок с лишним миллисекунд на любой запрос, включая `/health`. Причина — не модель: при нескольких
воркерах uvicorn 0.54.0 сам создаёт слушающий сокет и передаёт его процессам, а стандартный цикл
`asyncio` включает `TCP_NODELAY` только на соединениях, у сокета которых явно указан протокол TCP.
Проверка на двух серверах `asyncio`:

```text
create_server(host, port): proto, TCP_NODELAY = (6, 1)
create_server(sock=...):   proto, TCP_NODELAY = (0, 0)
```

Без `TCP_NODELAY` вторая часть ответа ждёт подтверждения первой, а клиент откладывает подтверждение —
известное сочетание алгоритма Нейгла и отложенных подтверждений TCP, оно и даёт около 40 мс.
С `uvloop` (он входит в `uvicorn[standard]`) задержки нет. Этот эффект нельзя было предсказать, читая
код сервиса; его показал только замер `/health` — эндпоинта, который не делает ничего.

## 9. Тестирование через `TestClient`

`TestClient` вызывает приложение напрямую, без сети и без запуска сервера. В `starlette 1.10.0` он
работает поверх пакета `httpx2`; с прежним `httpx` импорт выдаёт предупреждение
`Using httpx with starlette.testclient is deprecated; install httpx2 instead`, а при `-W error` — падает.

```python
import pytest
from fastapi.testclient import TestClient

from app import app

ITEM = {"store_id": "S01", "product": "Мышь", "date": "2025-01-15", "price": 1490, "promo": False}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "model")
    with TestClient(app) as test_client:  # вход в with запускает lifespan: модель загружена
        yield test_client


def test_batch_matches_single_requests(client):
    items = [ITEM, {**ITEM, "store_id": "S08"}, {**ITEM, "promo": True, "price": 1270}]
    single = [client.post("/predict", json=item).json()["units"] for item in items]
    batch = client.post("/predict/batch", json={"items": items}).json()
    assert [row["units"] for row in batch["predictions"]] == single


def test_service_without_model_is_alive_but_not_ready(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))  # пустой каталог: модели нет
    with TestClient(app) as test_client:
        assert test_client.get("/health").status_code == 200
        assert test_client.get("/ready").status_code == 503
        assert test_client.post("/predict", json=ITEM).status_code == 503


def test_unexpected_error_does_not_leak_details(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "model")
    with TestClient(app, raise_server_exceptions=False) as test_client:
        monkeypatch.setattr(app.state.model.pipeline, "predict", lambda frame: 1 / 0)
        response = test_client.post("/predict", json=ITEM)
    assert response.status_code == 500
    assert response.json() == {"detail": "внутренняя ошибка сервиса"}
```

Вместе с тестом верного ответа и параметризованным тестом четырёх неверных запросов (`price: 0`,
`store_id: "1"`, `date: "вчера"`, лишнее поле) — `8 passed in 3.56s` при запуске
`uv run python -W error -m pytest -q`.

На что обратить внимание:

- **`with TestClient(app)`.** `lifespan` выполняется только при входе в `with`. Без него `app.state.model`
  не существует, и первый же запрос падает с `AttributeError` — частая причина «тесты не работают»;
- **окружение до старта.** `MODEL_DIR` читается в `lifespan`, поэтому `monkeypatch.setenv` стоит до `with`;
- **`raise_server_exceptions=False`.** По умолчанию `TestClient` пробрасывает исключение обработчика
  в тест — удобно при отладке, но ответ 500 так не проверить;
- **пакет против одиночных.** Тест сверяет два пути получения одного числа; он поймает и перепутанный
  порядок строк в пакете.

Тесты проверяют договор, а не модель: коды ответов, форму тела, границы входа. Качество прогноза
проверяется при обучении (темы 7.1 и 7.2), время ответа — нагрузочным замером.

## Типичные ошибки

| Симптом | Причина | Решение |
|---|---|---|
| Время ответа вдвое больше времени предсказания; память растёт под нагрузкой | модель загружается в обработчике | загрузка в `lifespan`, объект — в `app.state` |
| `AttributeError: 'State' object has no attribute 'model'` в тестах | `TestClient(app)` без `with`: `lifespan` не выполнился | `with TestClient(app) as client:` |
| `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated` | установлен `httpx`, но не `httpx2` | `uv add --dev httpx2` |
| Запрос с `"promo": "yes"` или `"price": "1490"` принят | мягкий режим Pydantic приводит типы | `StrictBool`, `Field(strict=True)` |
| Со `ConfigDict(strict=True)` отклоняются верные даты | в строгом режиме строка не превращается в `date` | строгость по полям, а не на модель |
| Опечатка в имени поля не замечена, прогноз «не учитывает цену» | лишние поля отбрасываются | `ConfigDict(extra="forbid")` |
| Для нового магазина сервис возвращает правдоподобное число | `handle_unknown="ignore"` и никакой проверки | сверять с известными категориями: предупреждение или отказ |
| `/health` отвечает секунду, оркестратор перезапускает сервис под нагрузкой | модель считает в `async def` и занимает цикл событий | обработчик `def`; число воркеров по числу ядер |
| Четыре воркера не быстрее одного | потоки OpenMP делят те же ядра | `OMP_NUM_THREADS=1` и замер |
| Каждый запрос на 40 мс дольше при `--workers` больше 1 | `uvicorn` без `uvloop`: на соединениях нет `TCP_NODELAY` | `uvicorn[standard]`; проверить замером `/health` |
| Потребитель получает HTML или трассировку при ошибке | нет обработчика непредвиденных исключений | `@app.exception_handler(Exception)` с JSON-ответом |
| В `/docs` нет кода 503 | FastAPI не видит `HTTPException` внутри вызываемых функций | `responses={503: …}` в декораторе |
| Запрос на 100 000 строк «подвешивает» сервис | размер пакета не ограничен | `Field(max_length=…)` у списка |

## Контрольные вопросы

1. Что проверяет схема Pydantic и чего она проверить не может? Приведите пример запроса, который проходит
   схему, но не имеет смысла.
2. Почему `ConfigDict(strict=True)` на модели запроса сломал приём дат? Как задать строгость иначе?
3. Чем `/health` отличается от `/ready`? Что произойдёт, если `/health` будет проверять доступность базы?
4. Сколько стоила бы загрузка модели в каждом запросе по сравнению с самим предсказанием?
5. Что вернул сервис для магазина `S99` и почему модель не упала? Какие три политики возможны?
6. Откуда в `openapi.json` взялись границы цены и пример запроса? Почему код 503 пришлось указать вручную?
7. Почему при одном воркере `async def` дал больше запросов в секунду, чем `def`? Почему его всё равно
   не стоит использовать для CPU-модели?
8. Во сколько раз пакет из 100 строк дешевле ста одиночных запросов и за счёт чего?
9. Что такое p99 и почему обещание о задержке не дают по среднему?
10. Что показал замер `/health` при четырёх воркерах без `uvloop` и в чём была причина?

## Домашнее задание

1. Обучите модель скриптом из раздела 1 и поднимите сервис. Приведите ответы на шесть запросов: верный,
   с акцией, с незнакомым магазином, с отрицательной ценой, с лишним полем, с пустым пакетом.
2. Добавьте в `SaleRequest` проверку, которой нет в уроке: дата не раньше последнего дня обучения
   (`trained_on` из `metadata.json`) или цена в пределах ±50 % от цены товара в обучающих данных. Решите,
   это отказ или предупреждение, и обоснуйте.
3. Сохраните `openapi.json` в репозиторий. Измените схему двумя способами — совместимым и несовместимым —
   и покажите разницу файлов для каждого.
4. Повторите замеры разделов 7 и 8 на своей машине: одна строка и пакет из 100; `def` и `async def`
   при 16 клиентах с опросом `/health`; один воркер и число воркеров по числу ядер; с `OMP_NUM_THREADS=1`
   и без; `--loop asyncio --workers 2` против `uvloop`. Запишите число ядер и версии пакетов. Если
   какой-то эффект из урока не воспроизвёлся — так и запишите.
5. Напишите тесты через `TestClient`: по одному на каждый код ответа из таблицы раздела 5 и тест
   равенства пакетного и одиночных прогнозов.

**Критерии приёмки:**

- [ ] модель загружается один раз в `lifespan`; путь к ней задаётся переменной окружения;
- [ ] `/health` отвечает 200 без модели, `/ready` и `/predict` — 503;
- [ ] в ответе на каждый прогноз есть версия модели; незнакомая категория отражена в ответе;
- [ ] запросы со строкой вместо числа и с лишним полем отклоняются с кодом 422;
- [ ] `openapi.json` лежит в репозитории; показаны совместимое и несовместимое изменения;
- [ ] таблица замеров содержит p50, p95, p99 и запросы в секунду не менее чем по 2 000 запросам после
      прогрева, с указанием числа ядер, воркеров и значения `OMP_NUM_THREADS`;
- [ ] в таблице есть время ответа `/health` под нагрузкой для `def` и `async def`;
- [ ] `uv run python -W error -m pytest -q` проходит; тесты не запускают сервер и не используют сеть;
- [ ] серверы, запущенные для замеров, остановлены; файл модели не попал в Git.

## Дополнительные материалы

- [Документация FastAPI](https://fastapi.tiangolo.com/) — обработчики, `lifespan`, обработка ошибок,
  тестирование, параллельность и `async`
- [Документация Pydantic](https://docs.pydantic.dev/latest/) — поля и ограничения, строгий режим,
  JSON Schema
- [Документация Uvicorn](https://www.uvicorn.org/) — параметры запуска, воркеры, циклы событий
- [Спецификация OpenAPI](https://spec.openapis.org/oas/latest.html)
- [Документация scikit-learn: сохранение моделей](https://scikit-learn.org/stable/model_persistence.html)

**Следующий урок:** 7.4.2 — Docker контейнеризация, Kubernetes deployment
