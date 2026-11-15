# Урок 7.1.2: Apache Airflow для оркестрации пайплайнов

> **Модуль 7:** ML Engineering & MLOps · **Тема 7.1:** ML Pipelines
> **Тип:** теория · **Время:** 120 минут + 120 минут на домашнее задание

## Цели урока

После урока вы сможете:

- объяснить, какие задачи оркестратор решает сверх cron, и определить, когда он избыточен;
- описать пайплайн как DAG в TaskFlow API: зависимости, повторы, таймауты, ветвление;
- не путать логическую дату, интервал данных и момент запуска;
- писать идемпотентные задачи, которые безопасно перезапускать за любую дату;
- отлаживать DAG без планировщика и проверять его структуру тестами.

## Зачем это нужно

В уроке 1.4.3 ночное переобучение было строкой crontab: «выгрузка в 02:05, обучение в 02:30, выгрузка
к этому времени обычно заканчивается». Слово «обычно» — и есть проблема. Оркестратор заменяет расчёт
на время явными зависимостями: обучение начнётся, когда выгрузка **успешно завершилась**, упавший шаг
будет повторён, пропущенные дни — досчитаны, а история запусков видна на одном экране.

## 1. Что не умеет cron

| Потребность | cron / таймер systemd | Оркестратор |
|---|---|---|
| Запустить шаг B после успеха шага A | расчёт на время или один большой скрипт | зависимости между задачами |
| Повторить упавший шаг | вручную | `retries`, `retry_delay` у каждой задачи |
| Досчитать пропущенные дни | вручную, меняя дату в команде | catchup и backfill |
| Перезапустить один шаг за прошлую дату | искать команду и параметры | «очистить» задачу — она выполнится заново |
| Узнать, что и когда упало | читать лог-файлы на сервере | состояние каждой задачи каждого запуска, логи в интерфейсе |
| Ограничить одновременные запуски | `flock` | `max_active_runs`, пулы |
| Исполнять шаги на разных машинах | нет | исполнители (раздел 10) |

## 2. Установка и отдельное окружение

У Airflow больше сотни зависимостей с жёсткими требованиями к версиям, поэтому его ставят в **отдельное
окружение**, а не в окружение, где обучается модель. Официальный способ — установка с файлом ограничений
(constraints), в котором проект фиксирует проверенный набор версий всех зависимостей:

```bash
pip install "apache-airflow==3.4.0" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.4.0/constraints-3.12.txt"
```

Без файла ограничений установщик берёт самые свежие версии зависимостей, и однажды вышедшая новая версия
любой из них ломает установку, которая вчера работала. Команда выше **не проверялась запуском**: файл
лежит на GitHub, недоступном в среде подготовки курса. Примеры урока выполнены в проекте uv, где набор
версий подобран разрешением зависимостей и зафиксирован в `uv.lock`:

```bash
uv init --python 3.12 airflow-demo && cd airflow-demo
uv add apache-airflow scikit-learn pandas pytest
```

Получилось 145 пакетов, среди них: `apache-airflow 3.4.0` (`apache-airflow-core 3.4.0`,
`apache-airflow-task-sdk 1.4.0`, `apache-airflow-providers-standard 1.21.0`), `SQLAlchemy 2.1.8`,
`pydantic 2.14.3`, `fastapi 0.162.5`, а также `scikit-learn 1.9.1`, `pandas 3.1.1`, `numpy 2.5.4`.
В этот раз версии библиотек курса совпали с основным окружением; гарантии, что так будет со следующей
версией Airflow, нет. В рабочей установке используйте файл ограничений.

Синтаксис урока — Airflow 3. В статьях об Airflow 2 вы встретите `schedule_interval` (теперь
`schedule`), `execution_date` (теперь `logical_date`) и импорт `from airflow.decorators import dag, task`
(теперь `from airflow.sdk import dag, task`).

Настройки окружения (всё — внутри папки проекта):

```bash
export AIRFLOW_HOME="$PWD/airflow_home"             # база метаданных, конфиг, логи
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"       # где искать файлы DAG
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export SALES_DATA_DIR="$PWD/data"
export PYTHONPATH="$PWD/src"                        # пакет проекта с ML-кодом
```

Раскладка проекта: `src/sales_ml/steps.py` — шаги, `src/sales_data.py` — копия генератора темы 2.1,
`dags/sales_retrain.py` — DAG, `tests/test_dags.py` — тесты.

## 3. Понятия

| Понятие | Смысл |
|---|---|
| DAG | направленный граф без циклов: задачи и зависимости между ними; описывается файлом Python |
| Задача (task) | один шаг; минимальная единица запуска, повтора и наблюдения |
| Оператор | шаблон задачи: `PythonOperator`, `BashOperator`; декоратор `@task` создаёт задачу из функции |
| Сенсор | задача, которая ждёт события: появления файла, завершения другого DAG (`FileSensor`, `@task.sensor`) |
| Расписание | cron-выражение, интервал или объект `Timetable`; `None` — только ручной запуск |
| Запуск DAG (DAG run) | одно выполнение графа за конкретную логическую дату |
| Экземпляр задачи | задача в конкретном запуске; у него есть состояние: `success`, `failed`, `skipped`, `up_for_retry` |
| XCom | небольшое значение, которое задача передаёт следующим; хранится в базе метаданных |

## 4. ML-код — в пакете проекта

Файл DAG описывает **порядок** шагов. Сами шаги живут в пакете проекта и ничего не знают об Airflow:

```python
def _write_csv(table: pd.DataFrame, path: Path) -> str:
    """Запись через временный файл: читатель никогда не увидит наполовину записанную партицию."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(path)
    return str(path)


def extract(day: date, data_dir: str | Path) -> str:
    """Выгружает продажи за один день в партицию raw/date=YYYY-MM-DD/sales.csv."""
    sales = _source()                    # «хранилище»: make_sales() из sales_data.py
    part = sales[sales["date"] == pd.Timestamp(day)]
    if part.empty:
        raise ValueError(f"в источнике нет продаж за {day}")
    return _write_csv(part, partition_path(data_dir, day))
```

Остальные шаги устроены так же — явные входы, результат на диске, путь к нему в ответе:

| Функция | Что делает | Результат |
|---|---|---|
| `validate(path, day)` | даты, дубликаты ключа, отрицательные значения, доля пропусков; при нарушении — `ValueError` | тот же путь |
| `build_features(day, data_dir, history_days=90)` | склеивает партиции за 90 дней, добавляет `units_week_ago` и день недели | `features/date=.../features.csv` |
| `train(features_path, day, data_dir)` | обучает пайплайн `Ridge` + `log1p` (урок 7.1.1) на всех днях, кроме последних семи | `models/date=.../model.joblib` |
| `evaluate(model_path, features_path)` | MAE модели и MAE прогноза «как неделю назад» на последних семи днях | словарь метрик и `metrics.json` |
| `publish(model_path, data_dir)` | копирует модель в `published/` и обновляет указатель `current.json` | путь к опубликованной модели |

## 5. DAG ежедневного переобучения

```python
import os
from datetime import datetime, timedelta

from airflow.sdk import dag, task
from airflow.timetables.interval import CronDataIntervalTimetable

DATA_DIR = os.environ.get("SALES_DATA_DIR", "/opt/airflow/data")
MAX_MAE_RATIO = 0.9  # модель публикуется, если её MAE не больше 90 % от MAE прогноза «как неделю назад»


@dag(
    dag_id="sales_retrain",
    schedule=CronDataIntervalTimetable("@daily", timezone="UTC"),
    start_date=datetime(2024, 6, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5), "execution_timeout": timedelta(minutes=30)},
)
def sales_retrain():
    @task
    def extract(data_interval_start=None) -> str:
        from sales_ml import steps  # тяжёлые импорты — внутри задачи, а не на верхнем уровне файла

        return steps.extract(data_interval_start.date(), DATA_DIR)

    @task(retries=0)
    def validate(path: str, data_interval_start=None) -> str:
        from sales_ml import steps

        return steps.validate(path, data_interval_start.date())

    # features, train (execution_timeout=1 час), evaluate, publish — такие же обёртки над steps;
    # keep_current(metrics) только пишет метрики в лог

    @task.branch
    def decide(metrics: dict) -> str:
        good = metrics["mae"] <= MAX_MAE_RATIO * metrics["mae_week_ago"]
        return "publish" if good else "keep_current"

    raw = extract()
    feats = features(validate(raw))
    model = train(feats)
    metrics = evaluate(model, feats)
    decide(metrics) >> [publish(model), keep_current(metrics)]


sales_retrain()
```

Что здесь важно.

**TaskFlow API.** `@task` превращает функцию в задачу, а вызов `validate(raw)` — в зависимость
«`validate` после `extract`». Сама функция при разборе файла не выполняется: `raw` — не строка с путём,
а ссылка на будущий результат. Для задач без передачи данных зависимость задаётся оператором `>>`.

**XCom — только ссылки.** Возвращаемое значение задачи сохраняется в базе метаданных Airflow и передаётся
следующей задаче. Это место для пути к файлу, идентификатора модели, словаря из трёх метрик — но не для
таблицы на 50 тысяч строк: база метаданных не хранилище данных, а задачи могут исполняться на разных
машинах. Данные лежат в хранилище (диск, S3, база), через XCom передаётся **адрес**.

**Контекст.** Параметр `data_interval_start=None` Airflow заполняет сам из контекста запуска — так задача
узнаёт, за какой день работать (раздел 6). `datetime.now()` в задаче — ошибка: при перезапуске за прошлую
дату она обработает сегодняшний день.

**Повторы и таймауты.** `default_args` применяются ко всем задачам: две повторные попытки с паузой
5 минут — от временных сбоев сети и хранилища; `execution_timeout` — от зависания. У `validate` повторы
отключены: испорченные данные от повторной проверки не исправятся.

**Ветвление.** Задача `@task.branch` возвращает `task_id` той ветви, которую нужно выполнить; остальные
получают состояние `skipped`. Так реализовано правило «публиковать, только если качество не хуже порога».

**Код верхнего уровня.** Файл DAG разбирается процессом Airflow каждые несколько десятков секунд. Всё,
что стоит вне функций задач, выполняется при каждом разборе: запрос к базе или `import torch` на верхнем
уровне замедляют весь планировщик. Поэтому импорт `steps` — внутри задач.

## 6. Логическая дата и интервал данных

Самое частое место путаницы. Запуск DAG обрабатывает **интервал данных** и стартует, когда этот интервал
**закончился** — раньше данных за него ещё нет.

```text
        интервал данных запуска A         интервал данных запуска B
    ├───────────────────────────────┼───────────────────────────────┤
 25 июня 00:00                   26 июня 00:00                   27 июня 00:00
    ▲                               ▲
    │                               └── в этот момент запуск A стартует (run_after)
    └── data_interval_start = logical_date запуска A; задача extract выгружает 25 июня
```

Логическая дата — это метка «за какой период запуск», а не время, когда он выполнялся. Запуск
с `logical_date = 2024-06-25` исполняется 26 июня; при перезапуске через год он по-прежнему обработает
25 июня.

В Airflow 3 есть второй вид расписаний, и он используется по умолчанию для cron-строк (параметр
`[scheduler] create_cron_data_intervals` равен `False`): запуск привязан к моменту, а интервал данных
вырожден. Что видит задача — проверено запусками трёх DAG с одной задачей, печатающей контекст:

| Расписание и способ запуска | `logical_date` | `data_interval_start` | `data_interval_end` |
|---|---|---|---|
| `schedule="0 3 * * *"`, backfill | 27 июня 03:00 | 27 июня 03:00 | 27 июня 03:00 |
| `CronDataIntervalTimetable("0 3 * * *")`, backfill | 27 июня 03:00 | 27 июня 03:00 | 28 июня 03:00 |
| `CronDataIntervalTimetable("@daily")`, backfill | 25 июня 00:00 | 25 июня 00:00 | 26 июня 00:00 |
| `CronDataIntervalTimetable("@daily")`, `dags test ... 2024-07-01` | 1 июля 00:00 | 30 июня 00:00 | 1 июля 00:00 |

Выводы: при cron-строке «обработать вчерашний день» — это `logical_date` минус сутки, при интервальном
расписании — `data_interval_start`; у ручного запуска интервал — последний **завершившийся** к логической
дате. В DAG урока расписание задано явно объектом `CronDataIntervalTimetable`, а задачи читают
`data_interval_start`: поведение не зависит от настройки по умолчанию.

**Catchup и backfill.** `catchup=True` означает: при включении DAG планировщик создаст запуски за все
пропущенные интервалы от `start_date`. В Airflow 3 по умолчанию `catchup=False`: создаётся только запуск
за последний завершившийся интервал. Backfill — явный расчёт за диапазон дат одной командой (раздел 9).
Оба механизма работают только с идемпотентными задачами.

## 7. Идемпотентность

Задача идемпотентна, если повторный запуск **за ту же логическую дату** даёт тот же результат и не
оставляет следов первого запуска. Это главное требование к задачам: оркестратор перезапускает их при
повторах, при backfill и когда вы «очищаете» упавшую задачу.

| Не идемпотентно | Идемпотентно |
|---|---|
| дописать строки в общий файл: `to_csv("sales.csv", mode="a")` | перезаписать партицию `raw/date=2024-06-30/sales.csv` целиком |
| `INSERT INTO sales ...` | `DELETE ... WHERE date = :day` и `INSERT` в одной транзакции |
| «данные за вчера» через `datetime.now()` | дата из контекста запуска |
| `model_latest.joblib` | `models/date=2024-06-30/model.joblib` |

Проверка на `extract`: два запуска за одну дату, контрольная сумма и число строк партиции не меняются.

```bash
sha256sum data/raw/date=2024-06-30/sales.csv     # 83a3d23dc30d79c2...
airflow tasks test sales_retrain extract 2024-07-01
sha256sum data/raw/date=2024-06-30/sales.csv     # 83a3d23dc30d79c2...
wc -l < data/raw/date=2024-06-30/sales.csv       # 97: заголовок и 96 строк, дубликатов нет
```

С `mode="a"` после второго запуска в файле 194 строки — заголовок и все 96 пар записаны дважды, —
и модель обучилась бы на удвоенном дне. Запись через
временный файл с переименованием (раздел 4) закрывает ещё один сценарий: задача, убитая по таймауту
посреди записи, не оставляет половину партиции под настоящим именем.

## 8. Отладка без планировщика

Перед первым запуском загрузим историю — 121 партицию с 1 марта по 29 июня — тем же шагом (5 секунд):

```bash
uv run python -c "
from datetime import date, timedelta
from sales_ml import steps
for i in range(121):
    steps.extract(date(2024, 3, 1) + timedelta(days=i), 'data')"
uv run airflow db migrate            # создать базу метаданных (SQLite в AIRFLOW_HOME)
uv run airflow dags reserialize      # разобрать файлы DAG и записать их в базу
uv run airflow dags list             # sales_retrain | .../dags/sales_retrain.py | airflow | True | ...
uv run airflow dags list-import-errors     # No data found — ошибок импорта нет
```

До `dags reserialize` команда `dags list` отвечает `No data found`: в Airflow 3 список DAG берётся
из базы, а заполняет её отдельный процесс разбора, который здесь не запущен.

Весь DAG за одну логическую дату, в одном процессе, без планировщика:

```bash
uv run airflow dags test sales_retrain 2024-07-01
```

Вывод (16 секунд; оставлены ключевые строки, убраны метки времени и имена логгеров):

```text
[info ] [DAG TEST] starting task_id=extract map_index=-1
[info ] Done. Returned value was: <proj>/data/raw/date=2024-06-30/sales.csv
[info ] [DAG TEST] end task task_id=extract map_index=-1
...
[info ] Downstream tasks skipped       tasks_skipped=1
[info ] [DAG TEST] end task task_id=decide map_index=-1
[info ] [DAG TEST] starting task_id=publish map_index=-1
[info ] Done. Returned value was: <proj>/data/published/model.joblib
[info ] Marking run <DagRun sales_retrain @ 2024-07-01 00:00:00+00:00: manual__..., state:running, ...> successful
```

Логическая дата — 1 июля, выгружена партиция за 30 июня (раздел 6). Метрики запуска:
`{"mae": 1.329, "mae_week_ago": 1.9654, "n_valid": 669}` — отношение 0.68 меньше порога 0.9, выполнена
ветвь `publish`, а `keep_current` пропущена. Одна задача запускается командой
`airflow tasks test sales_retrain extract 2024-07-01` — без проверки зависимостей.

Что происходит при ошибке: за 28 февраля 2025 года в источнике данных нет.

```text
$ uv run airflow dags test sales_retrain 2025-03-01
[error] Task failed with exception
ValueError: в источнике нет продаж за 2025-02-28
[info ] Task instance state updated    new_state=up_for_retry
[warning] No tasks to run. unrunnable tasks: {<TaskInstance: sales_retrain.keep_current ...
```

Задача перешла в `up_for_retry`, и команда стала ждать `retry_delay` — 5 минут до каждой из двух
повторных попыток (ожидание было прервано вручную). Повторы честно отрабатываются и при отладке;
на время разработки удобно уменьшать `retry_delay`.

## 9. Запуск с планировщиком

`airflow standalone` поднимает на одной машине всё сразу: планировщик, процесс разбора DAG, API-сервер
с веб-интерфейсом и базу SQLite. Это режим для знакомства, не для работы.

```bash
export AIRFLOW__API__PORT=18080
uv run airflow standalone            # пароль пользователя admin — в выводе и в файле
                                     # airflow_home/simple_auth_manager_passwords.json.generated
```

Сервис ответил на `/api/v2/monitor/health` через 15 секунд. Досчитаем три прошлых дня:

```bash
uv run airflow dags unpause sales_retrain
uv run airflow backfill create --dag-id sales_retrain --from-date 2024-06-25 --to-date 2024-06-27
```

Состояние запусков — через REST API (без токена сервер отвечает `401`; токен выдаётся по паролю):

```bash
TOKEN=$(curl -s -X POST http://localhost:18080/auth/token -H 'Content-Type: application/json' \
  -d '{"username": "admin", "password": "..."}' | python -c "import json,sys; print(json.load(sys.stdin)['access_token'])")
curl -s -H "Authorization: Bearer $TOKEN" "http://localhost:18080/api/v2/dags/sales_retrain/dagRuns"
```

| `run_type` | `logical_date` | Интервал данных | `state` |
|---|---|---|---|
| `backfill` | 2024-06-25 00:00 | 25 июня — 26 июня | `success` |
| `backfill` | 2024-06-26 00:00 | 26 июня — 27 июня | `success` |
| `backfill` | 2024-06-27 00:00 | 27 июня — 28 июня | `success` |
| `scheduled` | 2026-11-12 00:00 | 12 ноября — 13 ноября 2026 | `running` |

Три запуска backfill выполнились по очереди (`max_active_runs=1`) меньше чем за полторы минуты. Задачи одного
из них: `extract`, `validate`, `features`, `train`, `evaluate`, `publish` — `success`, по 3.5–4 секунды
каждая; `decide` — 0.2 секунды; `keep_current` — `skipped`. Четыре секунды на задачу при миллисекундной
работе — цена запуска отдельного процесса на каждую задачу: дробить пайплайн на десятки крошечных задач
невыгодно. XCom задачи `evaluate` по адресу `.../taskInstances/evaluate/xcomEntries/return_value`:
`"value":{"mae":1.3308,"mae_week_ago":1.9624,"n_valid":668}`.

Четвёртая строка — работа планировщика: после `unpause` он сам создал запуск за последний завершившийся
интервал (`catchup=False`). Данных за 2026 год в источнике нет, `extract` находится в `up_for_retry` —
в интерфейсе это сразу видно, в cron об этом узнали бы из лог-файла.

По окончании остановите `airflow standalone` (Ctrl+C) и убедитесь, что процессов не осталось:
`pgrep -f airflow`.

## 10. Исполнители, пулы и тяжёлые вычисления

| Исполнитель | Где выполняются задачи | Когда выбирать |
|---|---|---|
| `LocalExecutor` (по умолчанию) | процессы на машине планировщика | одна машина, десятки задач |
| `CeleryExecutor` | постоянные воркеры на нескольких машинах, очередь задач | много коротких задач, постоянная нагрузка |
| `KubernetesExecutor` | отдельный под Kubernetes на каждую задачу | разные ресурсы и образы у задач, нагрузка всплесками |

Celery и Kubernetes ставятся отдельными пакетами-провайдерами и здесь **не проверялись запуском**.

Ограничения параллелизма: `parallelism` (32 задачи на всю установку), `max_active_tasks_per_dag` (16),
`max_active_runs` у DAG и **пулы** — именованные наборы слотов (в `default_pool` их 128). Пул из двух
слотов для задач, обращающихся к одной базе, гарантирует, что одновременно к ней пойдут не больше двух.

Воркер Airflow — не место для обучения модели. Тяжёлая задача в процессе воркера делит с ним память
и зависимости: `torch` и Airflow в одном окружении — источник конфликтов версий. Оркестратор должен
**запускать** вычисление, а не выполнять его: задача стартует контейнер с образом проекта
(`DockerOperator`, `KubernetesPodOperator` из провайдеров) или отправляет задание во внешнюю систему
и ждёт результата. Образ собирается так же, как в уроке 1.1.3; эти операторы **не проверялись запуском**.
В уроке шаги лёгкие (обучение — доли секунды), поэтому они выполняются прямо в задачах.

## 11. Альтернативы и когда оркестратор не нужен

Сравнение по документации; инструменты, кроме Airflow, **не запускались**.

| Инструмент | Главная идея | Сильная сторона | Ограничение |
|---|---|---|---|
| Airflow | DAG задач по расписанию | зрелость, сотни готовых интеграций, стандарт в индустрии | тяжёлая установка, динамические графы неудобны |
| Prefect | обычные функции Python с декораторами `@flow`, `@task` | низкий порог входа, динамические графы | меньше готовых интеграций |
| Dagster | граф **данных** (assets), а не задач | происхождение данных и проверки качества встроены | новая модель мышления, перенос существующих скриптов дороже |
| Luigi | задачи с целями (`output()`), как в Makefile | простота, нет отдельного сервиса планирования | нет расписаний, проект развивается медленно |

Оркестратор избыточен, если шагов два-три и они идут строго подряд (один скрипт под таймером systemd
решает задачу); если пайплайн запускается вручную раз в месяц; если команда — один человек и некому
следить за ещё одним сервисом. Признаки, что пора: появились зависимости между расписаниями, регулярные
перезапуски за прошлые даты, вопрос «а что вчера упало?» звучит чаще раза в неделю.

## 12. Тестирование DAG

```python
import pytest

pytest.importorskip("airflow")

from airflow.models import DagBag  # noqa: E402


@pytest.fixture(scope="module")
def dagbag():
    return DagBag()                              # разбирает все файлы из DAGS_FOLDER


def test_no_import_errors(dagbag):
    assert dagbag.import_errors == {}


def test_order(dagbag):
    dag = dagbag.get_dag("sales_retrain")
    assert dag.get_task("validate").upstream_task_ids == {"extract"}
    assert dag.get_task("train").upstream_task_ids == {"features"}
    assert dag.get_task("publish").upstream_task_ids == {"decide", "train"}
    assert dag.get_task("decide").downstream_task_ids == {"publish", "keep_current"}
# в файле ещё два теста: состав задач и настройки повторов и таймаутов
```

```text
4 passed in 2.15s
```

Первый тест ловит синтаксические ошибки, отсутствующие импорты и циклы: DAG с зависимостью
`a >> b >> a` попадает в `import_errors` с сообщением
`AirflowDagCycleException: Cycle detected in Dag: cycle_demo. Faulty task: b`. Логику шагов проверяют
обычные тесты пакета `sales_ml` — им Airflow не нужен.

## Типичные ошибки

| Симптом | Причина | Решение |
|---|---|---|
| После backfill в данных дубликаты | задача дописывает данные, а не перезаписывает партицию | идемпотентная запись по дате (раздел 7) |
| Перезапуск за прошлую дату обработал сегодняшние данные | `datetime.now()` в задаче | дата — только из контекста: `data_interval_start`, `logical_date` |
| DAG «опаздывает на сутки» | запуск за интервал стартует после его окончания | это штатное поведение; см. временную шкалу в разделе 6 |
| Ошибка сериализации XCom или распухшая база метаданных | через XCom передают таблицы и модели | передавать пути и идентификаторы |
| Воркер падает по памяти на обучении | тяжёлое вычисление в процессе воркера | запуск в контейнере или внешней системе |

## Контрольные вопросы

1. Назовите четыре возможности оркестратора, которых нет у cron.
2. DAG с `CronDataIntervalTimetable("@daily")` и запуск с `logical_date = 2024-06-25`. Когда он стартует
   и какой интервал данных обрабатывает?
3. Чем отличается поведение `schedule="0 3 * * *"` в Airflow 3 по умолчанию от интервального расписания?
4. Что такое идемпотентная задача? Как сделать идемпотентной вставку строк в таблицу базы данных?
5. Почему через XCom нельзя передавать таблицу признаков? Что передают вместо неё?
6. Что выполняется при разборе файла DAG, а что — при запуске задачи? Почему это важно?
7. Почему обучение модели выносят из процесса воркера?

## Домашнее задание

1. Создайте проект с Airflow в отдельном окружении, воспроизведите DAG `sales_retrain` и добейтесь
   успешного `airflow dags test sales_retrain 2024-07-01`.
2. Напишите DAG `dates_demo` с одной задачей, печатающей `logical_date`, `data_interval_start`
   и `data_interval_end`, в двух вариантах расписания: `"0 6 * * *"` и
   `CronDataIntervalTimetable("0 6 * * *", timezone="UTC")`. Запустите оба через `dags test`
   за 2024-07-01 и объясните разницу.
3. Сломайте идемпотентность: замените запись партиции на дозапись в общий файл, выполните задачу
   дважды и покажите удвоение строк. Верните правильный вариант.
4. Напишите тесты DAG: импорт без ошибок, состав задач, порядок, настройки повторов и таймаутов.

**Критерии приёмки:**

- [ ] Airflow установлен в отдельный проект uv; в отчёте указаны точные версии;
- [ ] `airflow dags test sales_retrain 2024-07-01` завершается успешно, в `data/published/` есть модель;
- [ ] в отчёте — вывод обоих вариантов `dates_demo` и объяснение разницы в два-три предложения;
- [ ] показано удвоение строк при дозаписи и его отсутствие при записи партиции;
- [ ] тесты DAG проходят; файл DAG не содержит импортов pandas и scikit-learn на верхнем уровне;
- [ ] `AIRFLOW_HOME`, данные и логи не попали в Git; процессы Airflow остановлены.

## Дополнительные материалы

- [Installation from PyPI](https://airflow.apache.org/docs/apache-airflow/stable/installation/installing-from-pypi.html)
  — установка с файлом ограничений
- [DAG Runs](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dag-run.html) —
  интервал данных, catchup, backfill
- [Pythonic DAGs with the TaskFlow API](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/taskflow.html)
- [Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html) —
  идемпотентность, код верхнего уровня, тестирование DAG

**Следующий урок:** 7.1.3 — Kubeflow Pipelines, MLflow Projects.
