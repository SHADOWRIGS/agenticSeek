# Урок 7.1.3: Kubeflow Pipelines, MLflow Projects

> **Модуль 7:** ML Engineering & MLOps · **Тема 7.1:** ML Pipelines
> **Тип:** теория · **Время:** 120 минут + 90 минут на домашнее задание

## Цели урока

После урока вы сможете:

- различать две задачи, которые называют словом «пайплайн»: воспроизводимый запуск и оркестрацию шагов;
- оформить проект как MLflow Project: точки входа, параметры, окружение — и запустить его одной командой
  из папки и из git-репозитория;
- описать пайплайн Kubeflow из компонентов с типизированными входами и выходами, скомпилировать его
  и выполнить локально без кластера;
- объяснить, что даёт исполнение шагов в контейнерах на кластере и чего оно стоит;
- выбрать инструмент под размер команды и задачу.

## Зачем это нужно

Под «ML-пайплайном» понимают три разные вещи:

| Смысл | Вопрос, на который отвечает | Инструмент |
|---|---|---|
| составная модель | как не допустить утечки и расхождения обучения с применением | `sklearn.pipeline.Pipeline` (урок 7.1.1) |
| оркестрация | в каком порядке, когда и где выполняются шаги, что делать при сбое | Airflow (урок 7.1.2), Kubeflow Pipelines |
| воспроизводимый запуск | как другому человеку или машине запустить этот код так же, как я | MLflow Projects, Makefile |

Эти инструменты не конкурируют, а складываются: `Pipeline` — внутри шага обучения, шаг — точка входа
проекта, проект — задача оркестратора. Урок — о двух инструментах с противоположных концов шкалы
сложности: MLflow Projects (один файл в репозитории) и Kubeflow Pipelines (платформа на кластере
Kubernetes).

Шаги — те же функции пакета `sales_ml.steps` из урока 7.1.2: `extract`, `validate`, `build_features`,
`train`, `evaluate`. Ни одна из них не меняется: меняется только обёртка.

## 1. Окружения и версии

Оба инструмента установлены в отдельные проекты uv — как Airflow в прошлом уроке и по той же причине:
у каждого свой набор зависимостей с требованиями к версиям.

```bash
uv init --python 3.12 mlflow-demo && cd mlflow-demo && uv add mlflow scikit-learn pandas
uv init --python 3.12 kfp-demo && cd kfp-demo && uv add kfp scikit-learn pandas
```

| Проект | Ключевые версии |
|---|---|
| MLflow | `mlflow 3.18.0`, `protobuf 7.36.2`, `scikit-learn 1.9.1`, `pandas 3.1.1`, `numpy 2.5.4` |
| Kubeflow Pipelines SDK | `kfp 2.18.0`, `kfp-pipeline-spec 2.18.0`, `kfp-server-api 2.18.0`, `kubernetes 30.1.0`, `protobuf 6.33.6`, `scikit-learn 1.9.1`, `pandas 3.1.1` |

Разрешение зависимостей выбрало разные основные версии `protobuf` — наглядная причина не смешивать
инструменты в одном окружении. Совместная установка в одно окружение не проверялась. Проект нельзя
называть именем пакета: `uv init kfp && uv add kfp` заканчивается ошибкой
`Requirement name kfp matches project name kfp, but self-dependencies are not permitted`.

## 2. MLflow Projects

MLflow состоит из нескольких частей; учёт экспериментов (Tracking) разбирается в теме 7.2. Projects —
самая простая часть: соглашение о том, как описать проект, чтобы его можно было запустить одной
командой, не читая README.

### Файл MLproject

В корне проекта лежит файл `MLproject` (без расширения, формат YAML):

```yaml
name: sales-retrain

python_env: python_env.yaml

entry_points:
  main:
    parameters:
      day: {type: str, default: "2024-06-30"}
      history_days: {type: int, default: 90}
      data_dir: {type: str, default: data}
    command: "python run_pipeline.py --day {day} --history-days {history_days} --data-dir {data_dir}"
  features:
    parameters:
      day: str
      data_dir: {type: str, default: data}
    command: "python run_pipeline.py --day {day} --data-dir {data_dir} --only-features"
```

| Раздел | Смысл |
|---|---|
| `entry_points` | именованные команды проекта; `main` запускается по умолчанию |
| `parameters` | имя, тип и значение по умолчанию; параметр без значения по умолчанию обязателен. Особо обрабатываются только типы `path` и `uri`; остальные обозначения документируют намерение, значение передаётся в команду как строка |
| `command` | строка оболочки; значения параметров подставляются на место `{имя}` |
| `python_env` | описание окружения; варианты — `conda_env` и `docker_env` |

Файл окружения `python_env.yaml`:

```yaml
python: "3.12"
build_dependencies:
  - pip
dependencies:
  - mlflow==3.18.0
  - scikit-learn==1.9.1
  - pandas==3.1.1
```

Точка входа — обычный скрипт с `argparse`; от MLflow в нём три строки:

```python
with mlflow.start_run():  # под `mlflow run` подключается к запуску, который создал MLflow
    for i in range(args.history_days + 7):
        day = args.day - timedelta(days=i)
        steps.validate(steps.extract(day, args.data_dir), day)
    features = steps.build_features(args.day, args.data_dir, args.history_days)
    if args.only_features:
        return
    model = steps.train(features, args.day, args.data_dir)
    metrics = steps.evaluate(model, features)
    mlflow.log_metrics({"mae": metrics["mae"], "mae_week_ago": metrics["mae_week_ago"]})
    mlflow.log_artifact(model)
```

### Запуск

```bash
mlflow run . --env-manager local -P day=2024-06-30
```

```text
INFO mlflow.projects.backend.local: === Running command 'python run_pipeline.py --day 2024-06-30
     --history-days 90 --data-dir data' in run with ID '976485d0d80543ebb3df8dbc31bac682' ===
metrics: {'mae': 1.329, 'mae_week_ago': 1.9654, 'n_valid': 669}
INFO mlflow.projects: === Run (ID '976485d0d80543ebb3df8dbc31bac682') succeeded ===
```

(Вывод сокращён: убраны метки времени и строки о создании базы. Запуск занял 18 секунд.)

Что произошло: MLflow прочитал `MLproject`, проверил наличие параметров и подставил их, создал запись о запуске
и выполнил команду. После запуска в папке проекта появились файл `mlflow.db` (SQLite) и каталог
`mlruns/` — в MLflow 3.18 запуски по умолчанию записываются в `sqlite:///mlflow.db`, а файлы-артефакты
складываются в `mlruns/<эксперимент>/<запуск>/artifacts/`. В записи о запуске без единой строки кода
с нашей стороны оказались параметры `day=2024-06-30`, `history_days=90`, `data_dir=data`, имя точки
входа и тип источника `PROJECT`; метрики `mae=1.329`, `mae_week_ago=1.9654` и файл `model.joblib`
записал скрипт. Оба пути добавьте в `.gitignore`.

```bash
mlflow run . --env-manager local -e features -P day=2024-06-29      # другая точка входа
mlflow run . --env-manager local -e features
# ERROR mlflow.cli: === No value given for missing parameters: 'day' ===
```

### Окружения

Флаг `--env-manager` определяет, в каком окружении выполнится команда:

| Значение | Что делает | Проверено в уроке |
|---|---|---|
| `local` | использует текущее окружение, `python_env.yaml` игнорируется | да |
| `uv` | создаёт окружение по `python_env.yaml` средствами uv | да: первый запуск 26 секунд вместе с установкой пакетов |
| `virtualenv` | то же через venv; нужную версию Python ставит pyenv | нет: не проверялось запуском, pyenv в среде подготовки курса не установлен |
| `conda` | окружение из `conda_env` | нет: не проверялось запуском |

Если флаг не указан и в `MLproject` есть `python_env`, выбирается `virtualenv`. Окружения создаются
в каталоге `~/.mlflow/envs` (переменная `MLFLOW_ENV_ROOT` меняет его) и переиспользуются, пока
не изменится файл окружения. `local` — для разработки; для воспроизводимого запуска на чужой машине
нужен один из остальных.

### Запуск из git

Проект можно запустить по адресу репозитория, указав коммит:

```bash
mlflow run https://github.com/<организация>/<репозиторий>.git -v <коммит> -P day=2024-06-27
```

Запуск с GitHub **не проверялся**: GitHub недоступен в среде подготовки курса. Тот же механизм проверен
на локальном репозитории:

```bash
mlflow run file:///path/to/sales-retrain -v e65a327bfc0f2f6ed12439d7f1b0fef87b59fffc \
  --env-manager local -P day=2024-06-27
# metrics: {'mae': 1.2929, 'mae_week_ago': 1.9398, 'n_valid': 667}
```

MLflow клонировал репозиторий во временный каталог, переключился на коммит и выполнил точку входа.
В записи о запуске появились теги `mlflow.source.git.repoURL` и `mlflow.source.git.commit` — по ним
любой результат можно связать с точной версией кода. Это и есть главная ценность: «модель от 27 июня»
превращается в «коммит `e65a327`, параметры такие-то».

### Связь с шаблоном проекта и Makefile

В шаблоне из урока 1.3.4 единая точка входа — `make train CONFIG=...`. `MLproject` решает ту же задачу
и добавляет три вещи: типизированные параметры с проверкой, описание окружения и автоматическую запись
о запуске. Они сочетаются: `Makefile` остаётся интерфейсом для человека, а цель вызывает `mlflow run`.

```makefile
train:
	mlflow run . --env-manager local -P day=$(DAY)
```

Чего MLflow Projects **не** делает: не запускает по расписанию, не повторяет при сбоях, не знает
о зависимостях между шагами. Это запуск, а не оркестрация. Оркестратор может вызывать `mlflow run`
как одну из задач.

## 3. Kubeflow Pipelines

Kubeflow Pipelines (KFP) — платформа поверх Kubernetes, в которой **каждый шаг пайплайна выполняется
в отдельном контейнере**. Отсюда все её свойства: у шага свой образ и свои зависимости, свои ресурсы
(в том числе видеокарта), а данные между шагами нельзя передать через память или локальный диск —
только через хранилище.

| Понятие | Смысл |
|---|---|
| Компонент | описание одного шага: образ, команда, типизированные входы и выходы |
| Параметр | небольшое значение — строка, число, словарь; передаётся по значению |
| Артефакт | файл или каталог в хранилище с типом и метаданными: `Dataset`, `Model`, `Metrics`; передаётся ссылкой |
| Пайплайн | граф компонентов; связи возникают, когда выход одного компонента подан на вход другому |
| Промежуточное представление (IR) | файл YAML, в который компилируется пайплайн; его исполняет сервер |
| Запуск | одно выполнение пайплайна с конкретными значениями параметров |

### Компоненты

Лёгкий компонент на Python — функция с аннотациями типов и декоратором:

```python
from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

IMAGE = "registry.example.com/sales-ml:1.0"  # образ проекта: Python, зависимости и пакет sales_ml


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def prepare(day: str, history_days: int, features: Output[Dataset]):
    """Выгружает партиции за нужный период и собирает таблицу признаков."""
    import shutil
    import tempfile
    from datetime import date, timedelta

    from sales_ml import steps

    last = date.fromisoformat(day)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(history_days + 7):
            steps.validate(steps.extract(last - timedelta(days=i), tmp), last - timedelta(days=i))
        shutil.copyfile(steps.build_features(last, tmp, history_days), features.path)
    features.metadata["day"] = day
```

Правила, непривычные после Airflow:

- **Функция должна быть самодостаточной.** В контейнер попадает только исходный текст функции, поэтому
  все импорты — внутри неё, а обращаться к переменным модуля нельзя. Код проекта приходит через образ
  (`base_image`) либо через список `packages_to_install`.
- **Аннотации типов — это интерфейс.** `day: str` — параметр. `Output[Dataset]` — выходной артефакт:
  платформа сама назначает ему место в хранилище, а компонент пишет файл по пути `features.path`.
  `Input[Model]` — входной артефакт: платформа скачает его и передаст путь. Возвращаемое значение
  `-> float` — выходной параметр с именем `Output`.
- **Передать можно только то, что объявлено.** Типы проверяются при компиляции: подать `Dataset` на вход,
  ожидающий `Model`, не получится — компиляция завершится `InconsistentTypeException`.

Остальные компоненты устроены так же: `train(day: str, features: Input[Dataset],
model: Output[Model])`, `publish(model: Input[Model], published: Output[Model])` и `evaluate(model:
Input[Model], features: Input[Dataset], metrics: Output[Metrics]) -> float`. Последний записывает метрики
вызовами `metrics.log_metric(name, value)` и возвращает отношение MAE модели к MAE прогноза «как неделю назад».

### Пайплайн

```python
@dsl.pipeline(name="sales-retrain", description="Ежедневное переобучение прогноза продаж")
def sales_retrain(day: str, history_days: int = 90, max_mae_ratio: float = 0.9):
    data = prepare(day=day, history_days=history_days)
    fit = train(day=day, features=data.outputs["features"])
    fit.set_cpu_limit("2").set_memory_limit("4G").set_retry(num_retries=2)
    score = evaluate(model=fit.outputs["model"], features=data.outputs["features"])
    score.set_caching_options(False)
    with dsl.If(score.outputs["Output"] <= max_mae_ratio, name="good-enough"):
        publish(model=fit.outputs["model"])


if __name__ == "__main__":
    compiler.Compiler().compile(sales_retrain, "sales_retrain.yaml")
```

Как и в TaskFlow API, тело функции не выполняет шаги, а описывает граф. Зависимости возникают
из передачи выходов; ресурсы, повторы и кэширование настраиваются у задачи; условие `dsl.If` — аналог
ветвления: `publish` выполнится, только если отношение ошибок не превышает порог.

### Компиляция

```bash
uv run python sales_kfp.py          # 2 секунды; результат — sales_retrain.yaml, 320 строк
```

Фрагменты результата (часть строк опущена):

```yaml
deploymentSpec:
  executors:
    exec-train:
      container:
        image: registry.example.com/sales-ml:1.0
        resources:
          cpuLimit: 2.0
          memoryLimit: 4.0
root:
  dag:
    tasks:
      condition-1:
        dependentTasks:
        - evaluate
        - train
        triggerPolicy:
          condition: inputs.parameter_values['pipelinechannel--evaluate-Output'] <=
            inputs.parameter_values['pipelinechannel--max_mae_ratio']
      train:
        retryPolicy:
          maxRetryCount: 2
schemaVersion: 2.1.0
sdkVersion: kfp-2.18.0
```

В файле три части: `components` (интерфейсы — входы и выходы с типами), `deploymentSpec` (что запускать:
образ, команда, ресурсы; исходный текст функции вшит в команду) и `root` (граф). Файл не зависит
от Python: его можно хранить в Git, сравнивать между версиями и загружать на сервер. Компиляция —
дешёвая проверка: ошибки типов и несостыкованные входы обнаруживаются на ней, а не через час работы
кластера. Добавьте её в тесты проекта.

### Локальный запуск

SDK умеет выполнить пайплайн без кластера. `SubprocessRunner` запускает каждый компонент отдельным
процессом в текущем окружении:

```python
from kfp import local

from sales_kfp import sales_retrain

local.init(runner=local.SubprocessRunner(use_venv=False), pipeline_root="./local_outputs")
sales_retrain(day="2024-06-30", history_days=90)
```

```text
INFO - Running pipeline: 'sales-retrain'
INFO - Executing task 'prepare'
INFO - Task 'prepare' finished with status SUCCESS
INFO - Task 'prepare' outputs:
    features: Dataset( name='features',
                       uri='<proj>/local_outputs/sales-retrain-2026-11-13-11-33-05-077610/prepare/features',
                       metadata={'day': '2024-06-30'} )
...
INFO - Task 'evaluate' finished with status SUCCESS
    Output: 0.6761982293680675
INFO - Executing task 'publish'
INFO - Task 'publish' finished with status SUCCESS
INFO - Pipeline 'sales-retrain' finished with status SUCCESS
```

(Вывод сокращён. Весь запуск — 14 секунд; пакет `sales_ml` доступен процессам через `PYTHONPATH`.)

Отношение ошибок 0.676 меньше порога, условие выполнилось, модель «опубликована». Метрики те же, что
в Airflow и MLflow: `mae` 1.329, `mae_week_ago` 1.9654 — шаги одни и те же. Каждый компонент получил
свой каталог в `local_outputs/<запуск>/<задача>/` с артефактами и файлом `executor_output.json`.

Что показал локальный запуск и чего он не показывает:

- он проверяет логику компонентов и стыковку входов с выходами, но не образы: SDK предупреждает
  `You may be attemping to run a task that uses custom or non-Python base image`;
- ресурсы игнорируются: `settings ['cpu_limit', 'memory_limit'] have no effect in the current local
  runner`;
- без `install_kfp_package=False` каждый компонент перед стартом ставит пакет `kfp` через pip, и в окружении
  uv первый запуск упал с `No module named pip`. В образе проекта `kfp` должен быть установлен заранее;
- `DockerRunner`, который выполняет компоненты в настоящих контейнерах, **не проверялся запуском**:
  образа `registry.example.com/sales-ml:1.0` не существует, это имя-заглушка.

### Запуск на кластере

> Раздел не проверялся запуском: в среде подготовки курса нет кластера Kubernetes. Описание — по
> документации Kubeflow Pipelines; методы SDK существуют в установленной версии 2.18.0.

Скомпилированный файл отправляется на сервер KFP:

```python
import kfp

client = kfp.Client(host="https://kfp.example.com")
client.create_run_from_pipeline_package(
    "sales_retrain.yaml",
    arguments={"day": "2024-06-30"},
    experiment_name="sales",
    enable_caching=True,
)
```

Что происходит дальше на кластере:

- сервер создаёт по поду на каждую задачу; под скачивает входные артефакты из объектного хранилища,
  выполняет команду и загружает выходные артефакты обратно;
- **кэширование**: если компонент с тем же образом, командой и теми же входами уже выполнялся, шаг
  не запускается, а его выходы берутся из прошлого запуска. Перезапуск пайплайна после падения
  на `train` не будет заново готовить данные. Шаги, зависящие от внешнего мира (выгрузка «свежих» данных
  без даты в параметрах), кэшировать нельзя — отсюда `set_caching_options(False)`;
- в интерфейсе виден граф запуска, логи каждого пода, артефакты с метаданными и метрики (`Metrics`),
  по которым запуски можно сравнивать;
- расписание задаётся отдельной сущностью — повторяющимся запуском (recurring run).

## 4. Что даёт кластер и чего он стоит

| Что даёт | Чего стоит |
|---|---|
| изоляция: у каждого шага свой образ, конфликт зависимостей между шагами невозможен | образы нужно собирать, версионировать и хранить в реестре |
| ресурсы по шагам: подготовка данных — на 2 CPU, обучение — на узле с видеокартой | старт пода — десятки секунд; мелкие шаги невыгодны |
| масштабирование: сто запусков подбора гиперпараметров параллельно | кластер нужно администрировать: обновления, права, квоты, мониторинг |
| воспроизводимость: шаг — это образ, команда и входы | отладка сложнее: лог пода вместо отладчика |
| кэш шагов и история артефактов | данные между шагами — только через хранилище: копирование вместо передачи в памяти |

Kubernetes — отдельная профессия. Команда без человека, который его сопровождает, платит за платформу
временем инженеров по ML, и обычно это плохая сделка. Обратная ситуация тоже типична: кластер в компании
уже есть, его сопровождают другие люди — тогда Kubeflow даёт изоляцию и ресурсы почти бесплатно.

## 5. Сравнение

| Критерий | Makefile + cron | MLflow Projects | Airflow | Kubeflow Pipelines |
|---|---|---|---|---|
| Что запускает | команды по времени | одну точку входа с параметрами | граф задач по расписанию | граф контейнеров |
| Где исполняется код | на этой машине | на этой машине, в описанном окружении | в воркерах; тяжёлое — во внешних системах | в подах Kubernetes |
| Как передаются данные | файлы на диске | файлы на диске, запись о запуске | данные — в хранилище, через XCom — ссылки | типизированные артефакты в объектном хранилище |
| Расписание, повторы, backfill | расписание есть, остальное вручную | нет | есть всё | расписание и повторы; расчёт за прошлую дату — запуск с другим параметром |
| Зависимости шагов | общие | общие, описаны в проекте | общие у воркеров | свои у каждого шага |
| Порог входа | минуты | час | дни | недели и кластер |
| Проверено в курсе | урок 1.4.3 | полностью, локально | локально, `LocalExecutor` | компиляция и локальный запуск; кластер — нет |
| Когда выбирать | один проект, одна машина | нужен воспроизводимый запуск и связь результата с кодом | много пайплайнов данных и ML, расписания, внешние системы | уже есть Kubernetes; шагам нужны разные образы и GPU |

## 6. Что выбрать команде

```text
Сколько человек и пайплайнов?
├── 1 человек, 1–2 пайплайна
│     Makefile (или MLproject) + таймер systemd. Шаги — функции с явными входами и выходами.
│     Оркестратор — когда появятся признаки из урока 7.1.2: зависимости, перезапуски, вопросы «что упало?».
├── около 5 человек, 5–20 пайплайнов
│     Один оркестратор на команду. Airflow — если вокруг много выгрузок и внешних систем,
│     или более лёгкая альтернатива (Prefect, Dagster). MLproject — как формат шага.
│     Тяжёлые шаги — в контейнерах. Свой кластер Kubernetes ради пайплайнов не заводить.
└── около 50 человек, сотни пайплайнов, выделенная платформенная команда
      Kubernetes уже есть. Kubeflow Pipelines или Airflow с KubernetesExecutor:
      изоляция зависимостей между командами и учёт ресурсов становятся важнее простоты.
```

В любом варианте верно одно: **шаги пишутся так, чтобы оркестратор можно было сменить**. Функции
`sales_ml.steps` за два урока отработали под `python -c`, Airflow, MLflow Projects и Kubeflow Pipelines
без единой правки — потому что у них явные входы, результат на диске и никакой зависимости от обёртки.

## Типичные ошибки

| Симптом | Причина | Решение |
|---|---|---|
| `mlflow run` падает с `No value given for missing parameters` | у параметра нет значения по умолчанию | передать `-P имя=значение` или задать `default` |
| У коллеги `mlflow run` даёт другой результат | запуск с `--env-manager local` в другом окружении | описать окружение в `python_env.yaml`, запускать с `uv` или `virtualenv` |
| В компоненте KFP `NameError` на переменную или импорт модуля | функция компонента не самодостаточна | импорты и константы — внутри функции; код проекта — в образе |
| Компонент локально падает с `No module named pip` | компонент пытается установить `kfp`, а pip в окружении нет | `install_kfp_package=False` и `kfp` в образе либо pip в окружении |
| Шаг «не выполняется», результат старый | сработал кэш: входы и компонент не изменились | дата и версия данных — в параметрах; для шагов с внешними эффектами кэш отключить |
| Пайплайн из десятков мелких шагов работает дольше скрипта | на каждый шаг — старт пода и копирование артефактов | укрупнить шаги: граница шага — там, где нужен другой образ, ресурс или точка перезапуска |

## Контрольные вопросы

1. Какие три разных смысла у слова «пайплайн» в ML и какой инструмент отвечает каждому?
2. Что описывает файл `MLproject`? Что MLflow записывает о запуске автоматически?
3. Чем `--env-manager local` отличается от `--env-manager uv` и когда уместен каждый?
4. Что даёт запуск проекта по адресу git-репозитория с указанием коммита?
5. Чем параметр компонента KFP отличается от артефакта?
6. Почему импорты в лёгком компоненте KFP находятся внутри функции?
7. Что содержит скомпилированный YAML и зачем его проверять в тестах?
8. Что проверяет локальный запуск через `SubprocessRunner` и чего он проверить не может?

## Домашнее задание

1. Оформите проект с шагами `sales_ml.steps` как MLflow Project с двумя точками входа: `main` (полный
   расчёт за дату) и `evaluate` (оценка уже обученной модели за указанную дату). Запустите обе.
2. Создайте локальный git-репозиторий с проектом и запустите его по адресу `file://...` с указанием
   коммита. Найдите в записи о запуске адрес репозитория и коммит.
3. Опишите тот же расчёт как пайплайн KFP из четырёх компонентов, скомпилируйте его и выполните локально
   через `SubprocessRunner`. Добавьте параметр пайплайна `valid_days`.
4. Напишите тест, который компилирует пайплайн во временный каталог (`tmp_path`) и проверяет, что в YAML
   есть все четыре компонента и параметр `day` без значения по умолчанию. Тест должен пропускаться,
   если `kfp` не установлен (`pytest.importorskip`).

**Критерии приёмки:**

- [ ] `mlflow run . --env-manager local` и `mlflow run . --env-manager local -e evaluate -P day=...` завершаются
      успешно; запуск без обязательного параметра завершается понятной ошибкой;
- [ ] в записи о запуске из git найдены теги `mlflow.source.git.repoURL` и `mlflow.source.git.commit`;
- [ ] пайплайн KFP компилируется; локальный запуск завершается со статусом `SUCCESS`;
- [ ] тест компиляции проходит в окружении с `kfp` и пропускается без него;
- [ ] `mlruns/`, `mlflow.db`, `local_outputs/`, скомпилированный YAML и данные не попали в Git.

## Дополнительные материалы

- [Документация MLflow](https://mlflow.org/docs/latest/) — раздел MLflow Projects: формат `MLproject`,
  окружения, запуск из git
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/) — компоненты, артефакты,
  компиляция, локальное исполнение, кэширование

**Следующий урок:** 7.1.4 — ПРАКТИКА: Создание end-to-end ML пайплайна.
