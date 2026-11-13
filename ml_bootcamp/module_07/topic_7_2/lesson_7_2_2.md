# Урок 7.2.2: MLflow Tracking, Weights & Biases для экспериментов

> **Модуль 7:** ML Engineering & MLOps · **Тема 7.2:** Версионирование и эксперименты
> **Тип:** теория · **Время:** 120 минут + 90 минут на домашнее задание

## Цели урока

После урока вы сможете:

- перечислить, что обязано попасть в запись об эксперименте, чтобы результат можно было повторить;
- записывать параметры, метрики, теги, артефакты и модели в MLflow Tracking — локально и через сервер;
- связать запуск с коммитом Git и версией данных DVC;
- оценить, что `autolog` записывает сам и чего не записывает;
- оформить подбор гиперпараметров Optuna вложенными запусками, найти лучший запуск запросом;
- вести реестр моделей: версии, псевдонимы `champion` и `challenger`, продвижение и откат;
- записать запуск в Weights & Biases в офлайн-режиме и выбрать инструмент под команду.

## Зачем это нужно

К четвёртому модулю у каждого накапливается папка вида `model_final_v2_new.joblib` и таблица с метриками,
в которой уже непонятно, какая строка получена каким кодом. `metadata.json` из практики 7.1.4 решает
это для одного запуска; `dvc exp` из прошлого урока — для одного человека и одного репозитория. Трекер
экспериментов — общий журнал: каждый запуск обучения оставляет запись, по которой любой участник команды
ответит, что запускалось, на чём, с каким результатом — и получит саму модель.

Урок продолжает проект `sales-dvc`: данные и стадии — из урока 7.2.1, таблицы `data/prepared/train.csv`
и `valid.csv` — выход стадии `prepare` (обучающее окно 56 дней, проверочное — неделя до 30 июня 2024).

## 1. Что такое эксперимент и что записывать

**Запуск** (run) — одно выполнение кода обучения. **Эксперимент** — набор запусков, решающих одну задачу
и сравнимых между собой. Запись о запуске полна, если по ней можно получить тот же результат заново:

| Что | Зачем | Чем записывается |
|---|---|---|
| Код | тот же алгоритм и те же признаки | хеш коммита Git и признак незакоммиченных правок |
| Данные | та же выборка | хеш данных (DVC, `data_hash` из практики 7.1.4), период, число строк |
| Параметры | включая значения по умолчанию | итоговый конфиг целиком (урок 1.3.3) |
| Окружение | версии библиотек меняют числа | `uv.lock` или его экспорт, версия Python (урок 7.2.3) |
| Seed | случайность разбиения, инициализации, подбора | параметр запуска |
| Метрики | с указанием выборки, на которой посчитаны | `valid_mae`, а не `mae` |
| Артефакты | модель, графики, отчёт о данных | файлы запуска |
| Статус | неудачные запуски — тоже результат | `FINISHED` / `FAILED` |

## 2. Анатомия MLflow Tracking

| Понятие | Смысл | Тип значения |
|---|---|---|
| Эксперимент | именованная группа запусков; у него своё место для артефактов | — |
| Запуск | запись об одном выполнении: идентификатор, статус, время | — |
| Параметр | входная настройка; записывается один раз и не меняется | строка |
| Метрика | число; у одного имени может быть история значений по шагам | число + шаг + время |
| Тег | произвольная пометка; можно менять после запуска | строка |
| Артефакт | файл или каталог | файл |
| Модель | артефакт с описанием: формат, сигнатура, зависимости | каталог |

Хранилищ два, и это разделение определяет всё развёртывание:

- **хранилище метаданных** (backend store) — база данных: эксперименты, запуски, параметры, метрики, теги,
  реестр моделей. По ней работают поиск и сравнение;
- **хранилище артефактов** (artifact store) — файлы: каталог на диске или объектное хранилище.

Окружение — тот же проект, что в уроке 7.2.1; трекеры добавлены к DVC без конфликтов версий:

```bash
uv add mlflow optuna lightgbm xgboost wandb
```

Версии урока: `mlflow 3.18.0`, `wandb 0.31.0`, `optuna 5.0.0`, `lightgbm 4.7.0`, `xgboost 3.4.1`,
`skops 0.16.0`, `scikit-learn 1.9.1`, `pandas 3.1.1`, `numpy 2.5.4`, `dvc 3.67.1`, Python 3.12.13.
Весь код урока выполнен; исключения помечены.

## 3. Первый запуск

```python
import mlflow
from sklearn.metrics import mean_absolute_error

mlflow.set_experiment("sales-forecast")
with mlflow.start_run(run_name="ridge-alpha-10") as run:
    mlflow.log_params({"model": "ridge", "alpha": 10.0, "min_frequency": 20})
    mlflow.set_tags(common_tags("baseline"))                      # раздел 4
    model = build_ridge(alpha=10.0, min_frequency=20).fit(X_train, y_train)
    mlflow.log_metric("valid_mae", mean_absolute_error(y_valid, model.predict(X_valid)))
    for days in (7, 14, 28, 56):          # метрика с шагом: ошибка при разной длине обучающего окна
        recent = train[train["date"] > train["date"].max() - pd.Timedelta(days=days)]
        short = build_ridge(alpha=10.0).fit(recent[FEATURES], recent["units"])
        mlflow.log_metric("valid_mae_by_train_days", mean_absolute_error(y_valid, short.predict(X_valid)), step=days)
    mlflow.log_artifact("params.yaml")                            # файл с диска
    mlflow.log_dict(errors_by_dow, "valid_mae_by_dow.json")       # объект Python -> файл в артефактах
```

`build_ridge` — пайплайн из урока 7.2.1 (one-hot, масштабирование, `Ridge` на `log1p` целевой переменной);
признаки — `store_id`, `product`, `dow`, `price`, `promo`. Запись о запуске, прочитанная через
`MlflowClient().get_run(run_id)`:

```text
run_id: 2f5123e20698437f82c8ec6a898cc398   status: FINISHED
params:  {'model': 'ridge', 'alpha': '10.0', 'min_frequency': '20'}
metrics: {'valid_mae': 1.3364638406013931, 'valid_mae_by_train_days': 1.3364638406013931}
история valid_mae_by_train_days: [(7, 1.3537), (14, 1.3498), (28, 1.3366), (56, 1.3365)]
tags:
  stage = baseline
  git_commit = 2b7b647e6a32827451f18fffc7d717279f6831c6
  git_dirty = False
  data_version = 0ed16eacc2a51a921f06bde7709ef8c1.dir
  mlflow.runName = ridge-alpha-10
  mlflow.source.name = <файл скрипта>        mlflow.source.type = LOCAL
  mlflow.source.git.commit = 2b7b647e6a32827451f18fffc7d717279f6831c6
  mlflow.source.git.branch = main            mlflow.user = <пользователь>
artifacts: ['params.yaml', 'valid_mae_by_dow.json']
```

Что видно:

- параметры хранятся **строками** (`'10.0'`), метрики — числами. У метрики с шагом в самой записи лежит
  последнее значение, история — отдельным запросом `client.get_metric_history(run_id, имя)`;
- теги с префиксом `mlflow.` проставлены автоматически, в том числе коммит и ветка — потому что скрипт
  запущен из репозитория Git;
- без настройки MLflow 3.18 пишет метаданные в файл `mlflow.db` (SQLite) в текущем каталоге, а артефакты —
  в `mlruns/<эксперимент>/<запуск>/artifacts/`. Оба пути — в `.gitignore`.

Исключение внутри `with` не теряет запись: запуск получает статус `FAILED`, а параметры и теги,
записанные до сбоя, остаются.

```text
                             run_id tags.mlflow.runName    status  metrics.valid_mae
0  46645ce8976e4997a7c223fbe52b8838        ridge-broken    FAILED                NaN
1  2f5123e20698437f82c8ec6a898cc398      ridge-alpha-10  FINISHED           1.336464
```

## 4. Связь запуска с кодом и данными

Автоматического тега `mlflow.source.git.commit` недостаточно: он не говорит, были ли в рабочем каталоге
незакоммиченные правки, и ничего не знает о данных. Три функции закрывают оба пробела:

```python
def git_state() -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    return commit.stdout.strip(), bool(dirty.stdout.strip())


def data_version() -> str:
    return yaml.safe_load(Path("data/raw.dvc").read_text())["outs"][0]["md5"]


def common_tags(stage: str) -> dict:
    commit, dirty = git_state()
    return {"stage": stage, "git_commit": commit, "git_dirty": dirty, "data_version": data_version()}
```

`data_version` — хеш каталога из файла `data/raw.dvc`: тот самый, по которому DVC достанет данные
из хранилища. Теперь запись о запуске превращается в инструкцию по воспроизведению:

```bash
git checkout 2b7b647 && uv sync --locked && uv run dvc checkout    # код, окружение, данные запуска
```

`git_dirty = True` означает, что этой инструкции верить нельзя: результат получен кодом, которого нет
ни в одном коммите. При первом прогоне примера, пока скрипт не был закоммичен, а `mlflow.db` не добавлен
в `.gitignore`, тег показал именно `True`. Разумное правило: запуски с `git_dirty = True` допустимы
при отладке, но в отчёт и в реестр моделей не попадают.

## 5. Запись пайплайна практики 7.1.4

Пайплайн из практики уже складывает всё нужное в каталог запуска — трекеру остаётся это забрать:

```python
from ml_pipeline import run, sales_source

mlflow.set_experiment("sales-pipeline")
for day, alpha in [("2024-06-30", 1.0), ("2024-06-30", 100.0), ("2024-03-31", 1.0)]:
    with mlflow.start_run(run_name=f"{day}-alpha-{alpha}"):
        result = run(date.fromisoformat(day), "work", sales_source, {"alpha": alpha})
        metadata = json.loads((result["run_dir"] / "metadata.json").read_text())
        mlflow.log_params({"run_date": day, **metadata["config"]})       # итоговый конфиг, а не только alpha
        mlflow.log_metrics(result["metrics"])
        mlflow.set_tags({"data_hash": metadata["data_hash"], "published": result["published"]})
        mlflow.log_artifacts(str(result["run_dir"]))     # model.joblib, metrics.json, metadata.json
```

| Запуск | `mae` | `alpha` | `data_hash` | `published` |
|---|---|---|---|---|
| `2024-03-31-alpha-1.0` | 1.602 | 1.0 | `683dbd76d397…` | True |
| `2024-06-30-alpha-1.0` | 1.334 | 1.0 | `020f582708dc…` | True |
| `2024-06-30-alpha-100.0` | 1.378 | 100.0 | `020f582708dc…` | True |

Метрики совпали с ориентирами практики 7.1.4, хеш данных за 30 июня — тот же `020f5827…`. Два наблюдения.
Каталог запуска в пайплайне один на дату, а запусков трекера за дату может быть сколько угодно: второй
запуск за 30 июня перезаписал каталог, но артефакты первого сохранились в MLflow. И по одинаковому
`data_hash` сразу видно, какие запуски можно сравнивать между собой напрямую.

## 6. Автоматическая запись: `autolog`

Одна строка перед обучением — и MLflow сам перехватывает вызовы `fit`:

```python
mlflow.sklearn.autolog()          # есть также mlflow.lightgbm.autolog(), mlflow.xgboost.autolog()
with mlflow.start_run(run_name="autolog-ridge"):
    model = build_ridge(alpha=1.0).fit(X_train, y_train)
mlflow.sklearn.autolog(disable=True)
```

Что оказалось в запусках (проверено с `scikit-learn 1.9.1`, `lightgbm 4.7.0`, `xgboost 3.4.1`):

| Что обучали | Параметры | Метрики | Артефакты |
|---|---|---|---|
| `Pipeline` с `Ridge` | 39 — параметры всех шагов: `features__cat__min_frequency`, … | 5, все на **обучающей** выборке: `training_mean_absolute_error` 1.4298, `training_r2_score` 0.6324, … | `estimator.html`, модель, описание наборов данных |
| `GridSearchCV`, 3 значения `alpha` | 11 и `best_model__regressor__alpha` | `best_cv_score`, метрики на обучении | `cv_results.csv`, две модели; 3 **дочерних запуска** с `mean_test_score` |
| `LGBMRegressor` с `eval_set` | 20 | `valid-l1`, `valid-poisson` — история из 200 значений, шаг = итерация | важности признаков (JSON и PNG), модель |
| `XGBRegressor` с `eval_set` | 37 | `validation_0-mae` — 200 значений | важности признаков, модель, описание наборов данных |

Чего `autolog` **не** записал и записать не может:

- **метрику на проверочном окне** для моделей sklearn. Вызов
  `mean_absolute_error(y_valid, model.predict(X_valid))` внутри запуска в журнал не попал — остались
  только `training_*`. Запуск, в котором есть только ошибка на обучении, хуже, чем запуск без метрик:
  по нему выберут переобученную модель. У бустингов метрика на `eval_set` записывается;
- версию данных, признак «грязного» дерева, период обучения, смысл запуска — всё, что знает только автор;
- подготовку признаков вне оценщика: всё, что сделано в pandas до `fit`, для `autolog` не существует.

Побочные эффекты, замеченные в прогоне: каждая модель сохраняется целиком вместе со списком зависимостей
(запуски становятся медленнее и тяжелее); шаг пайплайна с именем `model` записался параметром `model`
и смешался с нашим параметром `model` при группировке; для LightGBM описание набора данных не записалось
(`Failed to log dataset information ... 'Series' object has no attribute 'flatten'`). Вывод: `autolog` —
хорошее дополнение (кривые обучения бустинга, `cv_results.csv`), но не замена явной записи того, за что
вы отвечаете: метрики на правильной выборке, данные, код.

## 7. Вложенные запуски и Optuna

Подбор гиперпараметров (урок 4.4.2) — это десятки обучений одной гипотезы. Каждая попытка — дочерний
запуск, весь подбор — родительский:

```python
def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    with mlflow.start_run(run_name=f"trial-{trial.number:02d}", nested=True):
        mlflow.log_params({"model": "lgbm", **params})
        mlflow.set_tag("stage", "tuning")
        model = build_lgbm(**params).fit(X_train, y_train)      # n_estimators=300, random_state=0, n_jobs=1
        mae = mean_absolute_error(y_valid, model.predict(X_valid))
        mlflow.log_metric("valid_mae", mae)
    return mae


with mlflow.start_run(run_name="optuna-lgbm") as parent:
    mlflow.set_tags(common_tags("tuning"))
    mlflow.log_params({"model": "lgbm", "n_trials": 20, "sampler": "TPE", "sampler_seed": 0})
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=20)
    mlflow.log_params({f"best_{name}": value for name, value in study.best_params.items()})
    mlflow.log_metric("valid_mae", study.best_value)
```

Дочерний запуск получает тег `mlflow.parentRunId`; в интерфейсе попытки сворачиваются под родителем.
Seed сэмплера записан параметром: без него подбор не повторить. Три лучшие попытки:

```text
  tags.mlflow.runName  metrics.valid_mae params.learning_rate params.num_leaves params.min_child_samples
0            trial-15           1.398921   0.0675044047403687                 5                       69
1            trial-16           1.412719  0.09098367239535464                12                       94
2            trial-17           1.413259   0.0943055092555128                13                       88
```

Честный результат: лучший бустинг (1.399) проиграл линейной модели (1.336). Журнал для того и нужен,
чтобы это было видно. Оговорка из урока 4.4.2 остаётся в силе: лучшая из двадцати попыток выбрана по тому
же проверочному окну, поэтому её оценка оптимистична; для итогового сравнения нужна отложенная неделя.

## 8. Поиск и сравнение запусков

`mlflow.search_runs` возвращает `DataFrame`: по строке на запуск, столбцы — `params.*`, `metrics.*`,
`tags.*` и атрибуты.

```python
runs = mlflow.search_runs(
    experiment_names=["sales-forecast"],
    filter_string="params.model = 'lgbm' and tags.stage = 'tuning' "
    "and metrics.valid_mae < 1.42 and attributes.status = 'FINISHED'",
    order_by=["metrics.valid_mae ASC"],
    max_results=5,
)
```

```text
                             run_id tags.mlflow.runName  metrics.valid_mae params.learning_rate params.num_leaves
0  f4dbd7ce54e34847bc69f01145e131a7            trial-15           1.398921   0.0675044047403687                 5
1  dcd90bd2325045399907e3a964577fae         optuna-lgbm           1.398921                  NaN               NaN
2  c20e637b8adf45899477487c654634bb            trial-16           1.412719  0.09098367239535464                12
3  f8c6fd36782a46a28783728c8266c975            trial-17           1.413259   0.0943055092555128                13
4  145fd6785d994869b2950370b60d674d            trial-10           1.413509    0.041441110433563                11
```

Особенности языка запросов, на которые наталкиваются первыми:

- параметры сравниваются как строки: `params.alpha = '10.0'`; числовые сравнения — только для метрик;
- условия соединяются только через `and`: запрос с `or` завершился ошибкой
  `Invalid clause(s) in filter string: 'or'`. Объединение делается в pandas;
- в выдачу попал родительский запуск: он тоже удовлетворяет условиям. Только попытки одного подбора —
  `tags.mlflow.parentRunId = '<идентификатор родителя>'`;
- «лучший запуск» — первая строка при `order_by` и `max_results=1`. Направление задаёте вы: MLflow
  не знает, что `mae` лучше меньше, а `r2` — больше.

Дальше — обычный pandas: `runs.groupby("params.model")["metrics.valid_mae"].agg(["count", "min"])`.
После всех примеров урока в эксперименте 34 запуска (32 `FINISHED`, 2 `FAILED`); `mlflow.db` занял
1.4 МБ, `mlruns/` — 3.1 МБ.

## 9. Модели

Модель можно сохранить файлом через `log_artifact`, но `log_model` добавляет к файлу описание, по которому
модель загрузит любой — не зная, чем она обучена:

```python
from mlflow.models import infer_signature

signature = infer_signature(X_valid, model.predict(X_valid))
info = mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=X_valid.head(3))
info.model_uri        # 'models:/m-a6eba7934de54120959c6748f99c7c71'
```

```text
inputs:  ['store_id': string (required), 'product': string (required), 'dow': long (required),
          'price': double (required), 'promo': boolean (required)]
outputs: [Tensor('float64', (-1,))]
```

Каталог модели: `MLmodel` (описание: формат, сигнатура, версия sklearn и Python, идентификатор запуска),
`model.skops`, `input_example.json`, `serving_input_example.json`, `requirements.txt`, `python_env.yaml`,
`conda.yaml`. В MLflow 3 модель — самостоятельная сущность с идентификатором `m-…`: она лежит
в `mlruns/<эксперимент>/models/m-…/artifacts/`, а не среди артефактов запуска (`list_artifacts` для
запуска вернул пустой список), но помнит запуск-источник. Адрес `runs:/<run_id>/model` тоже работает.
В `requirements.txt` — 172 пакета: MLflow обнаружил проект uv и сам выполнил `uv export`.

```python
loaded = mlflow.pyfunc.load_model(info.model_uri)       # единый интерфейс: predict(DataFrame)
np.array_equal(loaded.predict(X_valid), model.predict(X_valid))      # True
```

Сигнатура — не документация, а проверка на входе:

| Вход | Результат |
|---|---|
| нет столбца `price` | `MlflowException: Failed to enforce schema of data ...` |
| в `price` строка `"дорого"` | `MlflowException: Failed to enforce schema ...` |
| лишние столбцы `date`, `units` | предупреждение `Found extra inputs ... will be ignored`, прогноз выдан |

Две особенности версии 3.18, обнаруженные при подготовке урока:

- **формат по умолчанию — skops**, а не pickle. Он безопаснее (при загрузке не выполняется произвольный
  код), но принимает только известные типы. Пайплайн с LightGBM записать не удалось:
  `The saved sklearn model references untrusted types ... ['collections.OrderedDict',
  'lightgbm.basic.Booster', 'lightgbm.sklearn.LGBMRegressor']`; для пайплайна практики 7.1.4 — то же
  с `ml_pipeline.DateFeatures` и `ml_pipeline.RareCategoryGrouper`. Решения проверены оба: перечислить
  типы в `skops_trusted_types=[...]` либо указать `serialization_format="cloudpickle"` (MLflow
  предупредит о рисках pickle). Для собственных классов модуль с ними должен импортироваться при загрузке;
- для целочисленного столбца `dow` MLflow выдаёт подсказку `Inferred schema contains integer column(s)`:
  целый столбец не может содержать пропуск, и строка с пропуском на входе не пройдёт проверку. Если пропуски
  возможны — приведите столбец к `float` до вывода сигнатуры (так сделано с `price`).

## 10. Реестр моделей: версии и псевдонимы

Реестр отвечает на вопрос «какая модель сейчас рабочая». **Зарегистрированная модель** — имя; **версия** —
конкретная записанная модель под этим именем, номера выдаются по порядку; **псевдоним** (alias) —
именованный указатель на версию.

```python
client = mlflow.MlflowClient()
name = "sales-forecast"

# версия 1: при записи модели
mlflow.sklearn.log_model(poisson, name="model", signature=..., registered_model_name=name)
client.set_registered_model_alias(name, "champion", 1)

# версия 2: уже записанная модель
v2 = mlflow.register_model(info.model_uri, name)
client.set_registered_model_alias(name, "challenger", v2.version)
client.set_model_version_tag(name, v2.version, "valid_mae", "1.3365")
```

Потребитель модели не знает номеров версий — он просит модель по псевдониму:

```python
champion = mlflow.pyfunc.load_model("models:/sales-forecast@champion")
```

Сценарий, выполненный в уроке:

| Шаг | Версии и `valid_mae` | `champion` | `challenger` |
|---|---|---|---|
| зарегистрирована регрессия Пуассона | v1: 1.3880 | v1 | — |
| зарегистрирован `Ridge` на `log1p` | v1, v2: 1.3365 | v1 | v2 |
| претендент лучше на 0.05 — **продвижение** | | v2 | — |
| зарегистрирован лучший LightGBM | v1, v2, v3: 1.3989 | v2 | v3 |
| претендент хуже — псевдоним не двигаем | | v2 | v3 |
| **откат**: `champion` возвращён на v1 | | v1 | v3 |

Продвижение и откат — одна и та же операция:

```python
client.set_registered_model_alias(name, "champion", 2)      # псевдоним переезжает, версии не меняются
client.delete_registered_model_alias(name, "challenger")
client.get_model_version_by_alias(name, "champion").version             # '2'
client.get_registered_model(name).aliases                    # {'challenger': 3, 'champion': 2}
```

После отката `models:/sales-forecast@champion` снова выдаёт прогнозы с MAE 1.3880 — без пересборки
и выкладки чего-либо: код потребителя не менялся. Замечания по API 3.18:

- запрос несуществующего псевдонима — `MlflowException: Registered model alias nobody not found`;
- `client.search_model_versions("name = 'sales-forecast'")` вернул версии с **пустым** списком `aliases` —
  псевдонимы читайте через `get_registered_model(name).aliases` или `get_model_version_by_alias`;
- стадии (`Staging`, `Production`) из старых руководств устарели: вызов `transition_model_version_stage`
  пока работает, но выдаёт `FutureWarning: ... deprecated since 2.9.0. Model registry stages will be
  removed in a future major release`. В новом коде — только псевдонимы и теги версий;
- неудачная попытка записать модель LightGBM версию в реестре не создала, но оставила запуск `FAILED`.

Правило продвижения («претендент лучше чемпиона минимум на δ на тех же данных») — это правило публикации
из практики 7.1.4, перенесённое в реестр. Его вы реализуете в практике 7.2.4.

## 11. Сервер трекинга

Локальные `mlflow.db` и `mlruns/` видны только вам. Общий журнал — это сервер:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./artifacts \
  --host 127.0.0.1 --port 5711
curl http://127.0.0.1:5711/health        # OK
```

Клиенту достаточно адреса — код записи не меняется:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5711 uv run python train.py
```

```text
artifact_uri: mlflow-artifacts:/1/ef4ea5c2103a46ffab35c0f0e47ced97/artifacts
model_uri: models:/m-35ae9b8915e94dd1bdcd85ffbaf0f11a   version: 1
🏃 View run ridge-via-server at: http://127.0.0.1:5711/#/experiments/1/runs/ef4ea5c2103a46ffab35c0f0e47ced97
```

Адрес `mlflow-artifacts:/…` означает, что клиент отдаёт файлы серверу по HTTP, а тот кладёт их в свой
каталог `artifacts/1/<запуск>/artifacts/params.yaml` и `artifacts/1/models/m-…/artifacts/model.skops`.
Клиенту не нужен доступ к диску или бакету сервера. Через тот же адрес модель загружена обратно
по псевдониму (`models:/sales-forecast@champion`), прогнозы совпали. Всё, что умеет клиент, доступно
и по REST: `POST /api/2.0/mlflow/experiments/search` вернул JSON с экспериментами.

Что проверено и что нет:

- проверено: запуск сервера (готов к запросам через 20 секунд), запись запуска, артефактов и модели,
  реестр, загрузка модели, REST API, остановка сервера. Страницы веб-интерфейса в браузере не открывались;
- при старте сервер сообщает: `Security middleware enabled with default settings (localhost-only)`. Доступ
  с других машин требует `--host 0.0.0.0` и явного списка `--allowed-hosts`;
- **не проверялось запуском**: сервер на отдельной машине, PostgreSQL как хранилище метаданных, бакет
  как хранилище артефактов, аутентификация (`--app-name basic-auth`) — в среде подготовки нет ни второй
  машины, ни облачного хранилища. Для команды нужны все четыре: SQLite не рассчитан на одновременную
  запись многих клиентов, а сервер без аутентификации открыт каждому в сети.

Развернуть сервер вместе с базой в Docker Compose (урок 1.1.3) — задание на усложнение практики 7.2.4.

## 12. Weights & Biases

W&B — трекер как облачный сервис: журнал, графики и хранилище артефактов находятся на серверах компании,
в проекте — только клиентская библиотека. Понятия те же: проект (эксперимент), запуск, `config`
(параметры), `log` (метрики с шагом), `summary` (итоговые значения), артефакты.

Сервер `api.wandb.ai` из среды подготовки недоступен, поэтому показан **офлайн-режим** — он полезен
и сам по себе: на кластерах без выхода в интернет запуски записываются локально и отправляются позже.

```python
import os

os.environ["WANDB_MODE"] = "offline"

import wandb

config = {"model": "ridge", "alpha": 10.0, "min_frequency": 20, **common_tags("baseline")}
run = wandb.init(project="sales-forecast", name="ridge-alpha-10", config=config, tags=["baseline"])
for days in (7, 14, 28, 56):
    ...
    run.log({"train_days": days, "valid_mae": mae})          # каждый вызов — следующий шаг
run.summary["best_valid_mae"] = 1.3365
artifact = wandb.Artifact("sales-model", type="model")
artifact.add_file("models/model.joblib")
run.log_artifact(artifact)
run.finish()
```

```text
wandb: Tracking run with wandb version 0.31.0
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online
wandb: Run data is saved locally in <проект>/wandb/offline-run-20261113_120843-a2va7s7o
wandb: Run history:
wandb: train_days ▁▂▄█
wandb:  valid_mae █▆▁▁
wandb: Run summary:
wandb: best_valid_mae 1.3365
wandb:      valid_mae 1.33646
wandb: You can sync this run to the cloud by running:
wandb: wandb sync <проект>/wandb/offline-run-20261113_120843-a2va7s7o
```

(Вывод сокращён.) В каталоге `wandb/offline-run-…/` появились двоичный журнал `run-a2va7s7o.wandb`,
`files/requirements.txt` (окружение записывается само), `files/output.log` (вывод программы) и логи
клиента. Каталог `wandb/` — в `.gitignore`.

> Всё дальнейшее в этом разделе не проверялось запуском: в среде подготовки нет доступа к серверу W&B.
> Описание — по документации W&B.

- `wandb login` сохраняет ключ API; `wandb sync <каталог>` отправляет офлайн-запуск на сервер;
- веб-интерфейс: таблица запусков с фильтрами и группировкой, графики метрик с наложением запусков,
  сравнение конфигов, системные метрики (загрузка процессора, памяти, видеокарты) без единой строки кода;
- **Sweeps** — подбор гиперпараметров, которым управляет сервер: он раздаёт значения агентам на разных
  машинах. Без сервера не работает; Optuna с записью попыток — замена, не требующая сети;
- **Artifacts** — версионируемые наборы файлов с графом происхождения «данные → запуск → модель»;
  **Registry** — аналог реестра моделей с псевдонимами;
- **Reports** — страницы с живыми графиками поверх запусков: отчёт об эксперименте без скриншотов.

## 13. Сравнение инструментов

| Критерий | MLflow | Weights & Biases | «Своё на файлах»: `metadata.json` + DVC |
|---|---|---|---|
| Где хранятся данные | у вас: база и каталог или бакет | на серверах W&B; свой сервер — отдельная платная поставка | в репозитории и хранилище DVC |
| Стоимость | бесплатно; платите за сервер и его сопровождение | бесплатно для личного использования, для команд — по подписке; условия меняются, проверяйте на сайте | бесплатно |
| Совместная работа | общий сервер; права и аутентификацию настраиваете сами | из коробки: команды, права, отчёты, комментарии | через Git: pull request и `dvc exp push` |
| Порог входа | минуты локально; дни — сервер для команды | минуты, если можно отправлять данные наружу | ноль новых инструментов, но всё пишете сами |
| Интерфейс | таблица, графики, сравнение запусков | самый развитый: панели, отчёты, системные метрики | `dvc exp show`, `dvc metrics diff`, pandas |
| Реестр моделей | есть: версии, псевдонимы, теги | есть | нет: тег Git и файл-указатель |
| Работа без сети | полностью | офлайн-запись, просмотр — после синхронизации | полностью |
| Проверено в уроке | локально и через локальный сервер | только офлайн-запись | уроки 7.1.4 и 7.2.1 |
| Когда выбирать | данные нельзя отдавать наружу; нужен реестр; есть кому сопровождать сервер | важнее всего скорость и удобство анализа; данные можно хранить в облаке | один человек, один репозиторий, мало запусков |

Инструменты не исключают друг друга: версия данных из DVC — тег запуска в MLflow, а функция записи,
спрятанная за собственной обёрткой (практика 7.2.4), позволяет сменить трекер, не трогая код обучения.

## 14. Дисциплина ведения журнала

- **Эксперимент — задача, а не день и не человек.** `sales-forecast`, а не `ivan-tuesday`. Всё, что имеет
  смысл сравнивать по одной метрике на одних данных, — в одном эксперименте.
- **Один запуск — одна гипотеза.** Изменили и признаки, и модель, и окно — не узнаете, что помогло.
  Имя запуска описывает гипотезу: `ridge-alpha-10`, а не `test3`.
- **Обязательные теги.** `stage`, `git_commit`, `git_dirty`, `data_version` — у каждого запуска; проще
  всего обеспечить это обёрткой над `start_run`.
- **Неудачные запуски остаются.** Упавший запуск и гипотеза, ухудшившая метрику, — знание, за которое
  уже заплачено. Удаляют только мусор отладки, и для него заводят отдельный эксперимент.
- **Параметры — все.** Значение по умолчанию, не попавшее в журнал, изменится в новой версии библиотеки,
  и два «одинаковых» запуска разойдутся. Записывайте итоговый конфиг целиком.
- **Имя метрики называет выборку:** `valid_mae`, `test_mae`. Метрика `mae` без уточнения через месяц
  будет означать разное в разных запусках.
- **Секреты не записываются.** Параметры и теги видны всем, у кого есть доступ к журналу. Не пишите
  в них содержимое `os.environ` и строки подключения целиком (урок 1.3.3).

## Типичные ошибки

| Симптом | Причина | Решение |
|---|---|---|
| В журнале отличные метрики, на новых данных модель плохая | записана метрика на обучающей выборке (`training_*` от `autolog`) | явно записывать `valid_*`; выборку называть в имени метрики |
| Два запуска с одинаковыми параметрами дают разные числа | записаны не все параметры; другой коммит, данные или seed | итоговый конфиг целиком; теги коммита и версии данных; seed — параметр |
| Коллега видит запуск, но не может скачать модель | артефакты записаны на локальный диск автора (`mlruns/`) | сервер с `--artifacts-destination` или общий бакет |
| В эксперименте тысячи запусков, найти нужный нельзя | нет тегов и осмысленных имён; отладка смешана с результатами | обязательные теги; отдельный эксперимент для отладки; вложенные запуски для подбора |
| В параметрах запуска виден пароль или токен | в журнал записан весь конфиг или окружение без отбора | секреты — только из окружения в момент использования; фильтровать ключи перед `log_params` |
| `MlflowException: ... untrusted types` при `log_model` | формат skops не знает класс из сторонней библиотеки или вашего модуля | `skops_trusted_types=[...]` либо `serialization_format="cloudpickle"` |
| `Failed to enforce schema` при прогнозе | вход не соответствует сигнатуре: нет столбца, другой тип, пропуск в целочисленном столбце | приводить типы до вывода сигнатуры; тот же код подготовки на обучении и применении |
| `search_runs` с условием `or` падает | язык запросов поддерживает только `and` | два запроса и `pd.concat` либо фильтр в pandas |
| Запуск воспроизвести нельзя, хотя коммит записан | в рабочем каталоге были незакоммиченные правки | тег `git_dirty`; не регистрировать модели из «грязных» запусков |
| Запуск остался в статусе `RUNNING` навсегда | `start_run()` без `with`, процесс убит | контекстный менеджер; для убитых процессов — `client.set_terminated(run_id, "KILLED")` |

## Контрольные вопросы

1. Какие восемь составляющих делают запись о запуске достаточной для воспроизведения?
2. Чем параметр отличается от метрики и от тега? В каком виде хранятся значения параметров?
3. Какие два хранилища есть у MLflow и что лежит в каждом?
4. Почему автоматического тега с коммитом недостаточно и что добавляет `git_dirty`?
5. Что `autolog` записал для пайплайна sklearn и какой важной метрики в записи не оказалось?
6. Как устроены вложенные запуски и как выбрать все попытки одного подбора?
7. Что даёт сигнатура модели при загрузке через `pyfunc`?
8. Чем псевдоним отличается от версии? Почему продвижение и откат — одна операция?
9. Что изменяется в коде обучения при переходе с локального хранилища на сервер трекинга?
10. Что из возможностей W&B работает без сети, а что требует сервера?

## Домашнее задание

1. В проекте с DVC из урока 7.2.1 заведите эксперимент MLflow. Напишите функции `git_state`,
   `data_version`, `common_tags` и запишите базовый запуск: параметры, `valid_mae`, метрику с шагом,
   артефакт. Добейтесь `git_dirty = False`.
2. Запишите заведомо падающий запуск и покажите его статус в выдаче `search_runs`.
3. Проведите подбор гиперпараметров Optuna (не меньше 15 попыток) с вложенными запусками. Запросом
   получите три лучшие попытки именно этого подбора.
4. Запишите две модели с сигнатурой и примером входа, зарегистрируйте обе, назначьте псевдонимы
   `champion` и `challenger`. Реализуйте продвижение и откат; после каждого шага выведите
   `get_registered_model(name).aliases` и MAE модели, загруженной по псевдониму `champion`.
5. Поднимите локальный сервер трекинга на нестандартном порту, запишите через него запуск с моделью,
   остановите сервер. Убедитесь, что артефакты лежат в каталоге сервера.
6. Запишите тот же базовый запуск в W&B в офлайн-режиме; опишите в README, какие файлы появились.

**Критерии приёмки:**

- [ ] у каждого запуска есть теги `stage`, `git_commit`, `git_dirty`, `data_version`; по тегам базового
      запуска командой из раздела 4 восстанавливаются код и данные;
- [ ] в журнале есть запуск со статусом `FAILED`, и он не удалён;
- [ ] метрики названы по выборке (`valid_*`); метрик на обучающей выборке под именем без префикса нет;
- [ ] попытки подбора — дочерние запуски одного родителя; seed сэмплера записан;
- [ ] модель, загруженная по `models:/<имя>@champion`, даёт те же прогнозы, что исходная;
- [ ] откат возвращает псевдоним на прошлую версию, история версий при этом не меняется;
- [ ] сервер остановлен; `mlflow.db`, `mlruns/`, `wandb/`, каталог сервера не попали в Git;
- [ ] в параметрах и тегах нет секретов.

## Дополнительные материалы

- [Документация MLflow](https://mlflow.org/docs/latest/) — разделы Tracking, Models, Model Registry
  (псевдонимы и теги версий), Tracking Server
- [Документация Weights & Biases](https://docs.wandb.ai/) — Experiments, офлайн-режим и `wandb sync`,
  Sweeps, Artifacts, Reports

**Следующий урок:** 7.2.3 — Репроизводимость: Docker, environment management.
