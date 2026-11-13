# Урок 7.1.1: Scikit-learn pipelines, custom transformers

> **Модуль 7:** ML Engineering & MLOps · **Тема 7.1:** ML Pipelines
> **Тип:** теория · **Время:** 120 минут + 120 минут на домашнее задание

## Цели урока

После урока вы сможете:

- объяснить на числах, чем опасна предобработка вне пайплайна: утечка при оценке качества и расхождение
  между обучением и применением модели;
- читать и собирать составные объекты `Pipeline`, `ColumnTransformer`, `FeatureUnion`, обращаться
  к любому их параметру по имени и получать имена итоговых признаков;
- писать свои трансформеры по контракту scikit-learn и проверять контракт автоматически;
- преобразовывать целевую переменную, кэшировать дорогие шаги и подбирать гиперпараметры всего пайплайна,
  включая вопрос «нужен ли этот шаг вообще»;
- сохранять обученный пайплайн и понимать риски каждого формата.

## Зачем это нужно

В модулях 2–4 вы писали `make_pipeline(StandardScaler(), LogisticRegression())` как заклинание: «так надо,
чтобы не было утечки». В этом модуле модель перестаёт быть ячейкой ноутбука и становится тем, что
каждую ночь переобучается и каждый день отвечает на запросы. С этого момента вопрос «что именно является
моделью» получает практический ответ: **модель — это всё, что стоит между сырыми данными и прогнозом**.
Заполнение пропусков, кодирование категорий, признаки из дат — часть модели, и выкатывать, версионировать
и проверять их нужно вместе с весами. `Pipeline` — способ сделать это одним объектом.

Версии: Python 3.12, `scikit-learn 1.9.1`, `pandas 3.1.1`, `numpy 2.5.4`, `joblib 1.6.0`, `optuna 5.0.0`,
`skops 0.16.0`. Все числа в уроке получены на этих версиях.

## 1. Данные и задача темы

Сквозная задача темы 7.1 — прогноз дневных продаж: сколько единиц товара продаст магазин в конкретный день.
Данные — генератор `sales_data.py` из темы 2.1: 8 магазинов, 12 товаров, одна строка на пару
«магазин — товар» в день, два года.

```python
import sys

sys.path.insert(0, "../../module_02/topic_2_1")
from sales_data import make_sales

sales = make_sales()                                   # 2023-01-01 … 2024-12-31, 70 176 строк
sales["units_week_ago"] = sales.groupby(["store_id", "product"])["units"].shift(7)
sales = sales.dropna(subset=["units"])                 # 335 строк без целевой переменной

FEATURES = ["date", "store_id", "city", "category", "product", "price", "promo", "units_week_ago"]
X, y = sales[FEATURES], sales["units"]
is_train = sales["date"] < "2024-07-01"
X_train, y_train, X_test, y_test = X[is_train], y[is_train], X[~is_train], y[~is_train]
```

Обучение — 52 263 строки (полтора года), тест — 17 578 строк (последние полгода): разбиение по времени,
как и положено для прогноза (урок 2.3.1). Признак `units_week_ago` — продажи той же пары неделю назад;
у 916 обучающих строк он пропущен. Целевая переменная — счётчик: среднее 4.39, максимум 46.

Запускайте примеры с `PYTHONDONTWRITEBYTECODE=1`, чтобы импорт из чужой папки не оставил в ней `__pycache__`.

## 2. Три причины держать предобработку и модель в одном объекте

### Причина 1. Оценка качества без утечки

Любой шаг, который что-то **вычисляет по данным**, должен обучаться только на обучающей части каждого
фолда. Насколько это важно, зависит от шага. Возьмём 200 строк и добавим 5 000 столбцов чистого шума,
отберём 20 признаков, сильнее всего связанных с целевой переменной, и оценим линейную регрессию
кросс-валидацией (5 фолдов, R²):

```python
# Неправильно: отбор видел все строки, включая проверочные части будущих фолдов
X_sel = SelectKBest(f_regression, k=20).fit_transform(X_small, y_small)
cross_val_score(LinearRegression(), X_sel, y_small, cv=cv, scoring="r2")

# Правильно: отбор — шаг модели и заново обучается в каждом фолде
pipe = make_pipeline(SelectKBest(f_regression, k=20), LinearRegression())
cross_val_score(pipe, X_small, y_small, cv=cv, scoring="r2")
```

| Способ оценки | 4 настоящих признака + шум | Только шум |
|---|---|---|
| отбор до кросс-валидации | 0.631 | 0.318 |
| отбор внутри пайплайна | 0.345 | −0.673 |
| та же модель на 5 000 новых строк | 0.418 | −0.267 |

На чистом шуме «неправильная» оценка показывает R² = 0.32 для модели, которая на новых данных хуже
константы. Механизм: среди 5 000 случайных столбцов всегда найдутся двадцать, случайно коррелирующих
с ответом на этих 200 строках — в том числе на тех, по которым потом «проверяется» качество.

Не каждая утечка так заметна. Масштабирование до разбиения (300 строк, `Ridge`) даёт 0.3607 против
0.3608 внутри пайплайна. Правило от этого не меняется — оно одно для всех шагов, и проще ему следовать,
чем каждый раз оценивать, велик ли вред. Сильнее всего протекают шаги, использующие целевую переменную:
отбор признаков, target encoding (урок 2.3.2), балансировка классов.

### Причина 2. Одинаковая предобработка при обучении и применении

Расхождение между обучением и применением (training-serving skew) возникает, когда предобработку
в сервисе пишут второй раз — руками. Пример: категории закодированы номерами через
`df[col].astype("category").cat.codes`, бустинг обучен. Сервис получает запрос по одному магазину
и кодирует его тем же кодом:

| Что пришло в сервис | MAE, коды как при обучении | MAE, коды посчитаны по запросу |
|---|---|---|
| все строки магазина `S05` | 1.642 | 2.350 |
| строки трёх товаров из двенадцати | 2.356 | 3.818 |

В запросе единственный магазин, поэтому `S05` получает код 0 — а при обучении код 0 был у самого
крупного магазина `S01`. Средний прогноз — 5.64 при фактических 3.79. Ошибки нет, исключения нет,
числа правдоподобные. Тот же бустинг с `OrdinalEncoder` внутри пайплайна даёт на трёх товарах те же
2.356, что и на полных данных: кодировщик запомнил соответствие при обучении.

Более тихий вариант той же ошибки — вызвать `fit_transform` на новых данных. Если пересчитывать медианы
и масштаб по каждой дневной выгрузке, MAE линейной модели из раздела 7 меняется с 1.731 на 1.745:
незаметно на глаз и поэтому живёт годами.

### Причина 3. Один артефакт

Обученный пайплайн — один объект: его можно сохранить в один файл, передать в сервис, сравнить с прошлой
версией, откатить. Набор «модель + scaler.pkl + словарь кодов + функция в ноутбуке» рано или поздно
разъезжается по версиям.

## 3. Устройство Pipeline

`Pipeline` — список пар «имя — объект». Все шаги, кроме последнего, обязаны быть трансформерами
(`fit` + `transform`); последний — любой объект с `fit`.

```python
numeric = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("clip", QuantileClipper(0.01, 0.99)),          # свой трансформер, раздел 6
    ("scale", StandardScaler()),
])
pre = ColumnTransformer(
    [
        ("date", DateFeatures(), ["date"]),          # свой трансформер, раздел 6
        ("num", numeric, ["price", "units_week_ago"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["store_id", "product"]),
        ("unused", "drop", ["city", "category"]),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
model = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])
model.fit(X_train, y_train)                          # R² на тесте 0.654, MAE 1.955
```

Что происходит внутри:

| Вызов | Шаги, кроме последнего | Последний шаг |
|---|---|---|
| `model.fit(X, y)` | по очереди `X = step.fit_transform(X, y)` | `fit(X, y)` |
| `model.predict(X)` | по очереди `X = step.transform(X)` | `predict(X)` |
| `model.score`, `predict_proba`, `transform` | так же: только `transform` | одноимённый метод |

При `predict` ни один шаг не обучается — поэтому статистики, найденные на обучении, применяются к новым
данным без изменений. `make_pipeline(...)` делает то же самое, но имена шагов придумывает сам
(`standardscaler`, `ridge`); в коде, который живёт дольше ноутбука, давайте имена явно.

**Доступ к шагам.** `model["pre"]`, `model.named_steps["pre"]`, `model[0]` — один и тот же объект;
`model[:-1]` — пайплайн без последнего шага (всё, кроме модели). Обученные статистики вложенных шагов:

```python
model["pre"].named_transformers_["num"]["clip"].lower_     # array([1270., 0.])
model["pre"].named_transformers_["num"]["clip"].upper_     # array([74990., 18.])
```

**Параметры.** Составной объект отдаёт параметры всех вложенных объектов через `get_params()`; имя
строится из имён шагов через двойное подчёркивание. У нашего пайплайна 52 параметра, среди них
`model__alpha`, `pre__num__clip__upper`, `pre__num__clip` (сам шаг — тоже параметр):

```python
model.set_params(model__alpha=10.0, pre__num__clip__upper=0.95)
model.set_params(model__alhpa=3)
# ValueError: Invalid parameter 'alhpa' for estimator Ridge(). Valid parameters are: ['alpha', 'copy_X', ...
```

**Клонирование.** `sklearn.base.clone(model)` создаёт необученную копию: вызывает конструкторы заново
с теми же параметрами, обученные атрибуты не переносит. Так поступают `cross_val_score`, `GridSearchCV`
и все ансамбли: исходный объект они не трогают, а обучают клоны. На `get_params` / `set_params` / `clone`
держится вся библиотека — и из них же следует контракт для своих классов (раздел 6).

**Отображение.** В ноутбуке пайплайн рисуется схемой; в логах и терминале полезнее текст:

```python
sklearn.set_config(display="text")
print(model)
```

```text
Pipeline(steps=[('pre',
                 ColumnTransformer(remainder='passthrough',
                                   transformers=[('date', DateFeatures(),
                                                  ['date']),
                                                 ('num',
                                                  Pipeline(steps=[('impute',
                                                                   SimpleImputer(strategy='median')),
                                                                  ('clip',
                                                                   QuantileClipper()),
                                                                  ('scale',
                                                                   StandardScaler())]),
                                                  ['price', 'units_week_ago']),
                                                 ...
                                   verbose_feature_names_out=False)),
                ('model', Ridge())])
```

Печатаются только параметры, отличающиеся от значений по умолчанию. Эту строку стоит писать в лог
каждого обучения.

## 4. ColumnTransformer и FeatureUnion

`ColumnTransformer` применяет разные преобразования к разным столбцам и склеивает результаты по
горизонтали. Каждый элемент — тройка «имя, трансформер, столбцы».

| Что | Варианты |
|---|---|
| Трансформер | объект, вложенный `Pipeline`, строка `"drop"` или `"passthrough"` |
| Столбцы | список имён, срез, булева маска, функция от таблицы — например, `make_column_selector` |
| `remainder` | что делать со столбцами, не попавшими ни в один элемент: `"drop"` (по умолчанию), `"passthrough"` или трансформер |
| `verbose_feature_names_out` | `True` — к именам признаков добавляется имя элемента: `num__price`; `False` — имена как есть, но тогда они обязаны быть уникальными |

В примере выше `promo` не указан нигде и проходит как есть (`remainder="passthrough"`), а `city`
и `category` отброшены явно. Явное `"drop"` лучше молчаливого: когда в данных появится новый столбец,
`remainder="passthrough"` отправит его в модель, а `remainder="drop"` — молча выбросит; оба поведения
однажды удивят. Выберите одно и перечисляйте столбцы явно.

Выбор столбцов по типу:

```python
from sklearn.compose import make_column_selector as selector

ColumnTransformer([
    ("num", StandardScaler(), selector(dtype_include="number")),
    ("cat", OneHotEncoder(handle_unknown="ignore"), selector(dtype_include=["object", "string"])),
])
# num: ['price', 'units_week_ago'];  cat: ['store_id', 'city', 'category', 'product']
```

В pandas 3 текстовые столбцы имеют тип `str`, и селектор по `"string"` их находит. Булев `promo` не попал
ни в `number`, ни в строки и ушёл в `remainder` — выбор по типу удобен в исследовании, но в рабочем коде
надёжнее список имён.

`FeatureUnion` — другой способ склейки: **все** трансформеры получают **одни и те же** столбцы.

```python
union = FeatureUnion([
    ("scaled", StandardScaler()),
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
])
union.fit(X_train[["price"]]).get_feature_names_out()      # ['scaled__price', 'log__price']
```

## 5. Имена признаков и вывод pandas

После `OneHotEncoder` и `ColumnTransformer` матрица признаков — массив без подписей. Имена нужны, чтобы
подписать коэффициенты и важности, сравнить состав признаков двух версий модели и найти, какой шаг
испортил столбец. Два инструмента:

```python
model[:-1].get_feature_names_out()
# ['dow', 'is_weekend', 'doy_sin', 'doy_cos', 'days', 'price', 'units_week_ago',
#  'store_id_S01', ..., 'product_Чайник', 'promo']          — 28 признаков

model.set_output(transform="pandas")       # все шаги возвращают DataFrame
model.fit(X_train, y_train)
model[:-1].transform(X_test.head(3))
```

```text
       dow  is_weekend  doy_sin  doy_cos  days  price  units_week_ago  store_id_S01  store_id_S02
52512    0           0   -0.006     -1.0   547 -0.498          -0.843           1.0           0.0
52513    0           0   -0.006     -1.0   547 -0.662          -0.293           1.0           0.0
52514    0           0   -0.006     -1.0   547 -0.565           1.082           1.0           0.0
```

Таблица сохраняет индекс исходных строк — любую строку можно проследить от сырых данных до входа модели.
Коэффициенты с подписями:

```python
pd.Series(model[-1].coef_, index=model[:-1].get_feature_names_out()).sort_values(key=abs)
# product_Ноутбук −10.888, product_Мышь 6.865, product_Смартфон −5.261, ..., price 3.562, promo 3.471
```

Сразу видна особенность данных: цена почти однозначно определяется товаром и акцией, поэтому модель делит
один эффект между `price` и столбцами товаров (урок 3.1.2). Без имён это были бы «признаки 5 и 21».

Ограничения: вывод pandas требует плотных матриц (`sparse_output=False`), а настройка хранится в каждом
шаге отдельно. Если после `set_output` заменить шаг через `set_params`, новый шаг вернёт массив, следующий
за ним — таблицу с новым индексом, и склейка упадёт:

```text
ValueError: Concatenating DataFrames from the transformer's output lead to an inconsistent number of
samples. The output may have Pandas Indexes that do not match, ...
```

Лечится повторным `model.set_output(transform="pandas")` после замены шага или глобальной настройкой
`sklearn.set_config(transform_output="pandas")`.

## 6. Свои трансформеры

### Простой случай: FunctionTransformer

Преобразование без состояния — функция, обёрнутая в трансформер:

```python
log_price = FunctionTransformer(np.log1p, inverse_func=np.expm1, feature_names_out="one-to-one")
```

Функция должна быть импортируемой (не `lambda`), иначе пайплайн не сохранится в файл.

### Контракт

Как только преобразование что-то запоминает на обучении, нужен класс. Требования следуют из того,
как библиотека клонирует объекты и перебирает параметры:

1. Класс наследует `BaseEstimator` (даёт `get_params`, `set_params`, печать) и `TransformerMixin`
   (даёт `fit_transform` и `set_output`). `BaseEstimator` в списке предков — последний.
2. `__init__` только сохраняет аргументы в одноимённые атрибуты: `self.lower = lower`. Никаких проверок,
   преобразований, вычислений. Все аргументы — именованные, со значениями по умолчанию.
3. Всё, что вычислено по данным, записывается в `fit` в атрибуты с подчёркиванием на конце: `lower_`.
   По их наличию `check_is_fitted` отличает обученный объект от необученного.
4. `fit(X, y=None)` возвращает `self`; проверка параметров — в `fit`, а не в `__init__`.
5. `transform` не обучается и не изменяет ни `X`, ни состояние объекта.
6. Для имён признаков нужен `get_feature_names_out`; если столбцы не меняются — готовый
   `OneToOneFeatureMixin`.

Трансформер с состоянием — обрезка выбросов по квантилям обучающей выборки:

```python
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Обрезает значения каждого столбца по квантилям, найденным на обучающей выборке."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError(f"нужно 0 <= lower < upper <= 1, получено {self.lower} и {self.upper}")
        X = validate_data(self, X, reset=True, ensure_all_finite="allow-nan")
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        return np.clip(X, self.lower_, self.upper_)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
```

`validate_data` превращает вход в числовой массив, а заодно запоминает число и имена столбцов
(`n_features_in_`, `feature_names_in_`) при `reset=True` и сверяет их при `reset=False`: если при
применении столбцы придут в другом порядке, будет ошибка, а не тихо перепутанные признаки. `np.clip`
возвращает новый массив — вход не изменяется.

Трансформер для дат. Числового массива здесь нет, поэтому вход проверяется вручную, а имена новых
признаков объявляются явно:

```python
class DateFeatures(TransformerMixin, BaseEstimator):
    """Признаки из одного столбца дат: день недели, выходной, сезонность, номер дня от origin."""

    def __init__(self, origin="2023-01-01"):
        self.origin = origin

    def fit(self, X, y=None):
        self._dates(X)
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        dates = self._dates(X)
        angle = 2 * np.pi * dates.dt.day_of_year / 365.25
        return pd.DataFrame(
            {
                "dow": dates.dt.day_of_week,
                "is_weekend": (dates.dt.day_of_week >= 5).astype(int),
                "doy_sin": np.sin(angle),
                "doy_cos": np.cos(angle),
                "days": (dates - pd.Timestamp(self.origin)).dt.days,
            },
            index=dates.index,
        )

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["dow", "is_weekend", "doy_sin", "doy_cos", "days"], dtype=object)

    @staticmethod
    def _dates(X) -> pd.Series:
        if not isinstance(X, pd.DataFrame) or X.shape[1] != 1:
            raise ValueError("DateFeatures ожидает таблицу из одного столбца с датами")
        return pd.to_datetime(X.iloc[:, 0])
```

Классы кладут в модуль проекта (здесь — `sales_features.py`), а не в ноутбук: сохранённый пайплайн
хранит только путь импорта класса (раздел 10).

### Что будет, если нарушить контракт

```python
class BadClipper(TransformerMixin, BaseEstimator):
    def __init__(self, quantiles=(0.01, 0.99)):
        self.quantiles = sorted(quantiles)        # логика в __init__
```

```text
>>> clone(BadClipper((0.99, 0.01)))
RuntimeError: Cannot clone object BadClipper(quantiles=[0.01, 0.99]), as the constructor either does
not set or modifies parameter quantiles
```

Такой объект нельзя использовать ни в `cross_val_score`, ни в `GridSearchCV`. Хуже, когда ошибки нет:
если `__init__` молча приводит параметр к другому значению, `set_params` при переборе обойдёт это
приведение, и модель обучится не с теми параметрами, которые записаны в отчёте.

### Автоматическая проверка контракта

```python
from sklearn.utils.estimator_checks import check_estimator

results = check_estimator(QuantileClipper(), on_fail=None)     # список словарей, по одному на проверку
# 46 проверок: 45 passed, 1 skipped (поддержка Array API не включена)
```

В тестах проекта — по одному тесту pytest на проверку:

```python
from sklearn.utils.estimator_checks import parametrize_with_checks


@parametrize_with_checks([QuantileClipper()])
def test_sklearn_contract(estimator, check):
    check(estimator)
```

```text
45 passed, 1 skipped in 4.14s
```

Проверки подают на вход числовые массивы, поэтому рассчитаны на трансформеры общего назначения.
`DateFeatures` из 47 проверок не проходит 32 — все с его же сообщением «ожидает таблицу из одного столбца
с датами». Для таких классов контракт проверяют своими тестами (клонирование, неизменность входа, работа
внутри `Pipeline`), а заведомо неприменимые общие проверки перечисляют в параметре
`expected_failed_checks`.

## 7. Преобразование целевой переменной

Трансформеры пайплайна меняют только `X`. Для `y` есть `TransformedTargetRegressor`: он преобразует
целевую переменную перед обучением и применяет обратное преобразование к прогнозу.

```python
ttr = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
ttr.fit(X_train, y_train)
ttr.predict(X_test)            # в исходных единицах — штуках
```

| Модель | MAE на тесте | R² | Отрицательных прогнозов |
|---|---|---|---|
| прогноз средним по обучению | 3.272 | — | 0 |
| «как неделю назад» (`units_week_ago`) | 2.579 | — | 0 |
| `Ridge` | 1.955 | 0.654 | 253 |
| `Ridge` + `log1p` целевой переменной | 1.731 | 0.746 | 0 |
| `PoissonRegressor(solver="newton-cholesky")` | 1.708 | 0.768 | 0 |
| `HistGradientBoostingRegressor(loss="poisson")` | 1.825 | 0.706 | 0 |

Продажи устроены мультипликативно (выходные — «на 35 % больше», акция — «в 1.8 раза больше»), и логарифм
превращает произведение в сумму, с которой линейная модель справляется. Обратное преобразование
гарантирует неотрицательный прогноз. Пуассоновская регрессия (урок 3.1.3) делает то же самое без
преобразования целевой переменной и немного точнее. Бустинг здесь проигрывает линейным моделям: в данных
есть рост продаж во времени, а деревья не умеют продолжать тренд за пределы обучающего периода.

Обратите внимание на имена параметров: пайплайн теперь вложен в `regressor`, поэтому
`regressor__model__alpha`, `regressor__pre__num__clip__upper`.

## 8. Кэширование шагов

При подборе гиперпараметров модели шаги предобработки переобучаются заново для каждой комбинации, хотя
их параметры не менялись. Параметр `memory` сохраняет результат `fit_transform` шагов на диск и берёт его
оттуда, если шаг с теми же параметрами уже обучался на тех же данных.

```python
heavy = Pipeline(
    [
        ("pre", pre),
        ("select", SelectFromModel(RandomForestRegressor(50, min_samples_leaf=5, random_state=0))),
        ("model", Ridge()),
    ],
    memory="cache_dir",
)
GridSearchCV(heavy, {"model__alpha": [0.01, 0.1, 1, 10, 100, 1000]}, cv=TimeSeriesSplit(5), ...)
```

| Пайплайн | Без `memory` | С `memory` | Повторный запуск с готовым кэшем |
|---|---|---|---|
| с отбором признаков случайным лесом (дорогой шаг) | 135.9 с | 32.0 с | 6.4 с |
| только `pre` и `Ridge` (дешёвые шаги) | 2.7 с | 6.2 с | — |

Шесть значений `alpha` на пяти фолдах — 30 обучений; лес с кэшем обучается 5 раз вместо 30. Но кэш
не бесплатен: чтобы узнать, были ли уже такие данные, joblib считает их хеш, а результат шага пишет
на диск (здесь — 185 МБ). На дешёвых шагах это дороже самого шага: пайплайн стал вдвое медленнее.
Кэшируйте, когда шаг заметно дороже чтения и записи своего результата; каталог кэша сам не очищается.

## 9. Подбор гиперпараметров по всему пайплайну

Параметры любого шага — обычные параметры поиска. Сам шаг — тоже параметр: его можно заменить другим
объектом или строкой `"passthrough"` («пропустить шаг»). Так проверяется вопрос «нужна ли обрезка
выбросов вообще»:

```python
search = GridSearchCV(
    ttr,
    {
        "regressor__pre__num__clip": ["passthrough", QuantileClipper(0.01, 0.99), QuantileClipper(0.05, 0.95)],
        "regressor__pre__num__scale": [StandardScaler(), "passthrough"],
        "regressor__model__alpha": [0.1, 10.0, 1000.0],
    },
    cv=TimeSeriesSplit(n_splits=5),
    scoring="neg_mean_absolute_error",
)
```

MAE на кросс-валидации (со `StandardScaler`; 18 комбинаций, 7.5 с):

| Шаг `clip` | `alpha=0.1` | `alpha=10` | `alpha=1000` |
|---|---|---|---|
| `"passthrough"` | 1.5698 | **1.5682** | 1.7898 |
| `QuantileClipper(0.01, 0.99)` | 1.5704 | 1.5693 | 1.7790 |
| `QuantileClipper(0.05, 0.95)` | 1.5710 | 1.5712 | 1.7923 |

Честный результат: обрезка здесь не нужна. Различия — в четвёртом знаке при разбросе между фолдами
от 1.31 до 1.76; в этих данных нет выбросов, которые мешали бы модели. Шаг, не дающий выигрыша,
из пайплайна убирают — каждый шаг нужно поддерживать и объяснять.

То же через Optuna (урок 4.4.2): условные параметры и непрерывные диапазоны.

```python
def objective(trial):
    params = {"regressor__model__alpha": trial.suggest_float("alpha", 1e-2, 1e3, log=True)}
    if trial.suggest_categorical("clip", [True, False]):
        params["regressor__pre__num__clip"] = QuantileClipper(0.0, trial.suggest_float("clip_upper", 0.9, 0.999))
    else:
        params["regressor__pre__num__clip"] = "passthrough"
    model = clone(ttr).set_params(**params)
    return -cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error").mean()
```

30 попыток за 8.3 с: лучший результат — 1.5653 без обрезки при `alpha=2.68`; у исходных параметров 1.5670.
Вывод тот же. Главное в обоих примерах — перебирается **весь пайплайн**, поэтому каждая комбинация
оценена без утечки.

## 10. Сериализация

```python
joblib.dump(ttr, "model.joblib")        # 5.7 КБ
model = joblib.load("model.joblib")     # прогнозы совпадают с исходными побитно
```

`joblib` (как и `pickle`, на котором он основан) сохраняет не «модель», а инструкцию по восстановлению
объектов Python. Отсюда три риска.

**Версии.** Файл содержит внутреннее состояние объектов конкретной версии библиотеки. Тот же файл,
загруженный в окружении со `scikit-learn 1.6.1`:

```text
InconsistentVersionWarning: Trying to unpickle estimator Pipeline from version 1.9.1 when using
version 1.6.1. This might lead to breaking code or invalid results. Use at your own risk.
```

В этом опыте прогнозы совпали, но гарантий нет: возможны и ошибка, и тихо изменившийся результат.
Рядом с файлом модели всегда кладут метаданные:

```python
meta = {
    "python": platform.python_version(),
    "versions": {m.__name__: m.__version__ for m in (sklearn, pd, np, joblib)},
    "features": FEATURES,
    "train_period": ["2023-01-01", "2024-06-30"],
}
```

**Код классов не сохраняется.** В файле записан только путь `sales_features.QuantileClipper`. Если
класс был объявлен в ноутбуке, сервис его не найдёт; если класс в модуле изменился — загруженный
объект молча получит новое поведение. Свои трансформеры версионируются вместе с моделью.

**Безопасность.** Загрузка pickle исполняет код, записанный в файле:

```python
class Evil:
    def __reduce__(self):
        return (print, ("этот код выполнен при загрузке файла",))

pickle.loads(pickle.dumps(Evil()))      # печатает строку; на месте print может быть что угодно
```

Правило: загружать через `pickle`/`joblib` только файлы, которые создали вы сами или ваш пайплайн.

Альтернатива — `skops`: формат без исполнения кода при загрузке и с явным списком доверенных типов.

```python
import skops.io as sio

sio.dump(ttr, "model.skops")                             # 267 КБ
sio.get_untrusted_types(file="model.skops")
# ['numpy.dtype', 'sales_features.DateFeatures', 'sales_features.QuantileClipper']
sio.load("model.skops")                                  # UntrustedTypesFoundException
model = sio.load("model.skops", trusted=["numpy.dtype", "sales_features.DateFeatures",
                                         "sales_features.QuantileClipper"])
```

Типы scikit-learn доверенные по умолчанию; свои классы нужно просмотреть и перечислить явно. От
несовместимости версий `skops` не спасает — версии всё равно фиксируются.

## Типичные ошибки

| Симптом | Причина | Решение |
|---|---|---|
| На кросс-валидации отлично, на новых данных плохо | отбор признаков, кодирование или балансировка выполнены до разбиения | все шаги, которые обучаются, — внутри `Pipeline` |
| Прогнозы в сервисе правдоподобные, но хуже, чем при проверке | предобработка в сервисе написана заново или вызывает `fit_transform` | в сервис передаётся весь обученный пайплайн; на новых данных — только `predict` / `transform` |
| `RuntimeError: Cannot clone object ...` | `__init__` изменяет параметр или сохраняет его под другим именем | в `__init__` только `self.x = x`; проверки и приведения — в `fit` |
| Перебор параметров «не влияет» на свой трансформер | параметр обработан в `__init__`, а `set_params` его обходит | то же |
| После `transform` изменилась исходная таблица | трансформер пишет в `X` на месте (`X[col] = ...`, `inplace=True`) | работать с копией или строить новый объект |
| `ValueError: Invalid parameter ...` | опечатка или пропущен уровень вложенности | список имён: `model.get_params().keys()` |
| `Concatenating DataFrames ... Indexes that do not match` | шаг заменён после `set_output`, или свой трансформер сбрасывает индекс | повторить `set_output`; возвращать таблицу с индексом входа |
| При применении — ошибка или пропуски из-за новой категории | `OneHotEncoder` без `handle_unknown` | `handle_unknown="ignore"` или `"infrequent_if_exist"` |
| `AttributeError: Can't get attribute ...` при загрузке модели | класс трансформера объявлен в ноутбуке или переименован | классы — в модуле проекта, версия кода записана рядом с моделью |
| `InconsistentVersionWarning` при загрузке | модель сохранена другой версией библиотеки | фиксировать версии; переобучить модель при обновлении |

## Контрольные вопросы

1. Почему отбор признаков до кросс-валидации завышает оценку даже на чистом шуме? Почему масштабирование
   в той же ситуации почти не вредит?
2. Что такое training-serving skew? Приведите пример, при котором сервис не падает, а прогнозы неверны.
3. Какие методы шагов вызывает `Pipeline` при `fit` и при `predict`?
4. Как называется параметр `upper` обрезки выбросов в объекте `ttr` из раздела 7 и почему?
5. Что делает `clone` и почему из-за него в `__init__` нельзя размещать логику?
6. Чем `ColumnTransformer` отличается от `FeatureUnion`?
7. Зачем атрибутам, вычисленным в `fit`, подчёркивание на конце?
8. Когда `memory` замедляет подбор гиперпараметров?
9. Что означает значение `"passthrough"` в сетке параметров?
10. Назовите три риска сохранения модели через `joblib`. Какой из них закрывает `skops`?

## Домашнее задание

1. Воспроизведите опыт с утечкой из раздела 2 для `k` = 5, 20 и 100. Постройте таблицу из трёх оценок
   для каждого `k` и объясните, как разрыв зависит от `k`.
2. Напишите трансформер `FrequencyEncoder`: для каждого столбца запоминает на обучении долю строк
   с каждым значением и заменяет значение этой долей; значение, которого не было при обучении, получает 0.
   Имена признаков сохраняются. Напишите тесты: обученные атрибуты появляются только после `fit`;
   `clone` работает; вход не изменяется; незнакомая категория даёт 0; трансформер работает внутри
   `ColumnTransformer` и `GridSearchCV`; с `set_output(transform="pandas")` возвращается таблица
   с индексом входа.
3. Запустите `check_estimator(FrequencyEncoder(), on_fail=None)` и для каждой непройденной проверки
   запишите, почему она не проходит: это ошибка реализации или проверка неприменима.
4. Соберите пайплайн с `FrequencyEncoder` для `store_id` и `product` вместо `OneHotEncoder` и подберите
   поиском: кодировщик (частотный или one-hot), `alpha`, наличие шага масштабирования. Разбиение —
   `TimeSeriesSplit(5)`.
5. Сохраните лучший пайплайн через `joblib` вместе с файлом метаданных. Отдельным скриптом (новый процесс)
   загрузите модель и убедитесь, что прогнозы на `X_test` совпадают с исходными.

**Критерии приёмки:**

- [ ] в таблице задания 1 три значения `k` и три оценки для каждого; сделан вывод;
- [ ] `FrequencyEncoder` лежит в модуле, а не в ноутбуке; `__init__` только сохраняет параметры;
- [ ] тесты проходят командой `uv run python -W error -m pytest -q`;
- [ ] каждая непройденная проверка `check_estimator` объяснена;
- [ ] в отчёте — таблица результатов поиска и вывод, какой кодировщик лучше и на сколько относительно
      разброса между фолдами;
- [ ] рядом с `model.joblib` лежит `model_meta.json` с версиями Python и библиотек, списком признаков
      и периодом обучения;
- [ ] скрипт загрузки запускается в новом процессе и печатает, что прогнозы совпали;
- [ ] `ruff check` и `ruff format --check` проходят.

## Дополнительные материалы

- [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html) — руководство
  scikit-learn по `Pipeline`, `ColumnTransformer`, `FeatureUnion`, `TransformedTargetRegressor`
- [Developing scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html) — полный
  контракт для своих классов
- [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) —
  утечки при предобработке и управление случайностью
- [Model persistence](https://scikit-learn.org/stable/model_persistence.html) — сравнение способов
  сохранения моделей, включая `skops`

**Следующий урок:** 7.1.2 — Apache Airflow для оркестрации пайплайнов.
