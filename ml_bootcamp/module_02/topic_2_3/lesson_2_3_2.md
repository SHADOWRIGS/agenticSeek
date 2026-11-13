# Урок 2.3.2: Target encoding, mean encoding, likelihood encoding

> **Модуль 2:** Data-Centric Machine Learning · **Тема 2.3:** Feature Engineering мастерство
> **Тип:** теория · **Время:** 120 минут + 120 минут на домашнее задание

## Цели урока

После урока вы сможете:

- выбирать способ кодирования категориального признака в зависимости от числа категорий и модели;
- объяснить, почему наивный target encoding — это утечка целевой переменной, и показать её числами;
- реализовать кодирование по схеме out-of-fold и сверить его с `sklearn.preprocessing.TargetEncoder`;
- подобрать силу сглаживания и корректно обработать новые категории и пропуски;
- встроить кодирование в `Pipeline`, чтобы оно обучалось только на обучающих фолдах.

## Зачем это нужно

Город, модель устройства, идентификатор продавца, почтовый индекс, товар — категориальные признаки с сотнями
и тысячами значений есть почти в каждой прикладной задаче, и часто именно в них сосредоточен сигнал.
One-hot превращает такой признак в тысячи столбцов. Target encoding заменяет его одним числом — средним
значением целевой переменной в категории.

Это одновременно самый сильный и самый опасный приём работы с категориями: в признак подмешивается сам
ответ. Ошибка в одну строку даёт модель, которая блестяще выглядит на обучении и хуже случайной работает
на новых данных. В этом уроке вы увидите такую модель и научитесь её не строить.

## 1. Данные

Синтетические заказы интернет-магазина; предсказываем отмену заказа. Сигнал есть в городе (300 значений)
и сумме заказа. Идентификатор устройства (8 000 значений) с отменой **не связан** — это ловушка.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)


def make_orders_sample(n=20_000, n_cities=300, n_devices=8_000, seed=42):
    rng = np.random.default_rng(seed)
    city_effect = rng.normal(0, 1, n_cities)
    popularity = rng.pareto(1.2, n_cities) + 0.05
    city = rng.choice(n_cities, size=n, p=popularity / popularity.sum())
    device = rng.integers(0, n_devices, size=n)
    amount = rng.lognormal(7, 1, size=n).round(0)
    logit = -1 + city_effect[city] + 0.5 * (np.log(amount) - 7)
    cancelled = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "city": [f"C{code:03d}" for code in city],
            "device_id": [f"D{code:04d}" for code in device],
            "amount": amount,
            "cancelled": cancelled.astype(int),
        }
    )


orders = make_orders_sample()
train, test = train_test_split(orders, test_size=0.25, random_state=42, stratify=orders["cancelled"])
train, test = train.reset_index(drop=True), test.reset_index(drop=True)
y_train, y_test = train["cancelled"], test["cancelled"]
CATS = ["city", "device_id"]
```

В обучающей части 15 000 строк, отменяется 25.4 % заказов. Городов 291, и распределены они неравномерно,
как в жизни: в крупнейшем 2 068 заказов, в половине городов — не больше 13, в 119 городах — меньше 10.
Устройств 6 768, в среднем 2.2 заказа на каждое; 16 % тестовых заказов сделаны с устройств, которых
в обучении нет.

Качество измеряем ROC AUC на обучении и на тесте; модель — градиентный бустинг.

```python
def auc_report(model, x_train, x_test):
    model.fit(x_train, y_train)
    return (
        round(roc_auc_score(y_train, model.predict_proba(x_train)[:, 1]), 3),
        round(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]), 3),
    )


def boosting():
    return HistGradientBoostingClassifier(random_state=42)


auc_report(boosting(), train[["amount"]], test[["amount"]])     # (0.634, 0.604)
```

Одна сумма заказа даёт 0.604 на тесте. Всё, что выше, — заслуга категориальных признаков.

## 2. Простые кодирования и их пределы

```python
onehot = make_pipeline(
    ColumnTransformer(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore"), CATS),
            ("amount", make_pipeline(FunctionTransformer(np.log), StandardScaler()), ["amount"]),
        ]
    ),
    LogisticRegression(max_iter=1000),
)
auc_report(onehot, train, test)                                 # (0.889, 0.752)

ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_ordinal, test_ordinal = train[["amount"]].copy(), test[["amount"]].copy()
train_ordinal[CATS] = ordinal.fit_transform(train[CATS])
test_ordinal[CATS] = ordinal.transform(test[CATS])
auc_report(boosting(), train_ordinal, test_ordinal)             # (0.801, 0.731)

train_count, test_count = train[["amount"]].copy(), test[["amount"]].copy()
for column in CATS:
    counts = train[column].value_counts()
    train_count[column] = train[column].map(counts)
    test_count[column] = test[column].map(counts).fillna(0)
auc_report(boosting(), train_count, test_count)                 # (0.773, 0.724)
```

| Кодирование | Что получает модель | AUC train | AUC test | Ограничения |
|---|---|---|---|---|
| One-hot + логистическая регрессия | столбец на каждую категорию | 0.889 | 0.752 | 7 059 столбцов; в плотном виде — 0.85 ГБ |
| Ordinal + бустинг | номер категории | 0.801 | 0.731 | порядок номеров случаен; дереву нужно много разбиений |
| Count + бустинг | число строк с этой категорией | 0.773 | 0.724 | разные категории с одной частотой неразличимы |

One-hot работает, пока матрица разреженная, а модель линейная; деревьям он вреден: каждое разбиение
отделяет одну категорию, и на редкие категории глубины не хватает. Ordinal подходит деревьям при десятках
категорий и настоящим порядковым признакам. Count (или frequency — доля вместо числа) дёшев и полезен,
когда информативна сама популярность категории. Все три способа ничего не знают о связи категории с ответом.

## 3. Идея и формула

Заменим категорию средним значением целевой переменной среди строк с этой категорией. Для бинарной цели
это оценка вероятности P(y = 1 | категория).

**Target encoding, mean encoding, likelihood encoding** (а также impact coding и target statistics
в CatBoost) — названия одного семейства. «Mean» — потому что кодировка есть среднее; «likelihood» —
потому что для бинарной цели это оценка вероятности. Варианты различаются не названием, а двумя решениями:
как сглаживать оценку и как не допустить утечки.

Среднее по двум заказам — плохая оценка. Поэтому его **сглаживают** к глобальному среднему:

```
кодировка(c) = (n_c · mean_c + m · prior) / (n_c + m) = (sum_c + m · prior) / (n_c + m)
```

Здесь `n_c` — число строк категории, `mean_c` — среднее целевой переменной в ней, `prior` — среднее по
всем данным, `m` — сила сглаживания. Смысл `m`: к наблюдениям категории добавили `m` воображаемых строк со
средним значением. При `n_c ≫ m` кодировка равна среднему категории, при `n_c ≪ m` — глобальному среднему.

```python
def fit_target_encoding(categories, target, smooth):
    """Таблица «категория -> сглаженное среднее целевой переменной» и глобальное среднее."""
    prior = target.mean()
    stats = target.groupby(categories).agg(["sum", "count"])
    return (stats["sum"] + smooth * prior) / (stats["count"] + smooth), prior


def apply_target_encoding(categories, encoding, prior):
    return categories.map(encoding).fillna(prior)
```

`fillna(prior)` — обработка **новых категорий**: о категории, которой не было в обучении, ничего не
известно, и лучшая оценка для неё — глобальное среднее. **Пропуски** в категориальном столбце перед
кодированием заменяют отдельным значением (`fillna("нет данных")`): пропуск сам бывает информативен.

## 4. Наивный вариант — утечка

Посчитаем кодировку по обучающей выборке и применим к ней же — так делают чаще всего:

```python
encoding, prior = fit_target_encoding(train["device_id"], y_train, smooth=0)
naive_train = apply_target_encoding(train["device_id"], encoding, prior)
naive_test = apply_target_encoding(test["device_id"], encoding, prior)
roc_auc_score(y_train, naive_train), roc_auc_score(y_test, naive_test)     # (0.905, 0.502)
```

Признак, который по построению не связан с ответом, «предсказывает» его на обучении с AUC 0.905. На тесте —
0.502, монетка. Причина проста: на устройство приходится около двух заказов, и «среднее по категории» —
это почти сама целевая переменная строки. У устройства с единственным заказом кодировка равна ответу.

Модель не может знать, что признак фальшивый. Она видит столбец, который отлично разделяет классы,
и строит решение на нём:

| Признаки | AUC train | AUC test |
|---|---|---|
| сумма | 0.634 | 0.604 |
| сумма + наивный target encoding города и устройства (`m = 10`) | 0.955 | 0.581 |

Добавив признаки, мы получили модель **хуже**, чем без них: она игнорирует настоящий сигнал города и суммы
ради утечки. Сглаживание не спасает — здесь `m = 10`. Утечка тем сильнее, чем меньше строк в категории:
для одного города (сотни строк в крупных категориях) наивная схема почти безвредна — 0.762 на тесте.
Но надеяться на это нельзя: редкие категории есть почти всегда.

## 5. Схема out-of-fold

Правило то же, что во временных рядах: **значение признака для строки не должно зависеть от её собственной
целевой переменной**. Разобьём обучающую выборку на фолды и закодируем каждый фолд по статистике остальных:

```python
def oof_target_encode(categories, target, cv, smooth):
    """Каждая строка кодируется по статистике, посчитанной без её фолда."""
    result = pd.Series(np.nan, index=categories.index)
    for fit_idx, encode_idx in cv.split(categories, target):
        encoding, prior = fit_target_encoding(categories.iloc[fit_idx], target.iloc[fit_idx], smooth)
        encoded = apply_target_encoding(categories.iloc[encode_idx], encoding, prior)
        result.iloc[encode_idx] = encoded.to_numpy()
    return result


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
oof_train = oof_target_encode(train["device_id"], y_train, cv, smooth=0)
roc_auc_score(y_train, oof_train)                                          # 0.504
```

Шумовой признак остался шумом и на обучении: 0.504. Тестовую выборку кодируют статистикой **всей**
обучающей — без фолдов, потому что её целевая переменная в расчёте не участвует.

Соберём сравнение схем. Функция `scheme` кодирует обучающую часть; тест всегда кодируется одинаково.

```python
def encode_frames(scheme, smooth=10.0):
    encoded_train, encoded_test = train[["amount"]].copy(), test[["amount"]].copy()
    for column in CATS:
        encoding, prior = fit_target_encoding(train[column], y_train, smooth)
        encoded_test[column] = apply_target_encoding(test[column], encoding, prior)
        encoded_train[column] = scheme(train[column], y_train, smooth)
    return encoded_train, encoded_test


def naive(categories, target, smooth):
    return apply_target_encoding(categories, *fit_target_encoding(categories, target, smooth))


def out_of_fold(categories, target, smooth):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return oof_target_encode(categories, target, folds, smooth)


auc_report(boosting(), *encode_frames(naive))           # (0.955, 0.581)
auc_report(boosting(), *encode_frames(out_of_fold))     # (0.783, 0.759)
```

Out-of-fold даёт 0.759 на тесте — лучше one-hot (0.752) при трёх столбцах вместо семи тысяч, и разрыв
между обучением и тестом сократился с 0.374 до 0.024.

## 6. Другие способы защиты

```python
def leave_one_out(categories, target, smooth):
    """Статистика категории без текущей строки."""
    prior = target.mean()
    group = target.groupby(categories)
    total, count = group.transform("sum"), group.transform("count")
    return (total - target + smooth * prior) / (count - 1 + smooth)


def naive_with_noise(categories, target, smooth, scale=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return naive(categories, target, smooth) * rng.normal(1, scale, len(categories))


def ordered(categories, target, smooth, seed=0):
    """Статистика по строкам, которые стоят раньше в случайной перестановке."""
    prior = target.mean()
    order = np.random.default_rng(seed).permutation(len(categories))
    shuffled_categories, shuffled_target = categories.iloc[order], target.iloc[order]
    group = shuffled_target.groupby(shuffled_categories)
    before_sum, before_count = group.cumsum() - shuffled_target, group.cumcount()
    return ((before_sum + smooth * prior) / (before_count + smooth)).sort_index()


for scheme in (leave_one_out, naive_with_noise, ordered):
    print(scheme.__name__, auc_report(boosting(), *encode_frames(scheme)))
```

| Схема (`m = 10`) | AUC train | AUC test | Комментарий |
|---|---|---|---|
| наивная | 0.955 | 0.581 | утечка |
| leave-one-out | 0.978 | 0.622 | утечка другого вида |
| наивная + шум 5 % | 0.942 | 0.615 | шум слишком слаб |
| наивная + шум 20 % / 50 % | 0.878 / 0.792 | 0.679 / 0.692 | утечка слабее, но и сигнал размыт |
| out-of-fold, 5 фолдов | 0.783 | 0.759 | |
| ordered | 0.781 | 0.758 | |

**Leave-one-out** выглядит как предельный случай out-of-fold, но устроен коварнее. Внутри категории
у строк с `y = 1` кодировка `(sum − 1) / (n − 1)`, у строк с `y = 0` — `sum / (n − 1)`: значение признака
однозначно выдаёт ответ. По отдельности столбец устройства выглядит невинно (AUC на обучении 0.505),
но бустинг находит закономерность и переобучается сильнее, чем на наивной схеме по обучающей метрике: 0.978.

**Шум** маскирует утечку, но вместе с ней и сигнал; как самостоятельная защита он проигрывает out-of-fold.

**Ordered target statistics** — подход CatBoost. Строки случайно переставляют, и каждая кодируется
только по тем, что стоят в перестановке раньше неё, — как во временном ряду, где прошлое известно,
а будущее нет. Первые строки каждой категории получают почти чистый `prior`, поэтому CatBoost усредняет
несколько перестановок. Качество — как у out-of-fold, а вся выборка кодируется за один проход.

## 7. Сила сглаживания

При схеме out-of-fold сглаживание перестаёт быть защитой от утечки и становится обычным гиперпараметром:

```python
for smooth in (0.0, 1.0, 10.0, 100.0, 1000.0):
    print(smooth, auc_report(boosting(), *encode_frames(out_of_fold, smooth)))
```

| `m` | 0 | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|---|
| AUC test | 0.758 | 0.758 | 0.759 | 0.754 | 0.733 |

Малые значения почти не различаются, а слишком сильное сглаживание стирает сигнал небольших городов.
Значение `m` подбирают кросс-валидацией вместе с остальными гиперпараметрами; разумная отправная точка —
от 5 до 30. У `TargetEncoder` из scikit-learn значение по умолчанию `smooth="auto"` выбирает силу
сглаживания по дисперсии целевой переменной внутри категорий (эмпирическая байесовская оценка).

## 8. `TargetEncoder` в scikit-learn

Сверим собственную реализацию с библиотечной. Разбиение передаём одно и то же:

```python
encoder = TargetEncoder(smooth=10.0, cv=cv, target_type="binary")
sklearn_oof = encoder.fit_transform(train[CATS], y_train)
own_oof = np.column_stack([oof_target_encode(train[column], y_train, cv, smooth=10.0) for column in CATS])
np.allclose(sklearn_oof, own_oof)                                            # True

sklearn_full = encoder.transform(train[CATS])
own_full = np.column_stack([naive(train[column], y_train, 10.0) for column in CATS])
np.allclose(sklearn_full, own_full), np.allclose(sklearn_oof, sklearn_full)  # (True, False)
```

Главная особенность класса: **`fit_transform(X, y)` и `fit(X, y).transform(X)` дают разные результаты.**

| Вызов | Что возвращает | Для чего |
|---|---|---|
| `fit_transform(X_train, y_train)` | out-of-fold кодировку (внутренний cross-fitting) | признаки для обучения модели |
| `fit(X_train, y_train).transform(X_train)` | кодировку по всей выборке — наивную | **не использовать** |
| `transform(X_test)` | кодировку по всей обучающей выборке | признаки новых данных |

Перепутав первые две строки, вы получите ту самую модель с AUC 0.955 на обучении и 0.581 на тесте.
В scikit-learn 1.9 разбиение задаётся объектом в параметре `cv`; прежние параметры `shuffle` и
`random_state` объявлены устаревшими. Без явного `cv` разбиение случайное, и результат `fit_transform`
меняется от запуска к запуску.

**Новые категории и пропуски:**

```python
small = pd.DataFrame({"city": ["A", "A", "B", "B", "B", None, None, "A"]})
small_target = np.array([1, 1, 0, 0, 1, 1, 1, 0])
small_encoder = TargetEncoder(smooth=2.0, cv=2, target_type="binary").fit(small, small_target)
small_encoder.transform(pd.DataFrame({"city": ["A", "B", None, "Z"]})).ravel().round(3)
# array([0.65 , 0.45 , 0.812, 0.625])
```

Пропуск — отдельная категория со своей кодировкой (0.812). Категория `"Z"`, которой не было при обучении,
получает глобальное среднее (0.625). Ошибки при этом нет — следить за долей новых категорий в продакшене
нужно самостоятельно.

**Регрессия и многоклассовая задача.** Тип задачи определяется по целевой переменной (`target_type="auto"`):

- регрессия — кодировка есть сглаженное среднее целевой переменной. Если распределение скошено, среднее
  неустойчиво к выбросам: кодируют логарифм цели или добавляют в признаки медиану и разброс по категории;
- многоклассовая задача — по столбцу на каждый класс: доля класса в категории (схема «один против всех»).
  Два признака и три класса дают шесть столбцов с именами вида `city_high`, `city_low`, `city_mid`.

## 9. Кодирование внутри Pipeline

Target encoding — обучаемое преобразование, такое же, как модель. При кросс-валидации оно должно обучаться
только на обучающих фолдах. `Pipeline` обеспечивает это автоматически: при `fit` он вызывает у кодировщика
`fit_transform` (со внутренним cross-fitting), при `predict` — `transform`.

```python
pipeline = make_pipeline(
    ColumnTransformer(
        [("target", TargetEncoder(smooth=10.0, cv=StratifiedKFold(5, shuffle=True, random_state=0)), CATS)],
        remainder="passthrough",
    ),
    HistGradientBoostingClassifier(random_state=42),
)
columns = [*CATS, "amount"]
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(pipeline, train[columns], y_train, cv=outer, scoring="roc_auc").mean()       # 0.743

pipeline.fit(train[columns], y_train)
roc_auc_score(y_test, pipeline.predict_proba(test[columns])[:, 1])                           # 0.759
```

Для сравнения — та же кросс-валидация, но таблицу закодировали целиком **до** неё:

```python
leaky = train[["amount"]].copy()
leaky[CATS] = TargetEncoder(smooth=10.0, cv=5).fit(train[CATS], y_train).transform(train[CATS])
cross_val_score(boosting(), leaky, y_train, cv=outer, scoring="roc_auc").mean()              # 0.938
```

Кросс-валидация обещает 0.938, на тесте будет 0.581. Это общий закон, а не особенность target encoding:
**любое преобразование, которое обучается на данных, должно находиться внутри `Pipeline`**. К масштабированию
и заполнению пропусков это тоже относится, просто там утечка обычно мала.

## 10. Библиотека category_encoders

Пакет `category_encoders` (`uv add category_encoders`) содержит десятки кодировщиков с интерфейсом
scikit-learn. Проверка на версии 2.11.2 с pandas 3.1:

```python
import category_encoders as ce

library_loo = ce.LeaveOneOutEncoder(cols=["city"]).fit_transform(train[["city"]], y_train)["city"]
np.allclose(library_loo, leave_one_out(train["city"], y_train, 0.0).fillna(y_train.mean()))   # True

library_ordered = ce.CatBoostEncoder(cols=["city"], a=10.0).fit_transform(train[["city"]], y_train)["city"]
by_city = y_train.groupby(train["city"])
own_ordered = (by_city.cumsum() - y_train + 10.0 * y_train.mean()) / (by_city.cumcount() + 10.0)
np.allclose(library_ordered, own_ordered)                                                     # True
```

Оба кодировщика совпали с нашими формулами. `CatBoostEncoder` идёт по строкам **в том порядке, в каком
они лежат в таблице**, и сам их не переставляет: для таблицы, отсортированной по времени, это честная
схема «только прошлое», иначе строки нужно перемешать заранее. `ce.TargetEncoder` и `ce.CountEncoder`
на pandas 3.1 работают, но выдают `Pandas4Warning` (при запуске с `python -W error` это ошибка).
И главное: `ce.TargetEncoder.fit_transform` не делает cross-fitting — это наивная схема, защищённая
только сглаживанием. Для новых проектов используйте `sklearn.preprocessing.TargetEncoder`.

## Типичные ошибки

| Симптом | Причина | Решение |
|---|---|---|
| AUC на обучении около 0.95, на тесте хуже, чем без признака | кодировка обучающей выборки посчитана по ней же | out-of-fold или `fit_transform` из scikit-learn |
| Кросс-валидация отличная, отложенная выборка плохая | таблица закодирована целиком до кросс-валидации | кодировщик внутри `Pipeline` |
| `fit(X, y).transform(X)` даёт утечку, хотя класс «защищён» | cross-fitting есть только в `fit_transform` | для обучающих данных — только `fit_transform` |
| Результат кодирования меняется от запуска к запуску | случайное внутреннее разбиение | передать в `cv` объект с `random_state` |
| Leave-one-out переобучается сильнее наивной схемы | значение признака внутри категории однозначно выдаёт ответ | out-of-fold или ordered |
| В продакшене много значений, равных глобальному среднему | новые категории | отслеживать долю новых категорий; переобучать кодировщик |
| Редкие категории получают крайние значения 0 и 1 | нет сглаживания | `smooth` от 5 до 30 или `"auto"` |

## Контрольные вопросы

1. Чем target encoding, mean encoding и likelihood encoding отличаются друг от друга?
2. Почему шумовой признак с 6 768 значениями после наивного кодирования получил AUC 0.905 на обучении?
3. Модель с наивным кодированием оказалась на тесте хуже модели без категориальных признаков. Почему
   лишний признак ухудшил, а не просто «не помог»?
4. Что означает параметр `m` в формуле сглаживания? Чему равна кодировка при `n_c = 0`?
5. Как при схеме out-of-fold кодируется тестовая выборка и почему не по фолдам?
6. Почему leave-one-out — утечка, хотя собственная целевая переменная строки в сумму не входит?
7. Чем `TargetEncoder.fit_transform(X, y)` отличается от `TargetEncoder.fit(X, y).transform(X)`?
8. Почему кодировщик должен находиться внутри `Pipeline`? Что покажет кросс-валидация, если это нарушить?

## Домашнее задание

Работайте в ноутбуке `homework/m02/hw_2_3_2.ipynb` с данными `make_orders_sample()`.

1. **Кардинальность и утечка.** Сгенерируйте шумовой признак с 10, 100, 1 000 и 10 000 значений.
   Для каждого посчитайте AUC наивной кодировки на обучении при `m = 0` и `m = 10`. Постройте график
   и объясните его форму.
2. **Число фолдов.** Сравните out-of-fold с 2, 5, 10 и 50 фолдами и leave-one-out: AUC на обучении
   и на тесте. При каком числе фолдов схема начинает вести себя как leave-one-out?
3. **Подбор сглаживания.** Поместите `TargetEncoder` в `Pipeline` и подберите `smooth` через
   `GridSearchCV` (значения 1, 5, 10, 30, 100 и `"auto"`). Совпал ли выбор с таблицей раздела 7?
4. **Ordered и порядок строк.** Отсортируйте обучающую выборку по целевой переменной и примените
   `ordered` без перестановки (или `ce.CatBoostEncoder`). Что произошло с AUC на тесте и почему?
5. **Регрессия.** Возьмите `make_sales()` из темы 2.1 и закодируйте `product` и `store_id` средним
   `units` по схеме out-of-fold. Сравните с ручным признаком `mean_28` из урока 2.3.1: в чём target
   encoding по случайным фолдам нарушает правила работы со временем?

**Критерии приёмки:**

- [ ] собственная функция out-of-fold сверена с `TargetEncoder` через `np.allclose`;
- [ ] в задании 1 есть график и объяснение, почему утечка растёт с кардинальностью;
- [ ] в задании 3 кодировщик находится внутри `Pipeline`, а не применяется до кросс-валидации;
- [ ] в задании 5 сформулировано, какая схема кодирования корректна для данных со временем;
- [ ] ноутбук выполняется целиком сверху вниз.

## Дополнительные материалы

- Руководство scikit-learn, раздел «Target Encoder»:
  https://scikit-learn.org/stable/modules/preprocessing.html#target-encoder
- Документация `TargetEncoder`:
  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
- Prokhorenkova L. и др. «CatBoost: unbiased boosting with categorical features», NeurIPS 2018 — ordered
  target statistics и анализ утечки целевой переменной.
- Документация category_encoders: https://contrib.scikit-learn.org/category_encoders/
- Micci-Barreca D. «A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification
  and Prediction Problems», ACM SIGKDD Explorations, 2001 — статья, с которой начался target encoding.

Примеры проверены на Python 3.12, pandas 3.1.1, NumPy 2.5.4, scikit-learn 1.9.1, category_encoders 2.11.2.

**Следующий урок:** 2.3.3 — Автоматический feature engineering с FeatureTools.
