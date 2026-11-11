# Урок 2.3.3: Автоматический feature engineering с FeatureTools

> **Модуль 2:** Data-Centric Machine Learning · **Тема 2.3:** Feature Engineering мастерство
> **Тип:** теория · **Время:** 120 минут + 90 минут на домашнее задание

## Цели урока

После урока вы сможете:

- объяснить идею Deep Feature Synthesis: таблицы, связи, примитивы, глубина;
- описать связанные таблицы как `EntitySet` и сгенерировать признаки функцией `dfs`;
- читать имена сгенерированных признаков и управлять их числом через примитивы и глубину;
- использовать cutoff times, чтобы признаки не заглядывали в будущее, и показать утечку числами;
- отбирать признаки и решать, когда автоматическая генерация оправдана, а когда нет.

## Зачем это нужно

Данные редко приходят одной таблицей. Есть клиенты, у клиентов заказы, в заказах позиции. Модель обучается
на таблице «одна строка — один клиент», значит, заказы и позиции нужно свернуть в признаки клиента: сколько
заказов, какой средний чек, какая доля дорогих товаров. Вручную это десятки однотипных `groupby` и `merge`,
в каждом из которых можно ошибиться со временем.

Featuretools автоматизирует именно эту часть: перебирает агрегаты по связям между таблицами и следит
за временем. Понимать его полезно в двух случаях. Первый — быстро получить сотню разумных признаков
и сильную точку отсчёта. Второй — понять на его примере, как выглядит дисциплина «признак на момент
времени», даже если библиотекой вы пользоваться не станете.

## 1. Окружение: отдельный проект

Основное окружение курса работает на pandas 3. Featuretools 1.31.0 (последняя версия на момент подготовки
урока) с pandas 3 не работает: генерация признаков завершается ошибкой `WoodworkNotInitError`. Кроме того,
библиотека Woodwork, на которую он опирается, импортирует модуль `pkg_resources`, удалённый из новых
версий setuptools.

Это обычная ситуация: инструмент развивается медленнее своих зависимостей. Решение — не откатывать pandas
во всём курсе, а завести под инструмент **отдельный проект со своими версиями**. Именно для этого вы
в уроке 1.1.1 договорились: один проект — одно окружение.

```bash
mkdir -p ~/projects/ft-lab && cd ~/projects/ft-lab
uv init --python 3.12
uv add "featuretools==1.31.0" "pandas<3" "numpy<2.5" "setuptools<81" scikit-learn
```

Проверенные версии: featuretools 1.31.0, woodwork 0.31.0, pandas 2.3.3, NumPy 2.4.6, setuptools 80.10.2,
scikit-learn 1.9.1, Python 3.12. Ограничение `numpy<2.5` убирает предупреждения, которые pandas 2.3
выдаёт на более новом NumPy. Сам featuretools на pandas 2.3 выдаёт много `FutureWarning` — мы их отключим.

Между окружениями данные передают файлами: признаки, посчитанные здесь, сохраните в Parquet
(`uv add pyarrow`) и читайте в основном проекте.

## 2. Данные и задача

Скопируйте [`shop_data.py`](shop_data.py) в проект. Это интернет-магазин: 1 500 клиентов, 13 543 заказа,
33 886 позиций в заказах за 2023–2024 годы.

```python
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import featuretools as ft
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from shop_data import make_labels, make_shop

customers, orders, items = make_shop()
```

| Таблица | Одна строка | Столбцы |
|---|---|---|
| `customers` | клиент | `customer_id`, `signup_date`, `city`, `channel` |
| `orders` | заказ | `order_id`, `customer_id`, `created_at`, `device` |
| `items` | позиция заказа | `item_id`, `order_id`, `created_at`, `product`, `category`, `price`, `quantity` |

**Задача:** на заданную дату предсказать, сделает ли клиент хотя бы один заказ в следующие 30 дней.
Клиенты в данных перестают покупать без объявления — как в жизни, где об уходе клиента не сообщает никто.

## 3. EntitySet: таблицы и связи

```python
es = ft.EntitySet(id="shop")
es = es.add_dataframe(
    dataframe_name="customers", dataframe=customers, index="customer_id", time_index="signup_date",
    logical_types={"city": "Categorical", "channel": "Categorical"},
)
es = es.add_dataframe(
    dataframe_name="orders", dataframe=orders, index="order_id", time_index="created_at",
    logical_types={"device": "Categorical"},
)
es = es.add_dataframe(
    dataframe_name="items", dataframe=items, index="item_id", time_index="created_at",
    logical_types={"product": "Categorical", "category": "Categorical"},
)
es = es.add_relationship("customers", "customer_id", "orders", "customer_id")
es = es.add_relationship("orders", "order_id", "items", "order_id")
```

| Параметр | Смысл |
|---|---|
| `index` | столбец с уникальным идентификатором строки |
| `time_index` | момент, когда строка **стала известна**. Основа защиты от утечек |
| `logical_types` | смысловой тип столбца: от него зависит, какие примитивы применимы |
| `add_relationship(родитель, ключ, потомок, ключ)` | связь «один ко многим» |

Типы столбцов определяет библиотека Woodwork. Строковый столбец с малым числом значений она может принять
за свободный текст, а числовой код — за количество. Типы категорий лучше указывать явно, а результат
проверять: `es["orders"].ww.schema`.

`time_index` нужен **каждой** таблице, в которой строки появляются со временем. Таблица без него
считается известной целиком и всегда: если убрать `time_index` у `orders`, то `COUNT(orders)` клиента 1294
на 1 июля и на 1 октября будет одинаковым — 20, все его заказы за два года, хотя на эти даты их было 11
и 15. Позиции заказа featuretools 1.31 отфильтровал бы и по времени родительского заказа, но полагаться
на это не стоит: в `items` время продублировано явно.

## 4. Deep Feature Synthesis

Алгоритм строит признаки из **примитивов** — элементарных операций двух видов:

| Вид | Что делает | Примеры |
|---|---|---|
| Агрегация | сворачивает строки потомка в одно значение для родителя | `count`, `sum`, `mean`, `max`, `std`, `mode`, `num_unique` |
| Трансформация | вычисляет новый столбец внутри одной таблицы | `month`, `weekday`, `is_weekend`, `time_since_previous` |

В версии 1.31 встроено 65 агрегаций и 138 трансформаций: `ft.primitives.list_primitives()`.

**Глубина** — сколько примитивов вложено друг в друга. «Deep» в названии — именно про это: агрегат
от агрегата.

```python
for depth in (1, 2, 3):
    definitions = ft.dfs(entityset=es, target_dataframe_name="customers", max_depth=depth, features_only=True)
    print(depth, len(definitions))
# 1 9
# 2 119
# 3 159
```

`features_only=True` возвращает только определения признаков, ничего не вычисляя, — удобно, чтобы оценить
масштаб заранее. Имена читаются изнутри наружу:

| Имя признака | Смысл |
|---|---|
| `COUNT(orders)` | число заказов клиента |
| `SUM(items.price)` | сумма цен всех позиций клиента: связь через заказы пройдена насквозь |
| `MEAN(orders.SUM(items.price))` | средний чек: сумма по позициям заказа, затем среднее по заказам клиента |
| `MODE(orders.WEEKDAY(created_at))` | день недели, в который клиент заказывает чаще всего |
| `NUM_UNIQUE(orders.MODE(items.category))` | число разных «главных категорий заказа» у клиента |

Расшифровку на английском даёт и сама библиотека: `ft.describe_feature(feature)`.

Число признаков растёт как произведение: столбцы × примитивы × пути по связям × глубина. Три таблицы
и набор по умолчанию дали 119 признаков; на десяти таблицах счёт идёт на тысячи. Средства управления:
`max_depth` (обычно 2), явные списки `agg_primitives` и `trans_primitives`, `ignore_columns`
и `max_features`.

## 5. Cutoff times: признаки на момент времени

Метка «сделает ли заказ в следующие 30 дней» привязана к дате. Признаки должны использовать только то,
что было известно на эту дату. **Cutoff time** — момент, после которого данные для строки не существуют.

```python
TRAIN_CUTOFFS = ["2024-07-01", "2024-08-01", "2024-09-01", "2024-10-01"]
train_parts = [make_labels(customers, orders, cutoff) for cutoff in TRAIN_CUTOFFS]
train_labels = pd.concat(train_parts, ignore_index=True)
test_labels = make_labels(customers, orders, "2024-11-01")
train_labels.head(3)
```

```
   customer_id       time  label
0         1409 2024-07-01  False
1         1482 2024-07-01  False
2         1488 2024-07-01  False
```

Один клиент входит в обучение четыре раза — с разными датами, признаками и метками. Проверка — на более
поздней дате, 1 ноября: это разбиение по времени из урока 2.3.1. Таблицу с датами передаём в `dfs`;
первый столбец — идентификатор, второй — время, остальные переносятся в результат без изменений.

```python
feature_matrix, definitions = ft.dfs(
    entityset=es, target_dataframe_name="customers", cutoff_time=train_labels, cutoff_time_in_index=True,
    max_depth=2,
)
feature_matrix, definitions = ft.encode_features(feature_matrix, definitions, top_n=10)
test_matrix = ft.calculate_feature_matrix(
    definitions, entityset=es, cutoff_time=test_labels, cutoff_time_in_index=True
)
feature_matrix.shape, test_matrix.shape      # ((6000, 215), (1500, 215))
```

Для каждой строки featuretools отбрасывает из всех таблиц записи с `time_index` позже её cutoff time
и только потом считает агрегаты. `encode_features` превращает категориальные признаки (`city`,
`MODE(...)`) в столбцы-индикаторы десяти самых частых значений. `calculate_feature_matrix` вычисляет
**те же самые** признаки для новых данных по сохранённым определениям: обучение и применение используют
один код.

```python
def to_xy(matrix):
    features = matrix.drop(columns="label").select_dtypes(["number", "bool"]).astype("float64")
    return features, matrix["label"].astype(int).to_numpy()


def test_auc(x_train, y_train, x_test, y_test):
    model = HistGradientBoostingClassifier(random_state=42).fit(x_train, y_train)
    return round(roc_auc_score(y_test, model.predict_proba(x_test[x_train.columns])[:, 1]), 3)


x_train, y_train = to_xy(feature_matrix)
x_test, y_test = to_xy(test_matrix)
test_auc(x_train, y_train, x_test, y_test)      # 0.908
```

214 автоматических признаков без единой строки ручного кода дают ROC AUC 0.908.

### Что будет без cutoff times

Сделаем так, как делают чаще всего: посчитаем признаки по всем данным и присоединим метки одной даты —
1 октября.

```python
october = make_labels(customers, orders, "2024-10-01")
no_cutoff = ft.calculate_feature_matrix(definitions, entityset=es)
leaky = no_cutoff.loc[october["customer_id"]].assign(label=october["label"].to_numpy())
honest = feature_matrix.xs(pd.Timestamp("2024-10-01"), level="time")

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for name, matrix in {"без cutoff": leaky, "с cutoff": honest}.items():
    x, y = to_xy(matrix)
    model = HistGradientBoostingClassifier(random_state=42)
    cv_auc = cross_val_score(model, x, y, cv=folds, scoring="roc_auc")
    print(name, round(cv_auc.mean(), 3), test_auc(x, y, x_test, y_test))
```

| Признаки для обучения | AUC на кросс-валидации | AUC на будущем периоде (1 ноября) |
|---|---|---|
| по всем данным, без cutoff | 0.917 | 0.872 |
| на момент 1 октября | 0.832 | 0.880 |

Без cutoff кросс-валидация показывает на 0.085 больше. Этот прирост фиктивный: `COUNT(orders)` включает
заказы октября, ноября и декабря, то есть сам ответ. У клиента 1294, например, признак равен 20, хотя
на 1 октября у него было 15 заказов. На будущем периоде модель с утечкой не лучше, а хуже честной.
По кросс-валидации вы выбрали бы её — и ошиблись бы на 0.045 в ожиданиях от продакшена.

Вывод тот же, что в двух предыдущих уроках, но теперь его обеспечивает инструмент: **cutoff time задаётся
для каждой строки обучения, а `time_index` — для каждой таблицы**. Если хотя бы одно из двух пропущено,
утечка возникает молча.

### Окно обучения

`training_window` ограничивает данные и с другой стороны — «не старше, чем»:

```python
es.add_last_time_indexes()
recent, recent_definitions = ft.dfs(
    entityset=es, target_dataframe_name="customers", cutoff_time=train_labels, cutoff_time_in_index=True,
    agg_primitives=["count"], trans_primitives=[], max_depth=1, training_window="90 days",
)
[f.get_name() for f in recent_definitions]      # ['city', 'channel', 'COUNT(orders)']
```

Здесь `COUNT(orders)` — число заказов за 90 дней до cutoff time: аналог скользящего окна из урока 2.3.1.
Несколько запусков с разными окнами дают признаки «за месяц», «за квартал», «за всё время».

## 6. Выбор примитивов

Набор по умолчанию не содержит ни одного примитива о том, **как давно** что-то происходило. Для задачи
об уходе клиентов это главный вопрос. Добавим его явно:

```python
AGG_PRIMITIVES = [
    "count", "sum", "mean", "max", "min", "std", "skew", "num_unique", "mode",
    "time_since_last", "time_since_first",
]
matrix_2, definitions_2 = ft.dfs(
    entityset=es, target_dataframe_name="customers", cutoff_time=train_labels, cutoff_time_in_index=True,
    agg_primitives=AGG_PRIMITIVES, max_depth=2,
)
matrix_2, definitions_2 = ft.encode_features(matrix_2, definitions_2, top_n=10)
test_matrix_2 = ft.calculate_feature_matrix(
    definitions_2, entityset=es, cutoff_time=test_labels, cutoff_time_in_index=True
)
x_train_2, _ = to_xy(matrix_2)
x_test_2, _ = to_xy(test_matrix_2)
x_train_2.shape[1], test_auc(x_train_2, y_train, x_test_2, y_test)      # (232, 0.918)
```

`TIME_SINCE_LAST(orders.created_at)` — число секунд от последнего заказа до cutoff time. Без cutoff time
этот примитив отсчитывал бы время от момента запуска кода — ещё одна причина задавать его всегда.

Выбор примитивов — это и есть feature engineering, только на уровень выше: вы решаете не «какой столбец
посчитать», а «какие вопросы к данным осмысленны в этой задаче». Собственные примитивы тоже можно
описывать — как классы-наследники `AggregationPrimitive` и `TransformPrimitive`.

## 7. Отбор признаков

Среди сотен автоматических признаков много пустых и дублирующих друг друга. В модуле `ft.selection`
есть четыре фильтра; каждый принимает матрицу и определения и возвращает их сокращёнными:

```python
selected, kept = ft.selection.remove_low_information_features(matrix_2, definitions_2)
selected, kept = ft.selection.remove_highly_null_features(selected, kept)
selected, kept = ft.selection.remove_single_value_features(selected, kept)
selected, kept = ft.selection.remove_highly_correlated_features(selected, kept)
test_selected = ft.calculate_feature_matrix(
    kept, entityset=es, cutoff_time=test_labels, cutoff_time_in_index=True
)
x_selected, _ = to_xy(selected)
x_test_selected, _ = to_xy(test_selected)
x_selected.shape[1], test_auc(x_selected, y_train, x_test_selected, y_test)      # (195, 0.92)
```

| Фильтр | Что убирает | Признаков осталось |
|---|---|---|
| до отбора | — | 232 |
| `remove_low_information_features` | столбцы из одних пропусков или с одним значением | 228 |
| `remove_highly_null_features` | пропусков больше 95 % (порог по умолчанию) | 228 |
| `remove_single_value_features` | одно уникальное значение | 228 |
| `remove_highly_correlated_features` | один из пары с корреляцией выше 0.95 (по умолчанию) | 195 |

Основную работу здесь делает корреляционный фильтр: `SUM(items.quantity)` и `COUNT(items)` — почти одно
и то же. Качество не упало (0.920). Эти фильтры не смотрят на целевую переменную, поэтому утечки
не создают; отбор по важности (permutation importance из урока 2.3.1) делают следующим шагом и только
на обучающих данных.

## 8. Сравнение с ручными признаками

Шесть признаков, которые аналитик написал бы первым делом: давность последнего заказа, число заказов
всего, за 30 и за 90 дней, средний чек, стаж клиента.

```python
def manual_features(labels):
    signup = customers.set_index("customer_id")["signup_date"]
    parts = []
    for moment, part in labels.groupby("time"):
        past = orders[orders["created_at"] < moment]
        spend = items[items["created_at"] < moment].groupby("order_id")["price"].sum()
        by_customer = past.assign(total=past["order_id"].map(spend)).groupby("customer_id")
        since = {days: past[past["created_at"] >= moment - pd.Timedelta(days=days)] for days in (30, 90)}
        table = pd.DataFrame(
            {
                "recency_days": (moment - by_customer["created_at"].max()).dt.total_seconds() / 86400,
                "orders_total": by_customer.size(),
                "orders_30d": since[30].groupby("customer_id").size(),
                "orders_90d": since[90].groupby("customer_id").size(),
                "mean_check": by_customer["total"].mean(),
            }
        )
        part = part.merge(table, left_on="customer_id", right_index=True, how="left")
        part["tenure_days"] = (moment - part["customer_id"].map(signup)).dt.days
        counts = ["orders_total", "orders_30d", "orders_90d"]
        part[counts] = part[counts].fillna(0)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


MANUAL = ["recency_days", "orders_total", "orders_30d", "orders_90d", "mean_check", "tenure_days"]
manual_train, manual_test = manual_features(train_labels), manual_features(test_labels)
test_auc(manual_train[MANUAL], manual_train["label"].astype(int), manual_test[MANUAL], y_test)      # 0.92
```

| Признаки | Число | AUC на будущем периоде |
|---|---|---|
| один ручной: `recency_days` | 1 | 0.914 |
| DFS, примитивы по умолчанию | 214 | 0.908 |
| DFS + `time_since_last`, `time_since_first` | 232 | 0.918 |
| DFS + время, после отбора | 195 | 0.920 |
| шесть ручных | 6 | 0.920 |

Один осмысленный признак оказался сильнее двухсот автоматических, а шесть ручных — не хуже лучшего набора
DFS. Автоматика догнала ручную работу только после того, как человек подсказал ей правильный вопрос.
При этом без подсказки она всё же получила 0.908 — через косвенные признаки вроде «самый частый месяц
заказа».

## 9. Когда автоматическая генерация оправдана

| Оправдана | Не оправдана |
|---|---|
| много связанных таблиц, ручной перебор агрегатов займёт дни | одна таблица: трансформации проще написать самому |
| нужна сильная точка отсчёта в первый день работы над задачей | нужна модель из десятка понятных признаков для бизнеса |
| данные со временем, и важна единая защита от утечек | данные не помещаются в память одной машины |
| обучение и применение должны считать признаки одним кодом | окружение нельзя отделить, а зависимости конфликтуют |

Цена автоматики: сотни признаков, которые нужно вычислять, хранить и объяснять; медленный расчёт (`dfs`
из раздела 6 занимает десятки секунд на шести тысячах строк); зависимость от библиотеки, отстающей
от pandas. Разумный порядок: запустить DFS, найти сильные признаки через permutation importance, понять,
**почему** они работают, — и переписать лучшие вручную.

**Альтернативы.** Для временных рядов есть tsfresh: он вычисляет сотни характеристик одного ряда
(статистики, автокорреляции, спектральные коэффициенты) и отбирает значимые статистическими тестами.
Он отвечает на другой вопрос: не «как свернуть связанные таблицы», а «как описать форму ряда числами».
Для таблиц «магазин — товар — день» из урока 2.3.1 ручные лаги и окна обычно и понятнее, и быстрее.

## Типичные ошибки

| Симптом | Причина | Решение |
|---|---|---|
| `ModuleNotFoundError: No module named 'pkg_resources'` | новая версия setuptools | `setuptools<81` в проекте с featuretools |
| `WoodworkNotInitError` при вызове `dfs` | pandas 3 | отдельный проект с `pandas<3` |
| `WoodworkNotInitError` после `drop` или `assign` над матрицей признаков | операции pandas теряют схему Woodwork | передавать в `encode_features` и `ft.selection` матрицу в том виде, в каком её вернул featuretools |
| Отличная кросс-валидация, слабый результат на новых данных | не задан `cutoff_time` или у таблицы нет `time_index` | cutoff time для каждой строки, `time_index` для каждой таблицы с событиями |
| `TIME_SINCE_LAST` даёт огромные одинаковые числа | без cutoff time отсчёт идёт от текущего момента | задать `cutoff_time` |
| Тысячи признаков, расчёт идёт часами | глубина 3 и все примитивы | `max_depth=2`, явные списки примитивов, `features_only=True` для оценки |
| Признаки на новых данных не совпадают с обучающими | вторая матрица получена новым вызовом `dfs` | `calculate_feature_matrix` с сохранёнными определениями |

## Контрольные вопросы

1. Что такое примитив? Чем агрегация отличается от трансформации?
2. Расшифруйте признак `MAX(orders.NUM_UNIQUE(items.category))`. Какова его глубина?
3. Почему число признаков растёт так быстро и какими параметрами его ограничивают?
4. Что такое cutoff time и чем он отличается от `time_index`?
5. Что произойдёт с признаком `COUNT(orders)`, если у таблицы `orders` не указать `time_index`?
6. Кросс-валидация без cutoff показала 0.917, с cutoff — 0.832. Какой модели верить и почему?
7. Почему `calculate_feature_matrix` надёжнее повторного вызова `dfs` на новых данных?
8. DFS с примитивами по умолчанию проиграл одному ручному признаку. Что из этого следует для практики?

## Домашнее задание

Работайте в отдельном проекте `ft-lab` с данными `make_shop()`.

1. **Утечка через таблицу без времени.** Создайте `EntitySet`, в котором у `orders` нет `time_index`.
   Сравните `COUNT(orders)` для клиента 1294 на 1 июля и 1 октября с правильным вариантом (11 и 15).
2. **Окна.** Постройте признаки с `training_window` 30, 90 и 365 дней, объедините их с признаками за всё
   время (добавьте к именам суффиксы) и оцените AUC на будущем периоде.
3. **Свой примитив.** Напишите агрегирующий примитив «доля заказов с телефона» и трансформирующий
   «заказ сделан ночью». Проверьте, что они появились в именах признаков.
4. **Важность.** Посчитайте permutation importance для лучшей модели DFS на отдельной валидационной дате.
   Перепишите пять самых важных признаков вручную на pandas и сравните качество с шестью ручными
   признаками урока.
5. **Передача в основной проект.** Сохраните матрицы признаков в Parquet, прочитайте их в основном
   окружении курса (pandas 3) и обучите там модель. Убедитесь, что AUC совпадает.

**Критерии приёмки:**

- [ ] в проекте `ft-lab` есть `pyproject.toml` и `uv.lock` с версиями из раздела 1;
- [ ] все матрицы признаков для моделей получены с `cutoff_time`;
- [ ] тестовая матрица везде получена через `calculate_feature_matrix`, а не повторным `dfs`;
- [ ] в задании 1 показаны числа обоих вариантов и объяснено, откуда взялась разница;
- [ ] в задании 4 для каждого переписанного признака указано, какому имени DFS он соответствует;
- [ ] итоговая таблица сравнивает все варианты на одной и той же дате проверки.

## Дополнительные материалы

- Документация Featuretools: https://featuretools.alteryx.com/en/stable/
- Документация Woodwork (типы столбцов): https://woodwork.alteryx.com/en/stable/
- Документация tsfresh: https://tsfresh.readthedocs.io/
- Kanter J. M., Veeramachaneni K. «Deep Feature Synthesis: Towards Automating Data Science Endeavors»,
  IEEE DSAA, 2015 — статья с описанием алгоритма.

**Следующий урок:** 2.3.4 — ПРАКТИКА: Feature engineering для Kaggle конкурса.
