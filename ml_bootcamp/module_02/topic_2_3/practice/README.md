# Практика 2.3.4: материалы

Постановка задачи: [lesson_2_3_4.md](../lesson_2_3_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `fe_lab.py` | заготовка стенда экспериментов: шесть функций с описанием поведения |
| `make_demo_data.py` | демо-данные «в духе конкурса» и оценка отправки на публичной и приватной частях |
| `tests/test_fe_lab.py` | 40 тестов — исполняемая спецификация функций `fe_lab.py` |

Стенд не зависит от данных: функции принимают таблицы и столбцы pandas. Сеть нужна только для того, чтобы
скачать данные настоящего конкурса.

## Запуск

```bash
uv add pandas numpy scikit-learn
uv add --dev pytest ruff
cd practice
uv run pytest -q                       # на заготовке все тесты падают с NotImplementedError
uv run python make_demo_data.py        # data/train.csv, test.csv, sample_submission.csv, solution.csv
```

По мере реализации функций тесты начинают проходить. Удобно работать по одной группе:

```bash
uv run pytest -q -k TestCountEncode
uv run pytest -q -k TestOutOfFoldTargetEncode
uv run pytest -q -k TestEvaluateFeatures
uv run pytest -q -k TestCompareFeatureSets
uv run pytest -q -k TestJournal
```

## Пример работы со стендом

```python
from pathlib import Path

import pandas as pd
from fe_lab import compare_feature_sets, evaluate_features, log_experiment, out_of_fold_target_encode

train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
features = train[["tenure_months", "monthly_fee", "support_calls", "churn"]].copy()
features["city_te"], test_city_te = out_of_fold_target_encode(train["city"], test["city"], train["churn"])

sets = {
    "числа": ["tenure_months", "monthly_fee", "support_calls"],
    "+ город": ["tenure_months", "monthly_fee", "support_calls", "city_te"],
}
print(compare_feature_sets(features, sets, "churn", baseline="числа"))

scores = evaluate_features(features, sets["+ город"], "churn")
journal = Path("experiments/journal.jsonl")
log_experiment(journal, "+ город", sets["+ город"], scores, note="target encoding города")
```

На демо-данных и эталонной реализации этот пример печатает:

```
             n_features      mean       std     delta  wins
feature_set
числа                 3  0.603144  0.013554  0.000000     0
+ город               4  0.703910  0.012784  0.100765     5
```

## Оценка отправки на демо-данных

Файл отправки — два столбца, `id` и `churn` (вероятность ухода), по строке на каждую строку `test.csv`:

```bash
uv run python make_demo_data.py --score data/sample_submission.csv
# public: 0.5000   private: 0.5000
```

`public` считается по 30 % тестовых строк, `private` — по остальным 70 %, как на Kaggle. Смотрите на
`private` один раз, в самом конце: это имитация итогового результата конкурса.

## Как подключить данные конкурса

1. Скачайте данные конкурса в папку `data/` (как — в разделе «Данные конкурса» урока).
2. Прочитайте `train.csv` и `test.csv` в pandas. Имя целевой переменной возьмите из `sample_submission.csv`:
   это столбец, которого нет в `test.csv`.
3. Передавайте в функции стенда свои столбцы. Для задачи регрессии укажите модель и метрику явно:

```python
from sklearn.ensemble import HistGradientBoostingRegressor

scores = evaluate_features(
    features,
    columns,
    "SalePrice_log",
    model=HistGradientBoostingRegressor(random_state=42),
    scoring="neg_root_mean_squared_error",
)
```

Папки `data/` и `experiments/` в Git не добавляйте: данные скачиваются или создаются заново, а журнал
экспериментов — рабочий файл. Итоговую таблицу экспериментов перенесите в отчёт.
