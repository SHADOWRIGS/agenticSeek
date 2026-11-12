# Практика 4.3.4: материалы

Постановка задачи: [lesson_4_3_4.md](../lesson_4_3_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `boosting_bench.py` | заготовка стенда: `make_models`, `split_train_valid_test`, `prepare_categoricals`, `fit_with_early_stopping`, `evaluate`, `benchmark`, `summarize` |
| `tests/test_boosting_bench.py` | 85 тестов: структура результатов, ранняя остановка в четырёх библиотеках, независимость от теста, воспроизводимость, категориальные признаки |

Данные курса для второй части практики — генераторы [`orders_data.py`](../orders_data.py) (эта тема),
`housing_data.py` (тема 4.2), `fraud_data.py` (тема 3.2) и `churn_data.py` (тема 3.1). Тестам они
не нужны: тесты работают на синтетических данных и встроенном наборе `load_wine` из scikit-learn
и не обращаются к сети.

## Запуск

Версии, на которых проверены заготовка и тесты: Python 3.12, `xgboost 3.4.1`, `lightgbm 4.7.0`,
`catboost 1.2.10`, `scikit-learn 1.9.1`, `pandas 3.1.1`, `numpy 2.5.4`, `pytest 9.2.0`.

```bash
uv add numpy pandas scikit-learn xgboost lightgbm catboost
uv add --dev pytest ruff
uv run pytest -q          # на заготовке не проходит ни один тест: 41 failed, 44 errors
```

По шагам:

```bash
uv run pytest -q -k TestMakeModels
uv run pytest -q -k TestSplit
uv run pytest -q -k TestPrepareCategoricals
uv run pytest -q -k TestEarlyStopping
uv run pytest -q -k TestEvaluate
uv run pytest -q -k TestBenchmark
uv run pytest -q -k TestSummarize
```

Общие фикстуры объявлены на уровне модуля: данные (`reg_data`, `clf_data`, `cat_data`, `cat_frames`),
обученные модели (`reg_fitted`, `clf_fitted`) и таблицы (`reg_bench`, `clf_bench`, `cat_bench`). Они
вызывают ваши функции, поэтому до их реализации тесты, которые используют эти фикстуры, завершаются
ошибкой на этапе подготовки (в отчёте pytest — `ERROR`, а не `FAILED`).

Эталонное решение проходит все тесты за 6 секунд. Тесты обучают около сотни маленьких моделей
(500 объектов, не больше 300 итераций, шаг 0.3), поэтому полный прогон дольше минуты означает, что
ранняя остановка у вас не работает.

Строгая проверка — без предупреждений библиотек (устаревшие аргументы `fit` в LightGBM, приведение
типов в pandas):

```bash
uv run python -W error -m pytest -q
```

Проверка стиля:

```bash
uv run ruff check .
uv run ruff format --check boosting_bench.py tests/test_boosting_bench.py
```

## Замечания

- CatBoost по умолчанию создаёт в текущей папке каталог `catboost_info`. `make_models` обязана
  отключить запись файлов (`allow_writing_files=False`) — это проверяет тест
  `test_no_warnings_and_no_files`.
- В LightGBM 4.7.0 валидационные данные передаются в `fit` аргументами `eval_X` и `eval_y`;
  `eval_set` объявлен устаревшим. Если у вас более ранняя версия, обновите её: стенд рассчитан
  на актуальный API.
- Число потоков в тестах равно одному (`n_jobs=1` по умолчанию в `make_models`), чтобы результаты
  воспроизводились точно; для замеров времени во второй части практики задайте `n_jobs` явно
  и укажите его в отчёте.
