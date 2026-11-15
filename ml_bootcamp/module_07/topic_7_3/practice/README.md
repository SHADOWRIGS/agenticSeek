# Практика 7.3.4: материалы

Постановка задачи: [lesson_7_3_4.md](../lesson_7_3_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `sales_features.py` | заготовка: восемь функций — `build_pair_features`, `point_in_time_join`, `validate_features`, `create_store`, `get_training_frame`, `publish_features`, `online_features`, `check_consistency`. Готовые части: константы (`KEYS`, `FEATURE_NAMES`, `FEATURE_COLUMNS`, `TIMESTAMP_DTYPE`, `LOAD_DELAY`, `FEATURE_VIEW`, `FEATURE_SERVICE`, `OFFLINE_FILE`, `DEFAULT_TTL`, `CHECKS`, `MAX_UNITS`, `MAX_NULL_SHARE`, `TOLERANCE`) и `DataValidationError` |
| `tests/test_sales_features.py` | 71 тест: таблица признаков и отметки времени (10), соединение по времени (9), проверка таблицы (20), создание хранилища (5), обучающая выборка (6), публикация со шлюзом (9), онлайн-признаки и свежесть (7), сверка хранилищ (5) |

Тесты не используют сеть и внешние сервисы: реестр Feast, офлайн-таблица (parquet) и онлайн-хранилище
(SQLite) создаются во временных каталогах pytest, для каждого теста свои. Данные тестов синтетические
и отсчитываются от текущего дня, поэтому результат не зависит от даты запуска.

## Окружение

Нужен **отдельный проект uv**, не основное окружение курса: `feast 0.68.0` требует `pandas<3`, а основная
среда курса — `pandas 3.1.1`.

Версии, на которых проверены заготовка и тесты: Python 3.12.13, `feast 0.68.0`, `pandas 2.3.3`,
`numpy 2.5.4`, `pyarrow 26.0.1`, `duckdb 2.0.0`, `ibis-framework 12.0.0`, `scikit-learn 1.9.1`,
`pytest 9.2.0`.

```bash
uv init --python 3.12
uv add "feast[duckdb]>=0.68" scikit-learn
uv add --dev pytest ruff
uv run pytest -q          # на заготовке: 71 failed
```

Нижняя граница версии обязательна: без неё в проекте с закреплённым `pandas 3` решатель зависимостей
молча ставит `feast 0.20.0`, который не импортируется. `scikit-learn` тестам не нужен — он понадобится
скрипту `run_store.py`. Для реализации проверки на Great Expectations или pandera добавьте их сами:
`uv add great_expectations pandera` (проверено с `great-expectations 1.25.0` и `pandera 0.36.0`).

## Запуск тестов

На заготовке все 71 тест падают с `NotImplementedError`.

По шагам:

```bash
uv run pytest -q -k test_features
uv run pytest -q -k test_join
uv run pytest -q -k test_validate
uv run pytest -q -k test_store
uv run pytest -q -k test_training_frame
uv run pytest -q -k test_publish
uv run pytest -q -k test_online
uv run pytest -q -k test_consistency
```

Эталонное решение проходит все тесты за 7 секунд. Строгая проверка — без предупреждений:

```bash
uv run python -W error -m pytest -q
```

В тестах задан один точечный фильтр предупреждений (`pytestmark` в начале файла):
`fetch_arrow_table() is deprecated` — его выдаёт `ibis-framework 12.0.0` при работе с `duckdb 2.0.0`
на каждом историческом запросе и материализации Feast; от вашего кода оно не зависит. Любое другое
предупреждение при `-W error` — из вашего решения.

Самое вероятное из них — `DeprecationWarning: The 'generic' unit for NumPy timedelta is deprecated`:
в связке `pandas 2.3.3` и `numpy 2.5.4` его вызывают `pd.Timedelta(days=1)` и `pd.Timedelta("1D")`.
Используйте `datetime.timedelta(days=1)`; там, где нужен именно `pd.Timedelta` (параметр `tolerance`
у `merge_asof`), — `pd.Timedelta(ttl)` от готового `timedelta`.

Проверка стиля:

```bash
uv run ruff check .
uv run ruff format --check sales_features.py tests/test_sales_features.py
```

## Замечания

- Фикстуры объявлены на уровне модуля: `sales` (продажи четырёх пар за 40 дней) и `features` (таблица
  признаков с произвольными значениями за 10 дней). Функции решения в фикстурах не вызываются — поэтому
  на заготовке тесты падают, а не завершаются ошибкой подготовки.
- Тесты шагов 4–7 готовят данные сами: помощники `fill_offline` и `fill_online` кладут таблицу
  в parquet-файл и в онлайн-хранилище в обход `publish_features`, а `raw_online` читает онлайн-хранилище
  средствами Feast. От ваших функций более ранних шагов зависят тесты `get_training_frame`
  (сравнение с `point_in_time_join`), `publish_features` (использует `validate_features`)
  и `check_consistency` (помощник `published` вызывает `create_store` и `publish_features`).
- Момент времени `now` во все функции передаётся параметром. Тесты вызывают их с моментами в прошлом
  и в будущем относительно системных часов — функция, читающая `datetime.now()`, тесты не пройдёт.
- Отметки времени в таблицах — `datetime64[us, UTC]`. С наносекундами (тип по умолчанию в pandas 2)
  материализация через DuckDB падает с `Unimplemented type for cast (TIMESTAMPTZ_NS -> TIMESTAMP)`.
- Пустую офлайн-таблицу в `create_store` создавайте со схемой Arrow: пустой столбец pandas типа `object`
  попадает в parquet с типом `null`, и исторический запрос падает с `IbisTypeError`.
- Feast 0.68.0 с онлайн-хранилищем SQLite не проверяет возраст значения при чтении: значение старше
  `ttl` возвращается со статусом `PRESENT`. Тест `test_online_stale_values_are_hidden` сначала
  убеждается в этом, а затем требует, чтобы устаревшее значение скрыла ваша функция.
- `get_historical_features` не сохраняет порядок строк. На малых запросах он обычно совпадает
  с исходным, поэтому ошибку ловит отдельный тест на 600 строках
  (`test_training_frame_restores_order_of_large_request`); порядок, в котором DuckDB возвращает строки,
  от запуска к запуску разный.
- Первый вызов `create_store` в процессе занимает около 0.7 секунды, остальные — сотые доли секунды;
  импорт `feast` — около 2.5 секунды.
- Feast печатает ход материализации в стандартный вывод; pytest его перехватывает.
- После работы не оставляйте в репозитории `registry.db`, `online_store.db`, parquet-файлы,
  `.pytest_cache`, `__pycache__`.
