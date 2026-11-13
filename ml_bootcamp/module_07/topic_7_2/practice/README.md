# Практика 7.2.4: материалы

Постановка задачи: [lesson_7_2_4.md](../lesson_7_2_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `tracking.py` | заготовка: восемь функций поверх MLflow — `collect_run_metadata`, `start_run`, `log_training_result`, `find_best_run`, `register_and_promote`, `load_champion`, `reproduce_command`, `compare_runs`. Готовые части: константы `REQUIRED_TAGS`, `PACKAGES`, `SECRET_MARKERS`, `REDACTED`, `MODEL_NAME` |
| `tests/test_tracking.py` | 49 тестов: метаданные запуска и «грязное» дерево Git, обязательные теги, запись параметров, метрик, модели и окружения, маскирование секретов, поиск лучшего запуска, регистрация и продвижение, загрузка по псевдониму, команда воспроизведения, таблица сравнения |

Тесты не используют сеть и сервер трекинга: хранилище MLflow — база SQLite и каталог артефактов
во временной папке pytest, одно на весь модуль тестов; каждый тест работает со своим экспериментом
и своей зарегистрированной моделью. Репозитории Git для тестов создаются во временных каталогах,
глобальная конфигурация Git не меняется. Нужен установленный `git`.

## Запуск тестов

Версии, на которых проверены заготовка и тесты: Python 3.12, `mlflow 3.18.0`, `scikit-learn 1.9.1`,
`pandas 3.1.1`, `numpy 2.5.4`, `pytest 9.2.0`, Git 2.43.

```bash
uv add mlflow scikit-learn pandas numpy
uv add --dev pytest ruff
uv run pytest -q          # на заготовке: 49 failed
```

На заготовке 48 тестов падают с `NotImplementedError`. Ещё один — `test_log_requires_active_run` — ждёт
`RuntimeError` и отдельно проверяет точный тип исключения: `NotImplementedError` — подкласс `RuntimeError`
и без этой проверки тест прошёл бы на пустой заготовке.

По шагам:

```bash
uv run pytest -q -k test_metadata
uv run pytest -q -k test_start_run
uv run pytest -q -k "test_log or test_secrets"
uv run pytest -q -k test_find_best
uv run pytest -q -k "version or champion or alias or improvement or registration or promotion or register"
uv run pytest -q -k test_reproduce
uv run pytest -q -k test_compare
```

Эталонное решение проходит все тесты за 14 секунд. Строгая проверка — без предупреждений библиотек:

```bash
uv run python -W error -m pytest -q
```

В тестах заданы три точечных фильтра предупреждений (`pytestmark` в начале файла). Все три выдаёт сам
MLflow 3.18 в сочетании с SQLAlchemy 2.1 и pandas 3.1, и от вашего кода они не зависят:
`The noload loader strategy is deprecated`, `Operation with range is deprecated`,
`Any type hint is inferred as AnyType`. Любое другое предупреждение при `-W error` — из вашего решения.

Проверка стиля:

```bash
uv run ruff check .
uv run ruff format --check tracking.py tests/test_tracking.py
```

## Замечания

- Функции не задают адрес хранилища: его выбирает вызывающий код (`mlflow.set_tracking_uri`
  или `MLFLOW_TRACKING_URI`). В тестах это делает фикстура `store`; она же на время тестов меняет текущий
  каталог на временный, потому что артефакты по умолчанию пишутся в `./mlruns`.
- Фикстуры объявлены на уровне модуля: `store` (хранилище, одно на модуль), `no_active_run` (после теста
  не остаётся активного запуска), `names` (имена эксперимента и модели для теста), `repo` (репозиторий Git
  с одним коммитом), `data_dir` (три файла данных). Помощники `log_run` и `light_run` создают запуски
  **через ваши** `start_run` и `log_training_result` — поэтому тесты реестра и сравнения не пройдут,
  пока не готовы первые три функции.
- `log_training_result` передаёт в `mlflow.sklearn.log_model` явный список `pip_requirements`. Без него
  MLflow определяет зависимости модели сам, и каждая запись модели занимает секунды.
- Формат моделей по умолчанию в MLflow 3.18 — skops: модель из стандартных классов scikit-learn
  записывается без настроек, а пайплайну с собственными трансформерами (практика 7.1.4) нужен
  `serialization_format="cloudpickle"` или список `skops_trusted_types` — для этого у функции есть
  `**log_model_kwargs`.
- `search_model_versions` в MLflow 3.18 возвращает версии с пустым списком псевдонимов; текущего
  обладателя псевдонима получайте через `get_model_version_by_alias`.
- Метрика «чемпиона» читается из тега его версии, а не пересчитывается: сравнение честно, только если
  обе метрики получены на одних данных. Что с этим делать — в идеях для усложнения.
- После работы не оставляйте в репозитории `mlruns/`, `mlflow.db`, `.pytest_cache`, `__pycache__`.
