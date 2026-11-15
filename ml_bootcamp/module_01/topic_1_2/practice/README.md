# Практика 1.2.4: эффективный ETL-пайплайн

Стартовый комплект к уроку [1.2.4](../lesson_1_2_4.md). Постановка задачи, правила преобразования данных
и критерии приёмки — в уроке. Здесь — как запустить и что измерять.

## Состав

| Файл | Назначение |
|---|---|
| `generate_data.py` | генерирует сырые данные: `products.csv` и файлы `events_NNN.csv` с «грязью» |
| `rates_service.py` | имитация внешнего сервиса курсов валют (задержка и временные ошибки); не менять |
| `etl_baseline.py` | наивный, медленный, но корректный пайплайн — точка отсчёта; не менять |
| `tests/test_etl.py` | тесты корректности: годятся и для baseline, и для вашей версии |
| `pyproject.toml` | зависимости (pandas, pyarrow, pytest, ruff), настройки Ruff и pytest |

## Запуск

Скопируйте папку в свой репозиторий курса и выполняйте команды внутри копии.

```bash
uv sync                                                        # создать окружение
uv run python generate_data.py --rows 50000 --out data/raw     # сгенерировать данные
uv run python etl_baseline.py --raw data/raw --out data/out    # запустить baseline
uv run pytest                                                  # тесты для baseline
uv run ruff check .                                            # линтер
```

Параметры генератора: `--rows` — число строк с данными во всех файлах, `--files` — число файлов
(по умолчанию 8), `--seed` — зерно генератора. При одинаковых параметрах данные получаются одинаковыми.

## Своя версия

Создайте `etl_fast.py` с функцией `run_pipeline(raw_dir: Path, out_dir: Path) -> dict[str, int]`
и проверьте её теми же тестами:

```bash
ETL_MODULE=etl_fast uv run pytest
uv run python etl_fast.py --raw data/raw --out data/out_fast
```

Тесты отключают задержку сервиса курсов (`RATES_LATENCY_MS=0`), поэтому идут быстро. В замерах задержку
не трогайте: по умолчанию каждый запрос к сервису занимает 5 мс.

## Что измерять

1. **Время** работы `run_pipeline` (`time.perf_counter`) на 50 000 и 1 000 000 строк.
2. **Пик памяти** процесса. `etl_baseline.py` печатает его сам, функцию `peak_memory_mb` можно
   импортировать: `from etl_baseline import peak_memory_mb`. Она учитывает только текущий процесс; если
   используете дочерние процессы, добавьте `resource.getrusage(resource.RUSAGE_CHILDREN)`.
3. **Профиль** до и после каждой оптимизации:

```bash
uv run python -m cProfile -o baseline.prof etl_baseline.py --raw data/raw --out data/out
uv run python -c "import pstats; pstats.Stats('baseline.prof').strip_dirs().sort_stats('tottime').print_stats(15)"
```

Baseline на 50 000 строк работает около 15 секунд (на машине авторов), и время растёт быстрее, чем объём
данных. Запускать его на миллионе строк не нужно.

Работа рассчитана на Linux и macOS (в Windows — через WSL, как во всём курсе): модуль `resource` в Windows
недоступен.
