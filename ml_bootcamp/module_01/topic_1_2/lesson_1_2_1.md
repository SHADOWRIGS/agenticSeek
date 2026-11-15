# Урок 1.2.1: Продвинутый Python: декораторы, генераторы, context managers

> **Модуль 1:** Foundations & Mindset Engineer · **Тема 1.2:** Python для ML инженера
> **Тип:** теория · **Время:** 120 минут + 90 минут на домашнее задание

## Цели урока

После урока вы сможете:

- писать декораторы с параметрами и без: замер времени, повтор при сбое, кэширование результата;
- читать данные, которые не помещаются в память, с помощью генераторов и собирать из них конвейеры;
- оформлять «подготовку и уборку» (таймер, временные файлы, фиксация seed) в виде context manager;
- объяснить, что именно происходит при `@decorator`, `yield` и `with`.

## Зачем это нужно

Модель — малая часть кода ML-проекта. Остальное — обвязка: загрузить данные, повторить запрос к хранилищу,
которое ответило ошибкой, замерить время шага, не упасть по памяти на файле в 20 ГБ, вернуть генератор
случайных чисел в прежнее состояние, не оставить на диске наполовину записанный файл с метриками.

Если писать эту обвязку «в лоб», она размазывается по всему проекту: в каждой функции свой `time.time()`,
свой `try/except` с повтором, свой `f.close()`. Три инструмента этого урока позволяют написать её один раз:

1. **Декоратор** добавляет поведение вокруг функции, не меняя её код.
2. **Генератор** отдаёт данные по одному элементу, а не все сразу.
3. **Context manager** гарантирует уборку после блока кода, даже если в нём произошла ошибка.

Примеры запускайте в проекте `ml-bootcamp` из урока 1.1.1: сохраните код в файл и выполните
`uv run python имя_файла.py`. Блоки кода продолжают друг друга — складывайте их в один файл: следующим
блокам нужны импорты и функции из предыдущих. Время и память в выводе зависят от машины.

## 1. Декораторы: поведение вокруг функции

Функция в Python — обычный объект: её можно передать в другую функцию и вернуть как результат. Декоратор —
это функция, которая принимает функцию и возвращает новую, с дополнительным поведением. Запись `@timer` над
`def build_features` означает ровно одно: `build_features = timer(build_features)`.

```python
import functools
import time
from collections.abc import Callable


def timer[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Печатает время выполнения функции."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            print(f"{func.__name__}: {time.perf_counter() - start:.3f} с")

    return wrapper


@timer
def build_features(n_rows: int) -> list[float]:
    """Считает признаки для n_rows объектов."""
    return [(i % 7) ** 0.5 for i in range(n_rows)]


features = build_features(2_000_000)
print(len(features), build_features.__name__, build_features.__doc__)
```

```text
build_features: 0.222 с
2000000 build_features Считает признаки для n_rows объектов.
```

На что обратить внимание:

- `wrapper` принимает любые аргументы (`*args, **kwargs`) и передаёт их исходной функции без изменений;
- `functools.wraps` копирует имя и docstring исходной функции. Без него `build_features.__name__` вернёт
  `wrapper`, и в логах и отчётах профилировщика все функции окажутся с одним именем;
- `try/finally` печатает время, даже если функция завершилась исключением;
- `time.perf_counter()` — часы для замеров: они не «прыгают» при синхронизации системного времени;
- `[**P, R]` — синтаксис Python 3.12 для обобщённых функций. Он сообщает редактору, что декоратор сохраняет
  параметры (`P`) и тип результата (`R`), поэтому подсказки по аргументам `build_features` продолжают работать.

## 2. Декоратор с параметрами: повтор при сбое

Сеть и хранилища отказывают: таймаут, обрыв соединения, «попробуйте позже». Задача, которая шесть часов
готовила датасет и упала на одном неудачном запросе, — типичная и обидная потеря. Декоратору `retry` нужны
настройки, поэтому появляется третий уровень вложенности: функция с параметрами возвращает декоратор.

```python
import random


def retry[**P, R](
    attempts: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (ConnectionError, TimeoutError),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Повторяет вызов при временных ошибках, увеличивая паузу между попытками."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            pause = delay
            for attempt in range(1, attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    print(f"{func.__name__}: попытка {attempt} не удалась ({exc}), ждём {pause:.1f} с")
                    time.sleep(pause)
                    pause *= backoff
            return func(*args, **kwargs)  # последняя попытка: исключение уходит вызывающему коду

        return wrapper

    return decorator


@timer
@retry(attempts=4, delay=0.1)
def download_batch(batch_id: int) -> list[float]:
    if random.random() < 0.6:  # имитация нестабильного хранилища
        raise ConnectionError("хранилище недоступно")
    return [batch_id * 0.1] * 3


random.seed(7)
print(download_batch(5))
```

```text
download_batch: попытка 1 не удалась (хранилище недоступно), ждём 0.1 с
download_batch: попытка 2 не удалась (хранилище недоступно), ждём 0.2 с
download_batch: 0.301 с
[0.5, 0.5, 0.5]
```

Три правила повторов:

- **повторяем только временные ошибки.** `ValueError` из-за битых данных от повтора не исчезнет, поэтому
  список `exceptions` перечисляется явно, а `except Exception` здесь — ошибка;
- **пауза растёт** (0.1 → 0.2 → 0.4 с): перегруженному сервису нужно время, а не новая волна запросов;
- **число попыток ограничено**, после последней исключение уходит наверх.

Декораторы применяются снизу вверх: `timer(retry(...)(download_batch))`. Поэтому `timer` здесь измеряет всё
время вместе с повторами и паузами. Поменяйте их местами — и время будет печататься для каждой попытки.

## 3. Кэширование результатов

Если функция при одинаковых аргументах возвращает одинаковый результат и считается долго, результат можно
запомнить. В стандартной библиотеке для этого есть `functools.lru_cache`.

```python
@functools.lru_cache(maxsize=10_000)
def user_features(user_id: int) -> tuple[float, float]:
    time.sleep(0.2)  # имитация тяжёлого запроса к базе
    return (user_id % 10 / 10, user_id % 3 / 3)


for label in ("первый вызов", "повторный вызов"):
    start = time.perf_counter()
    user_features(42)
    print(f"{label}: {time.perf_counter() - start:.6f} с")
print(user_features.cache_info())
```

```text
первый вызов: 0.200172 с
повторный вызов: 0.000003 с
CacheInfo(hits=1, misses=1, maxsize=10000, currsize=1)
```

Ограничения, о которых нужно помнить:

- аргументы должны быть хешируемыми. Список или DataFrame передать нельзя: будет
  `TypeError: unhashable type: 'list'`. Передавайте ключ (идентификатор, путь к файлу, кортеж);
- кэш живёт в памяти процесса. `maxsize` ограничивает число записей: при переполнении вытесняются давно
  не использовавшиеся. `functools.cache` — тот же кэш без ограничения, на долгой задаче он может съесть память;
- кэш отдаёт тот же самый объект. Если функция вернула список и вы его изменили, изменилось и содержимое
  кэша. Возвращайте неизменяемые объекты (кортежи);
- при перезапуске процесса кэш пропадает. Тяжёлые признаки, которые нужны между запусками, кэшируют на диск —
  такой декоратор вы напишете в домашнем задании.

## 4. Генераторы: данные по одному элементу

Функция с `yield` — генератор. Её вызов не выполняет тело, а возвращает объект-итератор. Тело выполняется
по шагам: до ближайшего `yield` при каждом запросе следующего элемента.

```python
from collections.abc import Iterator


def countdown(n: int) -> Iterator[int]:
    print("старт")
    while n > 0:
        yield n
        n -= 1
    print("финиш")


gen = countdown(2)  # ничего не напечатано: тело ещё не выполнялось
print(next(gen))    # «старт», затем 2
print(next(gen))    # 1
print(list(gen))    # «финиш», затем [] — элементы закончились
```

Для данных это означает: в памяти находится только текущий элемент. Создадим CSV на 500 тысяч строк
и посчитаем сумму покупок двумя способами. Пик памяти измерим модулем `tracemalloc` (о нём — в уроке 1.2.3).

```python
import csv
import tracemalloc
from collections.abc import Iterable
from pathlib import Path

path = Path("events.csv")
rng = random.Random(0)
events = ["view", "cart", "purchase"]
with path.open("w", encoding="utf-8") as f:
    f.write("user_id,event,amount\n")
    f.writelines(f"{i % 5000},{rng.choice(events)},{rng.uniform(1, 500):.2f}\n" for i in range(500_000))
print(f"размер файла: {path.stat().st_size / 1e6:.1f} МБ")


def read_rows(path: Path) -> Iterator[dict[str, str]]:
    """Читает CSV построчно: файл открыт, пока из генератора берут строки."""
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def purchase_amounts(rows: Iterable[dict[str, str]]) -> Iterator[float]:
    return (float(row["amount"]) for row in rows if row["event"] == "purchase")


tracemalloc.start()
rows = list(read_rows(path))  # весь файл в памяти
total = sum(float(row["amount"]) for row in rows if row["event"] == "purchase")
print(f"список:    сумма {total:.2f}, пик памяти {tracemalloc.get_traced_memory()[1] / 1e6:.1f} МБ")
del rows

tracemalloc.reset_peak()
total = sum(purchase_amounts(read_rows(path)))  # конвейер генераторов
print(f"генератор: сумма {total:.2f}, пик памяти {tracemalloc.get_traced_memory()[1] / 1e6:.1f} МБ")
tracemalloc.stop()
```

```text
размер файла: 8.9 МБ
список:    сумма 41585058.61, пик памяти 165.1 МБ
генератор: сумма 41585058.61, пик памяти 0.1 МБ
```

Файл в 8.9 МБ превратился в 165 МБ объектов Python. Файл в 5 ГБ таким способом на обычной машине
не прочитать, а конвейеру генераторов размер файла безразличен. Что здесь использовано:

- `yield from iterable` отдаёт все элементы вложенного итератора;
- `(... for row in rows if ...)` — генераторное выражение: тот же генератор, записанный в одну строку. Его
  же принимает `f.writelines`, поэтому и запись файла не потребовала списка из 500 тысяч строк;
- конвейер `purchase_amounts(read_rows(path))` ленивый: `sum` запрашивает число, фильтр запрашивает строку
  у файла. Промежуточных списков не существует.

Модели и базы данных удобнее кормить не по одной строке, а пачками. В Python 3.12 для этого есть
`itertools.batched`: она режет любой итератор на кортежи заданной длины.

```python
import itertools

for i, batch in enumerate(itertools.batched(purchase_amounts(read_rows(path)), 50_000)):
    print(f"батч {i}: {len(batch)} значений, среднее {sum(batch) / len(batch):.2f}")

squares = (i * i for i in range(1_000_000))
print(sum(squares), sum(squares))
path.unlink()
```

```text
батч 0: 50000 значений, среднее 249.37
батч 1: 50000 значений, среднее 249.34
батч 2: 50000 значений, среднее 250.68
батч 3: 16439 значений, среднее 250.36
333332833333500000 0
```

Последняя строка — главное ограничение: **генератор одноразовый**. Второй `sum` получил ноль, потому что
элементы уже израсходованы. У генератора также нет `len()` и индексации. Если данные нужны дважды, создайте
генератор заново или, если данные помещаются в память, соберите список.

## 5. Context managers: гарантированная уборка

Конструкция `with` решает задачу «подготовить — поработать — убрать за собой при любом исходе». Объект после
`with` должен иметь два метода: `__enter__` выполняется при входе в блок, `__exit__` — при выходе, в том
числе при исключении.

```python
from typing import Self


class Timer:
    """Замеряет время блока кода."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.elapsed = 0.0

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        return self  # это значение попадает в переменную после as

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.elapsed = time.perf_counter() - self._start
        print(f"[{self.name}] {self.elapsed:.3f} с ({'ошибка' if exc_type else 'ок'})")
        return False  # False — не подавлять исключение, если оно было


with Timer("подготовка признаков") as t:
    data = [x**0.5 for x in range(1_000_000)]
print(f"в метрики эксперимента: {t.elapsed:.3f}")
```

```text
[подготовка признаков] 0.093 с (ок)
в метрики эксперимента: 0.093
```

Если блок завершился исключением, `__exit__` всё равно будет вызван и получит его в аргументах
`exc_type, exc, tb` (при нормальном завершении все три равны `None`). Декоратор `timer` измеряет функцию
целиком, `Timer` — произвольный кусок кода.

## 6. contextlib: менеджер из генератора

Писать класс ради двух методов долго. `contextlib.contextmanager` делает менеджер из генератора: код до
`yield` — вход, код после — выход. Чтобы уборка выполнялась и при ошибке, `yield` оборачивают в `try/finally`.

```python
import tempfile
from contextlib import contextmanager

import numpy as np


@contextmanager
def fixed_seed(seed: int) -> Iterator[None]:
    """Фиксирует seed внутри блока, а после него возвращает генераторы в прежнее состояние."""
    py_state, np_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


with fixed_seed(42):
    a = np.random.rand(3)
with fixed_seed(42):
    b = np.random.rand(3)
print(a, np.array_equal(a, b))

with tempfile.TemporaryDirectory() as tmp:
    checkpoint = Path(tmp) / "features.npy"
    np.save(checkpoint, a)
    print(checkpoint.exists(), np.load(checkpoint).shape)
print(Path(tmp).exists())
```

```text
[0.37454012 0.95071431 0.73199394] True
True (3,)
False
```

- `fixed_seed` делает кусок кода воспроизводимым и не ломает случайность в остальной программе. Глобальное
  состояние `np.random` встречается в старом коде и библиотеках; в своём новом коде лучше создавать
  отдельный генератор `np.random.default_rng(seed)` и передавать его явно;
- `tempfile.TemporaryDirectory` удаляет папку со всем содержимым при выходе из блока (последняя строка
  вывода). Промежуточные файлы не копятся на диске, даже если расчёт упал;
- файлы, сокеты, соединения с базой, блокировки — всё это тоже открывают через `with`.

## Типичные ошибки

| Симптом | Причина | Решение |
|---|---|---|
| `func.__name__` возвращает `wrapper`, пропал docstring | в декораторе нет `functools.wraps` | добавьте `@functools.wraps(func)` над `wrapper` |
| Декорированная функция возвращает `None` | `wrapper` вызывает функцию, но не возвращает результат | `return func(*args, **kwargs)` |
| `TypeError: unhashable type: 'list'` при вызове кэшируемой функции | аргументы `lru_cache` должны быть хешируемыми | передавайте ключ или кортеж, а не сами данные |
| Память процесса растёт при долгой работе | `functools.cache` без ограничения размера | `lru_cache(maxsize=...)`, при необходимости `cache_clear()` |
| `retry` бесконечно долго повторяет заведомо неудачный вызов | ловится `Exception` целиком, включая ошибки в данных | перечисляйте только временные ошибки |
| Второй проход по генератору ничего не возвращает | генератор одноразовый | создайте его заново или соберите список |
| `ValueError: I/O operation on closed file.` | из блока `with` вернули ленивый объект чтения, файл уже закрыт | открывайте файл внутри генератора, как в `read_rows` |
| После ошибки в блоке `with` уборка не выполнилась | в `@contextmanager` вокруг `yield` нет `try/finally` | оберните `yield` в `try/finally` |

## Контрольные вопросы

1. Во что превращается запись `@retry(attempts=4)` над функцией? Сколько функций при этом вызывается?
2. Зачем в декораторе `functools.wraps` и что сломается без него?
3. Почему `retry` не должен перехватывать `Exception`?
4. В каком порядке применяются декораторы `@timer` и `@retry`, записанные друг над другом?
5. Чем `functools.cache` отличается от `lru_cache(maxsize=1000)` и когда разница важна?
6. Почему конвейер генераторов расходует память независимо от размера файла?
7. Что вернёт `__exit__`, если нужно подавить исключение, и почему обычно так делать не следует?
8. Зачем в `fixed_seed` конструкция `try/finally`?

## Домашнее задание

Соберите модуль с инструментами, которые пригодятся в практике 1.2.4.

1. Создайте `pipeline_utils.py` с инструментами из урока: `timer`, `retry`, `fixed_seed` и генератор
   `read_batches(path, batch_size)`, который читает CSV и отдаёт списки словарей длиной не больше `batch_size`.
2. Добавьте два новых инструмента:
   - context manager `atomic_write(path)`: отдаёт файл для записи во временный путь рядом с `path`, при
     успешном выходе из блока подменяет им целевой файл (`os.replace`), при исключении удаляет временный файл;
   - декоратор `disk_cache(cache_dir)`: сохраняет результат функции в файл (модуль `pickle`), имя файла
     строит из имени функции и её аргументов, при повторном вызове читает результат с диска.
3. Напишите `demo_1_2_1.py`, который:
   - построчно создаёт CSV на 1 000 000 строк с колонками `user_id`, `event`, `amount`;
   - считает средний `amount` по каждому типу события, читая файл батчами по 50 000 строк;
   - печатает пик памяти (`tracemalloc`) для чтения батчами и для чтения файла целиком в список;
   - оборачивает расчёт в `timer` и `disk_cache` и показывает, что второй запуск скрипта берёт результат
     из кэша.

**Критерии приёмки:**

- [ ] `uv run ruff check .` завершается без ошибок;
- [ ] у декорированных функций сохраняются `__name__` и docstring;
- [ ] `retry` не повторяет вызов при `ValueError` и пробрасывает исключение после последней попытки;
- [ ] пик памяти при чтении батчами как минимум в 10 раз меньше, чем при чтении в список;
- [ ] второй запуск `demo_1_2_1.py` выполняет расчёт заметно быстрее первого за счёт `disk_cache`;
- [ ] после исключения внутри `atomic_write` целевой файл не изменился, временного файла нет.

## Дополнительные материалы

- Модуль functools: https://docs.python.org/3/library/functools.html
- Модуль itertools: https://docs.python.org/3/library/itertools.html
- Модуль contextlib: https://docs.python.org/3/library/contextlib.html
- Учебник Python, глава о классах (разделы об итераторах и генераторах): https://docs.python.org/3/tutorial/classes.html

**Следующий урок:** 1.2.2 — Асинхронное программирование для ML пайплайнов.
