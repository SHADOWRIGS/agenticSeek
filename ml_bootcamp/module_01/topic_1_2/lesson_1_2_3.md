# Урок 1.2.3: Оптимизация кода: profiling, memory management

> **Модуль 1:** Foundations & Mindset Engineer · **Тема 1.2:** Python для ML инженера
> **Тип:** теория · **Время:** 120 минут + 90 минут на домашнее задание

## Цели урока

После урока вы сможете:

- измерять время выполнения кода и находить узкое место профилировщиком `cProfile`;
- измерять память: размер объекта, место выделения, пик потребления процесса;
- объяснить, как Python освобождает память и почему она иногда «не возвращается»;
- применять типовые приёмы ускорения и экономии памяти и проверять их эффект замером.

## Зачем это нужно

Две типичные ситуации. Первая: расчёт признаков идёт шесть часов, и никто не знает почему. Профилировщик
за минуту показывает, что 90% времени уходит на одну строку с поиском по списку. Вторая: обучение
на сервере убито системой с сообщением `Killed` — не хватило памяти, потому что датасет хранился списком
словарей и занимал в десять раз больше, чем файл на диске.

Интуиция в таких случаях подводит: медленным почти всегда оказывается не то место, на которое вы думали.
Поэтому оптимизация — это не «переписать красивее», а цикл из измерений.

Это обзорный урок: инструменты показаны на уровне, достаточном для практики 1.2.4. Углублённая работа
с `cProfile`, построчный профилировщик `line_profiler` и визуализация профилей — в уроке 6.3.1.

## 1. Порядок работы

1. **Убедитесь, что код работает правильно**, и сохраните эталонный результат. Быстрый неправильный код
   никому не нужен.
2. **Измерьте** общее время и пик памяти на данных реалистичного размера.
3. **Найдите узкое место** профилировщиком. Обычно несколько строк дают основную часть времени.
4. **Исправьте только его.** Сначала алгоритм и структуры данных, потом всё остальное.
5. **Измерьте снова** и сравните результат с эталонным. Если цель достигнута — остановитесь.

## 2. Замер времени

Для одного запуска достаточно `time.perf_counter()` или декоратора `timer` из урока 1.2.1. Для сравнения
двух коротких фрагментов нужен `timeit`: он выполняет код много раз и берёт лучший результат из серий,
отсекая шум системы. В ноутбуке то же самое делает `%timeit`.

```bash
uv run python -m timeit -s "data = list(range(2000))" "1999 in data"
uv run python -m timeit -s "data = set(range(2000))" "1999 in data"
```

```text
20000 loops, best of 5: 13 usec per loop
10000000 loops, best of 5: 21.9 nsec per loop
```

Поиск в списке просматривает элементы по одному, поиск во множестве — одно обращение к хеш-таблице.
На двух тысячах элементов разница около 600 раз, и она растёт вместе с размером данных.

## 3. cProfile: где теряется время

Сохраните пример в `slow_features.py`. Скрипт считает покупки по пользователям, пропуская заблокированных.

```python
import random


def load_events(n: int) -> list[tuple[int, str]]:
    rng = random.Random(0)
    return [(rng.randrange(20_000), rng.choice(["view", "cart", "purchase"])) for _ in range(n)]


def load_blacklist(n: int) -> list[int]:
    rng = random.Random(1)
    return [rng.randrange(20_000) for _ in range(n)]


def is_blocked(user_id: int, blacklist: list[int]) -> bool:
    return user_id in blacklist


def normalize(event: str) -> str:
    return event.strip().lower()


def build_features(events: list[tuple[int, str]], blacklist: list[int]) -> dict[int, int]:
    purchases: dict[int, int] = {}
    for user_id, event in events:
        if is_blocked(user_id, blacklist):
            continue
        if normalize(event) == "purchase":
            purchases[user_id] = purchases.get(user_id, 0) + 1
    return purchases


def main() -> None:
    events = load_events(200_000)
    blacklist = load_blacklist(2_000)
    features = build_features(events, blacklist)
    print(f"пользователей с покупками: {len(features)}")


if __name__ == "__main__":
    main()
```

Профилировщик запускается без изменения кода. Ключ `-s tottime` сортирует функции по собственному времени:

```bash
uv run python -m cProfile -s tottime slow_features.py
```

```text
пользователей с покупками: 17432
         3208772 function calls (3208744 primitive calls) in 3.791 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   200000    2.733    0.000    2.733    0.000 slow_features.py:14(is_blocked)
   402000    0.231    0.000    0.351    0.000 random.py:242(_randbelow_with_getrandbits)
        1    0.129    0.129    3.026    3.026 slow_features.py:22(build_features)
        1    0.121    0.121    0.753    0.753 slow_features.py:4(load_events)
   ...
```

| Колонка | Что означает |
|---|---|
| `ncalls` | сколько раз вызвана функция |
| `tottime` | время внутри самой функции, без вызванных из неё функций |
| `cumtime` | время функции вместе со всеми вложенными вызовами |
| `percall` | соответствующее время, делённое на число вызовов |

Читаем отчёт: `is_blocked` вызвана 200 тысяч раз и заняла 2.7 из 3.8 секунды. Сама функция — одна строка
`user_id in blacklist`, где `blacklist` — список. Исправление занимает одну строку: `load_blacklist` должна
возвращать множество (`{rng.randrange(20_000) for _ in range(n)}`, тип `set[int]`). После замены время работы
скрипта сократилось с 4.0 до 0.23 секунды, результат (`17432`) не изменился. Оптимизировать `normalize`
или генерацию данных после этого бессмысленно: это доли секунды.

Два замечания:

- под профилировщиком программа работает медленнее (исправленный скрипт — 0.97 с вместо 0.23 с), причём
  сильнее всего замедляются мелкие частые вызовы. Профиль показывает доли, а итоговый результат измеряют
  без профилировщика;
- профилировать можно часть программы, а отчёт сохранять в файл для последующего разбора:

```python
import cProfile
import pstats

from slow_features import main

with cProfile.Profile() as profiler:
    main()

stats = pstats.Stats(profiler).strip_dirs()
stats.sort_stats(pstats.SortKey.TIME).print_stats(3)  # три самые «дорогие» функции
profiler.dump_stats("features.prof")  # файл можно открыть позже: pstats.Stats("features.prof")
```

## 4. Сколько весит объект

`sys.getsizeof` возвращает размер самого объекта в байтах — без объектов, на которые он ссылается.
Примеры разделов 4–6 выполняйте в одном файле: следующие блоки используют импорты предыдущих.

```python
import sys

import numpy as np

n = 1_000_000
as_list = list(range(n))
as_numpy = np.arange(n, dtype=np.int64)

print(f"list, только ссылки:  {sys.getsizeof(as_list) / 1e6:5.1f} МБ")
print(f"list вместе с числами: {(sys.getsizeof(as_list) + sum(map(sys.getsizeof, as_list))) / 1e6:5.1f} МБ")
print(f"numpy int64:           {as_numpy.nbytes / 1e6:5.1f} МБ")
print(f"numpy int32:           {as_numpy.astype(np.int32).nbytes / 1e6:5.1f} МБ")
print(f"генератор:             {sys.getsizeof(i for i in range(n))} байт")

row = {"user_id": 1, "event": "purchase", "amount": 9.99}
print(sys.getsizeof(1), sys.getsizeof(9.99), sys.getsizeof("purchase"), sys.getsizeof(row))
```

```text
list, только ссылки:    8.0 МБ
list вместе с числами:  36.0 МБ
numpy int64:             8.0 МБ
numpy int32:             4.0 МБ
генератор:             192 байт
28 24 49 184
```

Выводы:

- в Python всё — объекты с заголовком: целое число занимает 28 байт, а не 8. Список хранит не числа,
  а ссылки на них (по 8 байт), поэтому миллион чисел в списке — 36 МБ, а в массиве NumPy — 8 МБ;
- `getsizeof` для контейнера показывает только «верхний слой». Словарь из трёх полей — 184 байта без учёта
  значений. Для честной оценки нужен `tracemalloc`;
- тип данных имеет значение: `int32` вдвое компактнее `int64`. К типам вернёмся в уроке 2.1.2.

## 5. tracemalloc: кто выделил память

`tracemalloc` отслеживает все выделения памяти интерпретатором: сколько занято сейчас, каким был пик
и какая строка кода выделила больше всего.

```python
import tracemalloc


def load_rows(n: int) -> list[dict[str, float | int | str]]:
    return [{"user_id": i, "event": "purchase", "amount": i * 0.5} for i in range(n)]


def total_amount(rows: list[dict[str, float | int | str]]) -> float:
    amounts = [row["amount"] for row in rows]  # лишняя копия колонки
    return sum(amounts)


tracemalloc.start()
rows = load_rows(300_000)
total = total_amount(rows)
current, peak = tracemalloc.get_traced_memory()
snapshot = tracemalloc.take_snapshot()
tracemalloc.stop()

print(f"занято сейчас {current / 1e6:.1f} МБ, пик {peak / 1e6:.1f} МБ")
top = snapshot.statistics("lineno")[0]
print(f"{top.size / 1e6:.1f} МБ, объектов: {top.count}, строка: {top.traceback.format()[-1].strip()[:42]}")
```

```text
занято сейчас 74.6 МБ, пик 77.2 МБ
74.6 МБ, объектов: 1199702, строка: return [{"user_id": i, "event": "purchase"
```

Триста тысяч маленьких словарей занимают 75 МБ — около 250 байт на строку из трёх полей. Пик выше текущего
значения из-за временного списка `amounts`: его можно убрать, передав в `sum` генераторное выражение.
`tracemalloc` замедляет программу, поэтому включают его только на время поиска проблемы.

## 6. Компактные объекты

Сравним способы хранить 500 тысяч записей с двумя полями.

```python
from dataclasses import dataclass


@dataclass
class Event:
    user_id: int
    amount: float


@dataclass(slots=True)
class EventSlots:
    user_id: int
    amount: float


def measure(factory) -> float:
    tracemalloc.start()
    events = [factory(i, i * 0.5) for i in range(500_000)]
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del events
    return current / 1e6


for name, factory in [
    ("dict", lambda u, a: {"user_id": u, "amount": a}),
    ("обычный класс", Event),
    ("класс со слотами", EventSlots),
    ("tuple", lambda u, a: (u, a)),
]:
    print(f"{name:18} {measure(factory):6.1f} МБ")

ids = np.arange(500_000, dtype=np.int64)
amounts = ids * 0.5
print(f"{'два массива NumPy':18} {(ids.nbytes + amounts.nbytes) / 1e6:6.1f} МБ")
```

```text
dict                124.2 МБ
обычный класс        72.2 МБ
класс со слотами     56.2 МБ
tuple                60.2 МБ
два массива NumPy     8.0 МБ
```

- список словарей — самый дорогой способ хранить табличные данные;
- `slots=True` добавляет классу атрибут `__slots__`: набор полей фиксируется, и объекту не нужен собственный
  словарь атрибутов. Плата: добавить объекту новое поле нельзя (`AttributeError`). В обычном классе
  то же самое делает строка `__slots__ = ("user_id", "amount")`;
- хранение «по колонкам» в массивах NumPy компактнее на порядок. Поэтому pandas и NumPy — основа работы
  с данными в модуле 2, а объекты Python оставляют для небольших структур.

## 7. Как Python освобождает память

У каждого объекта есть счётчик ссылок. Когда он падает до нуля, память освобождается сразу. `del x` удаляет
не объект, а имя: объект исчезнет, только если это была последняя ссылка. Объекты, которые ссылаются друг
на друга по кругу, счётчик освободить не может — их периодически находит сборщик мусора, модуль `gc`.

```python
import gc
import resource
import sys

import numpy as np

features = np.zeros((2_000, 1_000))  # 16 МБ
alias = features
print(sys.getrefcount(features) - 1)  # минус временная ссылка из самого вызова getrefcount
del alias
print(sys.getrefcount(features) - 1)


class Node:
    def __init__(self) -> None:
        self.payload = bytearray(10_000_000)
        self.other: Node | None = None


gc.collect()
gc.disable()  # отключаем автоматическую сборку, чтобы увидеть проблему
a, b = Node(), Node()
a.other, b.other = b, a  # цикл: a -> b -> a
del a, b  # имён нет, но объекты держат друг друга
print("объектов Node в памяти:", sum(isinstance(obj, Node) for obj in gc.get_objects()))
gc.collect()
print("после gc.collect():", sum(isinstance(obj, Node) for obj in gc.get_objects()))
gc.enable()

peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: килобайты, macOS: байты
print(f"пик памяти процесса: {peak_kb / 1024:.0f} МБ")
```

```text
2
1
объектов Node в памяти: 2
после gc.collect(): 0
пик памяти процесса: 44 МБ
```

Что из этого следует на практике:

- память большого объекта освобождается, когда исчезла **последняя** ссылка. Частая причина «утечки» —
  забытая ссылка: глобальная переменная, список с историей, кэш без ограничения;
- в Jupyter результаты ячеек сохраняются в служебных переменных `_` и `Out`. Вывели DataFrame в конце
  ячейки — на него появилась ещё одна ссылка, и `del df` память не освободит;
- сборщик мусора запускается автоматически; вызывать `gc.collect()` вручную имеет смысл после удаления
  больших структур с циклическими ссылками;
- `resource.getrusage` показывает пик памяти всего процесса с точки зрения операционной системы, включая
  память NumPy и других библиотек. Именно эта величина определяет, убьёт ли система вашу задачу. Модуль
  `resource` доступен в Linux и macOS.

## 8. Типовые приёмы

| Проблема | Приём | Где подробнее |
|---|---|---|
| Поиск `x in list` в цикле | `set` или `dict` | этот урок |
| Повторный расчёт одного и того же | кэширование: `functools.lru_cache`, кэш на диске | урок 1.2.1 |
| Весь файл читается в память | генераторы, обработка батчами | урок 1.2.1 |
| Программа ждёт сеть или диск | `asyncio`, потоки | урок 1.2.2 |
| Вычисления на чистом Python занимают одно ядро | процессы | урок 1.2.2 |
| Цикл по числам на Python | векторизация в NumPy и pandas | модуль 2 |
| Список словарей или объектов занимает гигабайты | хранение по колонкам, компактные типы, `__slots__` | урок 2.1.2 |
| Python исчерпан, а нужно ещё быстрее | Numba, Cython, C++ | модуль 6 |

Порядок в таблице не случаен: сначала алгоритм и структура данных, затем устранение лишней работы и только
потом параллелизм и низкоуровневые средства. Параллельный запуск квадратичного алгоритма на восьми ядрах
ускорит его в восемь раз, а замена списка на множество — в сотни.

## Типичные ошибки

| Симптом | Причина | Решение |
|---|---|---|
| Функцию оптимизировали, а программа не ускорилась | оптимизировали не узкое место | сначала профиль, потом правка самой «дорогой» функции |
| После оптимизации изменился результат | не было эталона для сравнения | сохраните результат до правок и сравнивайте после каждой |
| Замеры одного и того же кода заметно различаются | шум системы, один запуск | `timeit` с несколькими сериями; сравнивайте лучшие значения |
| Под `cProfile` соотношение времени не такое, как без него | накладные расходы на частые мелкие вызовы | итог измеряйте без профилировщика |
| `sys.getsizeof` показывает 8 МБ, а процесс занимает 40 МБ | не учтены вложенные объекты | измеряйте `tracemalloc` или пик процесса |
| После `del df` память не уменьшилась | на объект остались другие ссылки, в ноутбуке — `_` и `Out` | уберите ссылки; в Jupyter очистите историю вывода `%reset -f out` |
| `AttributeError` при добавлении атрибута объекту | класс объявлен со `__slots__` | добавьте атрибут в `__slots__` |
| Процесс завершился с сообщением `Killed` | закончилась оперативная память | измерьте пик, перейдите на батчи и компактные типы |

## Контрольные вопросы

1. Почему оптимизацию начинают с измерения, а не с переписывания «очевидно медленного» кода?
2. Чем `tottime` отличается от `cumtime` в отчёте `cProfile`?
3. Почему итоговое время нельзя измерять под профилировщиком?
4. Что именно возвращает `sys.getsizeof` для списка из миллиона чисел?
5. Почему миллион целых чисел в списке занимает в несколько раз больше памяти, чем в массиве NumPy?
6. Что делает `__slots__` и какое ограничение он накладывает?
7. В каком случае `del x` не освобождает память объекта?
8. Чем показания `tracemalloc` отличаются от `ru_maxrss`?

## Домашнее задание

Ускорьте скрипт и сократите потребление памяти, действуя по порядку из раздела 1.

Сохраните код в `slow_stats.py`. Он работает правильно, но медленно.

```python
import random


def make_events(n: int) -> list[dict]:
    rng = random.Random(0)
    return [{"user_id": rng.randrange(3_000), "amount": round(rng.uniform(1, 500), 2)} for _ in range(n)]


def unique_users(events: list[dict]) -> list[int]:
    users = []
    for event in events:
        if event["user_id"] not in users:
            users.append(event["user_id"])
    return users


def user_totals(events: list[dict], users: list[int]) -> dict[int, float]:
    totals = {}
    for user in users:
        totals[user] = sum(event["amount"] for event in events if event["user_id"] == user)
    return totals


def median_amount(events: list[dict]) -> float:
    amounts = []
    for event in events:
        amounts = amounts + [event["amount"]]
    return sorted(amounts)[len(amounts) // 2]


def main() -> None:
    events = make_events(40_000)
    users = unique_users(events)
    totals = user_totals(events, users)
    best = max(totals, key=totals.get)
    print(f"пользователей: {len(users)}, лучший: {best} ({totals[best]:.2f}), медиана: {median_amount(events)}")


if __name__ == "__main__":
    main()
```

1. Запустите скрипт, запишите его вывод (это эталон) и время работы.
2. Снимите профиль `cProfile`. Запишите три самые «дорогие» функции и объясните причину для каждой.
3. Исправляйте по одной функции, начиная с самой дорогой. После каждой правки повторяйте профиль
   и проверяйте, что вывод совпадает с эталоном.
4. Измерьте `tracemalloc` пик памяти для `make_events(400_000)`. Предложите представление тех же данных,
   которое занимает как минимум в пять раз меньше, и подтвердите это замером.

**Критерии приёмки:**

- [ ] вывод оптимизированного скрипта совпадает с эталоном (сумма лучшего пользователя — с точностью
      до копеек);
- [ ] скрипт работает минимум в 30 раз быстрее исходного (замер без профилировщика);
- [ ] в отчёте есть профили до и после каждой правки и таблица «функция — было — стало — что изменили»;
- [ ] пик памяти для 400 000 событий сокращён минимум в пять раз, оба замера приведены в отчёте;
- [ ] `uv run ruff check .` завершается без ошибок.

## Дополнительные материалы

- Профилировщики Python: https://docs.python.org/3/library/profile.html
- Модуль timeit: https://docs.python.org/3/library/timeit.html
- Модуль tracemalloc: https://docs.python.org/3/library/tracemalloc.html
- Модуль gc: https://docs.python.org/3/library/gc.html

**Следующий урок:** 1.2.4 — ПРАКТИКА: Написание эффективного ETL пайплайна.
