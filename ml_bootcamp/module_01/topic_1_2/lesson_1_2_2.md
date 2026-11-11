# Урок 1.2.2: Асинхронное программирование для ML пайплайнов

> **Модуль 1:** Foundations & Mindset Engineer · **Тема 1.2:** Python для ML инженера
> **Тип:** теория · **Время:** 120 минут + 90 минут на домашнее задание

## Цели урока

После урока вы сможете:

- отличить I/O-bound задачу от CPU-bound и выбрать для неё `asyncio`, потоки или процессы;
- объяснить, что такое GIL и почему потоки не ускоряют вычисления на чистом Python;
- писать корутины, запускать их одновременно, ограничивать нагрузку, ставить таймауты и обрабатывать ошибки;
- вынести блокирующий и вычислительный код из цикла событий в потоки и процессы.

## Зачем это нужно

Пайплайн подготовки данных большую часть времени ждёт: скачивает тысячи файлов из объектного хранилища,
обращается к API разметки или языковой модели, читает признаки из базы. Двадцать запросов по 200 мс,
выполненные по очереди, занимают 4 секунды. Те же запросы, отправленные одновременно, — 0.2 секунды.
На миллионе запросов это разница между «посчиталось за обед» и «считалось двое суток».

Обратная ситуация: токенизация или парсинг на чистом Python загружает одно ядро на 100%, а остальные
простаивают. Здесь одновременная отправка не поможет — нужны несколько процессов.

Главный навык урока — не синтаксис `async/await`, а умение определить, чего именно ждёт программа,
и выбрать подходящий инструмент.

## 1. Два вида медленного кода

| | I/O-bound | CPU-bound |
|---|---|---|
| На что уходит время | ожидание сети, диска, базы данных | вычисления на процессоре |
| Примеры в ML | скачивание датасета по частям, запросы к API модели, чтение из feature store | токенизация, парсинг логов, аугментации на чистом Python |
| Загрузка процессора | низкая: программа ждёт ответа | одно ядро занято на 100% |
| Что помогает | `asyncio`, потоки | процессы, векторизация (NumPy), компилируемый код |

Проверить, к какому виду относится ваша задача, можно командой `htop` (урок 1.4.2): если процесс работает
медленно, а процессор почти не загружен — это ожидание ввода-вывода.

## 2. GIL и три инструмента

В CPython, стандартной реализации Python, действует глобальная блокировка интерпретатора — GIL
(Global Interpreter Lock): в каждый момент времени байт-код Python выполняет только один поток. Следствия:

- потоки не ускоряют вычисления на чистом Python: они выполняются по очереди на одном ядре;
- на время ожидания ввода-вывода (сокет, файл, `time.sleep`) поток отпускает GIL, поэтому для I/O-bound
  задач потоки работают хорошо;
- многие библиотеки на C отпускают GIL на время тяжёлых операций — так делает, например, NumPy;
- у каждого процесса свой интерпретатор и свой GIL, поэтому процессы дают настоящий параллелизм.

Начиная с Python 3.13 существует экспериментальная сборка интерпретатора без GIL. В курсе используется
стандартный Python 3.12, и всё сказанное выше к нему относится полностью.

| Инструмент | Как переключаются задачи | Для чего | Цена |
|---|---|---|---|
| `asyncio` | один поток; задача сама отдаёт управление на `await` | тысячи одновременных сетевых запросов | весь код ввода-вывода должен быть асинхронным |
| Потоки: `ThreadPoolExecutor`, `asyncio.to_thread` | переключает операционная система | I/O через обычные блокирующие библиотеки | вычисления не ускоряются; общие данные нужно защищать |
| Процессы: `ProcessPoolExecutor` | независимые интерпретаторы | CPU-bound задачи | запуск процессов, копирование данных между ними |

## 3. Корутины и цикл событий

Функция, объявленная через `async def`, — корутина. Её вызов ничего не выполняет, а возвращает объект,
который нужно передать циклу событий. `await` означает: «здесь я жду, пока можно выполнять другие задачи».
Сеть имитируем через `asyncio.sleep` — для цикла событий это такое же ожидание, как ответ сервера.
Примеры разделов 3–5 и 7 выполняйте в одном файле: им нужны импорты из первого блока.

```python
import asyncio
import time


async def fetch_features(user_id: int) -> dict[str, float]:
    await asyncio.sleep(0.2)  # имитация сетевого запроса к feature store
    return {"user_id": user_id, "score": user_id / 10}


async def main() -> None:
    user_ids = range(20)

    start = time.perf_counter()
    rows = [await fetch_features(uid) for uid in user_ids]
    print(f"по очереди:   {len(rows)} ответов за {time.perf_counter() - start:.2f} с")

    start = time.perf_counter()
    rows = await asyncio.gather(*(fetch_features(uid) for uid in user_ids))
    print(f"одновременно: {len(rows)} ответов за {time.perf_counter() - start:.2f} с")
    print(rows[3])


asyncio.run(main())
```

```text
по очереди:   20 ответов за 4.01 с
одновременно: 20 ответов за 0.20 с
{'user_id': 3, 'score': 0.3}
```

- `asyncio.run(main())` создаёт цикл событий, выполняет корутину до конца и закрывает цикл. В программе это
  единственная точка входа в асинхронный код;
- `await` в цикле `for` ждёт каждый запрос до конца, поэтому первый вариант ничем не лучше синхронного кода;
- `asyncio.gather` запускает все корутины сразу и возвращает результаты в порядке аргументов, а не в порядке
  завершения;
- поток один: пока одна корутина ждёт на `await`, цикл событий выполняет другие. Ускоряется ожидание,
  а не вычисления.

В Jupyter цикл событий уже запущен, поэтому `asyncio.run()` там завершится ошибкой. В ячейке ноутбука
пишите просто `await main()`.

## 4. Ограничение нагрузки, ошибки и таймауты

Отправить сервису десять тысяч запросов одновременно — верный способ получить отказ или блокировку. В реальном
пайплайне нужны три вещи: ограничить число одновременных запросов, пережить ошибку отдельного запроса
и не ждать ответа вечно.

```python
class RateLimitError(Exception):
    pass


async def call_model_api(prompt_id: int, delay: float = 0.2) -> str:
    """Имитация обращения к API модели: каждый седьмой запрос получает отказ."""
    await asyncio.sleep(delay)
    if prompt_id % 7 == 3:
        raise RateLimitError(f"запрос {prompt_id} отклонён")
    return f"ответ {prompt_id}"


async def limited(prompt_id: int, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:  # внутри блока одновременно находится не больше пяти корутин
        return await call_model_api(prompt_id)


async def main() -> None:
    semaphore = asyncio.Semaphore(5)
    start = time.perf_counter()
    results = await asyncio.gather(*(limited(i, semaphore) for i in range(30)), return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    print(f"ответов: {len(results) - len(errors)}, ошибок: {len(errors)}, время: {time.perf_counter() - start:.2f} с")
    print(repr(errors[0]))

    try:
        async with asyncio.timeout(0.1):
            await call_model_api(1, delay=5)
    except TimeoutError:
        print("запрос отменён по таймауту")

    try:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(call_model_api(i, delay=0.1 if i == 3 else 1.0)) for i in range(5)]
    except* RateLimitError as group:
        print(f"TaskGroup: {group.exceptions[0]}; отменено задач: {sum(t.cancelled() for t in tasks)}")


asyncio.run(main())
```

```text
ответов: 26, ошибок: 4, время: 1.20 с
RateLimitError('запрос 3 отклонён')
запрос отменён по таймауту
TaskGroup: запрос 3 отклонён; отменено задач: 4
```

- **`asyncio.Semaphore(5)`** пропускает в блок не больше пяти корутин: 30 запросов прошли шестью волнами
  по 0.2 с. Лимит подбирают под ограничения сервиса;
- **`return_exceptions=True`** возвращает исключения как обычные результаты. Без этого флага первая же ошибка
  вылетит из `gather`, и вы потеряете ответы остальных запросов. Неудачные запросы затем повторяют — приём
  тот же, что в декораторе `retry` из прошлого урока, только пауза делается через `await asyncio.sleep()`;
- **`asyncio.timeout`** отменяет всё, что выполняется внутри блока дольше заданного времени;
- **`asyncio.TaskGroup`** — противоположная стратегия «всё или ничего»: при ошибке в одной задаче остальные
  отменяются, а ошибки приходят группой (их ловит `except*`). Подходит, когда частичный результат бесполезен.

## 5. Блокирующий код в асинхронной программе

Цикл событий переключает задачи только на `await`. Обычная блокирующая функция внутри корутины — `time.sleep`,
`requests.get`, `pandas.read_csv` — останавливает весь цикл: остальные корутины ждут вместе с ней.

```python
def load_from_disk(path: str) -> str:
    time.sleep(0.5)  # обычная блокирующая функция: чтение файла, запрос через requests и т.п.
    return f"данные из {path}"


async def bad(path: str) -> str:
    return load_from_disk(path)  # цикл событий стоит все 0.5 с


async def good(path: str) -> str:
    return await asyncio.to_thread(load_from_disk, path)  # функция выполняется в потоке, цикл свободен


async def main() -> None:
    paths = [f"part-{i}.parquet" for i in range(4)]
    for worker in (bad, good):
        start = time.perf_counter()
        await asyncio.gather(*(worker(p) for p in paths))
        print(f"{worker.__name__}: {time.perf_counter() - start:.2f} с")


asyncio.run(main())
```

```text
bad: 2.00 с
good: 0.50 с
```

`asyncio.to_thread` — мост к библиотекам, у которых нет асинхронной версии. Для CPU-bound функции он
не поможет: поток упрётся в GIL.

## 6. CPU-bound: потоки против процессов

Сохраните пример в отдельный файл. Функция `count_tokens` считает на чистом Python и ничего не ждёт.

```python
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def count_tokens(n_docs: int) -> int:
    """CPU-bound задача на чистом Python: «токенизация» n_docs документов."""
    total = 0
    for i in range(n_docs):
        total += len(f"document number {i} with some text".split())
    return total


def main() -> None:
    jobs = [1_500_000] * 4

    start = time.perf_counter()
    results = [count_tokens(n) for n in jobs]
    print(f"последовательно: {time.perf_counter() - start:.2f} с")

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(count_tokens, jobs)) == results
    print(f"4 потока:        {time.perf_counter() - start:.2f} с")

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(count_tokens, jobs)) == results
    print(f"4 процесса:      {time.perf_counter() - start:.2f} с")


if __name__ == "__main__":
    main()
```

```text
последовательно: 1.84 с
4 потока:        1.99 с
4 процесса:      0.51 с
```

Замер сделан на машине с четырьмя ядрами. Потоки из-за GIL не дали ничего, процессы ускорили расчёт
в 3.6 раза. У процессов есть цена, о которой нужно помнить:

- аргументы и результаты передаются между процессами через сериализацию (`pickle`). Передавать гигабайтный
  DataFrame в каждую задачу дороже, чем передать путь к файлу и прочитать его внутри процесса;
- функция и её аргументы должны быть сериализуемыми: функция объявлена на уровне модуля, не `lambda`;
- код запуска обязан находиться под `if __name__ == "__main__":` — на macOS и Windows дочерний процесс заново
  импортирует ваш модуль;
- задачи должны быть крупными. Если одна задача занимает микросекунды, накладные расходы съедят выигрыш.

Вызвать пул процессов из асинхронного кода можно так:
`await asyncio.get_running_loop().run_in_executor(pool, count_tokens, 1_500_000)`.

## 7. Конвейер на очереди

Типичная схема пайплайна: несколько загрузчиков скачивают файлы, один потребитель пишет результат в базу или
подаёт в модель. Связывает их `asyncio.Queue`. Ограничение `maxsize` важно: если потребитель не успевает,
загрузчики приостанавливаются на `put`, а не складывают в память весь датасет.

```python
async def download(shard_id: int) -> list[int]:
    await asyncio.sleep(0.1)  # имитация скачивания одного файла из объектного хранилища
    return list(range(shard_id * 1000, (shard_id + 1) * 1000))


async def producer(shard_ids: asyncio.Queue[int], batches: asyncio.Queue[list[int] | None]) -> None:
    while not shard_ids.empty():
        shard_id = shard_ids.get_nowait()
        await batches.put(await download(shard_id))  # если очередь заполнена, загрузчик ждёт потребителя


async def consumer(batches: asyncio.Queue[list[int] | None]) -> int:
    total = 0
    while (batch := await batches.get()) is not None:
        total += sum(batch)  # здесь могла бы быть запись в базу или инференс модели
    return total


async def main() -> None:
    shard_ids: asyncio.Queue[int] = asyncio.Queue()
    for shard_id in range(40):
        shard_ids.put_nowait(shard_id)
    batches: asyncio.Queue[list[int] | None] = asyncio.Queue(maxsize=8)

    start = time.perf_counter()
    consumer_task = asyncio.create_task(consumer(batches))
    async with asyncio.TaskGroup() as tg:
        for _ in range(8):
            tg.create_task(producer(shard_ids, batches))
    await batches.put(None)  # загрузчики закончили: сообщаем потребителю, что данных больше не будет
    print(f"обработано 40 файлов, сумма {await consumer_task}, время {time.perf_counter() - start:.2f} с")


asyncio.run(main())
```

```text
обработано 40 файлов, сумма 799980000, время 0.50 с
```

Сорок файлов по 0.1 с восемь загрузчиков скачали за 0.5 с, и в памяти одновременно находилось не больше
восьми необработанных батчей.

## 8. Настоящие HTTP-запросы

Для настоящих запросов нужна асинхронная библиотека, например `httpx` (`uv add httpx`) или `aiohttp`.
Библиотека `requests` блокирующая и в корутинах без `asyncio.to_thread` не годится. Структура кода та же:

```python
import asyncio

import httpx


async def fetch_json(client: httpx.AsyncClient, url: str, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


async def fetch_all(urls: list[str], max_concurrency: int = 10) -> list[dict]:
    semaphore = asyncio.Semaphore(max_concurrency)
    async with httpx.AsyncClient(timeout=10.0) as client:  # один клиент на все запросы
        return await asyncio.gather(*(fetch_json(client, url, semaphore) for url in urls))
```

Клиент создаётся один раз и переиспользует соединения. Создавать клиента на каждый запрос — частая причина
медленной работы.

## Типичные ошибки

| Симптом | Причина | Решение |
|---|---|---|
| `RuntimeWarning: coroutine '...' was never awaited`, результата нет | корутину вызвали без `await` | добавьте `await` или передайте корутину в `gather` |
| Асинхронный код работает не быстрее обычного | `await` в цикле по одному запросу либо блокирующий вызов внутри корутины | `gather` или `TaskGroup`; блокирующие функции — через `asyncio.to_thread` |
| `RuntimeError: asyncio.run() cannot be called from a running event loop` | код запущен в Jupyter, где цикл событий уже работает | в ноутбуке пишите `await main()` |
| Сервис отвечает ошибками или блокирует клиента | `gather` отправил все запросы одновременно | ограничьте параллелизм через `asyncio.Semaphore` |
| Одна ошибка — и пропали результаты всех запросов | исключение вылетело из `gather` | `return_exceptions=True` и разбор результатов |
| Потоки не ускорили обработку | задача CPU-bound, мешает GIL | `ProcessPoolExecutor` или векторизация |
| `Can't pickle local object` при работе с пулом процессов | в пул передана `lambda` или вложенная функция | объявите функцию на уровне модуля |
| Процессы работают медленнее обычного цикла | задачи мелкие или между процессами копируется много данных | укрупните задачи, передавайте пути к файлам, а не данные |

## Контрольные вопросы

1. Как по поведению программы понять, I/O-bound она или CPU-bound?
2. Что такое GIL? Почему он не мешает потокам ускорять сетевые запросы?
3. Чем `await fetch(x)` в цикле отличается от `asyncio.gather(*(fetch(x) for x in xs))`?
4. Что произойдёт с остальными корутинами, если внутри одной вызвать `time.sleep(5)`?
5. Зачем нужен `asyncio.Semaphore` и как выбрать его значение?
6. Чем поведение `gather(..., return_exceptions=True)` при ошибке отличается от поведения `TaskGroup`?
7. Почему передавать в `ProcessPoolExecutor` большой DataFrame в каждую задачу — плохая идея?
8. Зачем очереди в конвейере параметр `maxsize`?

## Домашнее задание

Напишите асинхронный загрузчик данных и сравните три способа параллельной работы.

1. В файле `async_loader.py` создайте корутину `fetch_shard(shard_id)`: она ждёт случайное время от 0.05
   до 0.3 с (`asyncio.sleep`), в 10% случаев бросает `ConnectionError`, иначе возвращает список чисел.
2. Напишите `load_all(shard_ids, max_concurrency, attempts)`, которая:
   - держит одновременно не больше `max_concurrency` запросов;
   - повторяет неудачный запрос до `attempts` раз с растущей паузой;
   - прерывает запрос, который длится дольше 0.25 с, и считает это неудачной попыткой;
   - возвращает результаты и список `shard_id`, которые так и не удалось загрузить.
3. Загрузите 200 шардов по очереди и с `max_concurrency` 5, 20 и 100. Запишите время в таблицу.
4. В файле `cpu_compare.py` обработайте восемь батчей текста CPU-bound функцией на чистом Python тремя
   способами: в цикле, в `ThreadPoolExecutor` и в `ProcessPoolExecutor`. Запишите время в таблицу.

**Критерии приёмки:**

- [ ] `uv run ruff check .` завершается без ошибок;
- [ ] при `max_concurrency=20` загрузка 200 шардов идёт минимум в 10 раз быстрее, чем по очереди;
- [ ] число одновременных запросов не превышает `max_concurrency` — докажите это счётчиком в коде;
- [ ] ошибки отдельных шардов не прерывают загрузку: функция возвращает список неудачных `shard_id`;
- [ ] в отчёте есть обе таблицы и объяснение в 3–5 предложениях, почему потоки не ускорили `cpu_compare.py`.

## Дополнительные материалы

- Модуль asyncio: https://docs.python.org/3/library/asyncio.html
- Модуль concurrent.futures: https://docs.python.org/3/library/concurrent.futures.html
- Глоссарий Python, статья «global interpreter lock»: https://docs.python.org/3/glossary.html

**Следующий урок:** 1.2.3 — Оптимизация кода: profiling, memory management.
