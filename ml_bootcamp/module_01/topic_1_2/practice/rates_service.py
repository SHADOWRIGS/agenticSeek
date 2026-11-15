"""Имитация внешнего сервиса курсов валют. Этот файл в практике менять нельзя.

Сервис ведёт себя как настоящий сетевой API:
- каждый запрос занимает время (по умолчанию 5 мс, задаётся переменной окружения RATES_LATENCY_MS);
- часть запросов завершается временной ошибкой ConnectionError: для некоторых пар
  (валюта, день) первая попытка в процессе всегда неудачна, повторная проходит.

Пайплайн должен пользоваться только функциями get_rate и get_rate_async.
"""

import asyncio
import os
import time
import zlib
from collections import Counter
from datetime import date

BASE_RATES = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067}

_attempts: Counter[tuple[str, date]] = Counter()


def _latency_seconds() -> float:
    return float(os.environ.get("RATES_LATENCY_MS", "5")) / 1000


def _checksum(currency: str, day: date) -> int:
    return zlib.crc32(f"{currency}:{day.isoformat()}".encode())


def _respond(currency: str, day: date) -> float:
    if currency not in BASE_RATES:
        raise ValueError(f"неизвестная валюта: {currency!r}")
    checksum = _checksum(currency, day)
    key = (currency, day)
    _attempts[key] += 1
    if _attempts[key] == 1 and checksum % 5 == 0:
        raise ConnectionError(f"сервис курсов временно недоступен ({currency}, {day})")
    if currency == "USD":
        return 1.0
    # Курс «плавает» в пределах одного процента и зависит только от валюты и дня.
    wobble = (checksum % 2001 - 1000) / 100_000
    return round(BASE_RATES[currency] * (1 + wobble), 6)


def get_rate(currency: str, day: date) -> float:
    """Сколько долларов США стоит одна единица валюты в указанный день (блокирующий вызов)."""
    time.sleep(_latency_seconds())
    return _respond(currency, day)


async def get_rate_async(currency: str, day: date) -> float:
    """То же самое, но в виде корутины для asyncio."""
    await asyncio.sleep(_latency_seconds())
    return _respond(currency, day)
