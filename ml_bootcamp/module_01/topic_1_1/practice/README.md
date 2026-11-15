# Практика 1.1.4: материалы

Постановка задачи: [lesson_1_1_4.md](../lesson_1_1_4.md).

## check_workspace.py

Скрипт самопроверки рабочего окружения. Использует только стандартную библиотеку Python.

```bash
cp check_workspace.py ~/projects/ml-bootcamp/
cd ~/projects/ml-bootcamp
uv run python check_workspace.py                # все проверки
uv run python check_workspace.py --skip-docker  # без проверок Docker
```

Что проверяется:

| Раздел | Проверки |
|---|---|
| Python и пакеты | версия Python не ниже 3.12; запуск из `.venv` проекта; импорт numpy, pandas, scikit-learn, matplotlib |
| Файлы проекта | наличие обязательных файлов; правила в `.gitignore` и `.dockerignore`; настройка Ruff; цели Makefile |
| Git | репозиторий, имя и email автора, remote `origin`, не меньше трёх коммитов, нет `.venv` и больших файлов, установлен хук pre-commit |
| Инструменты | доступны `uv` и `make`; `ruff check` проходит |
| Docker | Docker установлен, демон доступен без `sudo`, `compose.yaml` корректен |

Статусы: `PASS` — проверка пройдена, `WARN` — стоит обратить внимание, `FAIL` — нужно исправить.
Код возврата равен 1, если есть хотя бы один `FAIL`, — скрипт можно использовать в CI.
