# Урок 1.1.4: ПРАКТИКА: Создание и настройка своего ML workspace

> **Модуль 1:** Foundations & Mindset Engineer · **Тема 1.1:** Установка профессионального окружения
> **Тип:** практика · **Время:** 3–4 часа

## Задача

Собрать из результатов уроков 1.1.1–1.1.3 единое рабочее окружение, в котором новый человек проходит путь
от `git clone` до обученной модели тремя командами:

```bash
git clone git@github.com:<ваш-логин>/ml-bootcamp.git && cd ml-bootcamp
make setup
make train
```

Этот репозиторий — фундамент всего курса: в нём будут жить все практические проекты и домашние задания.
Потратить несколько часов на его настройку сейчас дешевле, чем чинить окружение посреди модуля про нейросети.

## Что должно получиться

```
ml-bootcamp/
├── .vscode/settings.json        # настройки редактора для всей команды
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml      # проверки перед коммитом
├── .python-version
├── Dockerfile                   # этапы base, dev, runtime
├── Makefile                     # единая точка входа
├── README.md
├── check_workspace.py           # самопроверка (из папки practice этого урока)
├── compose.yaml
├── pyproject.toml
├── uv.lock
├── train.py
├── data/.gitkeep
├── homework/
├── notebooks/00_environment_check.ipynb
├── projects/
├── src/
└── tests/test_smoke.py
```

## Требования

1. **Воспроизводимость.** Окружение полностью описано файлами в репозитории. После `git clone` и `make setup`
   проходят `make lint`, `make test` и `make train`.
2. **Чистота репозитория.** В Git нет `.venv`, данных, моделей, секретов, выводов ноутбуков и файлов больше 1 МБ.
3. **Автоматические проверки.** pre-commit запускает Ruff, очистку ноутбуков и проверку больших файлов.
4. **Контейнер.** Обучение запускается в Docker, результат сохраняется на хосте и принадлежит вашему пользователю.
5. **Единая точка входа.** Все типовые действия выполняются через `make <цель>`.
6. **Документация.** По `README.md` проект может запустить человек, который видит его впервые.

## Пошаговый план

### Шаг 1. Ревизия

Убедитесь, что домашние задания уроков 1.1.1–1.1.3 выполнены и их результаты лежат в одном репозитории.
Скопируйте в корень проекта файл [`practice/check_workspace.py`](practice/check_workspace.py) и запустите:

```bash
uv run python check_workspace.py
```

Скрипт напечатает список проверок со статусами `PASS`, `WARN` и `FAIL`. Строки `FAIL` — ваш план работ.
Если Docker на машине установить нельзя, добавьте флаг `--skip-docker` и отметьте это в `README.md`.

### Шаг 2. Тесты

```bash
uv add --dev pytest
```

Файл `tests/test_smoke.py` — дымовые тесты: они не проверяют качество модели, а отвечают на вопрос
«окружение вообще живо?».

```python
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_core_packages_import():
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401


def test_train_script_saves_model(tmp_path):
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "train.py")], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert "accuracy" in result.stdout
    assert (tmp_path / "models" / "wine_rf.joblib").is_file()
```

Второй тест запускает обучение во временной папке (`tmp_path` — встроенная фикстура pytest), поэтому не
оставляет после себя файлов в проекте.

### Шаг 3. Makefile

`make` есть на любой Linux-машине и на macOS; в WSL он ставится вместе с `build-essential`. Makefile служит
исполняемой документацией: открыв его, сразу видно, что в проекте можно делать.

```makefile
.PHONY: setup lint format test train notebook docker-build docker-train check

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -q

train:
	uv run python train.py

notebook:
	uv run jupyter lab

docker-build:
	docker compose build

docker-train:
	docker compose run --rm --user "$$(id -u):$$(id -g)" train

check:
	uv run python check_workspace.py
```

На что обратить внимание:

- отступы перед командами — **символы табуляции**, не пробелы; иначе `make` выдаст `missing separator`;
- `.PHONY` сообщает, что цели — это команды, а не файлы; без этого `make test` ничего не сделает, если
  в проекте появится файл или папка с именем `test`;
- `$$` в Makefile означает обычный `$` для оболочки.

### Шаг 4. README

Минимальный состав:

1. Что это за репозиторий и чей он.
2. Требования: `uv`, Git, Docker, `make`.
3. Быстрый старт — три команды из начала урока.
4. Таблица целей `make` с описанием.
5. Структура папок.
6. Запуск в Docker с результатами измерений из домашнего задания 1.1.3.

### Шаг 5. Проверка «с нуля»

Главный шаг, который чаще всего пропускают. Склонируйте репозиторий в другую папку и пройдите путь нового
участника:

```bash
cd /tmp
git clone git@github.com:<ваш-логин>/ml-bootcamp.git ml-bootcamp-check
cd ml-bootcamp-check
make setup && make lint && make test && make train
make docker-build && make docker-train
make check
```

Всё, что пришлось доделывать руками, — недостаток окружения: исправьте его в основном репозитории и повторите
проверку. Типичные находки: файл не закоммичен, потому что попал под правило `.gitignore`; путь к данным
указан абсолютный; шаг установки выполнялся вручную и нигде не записан.

### Шаг 6. Сдача

Откройте pull request `workspace` в ветку `main`. В описании приведите вывод `make check` и ссылку на
коммит, на котором выполнялась проверка «с нуля». Слейте PR и поставьте тег:

```bash
git tag -a v0.1-workspace -m "ML workspace ready"
git push origin v0.1-workspace
```

## Критерии приёмки

- [ ] `make check` завершается без строк `FAIL`;
- [ ] в свежем клоне проходят `make setup`, `make lint`, `make test`, `make train`;
- [ ] `make docker-train` создаёт файл модели в `models/` на хосте, владелец файла — ваш пользователь;
- [ ] попытка закоммитить файл больше 1 МБ отклоняется pre-commit;
- [ ] в `git ls-files` нет `.venv`, файлов из `data/` и `models/`, а в ноутбуках нет выводов ячеек;
- [ ] `README.md` содержит быстрый старт и таблицу целей `make`;
- [ ] в репозитории есть тег `v0.1-workspace`.

## Как оценить свою работу

| Уровень | Признаки |
|---|---|
| Базовый | все критерии приёмки выполнены на вашей машине |
| Уверенный | проверка «с нуля» проходит без ручных действий; история коммитов читается как рассказ о том, что и зачем сделано |
| Продвинутый | выполнено хотя бы одно задание из раздела «Идеи для усложнения» |

## Идеи для усложнения

1. **CI.** Добавьте GitHub Actions: при каждом pull request запускаются `make lint` и `make test`. Так
   проверка «с нуля» выполняется автоматически на чистой машине.
2. **Dev Container.** Опишите окружение в `.devcontainer/devcontainer.json` на основе этапа `dev` вашего
   Dockerfile, чтобы VS Code открывал проект сразу внутри контейнера.
3. **Справка в Makefile.** Сделайте цель `help`, которая печатает список целей с описаниями, и назначьте
   её целью по умолчанию.
4. **Расширение самопроверки.** Добавьте в `check_workspace.py` проверку, что в закоммиченных ноутбуках
   нет выводов ячеек (файл `.ipynb` — это JSON, у ячеек кода есть поле `outputs`).
5. **GPU-вариант.** Если у вас есть видеокарта NVIDIA, добавьте в `compose.yaml` сервис с доступом к GPU
   и цель `make gpu-check`, которая запускает в нём `nvidia-smi`.

## Что дальше

Workspace готов. В теме 1.2 вы начнёте наполнять его кодом: первым проектом станет ETL-пайплайн, а папки
`homework/` и `projects/` начнут заполняться.

**Следующий урок:** 1.2.1 — Продвинутый Python: декораторы, генераторы, context managers.
