# Урок 1.1.1: Установка Python, Jupyter, VS Code + расширения для ML

> **Модуль 1:** Foundations & Mindset Engineer · **Тема 1.1:** Установка профессионального окружения
> **Тип:** теория · **Время:** 90 минут + 60 минут на домашнее задание

## Цели урока

После урока вы сможете:

- объяснить, почему системный Python нельзя использовать для ML-проектов;
- установить нужную версию Python и создать изолированное окружение проекта с помощью `uv`;
- запустить JupyterLab и подключить к нему окружение проекта как ядро;
- настроить VS Code так, чтобы линтер, форматтер и ноутбуки работали «из коробки».

## Зачем это нужно

Типичная история новичка: поставил Python с сайта, через `pip install` добавил десяток библиотек, через месяц
новый проект потребовал другую версию NumPy — и сломались оба проекта. В команде проблема умножается:
«у меня работает» перестаёт быть аргументом, когда у каждого своя версия Python и свои версии пакетов.

Инженерный подход опирается на три правила:

1. **Системный Python не трогаем.** Он нужен операционной системе. Ставить в него пакеты — значит рисковать
   и системой, и проектом.
2. **Один проект — одно окружение.** Зависимости проекта живут в папке проекта (`.venv`) и не пересекаются
   с другими проектами.
3. **Окружение описано в файлах.** Версия Python и список пакетов зафиксированы в репозитории, и любой коллега
   (или вы сами через полгода) воспроизводит окружение одной командой.

## 1. Выбор инструментов

| Задача | Инструмент курса | Альтернативы |
|---|---|---|
| Установка версий Python | `uv` | pyenv, conda, установщик с python.org |
| Окружения и зависимости | `uv` | venv + pip, Poetry, conda |
| Интерактивные эксперименты | JupyterLab | ноутбуки внутри VS Code, Google Colab |
| Редактор | VS Code | PyCharm, Cursor, Neovim |
| Линтер и форматтер | Ruff | flake8 + black + isort |

Почему `uv`: один инструмент заменяет pyenv, venv, pip и pip-tools, работает в десятки раз быстрее pip и создаёт
lock-файл с точными версиями всех пакетов. С conda вы тоже встретитесь — она по-прежнему удобна, когда нужны
не-Python зависимости (например, конкретная сборка CUDA). Подробно управление зависимостями разберём в теме 1.3.

**Какую версию Python выбрать.** Не самую новую, а предпоследнюю или ту, что вышла больше года назад:
ML-библиотеки (PyTorch, TensorFlow, Numba) начинают поддерживать свежий Python с задержкой в несколько месяцев.
В курсе используем Python 3.12 — на нём работают все библиотеки программы.

## 2. Подготовка операционной системы

**Linux (Ubuntu/Debian):**

```bash
sudo apt update && sudo apt install -y build-essential curl git
```

**macOS:**

```bash
xcode-select --install
```

**Windows:** работаем через WSL2 — это полноценный Linux внутри Windows. Почти все серверы, на которых обучают
и разворачивают модели, работают на Linux, поэтому привыкать к нему лучше сразу. В PowerShell от имени
администратора:

```powershell
wsl --install
```

После перезагрузки откройте приложение Ubuntu и выполняйте дальше команды для Linux. Проекты храните внутри
файловой системы WSL (`~/projects`), а не на диске `C:` — так работа с файлами в разы быстрее.

## 3. Установка uv и Python

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Перезапустите терминал и проверьте:

```bash
uv --version
```

Установите Python и посмотрите, какие версии доступны:

```bash
uv python install 3.12
uv python list
```

Эти версии лежат в домашней папке и никак не влияют на системный Python.

## 4. Первый проект

```bash
mkdir -p ~/projects && cd ~/projects
uv init ml-bootcamp --python 3.12
cd ml-bootcamp
```

Что появилось в папке:

| Файл | Назначение |
|---|---|
| `pyproject.toml` | описание проекта и список зависимостей |
| `.python-version` | версия Python для этого проекта |
| `.gitignore` | что не должно попадать в Git (в том числе `.venv`) |
| `main.py`, `README.md` | заготовки, можно менять |

Добавим основные библиотеки и инструменты разработки:

```bash
uv add numpy pandas scikit-learn matplotlib
uv add --dev jupyterlab ipykernel ruff
```

После первой команды `uv add` появятся папка `.venv` (само окружение) и файл `uv.lock` (точные версии всех
пакетов, включая зависимости зависимостей). Флаг `--dev` помечает пакеты, нужные только при разработке:
на сервер с моделью JupyterLab ставить незачем.

Запуск кода в окружении проекта:

```bash
uv run python -c "import sys, numpy, pandas, sklearn; print(sys.version.split()[0], numpy.__version__, pandas.__version__, sklearn.__version__)"
```

`uv run` сам находит окружение проекта, активировать его вручную не нужно. Если удобнее работать по-старому:

```bash
source .venv/bin/activate   # теперь python и pip — из окружения проекта
deactivate                  # выйти из окружения
```

**Что коммитим в Git:** `pyproject.toml`, `uv.lock`, `.python-version`. **Что не коммитим:** `.venv`.
Коллега, склонировав репозиторий, выполнит `uv sync` и получит точно такое же окружение.

## 5. JupyterLab

```bash
uv run jupyter lab
```

В браузере откроется JupyterLab, ядро по умолчанию — Python из окружения проекта. Проверьте в первой ячейке:

```python
import sys
print(sys.executable)   # путь должен заканчиваться на ml-bootcamp/.venv/bin/python
```

Чтобы окружение было видно как отдельное ядро в любом Jupyter и в VS Code, зарегистрируйте его:

```bash
uv run python -m ipykernel install --user --name ml-bootcamp --display-name "Python (ml-bootcamp)"
uv run jupyter kernelspec list
```

Полезные приёмы в ноутбуке:

| Команда | Что делает |
|---|---|
| `%timeit expr` | замеряет время выполнения выражения |
| `%%time` | замеряет время выполнения всей ячейки |
| `%load_ext autoreload` и `%autoreload 2` | подхватывает изменения в ваших `.py` файлах без перезапуска ядра |
| `!команда` | выполняет команду оболочки |
| `объект?` | показывает справку по объекту |

**Правило курса:** ноутбук — для исследования, `.py` модули — для кода, который переиспользуется. Как только
функция понадобилась во втором ноутбуке, она переезжает в модуль.

## 6. VS Code

Установите VS Code с официального сайта. Пользователям Windows: редактор ставится в Windows, а с проектом в WSL
соединяется через расширение WSL — откройте папку проекта в терминале Ubuntu и выполните `code .`.

Расширения ставятся из терминала:

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension charliermarsh.ruff
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-vscode-remote.remote-containers
code --install-extension eamodio.gitlens
code --install-extension tamasfe.even-better-toml
```

| Расширение | Зачем |
|---|---|
| Python + Pylance | автодополнение, переходы по коду, проверка типов, отладчик |
| Jupyter | ноутбуки прямо в редакторе, просмотр переменных и датафреймов |
| Ruff | линтер и форматтер, исправляет код при сохранении |
| Remote - SSH | работа с кодом на удалённом сервере с GPU как с локальным |
| Dev Containers | разработка внутри Docker-контейнера (пригодится в уроке 1.1.3) |
| GitLens | история изменений каждой строки |
| Even Better TOML | подсветка и проверка `pyproject.toml` |

Настройки проекта хранятся в файле `.vscode/settings.json` — его коммитят, чтобы у всей команды редактор вёл
себя одинаково:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  },
  "notebook.formatOnSave.enabled": true,
  "editor.rulers": [120],
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true
}
```

Настройки Ruff добавьте в `pyproject.toml`:

```toml
[tool.ruff]
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

Наборы правил: `E` — стиль, `F` — ошибки вроде неиспользуемых импортов, `I` — сортировка импортов,
`B` — частые баги, `UP` — устаревший синтаксис. Проверка из терминала:

```bash
uv run ruff check .
uv run ruff format .
```

Интерпретатор выбирается через палитру команд (`Ctrl+Shift+P` → «Python: Select Interpreter») — укажите
`.venv` проекта. Там же полезно запомнить «Jupyter: Create Interactive Window» и «Python: Debug Python File».

## Типичные ошибки

| Симптом | Причина | Решение |
|---|---|---|
| `ModuleNotFoundError`, хотя пакет установлен | код запущен другим интерпретатором | запускайте через `uv run`; в VS Code и Jupyter проверьте выбранный интерпретатор и ядро |
| `error: externally-managed-environment` | попытка поставить пакет в системный Python | создайте окружение проекта, не используйте `sudo pip` |
| `uv: command not found` после установки | терминал не перечитал `PATH` | перезапустите терминал |
| В ноутбуке не видно пакетов проекта | выбрано чужое ядро | выберите ядро «Python (ml-bootcamp)» |
| Проект в WSL работает медленно | файлы лежат на диске Windows (`/mnt/c/...`) | перенесите проект в `~/projects` |
| Репозиторий весит сотни мегабайт | закоммичена папка `.venv` | добавьте `.venv` в `.gitignore`, удалите её из индекса: `git rm -r --cached .venv` |

## Контрольные вопросы

1. Чем опасна установка пакетов в системный Python?
2. В чём разница между `pyproject.toml` и `uv.lock`? Почему коммитят оба файла?
3. Что делает флаг `--dev` в команде `uv add` и когда он нужен?
4. Как убедиться, что ноутбук использует окружение вашего проекта?
5. Почему для ML-проекта не стоит брать самую свежую версию Python?
6. Зачем коммитить `.vscode/settings.json`?

## Домашнее задание

Подготовьте рабочее окружение и докажите, что оно воспроизводимо.

1. Установите `uv`, Python 3.12, VS Code и расширения из урока.
2. Создайте проект `ml-bootcamp`, добавьте библиотеки и инструменты разработки, настройте Ruff
   и `.vscode/settings.json`.
3. Создайте ноутбук `notebooks/00_environment_check.ipynb`, который:
   - печатает путь к интерпретатору и версии Python, NumPy, pandas, scikit-learn;
   - сравнивает через `%timeit` суммирование миллиона чисел циклом `for` и функцией `numpy.sum`;
   - строит любой график в matplotlib.
4. Проверьте воспроизводимость: удалите папку `.venv`, выполните `uv sync` и убедитесь, что ноутбук
   по-прежнему выполняется целиком (Run → Restart Kernel and Run All Cells).

**Критерии приёмки:**

- [ ] `uv run ruff check .` завершается без ошибок;
- [ ] в ноутбуке путь к интерпретатору ведёт в `.venv` проекта;
- [ ] ноутбук выполняется сверху вниз без ошибок после пересоздания окружения;
- [ ] в папке проекта есть `pyproject.toml`, `uv.lock`, `.python-version`, а `.venv` указана в `.gitignore`.

В отчёте запишите, во сколько раз `numpy.sum` оказался быстрее цикла, — к этому числу вернёмся в теме 2.2.

## Дополнительные материалы

- Документация uv: https://docs.astral.sh/uv/
- Документация Ruff: https://docs.astral.sh/ruff/
- Документация JupyterLab: https://jupyterlab.readthedocs.io/
- Python в VS Code: https://code.visualstudio.com/docs/python/python-tutorial

**Следующий урок:** 1.1.2 — Настройка Git, GitHub, создание репозитория курса.
