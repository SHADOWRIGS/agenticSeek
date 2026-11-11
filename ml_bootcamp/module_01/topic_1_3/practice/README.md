# Стартовый шаблон ML-проекта (Cookiecutter)

Минимальный рабочий шаблон к уроку [1.3.4](../lesson_1_3_4.md). Из него уже генерируется проект, в котором
работают установка окружения, обучение, тесты и линтер. Ваша задача в практике — превратить эту заготовку
в собственный шаблон.

## Что внутри

```text
practice/
├── cookiecutter.json                      переменные шаблона и значения по умолчанию
├── README.md                              этот файл, в проект не попадает
└── {{cookiecutter.project_slug}}/         всё, что окажется в сгенерированном проекте
    ├── configs/base.yaml                  базовый конфиг эксперимента
    ├── data/{raw,interim,processed}/      пустые папки с .gitkeep
    ├── models/                            сюда сохраняются результаты запусков
    ├── notebooks/
    ├── src/{{cookiecutter.package_name}}/
    │   ├── __init__.py
    │   ├── config.py                      схема конфига (pydantic), загрузка и сохранение
    │   └── train.py                       точка входа обучения
    ├── tests/                             тесты конфига и обучения
    ├── .gitignore
    ├── .python-version
    ├── Makefile                           команды install, train, test, lint, format, clean
    ├── pyproject.toml                     зависимости, сборка пакета, настройки Ruff и pytest
    └── README.md
```

Переменные шаблона (`cookiecutter.json`):

| Переменная | По умолчанию | Где используется |
|---|---|---|
| `project_name` | `ML Project` | заголовок README, описание пакета |
| `project_slug` | из `project_name`: `ml-project` | имя папки проекта и имя проекта в `pyproject.toml` |
| `package_name` | из `project_slug`: `ml_project` | имя Python-пакета в `src/` и в импортах |
| `author` | `Your Name` | `pyproject.toml`, README |
| `python_version` | `3.12` | `.python-version`, `requires-python` |
| `description` | `ML-проект, созданный из шаблона курса` | `pyproject.toml`, README |

## Как сгенерировать проект

Нужны `uv` и `make`. Cookiecutter отдельно ставить не нужно — `uvx` запустит его во временном окружении.

Интерактивно, из корня репозитория курса (проект появится в `~/projects`):

```bash
uvx cookiecutter ml_bootcamp/module_01/topic_1_3/practice -o ~/projects
```

Cookiecutter по очереди спросит значения переменных; Enter оставляет значение по умолчанию из скобок.

Без вопросов, со значениями по умолчанию и точечными переопределениями:

```bash
uvx cookiecutter --no-input ml_bootcamp/module_01/topic_1_3/practice -o ~/projects \
    project_name="Churn Prediction" author="Анна Иванова"
```

Проверьте, что проект живой:

```bash
cd ~/projects/churn-prediction
make install   # uv sync: создаёт .venv и uv.lock
make train     # обучение по configs/base.yaml
make test
make lint
```

Ожидаемый результат `make train`:

```text
uv run python -m churn_prediction.train --config configs/base.yaml
Запуск сохранён в models/20261111-012831_baseline
accuracy: 0.9825
f1_macro: 0.9812
```

Имя папки запуска состоит из даты, времени и `experiment_name` из конфига. В ней лежат `model.joblib`,
`metrics.json` и `config.yaml` — итоговый конфиг со всеми значениями по умолчанию. После первого
`make install` закоммитьте появившийся `uv.lock`.

## Как устроен шаблон

Cookiecutter копирует папку `{{cookiecutter.project_slug}}/` и пропускает через шаблонизатор Jinja имена
файлов и папок и содержимое каждого файла. Выражение `{{ cookiecutter.package_name }}` заменяется значением
переменной. Отсюда главное правило при доработке: последовательности `{{`, `{%` и `{#` в файлах шаблона —
это синтаксис Jinja. Если они нужны в сгенерированном файле как обычный текст (например, `{{config}}`
в justfile или `${{ secrets.TOKEN }}` в GitHub Actions), оберните их в `{% raw %}...{% endraw %}`. Одиночные
фигурные скобки (f-строки, словари, `$(VAR)` в Makefile) безопасны.

После любой правки шаблона генерируйте проект заново и прогоняйте в нём `make install lint test train` —
ошибка в шаблоне видна только в сгенерированном проекте.

## Что нужно доработать

Заготовка намеренно минимальна. Полный список требований и критерии приёмки — в уроке
[1.3.4](../lesson_1_3_4.md), коротко:

1. Скопируйте содержимое этой папки в собственный репозиторий шаблона и замените этот README своим.
2. Дополните структуру: `reports/figures/`, `.env.example`, пример ноутбука или инструкция по ноутбукам.
3. Добавьте хуки: проверку `package_name` до генерации и `git init` после неё.
4. Расширьте работу с конфигами: второй конфиг эксперимента, переопределение параметров из командной строки
   или переход на Hydra.
5. Сделайте запуски воспроизводимыми до конца: сохраняйте рядом с метриками хеш коммита Git.
6. Разнесите `train.py` на модули (данные, модель, оценка) и добавьте тесты на новый код.
7. Добавьте команду, которая проверяет сам шаблон: генерирует проект во временную папку и запускает в нём
   установку, линтер, тесты и обучение.
