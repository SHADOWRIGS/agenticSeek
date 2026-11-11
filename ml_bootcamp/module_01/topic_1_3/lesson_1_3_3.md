# Урок 1.3.3: Конфигурационные файлы для ML экспериментов

> **Модуль 1:** Foundations & Mindset Engineer · **Тема 1.3:** Инструменты репроизводимости
> **Тип:** теория · **Время:** 90 минут + 60 минут на домашнее задание

## Цели урока

После урока вы сможете:

- вынести параметры эксперимента из кода в конфигурационный файл и выбрать для него формат;
- описать схему конфига и ловить опечатки и недопустимые значения до начала обучения;
- собирать конфиг из нескольких источников: базовый файл, файл эксперимента, командная строка;
- использовать Hydra: группы конфигов, переопределения, серия запусков одной командой;
- сохранять итоговый конфиг рядом с результатами, чтобы любой запуск можно было повторить;
- хранить секреты отдельно от конфигов.

## Зачем это нужно

Типичный код начинающего:

```python
model = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)  # было 0.1, потом 0.01
```

После двадцати экспериментов лучший результат записан в блокноте как «accuracy 0.965», а с какими
параметрами он получен — неизвестно: код с тех пор меняли ещё десять раз. Каждый эксперимент — это правка
кода, история Git забита коммитами «поменял lr», а сравнить два запуска можно только по памяти.

Конфигурация эксперимента должна отвечать трём требованиям:

1. **Все параметры — в одном месте**, отдельно от кода. Новый эксперимент — это новый файл или аргумент
   командной строки, а не правка исходников.
2. **Каждый запуск оставляет запись**: с какими именно параметрами он выполнен.
3. **Ошибки в параметрах обнаруживаются сразу**, а не на третьем часу обучения.

Работаем в проекте `churn-model` из прошлых уроков. Добавим зависимости:

```bash
uv add pydantic pyyaml omegaconf hydra-core
```

## 1. Первый шаг: argparse

Самый простой способ убрать числа из кода — аргументы командной строки. Фрагмент скрипта
`train_argparse.py`, который обучает градиентный бустинг на встроенном датасете:

```python
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--n-estimators", type=int, default=100)
parser.add_argument("--max-depth", type=int, default=3)
parser.add_argument("--learning-rate", type=float, default=0.1)
args = parser.parse_args()
```

```bash
uv run python train_argparse.py --max-depth 2 --learning-rate 0.05   # accuracy: 0.9386
```

Для пяти параметров этого хватает. Для реального проекта — нет: параметров десятки, у них есть структура
(данные, модель, обучение), команда запуска не помещается на экран и нигде не сохраняется. Следующий шаг —
конфигурационный файл, а в командной строке остаётся только путь к нему.

## 2. Форматы: YAML, TOML, JSON

| Формат | Плюсы | Минусы | Где встречается |
|---|---|---|---|
| YAML | компактный, есть комментарии, удобен для вложенных структур | неочевидное определение типов, значимые отступы | конфиги экспериментов, Hydra, CI, Docker Compose, Kubernetes |
| TOML | явные типы, есть комментарии, чтение встроено в Python (`tomllib`) | глубокая вложенность читается хуже | `pyproject.toml`, настройки инструментов |
| JSON | строгий, поддерживается везде | нет комментариев, много скобок и кавычек | обмен данными между программами, метрики, метаданные |

Практическое разделение: конфиги, которые пишет человек, — YAML; результаты, которые пишет программа
(метрики, метаданные запуска), — JSON.

У YAML есть ловушки, о которые спотыкается каждый. Библиотека PyYAML сама решает, какой тип у значения:

```python
import yaml

print(yaml.safe_load("learning_rate: 1e-3\npython: 3.10\nscheduler: off\n"))
print(yaml.safe_load("learning_rate: 1.0e-3\npython: '3.10'\nscheduler: 'off'\n"))
```

```text
{'learning_rate': '1e-3', 'python': 3.1, 'scheduler': False}
{'learning_rate': 0.001, 'python': '3.10', 'scheduler': 'off'}
```

`1e-3` без точки прочитано как строка, версия `3.10` превратилась в число `3.1`, а `off` — в `False`
(так же ведут себя `no`, `yes`, `on`). Числа в экспоненциальной записи пишите с точкой, версии и подобные
строки — в кавычках. Строка вместо числа в `learning_rate` — классика: ошибка проявится глубоко внутри
библиотеки или, хуже, не проявится вовсе. Защита — схема конфига. И читайте YAML только через
`yaml.safe_load`: небезопасные загрузчики PyYAML способны выполнить код, записанный в файле.

## 3. Конфиг-файл и схема

Базовый конфиг `configs/base.yaml`:

```yaml
seed: 42

data:
  test_size: 0.2

model:
  n_estimators: 100
  max_depth: 3
  learning_rate: 0.1
```

Прочитать его в словарь — одна строка. Но словарь ничего не проверяет: опечатка `max_dept` тихо
проигнорируется, и модель обучится со значением по умолчанию. Нужна **схема** — описание того, какие
параметры существуют, какого они типа и в каких границах.

Простейшая схема — `dataclass` из стандартной библиотеки. Он отклонит неизвестный ключ
(`ModelConfig(max_dept=5)` → `TypeError: ... got an unexpected keyword argument 'max_dept'`) и позволяет
написать проверки значений в `__post_init__`. Но типы он не проверяет: `ModelConfig(max_depth="пять")`
создаётся без ошибки, а вложенные секции приходится разбирать вручную. Поэтому в курсе используем
pydantic. Файл `src/churn_model/config.py`:

```python
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    # extra="forbid": неизвестный ключ — ошибка; frozen=True: конфиг нельзя менять после загрузки
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(StrictModel):
    test_size: float = Field(default=0.2, gt=0, lt=1)


class ModelConfig(StrictModel):
    n_estimators: int = Field(default=100, ge=1)
    max_depth: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=0.1, gt=0, le=1)


class TrainConfig(StrictModel):
    seed: int = 42
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()


def load_config(path: str | Path) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return TrainConfig.model_validate(raw)
```

Pydantic проверяет типы и границы, разбирает вложенные секции и приводит значения к нужному типу: строка
`'1e-3'` из ловушки YAML станет числом `0.001`. Точка входа `src/churn_model/train.py`:

```python
import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from churn_model.config import TrainConfig, load_config


def train(cfg: TrainConfig) -> float:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.seed, stratify=y
    )
    model = GradientBoostingClassifier(**cfg.model.model_dump(), random_state=cfg.seed)
    model.fit(X_train, y_train)
    return float(model.score(X_test, y_test))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    accuracy = train(cfg)

    run_dir = Path("models") / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    # Сохраняем полный конфиг со значениями по умолчанию, а не исходный файл
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    (run_dir / "metrics.json").write_text(json.dumps({"accuracy": accuracy}, indent=2))
    print(f"accuracy: {accuracy:.4f}, результаты: {run_dir}")


if __name__ == "__main__":
    main()
```

```bash
uv run python -m churn_model.train   # accuracy: 0.9561, результаты: models/20261111-012325
```

Теперь испортим конфиг: в `configs/typo.yaml` запишем в секцию `model` строки `max_dept: 6`
и `learning_rate: 10`. Запуск с `--config configs/typo.yaml` (вывод сокращён):

```text
pydantic_core._pydantic_core.ValidationError: 2 validation errors for TrainConfig
model.learning_rate
  Input should be less than or equal to 1 [type=less_than_equal, input_value=10, input_type=int]
model.max_dept
  Extra inputs are not permitted [type=extra_forbidden, input_value=6, input_type=int]
```

Обе ошибки найдены за доли секунды, до загрузки данных, с точным указанием места.

## 4. OmegaConf: слияние конфигов и параметры из командной строки

Копировать весь конфиг ради изменения одного параметра неудобно: двадцать почти одинаковых файлов расходятся
между собой. Удобнее хранить базовый конфиг и маленькие файлы экспериментов, где записаны только отличия.
`configs/deep.yaml`:

```yaml
model:
  max_depth: 6
  learning_rate: 5e-2
```

Библиотека OmegaConf умеет сливать конфиги и разбирать переопределения вида `model.max_depth=5`. Меняем
в `config.py` импорт `yaml` на `from omegaconf import OmegaConf` и переписываем загрузку:

```python
BASE_CONFIG = Path("configs/base.yaml")


def load_config(path: str | Path, overrides: list[str] | None = None) -> TrainConfig:
    # Чем правее источник, тем выше его приоритет
    merged = OmegaConf.merge(
        OmegaConf.load(BASE_CONFIG),
        OmegaConf.load(path),
        OmegaConf.from_dotlist(overrides or []),
    )
    return TrainConfig.model_validate(OmegaConf.to_container(merged, resolve=True))
```

В `train.py` добавляем разбор позиционных аргументов —
`parser.add_argument("overrides", nargs="*")` — и передаём их в загрузку:
`cfg = load_config(args.config, args.overrides)`.

```bash
uv run python -m churn_model.train --config configs/deep.yaml model.n_estimators=300 seed=7
cat models/20261111-012350/config.yaml
```

```text
accuracy: 0.9649, результаты: models/20261111-012350

seed: 7
data:
  test_size: 0.2
model:
  n_estimators: 300
  max_depth: 6
  learning_rate: 0.05
```

В итоговом конфиге видно всё сразу: `test_size` пришёл из базового файла, `max_depth` и `learning_rate` —
из файла эксперимента, `n_estimators` и `seed` — из командной строки. Схема pydantic по-прежнему проверяет
результат слияния, так что опечатка в командной строке тоже будет поймана.

Ещё одна возможность OmegaConf — интерполяция, ссылки одних значений на другие. Запись
`run_name: ${model.name}_depth${model.max_depth}_seed${seed}` при обращении к `cfg.run_name` даст строку
вида `gb_depth3_seed42` и изменится автоматически, если поменять `max_depth`.

## 5. Hydra: композиция конфигов

Hydra построена поверх OmegaConf и берёт на себя всё, что мы делали вручную: поиск конфигов, слияние,
разбор командной строки, создание папки запуска и сохранение итогового конфига. Главная идея — **группы
конфигов**: взаимозаменяемые варианты одной части конфигурации лежат в одной папке. Создадим в корне проекта
папку `conf` с главным конфигом `conf/config.yaml`:

```yaml
defaults:
  - model: rf
  - _self_

seed: 42

data:
  test_size: 0.2
```

и группой `model` из двух файлов:

```yaml
# conf/model/rf.yaml
_target_: sklearn.ensemble.RandomForestClassifier
n_estimators: 200
max_depth: 5

# conf/model/gb.yaml
_target_: sklearn.ensemble.GradientBoostingClassifier
n_estimators: 100
max_depth: 3
learning_rate: 0.1
```

Список `defaults` говорит: в секцию `model` подставить файл `model/rf.yaml`. Ключ `_target_` — путь к классу,
объект которого нужно создать. Файл `src/churn_model/train_hydra.py`:

```python
import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# config_path задаётся относительно этого файла: src/churn_model -> корень проекта -> conf
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.seed, stratify=y
    )
    # instantiate создаёт объект класса из _target_ с параметрами из конфига
    model = instantiate(cfg.model, random_state=cfg.seed)
    model.fit(X_train, y_train)
    accuracy = float(model.score(X_test, y_test))

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    (run_dir / "metrics.json").write_text(json.dumps({"accuracy": accuracy}, indent=2))
    print(f"{type(model).__name__}: accuracy={accuracy:.4f}, результаты: {run_dir}")


if __name__ == "__main__":
    main()
```

В коде нет ни одного упоминания конкретной модели: какой класс создать и с какими параметрами, решает
конфиг. Запуск с заменой модели и точечными переопределениями:

```bash
uv run python -m churn_model.train_hydra
uv run python -m churn_model.train_hydra model=gb model.learning_rate=0.05 seed=7
```

Первая команда обучила `RandomForestClassifier` (accuracy 0.9561), вторая — `GradientBoostingClassifier`
(accuracy 0.9474). Результаты каждого запуска Hydra кладёт в отдельную папку `outputs/<дата>/<время>/`.

| Запись в командной строке | Что делает |
|---|---|
| `model=gb` | выбирает другой файл из группы `model` |
| `model.learning_rate=0.05` | меняет существующий параметр |
| `+model.min_samples_leaf=5` | добавляет параметр, которого нет в конфиге |
| `--cfg job` | печатает итоговый конфиг и выходит, ничего не запуская |
| `--multirun model=rf,gb model.max_depth=2,4` | запускает все комбинации значений |

Опечатку Hydra не пропустит: на `model.max_dept=4` она ответит `Key 'max_dept' is not in struct` и не начнёт
запуск. Серия экспериментов — одна команда:

```bash
uv run python -m churn_model.train_hydra --multirun model=rf,gb model.max_depth=2,4
```

Hydra сообщит `Launching 4 jobs locally`, выполнит четыре запуска по очереди и разложит результаты по папкам
`multirun/<дата>/<время>/0` … `3`. Папки `outputs/` и `multirun/` добавьте в `.gitignore`.

**Что выбрать.** Один файл конфига со схемой pydantic — для небольших проектов и сервисов: минимум
зависимостей, строгая проверка. Hydra — когда экспериментов много и они комбинируются из взаимозаменяемых
частей: модель, датасет, оптимизатор. Её цена — собственные правила и «магия», в которых нужно разобраться.
Оба подхода встречаются в индустрии.

## 6. Итоговый конфиг — рядом с результатами

Правило: **папка запуска самодостаточна**. В ней лежит всё, что нужно, чтобы понять, как получен результат,
и повторить его: итоговый конфиг, метрики, модель и хеш коммита Git. Итоговый конфиг — это полный набор
параметров после слияния и подстановки значений по умолчанию; не исходный файл, а то, с чем реально шло
обучение. Значения по умолчанию в коде меняются: если в файле эксперимента параметра не было, через месяц
тот же файл даст другой запуск, и вы об этом не узнаете.

Наш `train.py` сохраняет `config.yaml` и `metrics.json` сам. Сохранённый конфиг — полноценный входной файл,
запуск по нему даёт тот же результат:

```bash
uv run python -m churn_model.train --config models/20261111-012350/config.yaml
```

Hydra делает это автоматически. В папке каждого запуска есть подпапка `.hydra` с тремя файлами:
`config.yaml` — итоговый конфиг, `overrides.yaml` — что было передано в командной строке, `hydra.yaml` —
настройки самой Hydra. Повторить запуск по сохранённому конфигу:

```bash
uv run python -m churn_model.train_hydra \
    --config-path "$PWD/outputs/2026-11-11/01-24-13/.hydra" --config-name config
```

## 7. Секреты — не в конфигах

Конфиги коммитят в Git и копируют в папку каждого запуска. Пароль от базы или токен сервиса, записанный
в конфиг, окажется в истории репозитория и в десятках папок с результатами. Секреты передают через
переменные окружения. Пример — скрипт `secrets_demo.py`:

```python
import os

token = os.environ.get("MLFLOW_TRACKING_TOKEN")
if token is None:
    raise SystemExit("Не задана переменная окружения MLFLOW_TRACKING_TOKEN")
print(f"Токен получен, длина {len(token)}")
```

Чтобы не задавать переменные вручную в каждом терминале, их записывают в файл `.env` в корне проекта
строками вида `MLFLOW_TRACKING_TOKEN=very-secret-token`. `uv` умеет загружать такой файл при запуске:

```bash
uv run python secrets_demo.py                   # Не задана переменная окружения MLFLOW_TRACKING_TOKEN
uv run --env-file .env python secrets_demo.py   # Токен получен, длина 17
```

Правила работы с секретами:

- `.env` внесён в `.gitignore` с первого коммита;
- в репозитории лежит `.env.example` с именами переменных и пустыми значениями — документация для коллег;
- код проверяет наличие переменной в самом начале и падает с понятным сообщением;
- секреты не печатают в лог и не сохраняют в папку запуска;
- в конфиге допустима ссылка на переменную окружения — в OmegaConf и Hydra это запись
  `${oc.env:MLFLOW_TRACKING_TOKEN}`. Но если сохранить такой конфиг с подстановкой значений
  (`resolve=True`), секрет попадёт в файл.

## Типичные ошибки

| Симптом | Причина | Решение |
|---|---|---|
| `TypeError` при сравнении или умножении `learning_rate` | `1e-3` в YAML прочитан как строка | пишите `1.0e-3`; проверяйте конфиг схемой |
| Параметр из конфига «не действует» | опечатка в имени ключа, использовано значение по умолчанию | запретите неизвестные ключи: `extra="forbid"` в pydantic |
| `ValidationError` с перечнем полей при запуске | значение в конфиге не проходит проверку схемы | сообщение называет поле и причину; исправьте конфиг |
| `Key '...' is not in struct` в Hydra | переопределяется параметр, которого нет в конфиге | исправьте имя; чтобы добавить новый параметр, используйте префикс `+` |
| Hydra не находит конфиг | `config_path` считается от файла со скриптом, а не от текущей папки | проверьте относительный путь в декораторе `hydra.main` |
| Невозможно понять, с какими параметрами получена модель | конфиг не сохранялся или сохранён исходный файл без значений по умолчанию | сохраняйте итоговый конфиг в папку запуска |
| Старый конфиг даёт другой результат | в коде изменились значения по умолчанию | запускайте по сохранённому итоговому конфигу; значимые параметры указывайте явно |
| Токен оказался в репозитории | секрет записан в конфиг или `.env` не внесён в `.gitignore` | отзовите токен, перенесите секрет в переменные окружения |

## Контрольные вопросы

1. Почему недостаточно хранить гиперпараметры константами в начале скрипта?
2. Какой тип получит значение `lr: 1e-4` при чтении через PyYAML? Как это исправить двумя способами?
3. Что даёт схема конфига по сравнению со словарём? Чем pydantic здесь удобнее dataclass?
4. В каком порядке объединяются базовый конфиг, конфиг эксперимента и аргументы командной строки? Почему
   именно в таком?
5. Что такое группа конфигов в Hydra? Что делает запись `model=gb`, а что — `model.max_depth=4`?
6. Зачем сохранять итоговый конфиг, если исходный файл уже лежит в Git?
7. Сколько запусков выполнит команда с `--multirun model=rf,gb model.max_depth=2,4,8 seed=1,2`?
8. Почему токен доступа нельзя хранить в `configs/base.yaml`, даже если репозиторий приватный?

## Домашнее задание

Работайте в проекте `churn-model`.

1. Реализуйте `config.py` и `train.py` из разделов 3–4: схема pydantic, базовый конфиг, слияние с конфигом
   эксперимента и переопределениями из командной строки, папка запуска с `config.yaml` и `metrics.json`.
2. Добавьте в схему параметр `experiment_name` и используйте его в имени папки запуска.
3. Создайте два конфига экспериментов в `configs/` и выполните по запуску с каждым. Убедитесь, что запуск
   по сохранённому `config.yaml` даёт те же метрики.
4. Напишите тесты: базовый конфиг проходит проверку; неизвестный ключ и недопустимое значение приводят
   к `ValidationError`; переопределение из командной строки побеждает значение из файла.
5. Реализуйте вариант на Hydra из раздела 5 и запустите серию из шести экспериментов через `--multirun`.
   Напишите небольшой скрипт, который обходит папки серии, читает `.hydra/overrides.yaml` и `metrics.json`
   и печатает таблицу «параметры — accuracy».
6. Добавьте `.env.example`, внесите `.env`, `outputs/` и `multirun/` в `.gitignore`.

**Критерии приёмки:**

- [ ] в коде обучения нет числовых гиперпараметров — все значения приходят из конфига;
- [ ] `uv run python -m churn_model.train --config configs/<ваш>.yaml model.max_depth=4` создаёт папку запуска
      с `config.yaml`, в котором `max_depth: 4`;
- [ ] запуск по сохранённому `config.yaml` воспроизводит метрики исходного запуска;
- [ ] конфиг с опечаткой в имени параметра приводит к ошибке до начала обучения;
- [ ] `--multirun` создаёт шесть папок, скрипт печатает сводную таблицу;
- [ ] `make test` и `make lint` проходят, `git status` не показывает `.env`, `outputs/`, `multirun/`.

## Дополнительные материалы

- Документация Hydra: https://hydra.cc/docs/intro/
- Документация OmegaConf: https://omegaconf.readthedocs.io/
- Документация pydantic: https://docs.pydantic.dev/
- Документация PyYAML: https://pyyaml.org/wiki/PyYAMLDocumentation
- Спецификация TOML: https://toml.io/

**Следующий урок:** 1.3.4 — ПРАКТИКА: Создание шаблона для ML проекта.
