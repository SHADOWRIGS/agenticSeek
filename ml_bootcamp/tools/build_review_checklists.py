#!/usr/bin/env python3
"""Собирает чек-листы ревью практических проектов из критериев приёмки в уроках.

Результат: ml_bootcamp/launch/REVIEW_CHECKLISTS.md. Темы с файлом .draft пропускаются.
"""

import re
from pathlib import Path

COURSE_DIR = Path(__file__).resolve().parent.parent
OUTPUT = COURSE_DIR / "launch" / "REVIEW_CHECKLISTS.md"
DRAFT_MARKER = ".draft"

TITLE_RE = re.compile(r"^# Урок (\d+\.\d+\.\d+): ПРАКТИКА: (.+)$")
LEVELS_HEADING = "## Как оценить свою работу"

HEADER = """<!-- Файл сгенерирован tools/build_review_checklists.py, вручную не редактировать -->
# Чек-листы ревью практических проектов

Документ для проверяющих и наставников. Критерии взяты из уроков автоматически, поэтому всегда совпадают
с тем, что видит ученик. Обновление: `python3 ml_bootcamp/tools/build_review_checklists.py`

## Порядок ревью

1. **Запустить автоматические проверки** из папки проекта ученика: тесты (`uv run pytest -q`) и линтер
   (`uv run ruff check .`). Если они не проходят, работа возвращается без дальнейшего разбора — с указанием,
   что именно упало.
2. **Убедиться, что тесты не изменены.** Сравнить папку `tests/` в работе ученика с исходной из материалов
   курса. Изменённые тесты — повод вернуть работу.
3. **Пройти по критериям приёмки** проекта из списка ниже. Каждый пункт — «выполнено», «не выполнено»
   или «частично» с пояснением.
4. **Посмотреть код и выводы.** Ревью оценивает не только то, что работает, но и то, как сделано:
   читаемость, отсутствие копирования функций в ноутбук, обоснованность выводов.
5. **Определить уровень** по таблице уровней проекта, если она есть.
6. **Написать отзыв** по шаблону ниже.

## Шаблон отзыва

```
Проект: <номер и название>
Итог: принято / на доработку
Уровень: базовый / уверенный / продвинутый

Что получилось хорошо (1–3 пункта, конкретно, со ссылкой на место в работе):
-

Что обязательно исправить (только для «на доработку»; каждый пункт — что не так и как проверить исправление):
-

Что можно улучшить (не блокирует приёмку):
-

Вопрос на понимание (один вопрос по самому важному месту проекта):
-
```

## Правила для проверяющего

- Критерии приёмки — единственное основание вернуть работу. Вкусовые замечания идут в раздел
  «Что можно улучшить».
- Замечание должно быть проверяемым: «функция `fill_calendar` заполняет нулями исходные пропуски —
  см. тест `test_original_missing_values_are_not_filled`», а не «плохо обработаны пропуски».
- Готовое решение ученику не показывают. Подсказка — это направление: какой раздел урока перечитать,
  какой тест запустить отдельно.
- Формулировки в отчётах учеников проверяются так же строго, как код: связь не выдаётся за причину,
  у оценок указана неопределённость.

---
"""


def practice_lessons() -> list[Path]:
    lessons = sorted(COURSE_DIR.glob("module_*/topic_*/lesson_*_4.md"), key=lesson_sort_key)
    return [path for path in lessons if not (path.parent / DRAFT_MARKER).exists()]


def lesson_sort_key(path: Path) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", path.stem))


def extract_checklist(lines: list[str]) -> list[str]:
    """Пункты вида «- [ ] ...» вместе со строками-продолжениями."""
    items: list[str] = []
    in_item = False
    for line in lines:
        if line.startswith("- [ ]"):
            items.append(line)
            in_item = True
        elif in_item and line.startswith("  ") and line.strip():
            items[-1] += " " + line.strip()
        else:
            in_item = False
    return items


def extract_section(lines: list[str], heading: str) -> list[str]:
    """Содержимое раздела от заголовка до следующего заголовка того же уровня."""
    if heading not in lines:
        return []
    start = lines.index(heading) + 1
    section: list[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        section.append(line)
    while section and not section[-1].strip():
        section.pop()
    while section and not section[0].strip():
        section.pop(0)
    return section


def has_tests(lesson: Path) -> bool:
    practice = lesson.parent / "practice"
    return practice.is_dir() and any(practice.rglob("test_*"))


def build() -> None:
    blocks = [HEADER]
    count = 0
    for lesson in practice_lessons():
        lines = lesson.read_text(encoding="utf-8").splitlines()
        match = TITLE_RE.match(lines[0])
        if not match:
            continue
        number, title = match.groups()
        link = lesson.relative_to(COURSE_DIR).as_posix()
        checklist = extract_checklist(lines)
        levels = extract_section(lines, LEVELS_HEADING)

        block = [f"## {number}. {title}", "", f"Постановка: [{lesson.name}](../{link})", ""]
        if has_tests(lesson):
            block += ["Автоматическая проверка: тесты из папки `practice/` этого урока.", ""]
        else:
            block += ["Автоматических тестов нет: проверка по критериям и материалам из папки `practice/`.", ""]
        block += ["**Критерии приёмки**", "", *checklist, ""]
        if levels:
            block += ["**Уровни**", "", *levels, ""]
        blocks.append("\n".join(block))
        count += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(blocks).rstrip() + "\n", encoding="utf-8")
    print(f"Чек-листов: {count}, файл: {OUTPUT.relative_to(COURSE_DIR.parent)}")


if __name__ == "__main__":
    build()
