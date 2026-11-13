#!/usr/bin/env python3
"""Строит структуру папок и оглавления курса по docs/ML_BOOTCAMP_COURSE_PLAN.md."""

import re
from dataclasses import dataclass, field
from pathlib import Path

COURSE_DIR = Path(__file__).resolve().parent.parent
PLAN_PATH = COURSE_DIR.parent / "docs" / "ML_BOOTCAMP_COURSE_PLAN.md"
GENERATED_NOTE = "<!-- Файл сгенерирован tools/build_index.py, вручную не редактировать -->"
DRAFT_MARKER = ".draft"

ABOUT = [
    "## Как устроены материалы",
    "",
    "- `module_XX/topic_X_Y/lesson_X_Y_Z.md` — урок: цели, теория с проверенными примерами, типичные ошибки,",
    "  контрольные вопросы и домашнее задание с критериями приёмки.",
    "- Четвёртый урок каждой темы — практический проект. Его стартовые материалы лежат в `topic_X_Y/practice/`:",
    "  заготовки кода и тесты, описывающие ожидаемое поведение. Эталонных решений в репозитории нет.",
    "- Код уроков рассчитан на Python 3.12 и окружение из урока 1.1.1 (`uv`, Ruff).",
    "",
    "## Подготовка к запуску",
    "",
    "- [Программа: цель и результат каждого урока](launch/SYLLABUS.md)",
    "- [Текст лендинга](launch/LANDING.md)",
    "- [Готовность к запуску продаж](launch/LAUNCH_READINESS.md)",
    "- [Запуск: продукт первой продажи и шаги владельца](launch/GO_LIVE.md)",
    "- [Бесплатный образец: состав и тексты](launch/FREE_SAMPLE.md)",
    "- [Пакет публикаций для запуска](launch/CONTENT_PACK.md)",
    "- [Чек-листы ревью практических проектов](launch/REVIEW_CHECKLISTS.md)",
    "- [Нужно решение владельца](launch/OWNER_DECISIONS.md)",
    "",
    "Оглавления (`README.md` в папках курса, модулей и тем) создаются автоматически.",
    "После добавления урока обновите их: `python3 ml_bootcamp/tools/build_index.py`",
]

MODULE_RE = re.compile(r"^## МОДУЛЬ (\d+): (.+)$")
TOPIC_RE = re.compile(r"^### Тема (\d+)\.(\d+): (.+)$")
LESSON_RE = re.compile(r"^- \*\*Урок (\d+)\.(\d+)\.(\d+):\*\* (.+)$")


@dataclass
class Lesson:
    number: str
    title: str

    @property
    def filename(self) -> str:
        return f"lesson_{self.number.replace('.', '_')}.md"

    @property
    def is_practice(self) -> bool:
        return self.title.startswith("ПРАКТИКА")


@dataclass
class Topic:
    number: str
    title: str
    lessons: list[Lesson] = field(default_factory=list)

    @property
    def dirname(self) -> str:
        return f"topic_{self.number.replace('.', '_')}"


@dataclass
class Module:
    number: int
    title: str
    topics: list[Topic] = field(default_factory=list)

    @property
    def dirname(self) -> str:
        return f"module_{self.number:02d}"


def parse_plan(text: str) -> list[Module]:
    modules: list[Module] = []
    for line in text.splitlines():
        if m := MODULE_RE.match(line):
            modules.append(Module(int(m[1]), m[2].strip()))
        elif m := TOPIC_RE.match(line):
            modules[-1].topics.append(Topic(f"{m[1]}.{m[2]}", m[3].strip()))
        elif m := LESSON_RE.match(line):
            modules[-1].topics[-1].lessons.append(Lesson(f"{m[1]}.{m[2]}.{m[3]}", m[4].strip()))
    return modules


def is_draft(topic_dir: Path) -> bool:
    """Тема с файлом .draft ещё не прошла проверку: её уроки не считаются готовыми."""
    return (topic_dir / DRAFT_MARKER).exists()


def lesson_row(lesson: Lesson, topic_dir: Path, link_prefix: str) -> str:
    kind = "практика" if lesson.is_practice else "теория"
    if (topic_dir / lesson.filename).exists():
        title = f"[{lesson.title}]({link_prefix}{lesson.filename})"
        status = "черновик, на проверке" if is_draft(topic_dir) else "готов"
    else:
        title = lesson.title
        status = "в разработке"
    return f"| {lesson.number} | {title} | {kind} | {status} |"


def lessons_table(topic: Topic, topic_dir: Path, link_prefix: str) -> list[str]:
    rows = ["| Урок | Название | Тип | Статус |", "|---|---|---|---|"]
    rows += [lesson_row(lesson, topic_dir, link_prefix) for lesson in topic.lessons]
    return rows


def count_ready(topics: list[Topic], module_dir: Path) -> tuple[int, int]:
    total = sum(len(t.lessons) for t in topics)
    ready = sum(
        (module_dir / t.dirname / lesson.filename).exists()
        for t in topics
        if not is_draft(module_dir / t.dirname)
        for lesson in t.lessons
    )
    return ready, total


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(modules: list[Module]) -> None:
    root = [
        GENERATED_NOTE,
        "# Machine Learning Bootcamp",
        "",
        "Материалы курса. Полный план: [docs/ML_BOOTCAMP_COURSE_PLAN.md](../docs/ML_BOOTCAMP_COURSE_PLAN.md).",
        "",
        "| Модуль | Название | Готово уроков |",
        "|---|---|---|",
    ]
    for module in modules:
        module_dir = COURSE_DIR / module.dirname
        ready, total = count_ready(module.topics, module_dir)
        root.append(f"| {module.number} | [{module.title}]({module.dirname}/README.md) | {ready} / {total} |")

        module_readme = [GENERATED_NOTE, f"# Модуль {module.number}: {module.title}", ""]
        for topic in module.topics:
            topic_dir = module_dir / topic.dirname
            module_readme += [f"## [Тема {topic.number}: {topic.title}]({topic.dirname}/README.md)", ""]
            module_readme += lessons_table(topic, topic_dir, f"{topic.dirname}/") + [""]
            write(
                topic_dir / "README.md",
                [
                    GENERATED_NOTE,
                    f"# Тема {topic.number}: {topic.title}",
                    "",
                    f"Модуль {module.number}: {module.title}",
                    "",
                    *lessons_table(topic, topic_dir, ""),
                ],
            )
        write(module_dir / "README.md", module_readme[:-1])

    counts = [count_ready(m.topics, COURSE_DIR / m.dirname) for m in modules]
    all_ready = sum(ready for ready, _ in counts)
    total = sum(module_total for _, module_total in counts)
    root += ["", f"Всего готово: {all_ready} из {total} уроков.", "", *ABOUT]
    write(COURSE_DIR / "README.md", root)
    print(f"Модулей: {len(modules)}, тем: {sum(len(m.topics) for m in modules)}, уроков: {total}, готово: {all_ready}")


if __name__ == "__main__":
    build(parse_plan(PLAN_PATH.read_text(encoding="utf-8")))
