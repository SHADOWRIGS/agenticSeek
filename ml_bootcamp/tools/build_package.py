#!/usr/bin/env python3
"""Собирает архив продукта для загрузки на площадку продаж.

Запуск из корня репозитория:

    python3 ml_bootcamp/tools/build_package.py --owner "Имя правообладателя"

Результат — `ml_bootcamp/dist/classic-ml-tabular-<версия>.zip`. Сборка воспроизводима: при том же
коммите и тех же аргументах архив совпадает до байта, поэтому его контрольную сумму можно сверять.

В архив попадают только файлы, отслеживаемые Git, из модулей продукта и папки итогового проекта.
Перед упаковкой скрипт проверяет продукт и останавливается, если найдёт:
тему-черновик, отсутствующий урок, ссылку в Markdown, ведущую за пределы архива или в никуда,
либо файл, похожий на эталонное решение.
"""

import argparse
import hashlib
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from build_index import COURSE_DIR, DRAFT_MARKER, PLAN_PATH, Module, parse_plan

VERSION = "1.0"
PRODUCT_SLUG = "classic-ml-tabular"
PRODUCT_TITLE = "Классический ML для табличных данных"
PRODUCT_MODULES = (1, 2, 3, 4)
CAPSTONE_DIR = "capstone_classic_ml"
OWNER_PLACEHOLDER = "<правообладатель>"
FIXED_TIME = (2026, 1, 1, 0, 0, 0)  # одна дата у всех файлов: архив не зависит от времени сборки

# Названия модулей для покупателя: в плане курса они рабочие
MODULE_TITLES = {
    1: "Рабочее окружение и инженерные практики",
    2: "Данные: pandas, NumPy, признаки, визуализация",
    3: "Классический ML: линейные модели, метрики, кластеризация, рекомендации",
    4: "Ансамбли: деревья, случайный лес, бустинг, подбор гиперпараметров",
}

INDEX_RE = re.compile(r"module_\d+(/topic_\d+_\d+)?/README\.md")  # оглавления для архива создаются заново
SUSPICIOUS = re.compile(r"(reference|solution|эталон|решени)", re.IGNORECASE)
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=COURSE_DIR, capture_output=True, text=True, check=True)
    return result.stdout


def tracked_files(directory: str) -> list[Path]:
    """Пути отслеживаемых файлов папки относительно ml_bootcamp/."""
    output = git("ls-files", "-z", "--", directory)
    return sorted(Path(name) for name in output.split("\0") if name)


def lessons_table(module: Module, topic_index: int, link_prefix: str) -> list[str]:
    topic = module.topics[topic_index]
    rows = ["| Урок | Название | Тип |", "|---|---|---|"]
    for lesson in topic.lessons:
        kind = "практика" if lesson.is_practice else "теория"
        rows.append(f"| {lesson.number} | [{lesson.title}]({link_prefix}{lesson.filename}) | {kind} |")
    return rows


def generated_readmes(modules: list[Module], owner: str, build_id: str) -> dict[Path, str]:
    """Оглавления архива: корневое, по модулям и по темам."""
    files: dict[Path, str] = {}
    lessons_total = sum(len(topic.lessons) for module in modules for topic in module.topics)
    projects_total = sum(
        lesson.is_practice for module in modules for topic in module.topics for lesson in topic.lessons
    )

    root = [
        f"# {PRODUCT_TITLE}",
        "",
        f"{lessons_total} урока, {projects_total} практических проектов с автотестами и итоговый проект.",
        "",
        "## С чего начать",
        "",
        "1. Откройте урок 1.1.1 — в нём ставится рабочее окружение: Python 3.12, `uv`, Ruff.",
        "2. Идите по порядку: каждая тема — три теоретических урока и практический проект.",
        "3. В практическом уроке сказано, какие файлы взять из папки `practice/` и как запустить тесты.",
        "   На заготовке тесты падают — так задумано. Когда все проходят, проект готов.",
        "4. После четвёртого модуля — [итоговый проект](capstone_classic_ml/README.md).",
        "",
        "Те, у кого Python и Git уже настроены, могут начать со второго модуля.",
        "",
        "## Программа",
        "",
    ]
    for module in modules:
        title = MODULE_TITLES[module.number]
        root += [f"### [Модуль {module.number}. {title}]({module.dirname}/README.md)", ""]
        module_readme = [f"# Модуль {module.number}. {title}", ""]
        for index, topic in enumerate(module.topics):
            root += [
                f"**Тема {topic.number}. {topic.title}**",
                "",
                *lessons_table(module, index, f"{module.dirname}/{topic.dirname}/"),
                "",
            ]
            module_readme += [f"## [Тема {topic.number}. {topic.title}]({topic.dirname}/README.md)", ""]
            module_readme += [*lessons_table(module, index, f"{topic.dirname}/"), ""]
            topic_readme = [f"# Тема {topic.number}. {topic.title}", "", f"Модуль {module.number}. {title}", ""]
            topic_readme += lessons_table(module, index, "")
            files[Path(module.dirname) / topic.dirname / "README.md"] = "\n".join(topic_readme) + "\n"
        files[Path(module.dirname) / "README.md"] = "\n".join(module_readme[:-1]) + "\n"

    root += [
        "### [Итоговый проект: прогноз спроса на прокат велосипедов](capstone_classic_ml/README.md)",
        "",
        "Сквозная задача на все четыре модуля: от сырой выгрузки до отчёта для заказчика. Прогноз проверяется",
        "скриптом, работа целиком — по критериям приёмки.",
        "",
        "## Что нужно для работы",
        "",
        "- Компьютер с Linux, macOS или Windows с WSL; видеокарта не нужна.",
        "- Умение писать на Python функции и классы и работать в командной строке.",
        "- Версии библиотек, на которых проверены материалы, указаны в уроках и в описаниях практик.",
        "",
        "## Об отсылках к другим курсам",
        "",
        "В нескольких уроках упоминаются темы за пределами этого курса: нейронные сети, MLOps,",
        "высокопроизводительные вычисления, проектирование ML-систем — с номерами модулей 5–11. Они в этот",
        "курс не входят и для его прохождения не нужны.",
        "",
        "## Условия использования",
        "",
        "См. [LICENSE.md](LICENSE.md).",
        "",
        "---",
        "",
        f"Версия {VERSION}, сборка {build_id}.",
    ]
    files[Path("README.md")] = "\n".join(root) + "\n"

    files[Path("LICENSE.md")] = (
        "\n".join(
            [
                "# Условия использования",
                "",
                f"© {owner}, 2026. Все права защищены.",
                "",
                "Материалы курса — тексты уроков, код, тесты и данные — предоставляются покупателю для личного",
                "обучения. Разрешается:",
                "",
                "- хранить материалы на своих устройствах и работать с ними;",
                "- использовать в своих проектах, в том числе рабочих, код, который вы написали, проходя курс;",
                "- показывать свои решения практических проектов, например в портфолио, без текстов уроков.",
                "",
                "Без письменного согласия правообладателя не разрешается:",
                "",
                "- передавать материалы третьим лицам, публиковать их целиком или частями в открытом доступе;",
                "- перепродавать материалы или включать их в другие курсы.",
                "",
                "Код уроков и практик предоставляется «как есть», без гарантий.",
            ]
        )
        + "\n"
    )
    return files


def check_links(texts: dict[Path, str], names: set[Path]) -> list[str]:
    """Относительные ссылки Markdown должны вести на файлы архива."""
    problems = []
    folders = {parent for name in names for parent in name.parents}
    for path, text in sorted(texts.items()):
        for match in LINK_RE.finditer(FENCE_RE.sub("", text)):
            target = match.group(1).split("#")[0]
            if not target or target.startswith(("http://", "https://", "mailto:", "<")):
                continue
            parts: list[str] = []
            for part in (path.parent / target).parts:
                if part == "..":
                    if not parts:
                        parts = ["<за пределами архива>"]
                        break
                    parts.pop()
                elif part != ".":
                    parts.append(part)
            resolved = Path(*parts) if parts else Path(".")
            if resolved not in names and resolved not in folders:
                problems.append(f"{path}: ссылка «{target}» не ведёт на файл архива")
    return problems


def collect(modules: list[Module], owner: str, build_id: str) -> tuple[dict[Path, bytes], list[str]]:
    """Возвращает содержимое архива и список найденных проблем."""
    problems: list[str] = []
    content: dict[Path, bytes] = {}

    for module in modules:
        for topic in module.topics:
            topic_dir = COURSE_DIR / module.dirname / topic.dirname
            if (topic_dir / DRAFT_MARKER).exists():
                problems.append(f"тема {topic.number} помечена как черновик")
            for lesson in topic.lessons:
                if not (topic_dir / lesson.filename).exists():
                    problems.append(f"нет урока {lesson.number}")

    directories = [module.dirname for module in modules] + [CAPSTONE_DIR]
    for directory in directories:
        files = tracked_files(directory)
        if not files:
            problems.append(f"в папке {directory} нет отслеживаемых файлов")
        for path in files:
            if path.name == DRAFT_MARKER or INDEX_RE.fullmatch(path.as_posix()):
                continue
            if "{{" not in str(path) and SUSPICIOUS.search(path.name):
                problems.append(f"{path}: имя файла похоже на эталонное решение")
            content[path] = (COURSE_DIR / path).read_bytes()

    dirty = git("status", "--porcelain", "--", *directories).strip()
    if dirty:
        problems.append("в папках продукта есть незакоммиченные изменения:\n" + dirty)

    for path, text in generated_readmes(modules, owner, build_id).items():
        content[path] = text.encode("utf-8")

    texts = {path: data.decode("utf-8") for path, data in content.items() if path.suffix == ".md"}
    problems += check_links(texts, set(content))
    return content, problems


def write_zip(content: dict[Path, bytes], destination: Path, top_folder: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(content):
            info = zipfile.ZipInfo(f"{top_folder}/{path.as_posix()}", date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            executable = path.suffix == ".sh"
            info.external_attr = (0o755 if executable else 0o644) << 16
            archive.writestr(info, content[path])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--owner", default=OWNER_PLACEHOLDER, help="правообладатель для LICENSE.md")
    parser.add_argument("--out", type=Path, default=COURSE_DIR / "dist", help="папка для архива")
    args = parser.parse_args()

    plan = parse_plan(PLAN_PATH.read_text(encoding="utf-8"))
    modules = [module for module in plan if module.number in PRODUCT_MODULES]
    build_id = git("rev-parse", "--short", "HEAD").strip()

    content, problems = collect(modules, args.owner, build_id)
    if problems:
        print("Архив не собран. Найдены проблемы:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    top_folder = f"{PRODUCT_SLUG}-{VERSION}"
    destination = args.out / f"{top_folder}.zip"
    write_zip(content, destination, top_folder)

    lessons = sum(1 for path in content if re.fullmatch(r"lesson_\d+_\d+_\d+\.md", path.name))
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f"Архив: {destination}")
    print(f"Файлов: {len(content)}, уроков: {lessons}, размер: {destination.stat().st_size / 1e6:.1f} МБ")
    print(f"SHA-256: {digest}")
    if args.owner == OWNER_PLACEHOLDER:
        print("Внимание: правообладатель не указан — в LICENSE.md стоит заглушка. Передайте --owner.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
