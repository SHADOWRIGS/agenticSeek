#!/usr/bin/env python3
"""Самопроверка ML workspace (практика 1.1.4).

Запуск из корня проекта: uv run python check_workspace.py [--skip-docker]
"""

import argparse
import importlib
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
Result = tuple[str, str, str]

REQUIRED_FILES = [
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".vscode/settings.json",
    "Dockerfile",
    ".dockerignore",
    "compose.yaml",
    "Makefile",
    "README.md",
    "train.py",
]
REQUIRED_PACKAGES = ["numpy", "pandas", "sklearn", "matplotlib"]
GITIGNORE_ENTRIES = [".venv", "data", "models", ".env"]
DOCKERIGNORE_ENTRIES = [".venv", ".git", "data", "models", ".env"]
MAKE_TARGETS = ["setup", "lint", "test", "train", "docker-build"]
MAX_TRACKED_FILE_BYTES = 1_000_000


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)


def ignore_patterns(path: Path) -> set[str]:
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return {line.strip("/").removesuffix("/*") for line in lines if line and not line.startswith("#")}


def check_python(root: Path) -> Iterator[Result]:
    version = ".".join(map(str, sys.version_info[:3]))
    ok = sys.version_info >= (3, 12)
    yield (PASS if ok else FAIL, "Версия Python не ниже 3.12", version)

    in_project_venv = Path(sys.prefix).resolve() == (root / ".venv").resolve()
    yield (
        PASS if in_project_venv else FAIL,
        "Скрипт запущен интерпретатором из .venv проекта",
        sys.executable if in_project_venv else f"{sys.executable} — запускайте через uv run",
    )

    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
            yield (PASS, f"Пакет {name} импортируется", getattr(module, "__version__", ""))
        except ImportError as error:
            yield (FAIL, f"Пакет {name} импортируется", str(error))


def check_files(root: Path) -> Iterator[Result]:
    for name in REQUIRED_FILES:
        exists = (root / name).is_file()
        yield (PASS if exists else FAIL, f"Файл {name} существует", "" if exists else "не найден")

    for filename, required in ((".gitignore", GITIGNORE_ENTRIES), (".dockerignore", DOCKERIGNORE_ENTRIES)):
        path = root / filename
        if not path.is_file():
            continue
        missing = [entry for entry in required if entry not in ignore_patterns(path)]
        yield (
            PASS if not missing else FAIL,
            f"{filename} содержит обязательные правила",
            "" if not missing else "нет правил: " + ", ".join(missing),
        )

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        has_ruff = "ruff" in config.get("tool", {})
        yield (
            PASS if has_ruff else FAIL,
            "В pyproject.toml настроен Ruff",
            "" if has_ruff else "нет секции [tool.ruff]",
        )

    makefile = root / "Makefile"
    if makefile.is_file():
        targets = set(re.findall(r"^([A-Za-z0-9_-]+):", makefile.read_text(encoding="utf-8"), flags=re.MULTILINE))
        missing = [target for target in MAKE_TARGETS if target not in targets]
        yield (
            PASS if not missing else FAIL,
            "В Makefile есть обязательные цели",
            "" if not missing else "нет целей: " + ", ".join(missing),
        )


def check_git(root: Path) -> Iterator[Result]:
    if shutil.which("git") is None:
        yield (FAIL, "Git установлен", "команда git не найдена")
        return
    if run(["git", "rev-parse", "--is-inside-work-tree"], root).returncode != 0:
        yield (FAIL, "Проект является Git-репозиторием", "выполните git init")
        return
    yield (PASS, "Проект является Git-репозиторием", "")

    for key in ("user.name", "user.email"):
        value = run(["git", "config", key], root).stdout.strip()
        yield (PASS if value else FAIL, f"Задана настройка git {key}", value or "не задана")

    remote = run(["git", "remote", "get-url", "origin"], root).stdout.strip()
    yield (PASS if remote else WARN, "Настроен удалённый репозиторий origin", remote or "не настроен")

    commits = run(["git", "rev-list", "--count", "HEAD"], root)
    count = int(commits.stdout.strip()) if commits.returncode == 0 else 0
    yield (PASS if count >= 3 else FAIL, "В истории не меньше трёх коммитов", str(count))

    tracked = run(["git", "ls-files"], root).stdout.splitlines()
    venv_tracked = any(path.startswith(".venv/") for path in tracked)
    yield (
        PASS if not venv_tracked else FAIL,
        "Папка .venv не отслеживается Git",
        "" if not venv_tracked else "git rm -r --cached .venv",
    )

    large = [
        path for path in tracked if (root / path).is_file() and (root / path).stat().st_size > MAX_TRACKED_FILE_BYTES
    ]
    yield (PASS if not large else FAIL, "В репозитории нет файлов больше 1 МБ", ", ".join(large[:5]))

    hook = (root / ".git" / "hooks" / "pre-commit").is_file()
    yield (PASS if hook else FAIL, "Хук pre-commit установлен", "" if hook else "выполните uv run pre-commit install")


def check_tools(root: Path) -> Iterator[Result]:
    for tool in ("uv", "make"):
        found = shutil.which(tool)
        yield (PASS if found else FAIL, f"Команда {tool} доступна", found or "не найдена")

    lint = run([sys.executable, "-m", "ruff", "check", "."], root)
    yield (
        PASS if lint.returncode == 0 else FAIL,
        "ruff check проходит без ошибок",
        "" if lint.returncode == 0 else "запустите make lint",
    )


def check_docker(root: Path) -> Iterator[Result]:
    if shutil.which("docker") is None:
        yield (FAIL, "Docker установлен", "команда docker не найдена")
        return
    yield (PASS, "Docker установлен", run(["docker", "--version"], root).stdout.strip())

    daemon = run(["docker", "info"], root).returncode == 0
    yield (
        PASS if daemon else FAIL,
        "Демон Docker запущен и доступен без sudo",
        "" if daemon else "docker info завершился с ошибкой",
    )

    if (root / "compose.yaml").is_file():
        valid = run(["docker", "compose", "config", "-q"], root)
        yield (PASS if valid.returncode == 0 else FAIL, "compose.yaml корректен", valid.stderr.strip()[:200])


def main() -> int:
    parser = argparse.ArgumentParser(description="Самопроверка ML workspace")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="корень проекта (по умолчанию текущая папка)")
    parser.add_argument("--skip-docker", action="store_true", help="пропустить проверки Docker")
    args = parser.parse_args()
    root = args.root.resolve()

    sections: list[tuple[str, Callable[[Path], Iterator[Result]]]] = [
        ("Python и пакеты", check_python),
        ("Файлы проекта", check_files),
        ("Git", check_git),
        ("Инструменты", check_tools),
    ]
    if not args.skip_docker:
        sections.append(("Docker", check_docker))

    counts = {PASS: 0, WARN: 0, FAIL: 0}
    for title, check in sections:
        print(f"\n{title}")
        for status, name, details in check(root):
            counts[status] += 1
            print(f"  [{status}] {name}" + (f" — {details}" if details else ""))

    print(f"\nИтог: пройдено {counts[PASS]}, предупреждений {counts[WARN]}, ошибок {counts[FAIL]}")
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
