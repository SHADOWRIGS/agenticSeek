#!/usr/bin/env python3
"""Проверка архива продукта перед выпуском.

Распаковывает архив продукта и архив эталонных решений во временную папку и для каждой практики
с тестами pytest проверяет две вещи:

1. на заготовке, как её получит покупатель, тесты падают (кроме тестов готовых частей);
2. с эталонным решением все тесты проходят, причём с `-W error`.

Кроме того: тест на bash из практики 1.4.4 — на заготовке и на эталоне; проект из шаблона практики 1.3.4
создаётся и проходит `make install lint test train` (нужны `uvx`, `make` и доступ к PyPI).

Запускать из окружения, в котором установлены зависимости всего курса — так, как они перечислены
в уроках (см. launch/RELEASE_CHECK.md):

    uv run python ml_bootcamp/tools/release_check.py \\
        ml_bootcamp/dist/classic-ml-tabular-1.0.zip путь/к/reference-solutions.zip

Код возврата 0 — архив можно выпускать. Архив эталонных решений в репозитории не хранится.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

SUMMARY_RE = re.compile(r"(\d+) (passed|failed|error|errors)")
# Практика 1.2.4: тесты проверяют модуль, имя которого задано переменной окружения; заготовка —
# рабочая медленная версия, поэтому её тесты проходят
REFERENCE_ENV = {"topic_1_2": {"ETL_MODULE": "etl_fast"}}
STUB_PASSES = {"topic_1_2", "capstone_classic_ml"}


def run_pytest(directory: Path, strict: bool, extra_env: dict[str, str] | None = None) -> tuple[int, str, float]:
    """Запускает pytest в папке практики. Возвращает код возврата, итоговую строку и время в секундах."""
    # -p no:dash: плагин Dash для pytest вызывает предупреждение при загрузке (см. урок 2.4.1)
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:dash"]
    if strict:
        command += ["-W", "error"]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(extra_env or {})}
    started = time.perf_counter()
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - started
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    summary = lines[-1].strip("= ") if lines else result.stderr.strip().splitlines()[-1]
    return result.returncode, summary, elapsed


def counts(summary: str) -> dict[str, int]:
    found = {"passed": 0, "failed": 0, "error": 0}
    for number, kind in SUMMARY_RE.findall(summary):
        found["error" if kind.startswith("error") else kind] += int(number)
    return found


def practices(product_root: Path) -> list[tuple[str, Path]]:
    """Папки с тестами pytest: практики тем и итоговый проект."""
    found = []
    for tests in sorted(product_root.glob("module_*/topic_*/practice/tests")):
        if any(tests.glob("test_*.py")):
            found.append((tests.parent.parent.name, tests.parent))
    capstone = product_root / "capstone_classic_ml"
    if (capstone / "tests").is_dir():
        found.append((capstone.name, capstone))
    return found


def overlay_reference(name: str, practice_dir: Path, reference_root: Path) -> int:
    """Кладёт файлы эталона поверх заготовки. Возвращает число скопированных файлов."""
    source = reference_root / name / "practice"
    if not source.is_dir():
        return 0
    copied = 0
    for file in sorted(source.rglob("*")):
        if file.is_file():
            target = practice_dir / file.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
            copied += 1
    return copied


def run(command: list[str], directory: Path, extra_env: dict[str, str] | None = None) -> tuple[int, str]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(extra_env or {})}
    env.pop("VIRTUAL_ENV", None)  # сгенерированный проект создаёт собственное окружение
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, env=env)
    return result.returncode, (result.stdout + result.stderr).strip()


def check_bash_practice(product_root: Path, reference_root: Path) -> tuple[str, str, list[str]]:
    """Практика 1.4.4: синтаксис скриптов, тест на заготовке и на эталоне."""
    practice = product_root / "module_01" / "topic_1_4" / "practice"
    problems = []
    for script in sorted([*practice.glob("scripts/*.sh"), *practice.glob("tests/*.sh")]):
        if run(["bash", "-n", str(script)], practice)[0] != 0:
            problems.append(f"topic_1_4: синтаксическая ошибка в {script.name}")

    test = ["bash", "tests/test_cleanup_checkpoints.sh"]
    stub_code, _ = run(test, practice)
    if stub_code == 0:
        problems.append("topic_1_4: тест проходит на заготовке")
    reference = reference_root / "topic_1_4" / "practice" / "cleanup_solution.sh"
    if not reference.exists():
        return "падает" if stub_code else "проходит", "нет эталона", [*problems, "topic_1_4: нет эталона"]
    code, output = run(test, practice, {"CLEANUP_SCRIPT": str(reference)})
    if code != 0:
        problems.append("topic_1_4: эталон не проходит тест")
    last_line = output.splitlines()[-1] if output else ""
    return "падает" if stub_code else "проходит", last_line, problems


def check_template(product_root: Path, work: Path) -> tuple[str, list[str]]:
    """Практика 1.3.4: проект из шаблона создаётся, ставится, проходит линтер и тесты, обучает модель."""
    template = product_root / "module_01" / "topic_1_3" / "practice"
    output_dir = work / "generated"
    code, output = run(["uvx", "cookiecutter", "--no-input", str(template), "-o", str(output_dir)], work)
    if code != 0:
        return "проект не создан", [f"topic_1_3: cookiecutter завершился с ошибкой: {output.splitlines()[-1]}"]
    project = next(output_dir.iterdir())
    for target in ("install", "lint", "test", "train"):
        code, output = run(["make", target], project)
        if code != 0:
            last_line = output.splitlines()[-1] if output else ""
            return f"make {target}: ошибка", [f"topic_1_3: make {target} завершился с ошибкой: {last_line}"]
    return "install, lint, test, train", []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("product", type=Path, help="архив продукта")
    parser.add_argument("reference", type=Path, help="архив эталонных решений")
    parser.add_argument("--skip-template", action="store_true", help="не проверять шаблон проекта (нужна сеть)")
    args = parser.parse_args()

    failures = []
    total_tests = 0
    with tempfile.TemporaryDirectory(prefix="release_check_") as workdir:
        work = Path(workdir)
        with zipfile.ZipFile(args.product) as archive:
            archive.extractall(work / "product")
        with zipfile.ZipFile(args.reference) as archive:
            archive.extractall(work / "reference")
        product_root = next((work / "product").iterdir())
        reference_root = next((work / "reference").iterdir())

        print(f"{'Практика':<22}{'Заготовка':<34}{'Эталон':<26}{'Время, с':>9}")
        for name, directory in practices(product_root):
            code, summary, _ = run_pytest(directory, strict=False)
            stub = counts(summary)
            if name in STUB_PASSES:
                stub_ok = code == 0
                stub_text = f"проходит: {stub['passed']}"
            else:
                stub_ok = code != 0 and stub["failed"] + stub["error"] > 0
                stub_text = f"падает: {stub['failed'] + stub['error']}, проходит: {stub['passed']}"
            if not stub_ok:
                failures.append(f"{name}: заготовка ведёт себя не так, как ожидается ({summary})")

            if name == "capstone_classic_ml":
                total_tests += stub["passed"]
                print(f"{name:<22}{stub_text:<34}{'не требуется':<26}{'':>9}")
                continue

            copied = overlay_reference(name, directory, reference_root)
            if not copied:
                failures.append(f"{name}: в архиве эталонов нет решения")
                print(f"{name:<22}{stub_text:<34}{'нет эталона':<26}{'':>9}")
                continue
            code, summary, elapsed = run_pytest(directory, strict=True, extra_env=REFERENCE_ENV.get(name))
            reference = counts(summary)
            total_tests += reference["passed"]
            if code != 0:
                failures.append(f"{name}: эталон не проходит тесты ({summary})")
            print(f"{name:<22}{stub_text:<34}{summary.split(' in ')[0]:<26}{elapsed:>9.1f}")

        stub_text, reference_text, problems = check_bash_practice(product_root, reference_root)
        failures += problems
        print(f"{'topic_1_4 (bash)':<22}{stub_text:<34}{reference_text:<26}")
        if not args.skip_template:
            started = time.perf_counter()
            template_text, problems = check_template(product_root, work)
            failures += problems
            print(f"{'topic_1_3 (шаблон)':<22}{'—':<34}{template_text:<26}{time.perf_counter() - started:>9.1f}")

    print()
    print(f"Тестов pytest, пройденных эталонами и материалами итогового проекта: {total_tests}")
    if failures:
        print("Архив выпускать нельзя:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Все проверки пройдены: архив можно выпускать.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
