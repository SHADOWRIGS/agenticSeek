#!/usr/bin/env bash
# Тесты для scripts/cleanup_checkpoints.sh. Внешних зависимостей нет: только bash и coreutils.
# Тесты одновременно служат спецификацией: каждый описывает одно требование к скрипту.
#
# Запуск:            tests/test_cleanup_checkpoints.sh
# Другой скрипт:     CLEANUP_SCRIPT=/path/to/script.sh tests/test_cleanup_checkpoints.sh
#
# set -e здесь намеренно не включён: упавшая проверка не должна прерывать остальные тесты.
set -uo pipefail

tests_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CLEANUP_SCRIPT="${CLEANUP_SCRIPT:-$tests_dir/../scripts/cleanup_checkpoints.sh}"

tmp_root="$(mktemp -d)"
trap 'rm -rf "$tmp_root"' EXIT

tests_total=0
tests_failed=0
current_failures=0

# Результат последнего запуска скрипта.
run_status=0
run_stdout=''

fail() {
    current_failures=$((current_failures + 1))
    printf '    ОШИБКА: %s\n' "$1"
}

# Создаёт пустую папку для одного теста и печатает путь к ней.
new_case_dir() {
    mktemp -d "$tmp_root/case.XXXXXX"
}

# make_ckpt ФАЙЛ ДЕНЬ — создаёт файл с временем изменения 2026-01-ДЕНЬ 12:00.
make_ckpt() {
    mkdir -p "$(dirname "$1")"
    touch -d "2026-01-$2 12:00" "$1"
}

# Запускает проверяемый скрипт с переданными аргументами, сохраняет код возврата и stdout.
run_cleanup() {
    run_stdout="$(bash "$CLEANUP_SCRIPT" "$@" 2> "$tmp_root/stderr.txt")"
    run_status=$?
}

assert_status() {
    if [[ "$run_status" -ne "$1" ]]; then
        fail "ожидался код возврата $1, получен $run_status; stderr: $(head -n 1 "$tmp_root/stderr.txt")"
    fi
}

assert_exists() {
    local file
    for file in "$@"; do
        [[ -e "$file" ]] || fail "файл должен остаться, но удалён: ${file#"$tmp_root"/}"
    done
}

assert_missing() {
    local file
    for file in "$@"; do
        [[ ! -e "$file" ]] || fail "файл должен быть удалён, но остался: ${file#"$tmp_root"/}"
    done
}

assert_stdout_contains() {
    [[ "$run_stdout" == *"$1"* ]] || fail "в stdout ожидалась подстрока '$1'"
}

assert_stdout_not_contains() {
    [[ "$run_stdout" != *"$1"* ]] || fail "в stdout не должно быть подстроки '$1'"
}

run_test() {
    local name="$1"
    tests_total=$((tests_total + 1))
    current_failures=0
    printf '%s\n' "- $name"
    "$name"
    if (( current_failures > 0 )); then
        tests_failed=$((tests_failed + 1))
        printf '    НЕ ПРОЙДЕН\n'
    else
        printf '    ok\n'
    fi
}

# ---------------------------------------------------------------------------------------------------------
# Основная логика
# ---------------------------------------------------------------------------------------------------------

test_keeps_n_newest_and_deletes_the_rest() {
    local dir
    dir="$(new_case_dir)"
    make_ckpt "$dir/epoch_01.pt" 01
    make_ckpt "$dir/epoch_02.pt" 02
    make_ckpt "$dir/epoch_03.pt" 03
    make_ckpt "$dir/epoch_04.pt" 04
    make_ckpt "$dir/epoch_05.pt" 05

    run_cleanup --keep 2 "$dir"

    assert_status 0
    assert_exists "$dir/epoch_04.pt" "$dir/epoch_05.pt"
    assert_missing "$dir/epoch_01.pt" "$dir/epoch_02.pt" "$dir/epoch_03.pt"
}

test_default_keep_is_three() {
    local dir
    dir="$(new_case_dir)"
    make_ckpt "$dir/epoch_01.pt" 01
    make_ckpt "$dir/epoch_02.pt" 02
    make_ckpt "$dir/epoch_03.pt" 03
    make_ckpt "$dir/epoch_04.pt" 04
    make_ckpt "$dir/epoch_05.pt" 05

    run_cleanup "$dir"

    assert_status 0
    assert_exists "$dir/epoch_03.pt" "$dir/epoch_04.pt" "$dir/epoch_05.pt"
    assert_missing "$dir/epoch_01.pt" "$dir/epoch_02.pt"
}

# «Свежесть» определяется временем изменения файла, а не номером в имени:
# после дообучения с более раннего чекпоинта нумерация и время могут разойтись.
test_order_is_defined_by_mtime_not_by_name() {
    local dir
    dir="$(new_case_dir)"
    make_ckpt "$dir/epoch_01.pt" 09
    make_ckpt "$dir/epoch_02.pt" 02
    make_ckpt "$dir/epoch_03.pt" 03
    make_ckpt "$dir/epoch_04.pt" 08

    run_cleanup --keep 2 "$dir"

    assert_status 0
    assert_exists "$dir/epoch_01.pt" "$dir/epoch_04.pt"
    assert_missing "$dir/epoch_02.pt" "$dir/epoch_03.pt"
}

test_dry_run_deletes_nothing_and_lists_candidates() {
    local dir
    dir="$(new_case_dir)"
    make_ckpt "$dir/epoch_01.pt" 01
    make_ckpt "$dir/epoch_02.pt" 02
    make_ckpt "$dir/epoch_03.pt" 03
    make_ckpt "$dir/epoch_04.pt" 04

    run_cleanup --keep 2 --dry-run "$dir"

    assert_status 0
    assert_exists "$dir/epoch_01.pt" "$dir/epoch_02.pt" "$dir/epoch_03.pt" "$dir/epoch_04.pt"
    assert_stdout_contains "epoch_01.pt"
    assert_stdout_contains "epoch_02.pt"
    assert_stdout_not_contains "epoch_03.pt"
    assert_stdout_not_contains "epoch_04.pt"
}

test_real_run_reports_deleted_files() {
    local dir
    dir="$(new_case_dir)"
    make_ckpt "$dir/epoch_01.pt" 01
    make_ckpt "$dir/epoch_02.pt" 02
    make_ckpt "$dir/epoch_03.pt" 03

    run_cleanup --keep 1 "$dir"

    assert_status 0
    assert_stdout_contains "epoch_01.pt"
    assert_stdout_contains "epoch_02.pt"
    assert_stdout_not_contains "epoch_03.pt"
}

test_nothing_to_delete_when_files_not_more_than_keep() {
    local dir
    dir="$(new_case_dir)"
    make_ckpt "$dir/epoch_01.pt" 01
    make_ckpt "$dir/epoch_02.pt" 02

    run_cleanup --keep 5 "$dir"

    assert_status 0
    assert_exists "$dir/epoch_01.pt" "$dir/epoch_02.pt"
}

test_empty_directory_is_not_an_error() {
    local dir
    dir="$(new_case_dir)"

    run_cleanup "$dir"

    assert_status 0
}

test_files_not_matching_pattern_are_untouched() {
    local dir
    dir="$(new_case_dir)"
    make_ckpt "$dir/epoch_01.pt" 03
    make_ckpt "$dir/epoch_02.pt" 04
    make_ckpt "$dir/epoch_03.pt" 05
    # Эти файлы старше всех чекпоинтов, но под шаблон epoch_*.pt не подходят.
    make_ckpt "$dir/best.pt" 01
    make_ckpt "$dir/train.log" 01
    make_ckpt "$dir/config.yaml" 01

    run_cleanup --keep 1 "$dir"

    assert_status 0
    assert_exists "$dir/epoch_03.pt" "$dir/best.pt" "$dir/train.log" "$dir/config.yaml"
    assert_missing "$dir/epoch_01.pt" "$dir/epoch_02.pt"
}

test_custom_pattern() {
    local dir
    dir="$(new_case_dir)"
    make_ckpt "$dir/step_1000.ckpt" 01
    make_ckpt "$dir/step_2000.ckpt" 02
    make_ckpt "$dir/step_3000.ckpt" 03
    make_ckpt "$dir/epoch_01.pt" 01
    make_ckpt "$dir/epoch_02.pt" 02

    run_cleanup --keep 1 --pattern 'step_*.ckpt' "$dir"

    assert_status 0
    assert_exists "$dir/step_3000.ckpt" "$dir/epoch_01.pt" "$dir/epoch_02.pt"
    assert_missing "$dir/step_1000.ckpt" "$dir/step_2000.ckpt"
}

test_file_names_with_spaces() {
    local dir
    dir="$(new_case_dir)/my experiment"
    make_ckpt "$dir/epoch_01 copy.pt" 01
    make_ckpt "$dir/epoch_02 final.pt" 02
    make_ckpt "$dir/epoch_03.pt" 03

    run_cleanup --keep 1 "$dir"

    assert_status 0
    assert_exists "$dir/epoch_03.pt"
    assert_missing "$dir/epoch_01 copy.pt" "$dir/epoch_02 final.pt"
}

test_subdirectories_are_not_touched() {
    local dir
    dir="$(new_case_dir)"
    make_ckpt "$dir/epoch_01.pt" 05
    make_ckpt "$dir/epoch_02.pt" 06
    # Вложенный файл старше остальных и подходит под шаблон, но лежит не в самой папке.
    make_ckpt "$dir/archive/epoch_00.pt" 01

    run_cleanup --keep 1 "$dir"

    assert_status 0
    assert_exists "$dir/epoch_02.pt" "$dir/archive/epoch_00.pt"
    assert_missing "$dir/epoch_01.pt"
}

# ---------------------------------------------------------------------------------------------------------
# Аргументы (в заготовке уже реализовано — эти тесты защищают от случайной поломки)
# ---------------------------------------------------------------------------------------------------------

test_help_exits_with_zero() {
    run_cleanup --help
    assert_status 0
    assert_stdout_contains "--dry-run"
}

test_missing_directory_argument_is_usage_error() {
    run_cleanup --keep 2
    assert_status 2
}

test_nonexistent_directory_is_usage_error() {
    run_cleanup "$tmp_root/no_such_dir"
    assert_status 2
}

test_invalid_keep_is_usage_error_and_deletes_nothing() {
    local dir
    dir="$(new_case_dir)"
    make_ckpt "$dir/epoch_01.pt" 01
    make_ckpt "$dir/epoch_02.pt" 02

    run_cleanup --keep 0 "$dir"
    assert_status 2
    run_cleanup --keep -1 "$dir"
    assert_status 2
    run_cleanup --keep many "$dir"
    assert_status 2

    assert_exists "$dir/epoch_01.pt" "$dir/epoch_02.pt"
}

main() {
    if [[ ! -f "$CLEANUP_SCRIPT" ]]; then
        printf 'Не найден проверяемый скрипт: %s\n' "$CLEANUP_SCRIPT" >&2
        exit 2
    fi

    printf 'Проверяется: %s\n\n' "$CLEANUP_SCRIPT"

    run_test test_keeps_n_newest_and_deletes_the_rest
    run_test test_default_keep_is_three
    run_test test_order_is_defined_by_mtime_not_by_name
    run_test test_dry_run_deletes_nothing_and_lists_candidates
    run_test test_real_run_reports_deleted_files
    run_test test_nothing_to_delete_when_files_not_more_than_keep
    run_test test_empty_directory_is_not_an_error
    run_test test_files_not_matching_pattern_are_untouched
    run_test test_custom_pattern
    run_test test_file_names_with_spaces
    run_test test_subdirectories_are_not_touched
    run_test test_help_exits_with_zero
    run_test test_missing_directory_argument_is_usage_error
    run_test test_nonexistent_directory_is_usage_error
    run_test test_invalid_keep_is_usage_error_and_deletes_nothing

    printf '\nВсего тестов: %d, не пройдено: %d\n' "$tests_total" "$tests_failed"
    (( tests_failed == 0 ))
}

main "$@"
