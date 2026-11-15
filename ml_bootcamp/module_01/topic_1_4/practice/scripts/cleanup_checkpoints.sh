#!/usr/bin/env bash
# Удаляет старые чекпоинты в папке, оставляя N самых свежих по времени изменения.
#
# ЗАГОТОВКА. Разбор и проверка аргументов готовы. Функции с пометкой TODO нужно написать самостоятельно.
# Ожидаемое поведение описано в tests/test_cleanup_checkpoints.sh и в README.md.
set -euo pipefail

readonly SCRIPT_NAME="${0##*/}"
readonly EXIT_USAGE=2
readonly EXIT_NOT_IMPLEMENTED=99

keep=3
pattern='epoch_*.pt'
dry_run=0
ckpt_dir=''

usage() {
    cat <<EOF
Использование: $SCRIPT_NAME [опции] DIR

Оставляет в папке DIR последние N чекпоинтов (по времени изменения), остальные удаляет.
Обрабатываются только обычные файлы непосредственно в DIR, имя которых подходит под шаблон.
Вложенные папки и файлы с другими именами (best.pt, логи, конфиги) не затрагиваются.

Опции:
  -k, --keep N        сколько последних чекпоинтов оставить, целое число >= 1 (по умолчанию $keep)
  -p, --pattern GLOB  шаблон имени файла чекпоинта (по умолчанию '$pattern')
  -n, --dry-run       ничего не удалять, только показать, какие файлы были бы удалены
  -h, --help          показать эту справку

Коды возврата: 0 — успех, $EXIT_USAGE — ошибка в аргументах, другое ненулевое значение — ошибка при работе.

Пример:
  $SCRIPT_NAME --keep 5 --dry-run /data/experiments/resnet18/checkpoints
EOF
}

die_usage() {
    printf '%s: %s\n' "$SCRIPT_NAME" "$1" >&2
    printf 'Справка: %s --help\n' "$SCRIPT_NAME" >&2
    exit "$EXIT_USAGE"
}

not_implemented() {
    printf '%s: функция %s не реализована (найдите TODO в скрипте)\n' "$SCRIPT_NAME" "$1" >&2
    return "$EXIT_NOT_IMPLEMENTED"
}

parse_args() {
    while (( $# > 0 )); do
        case "$1" in
            -k|--keep)
                (( $# >= 2 )) || die_usage "опции $1 нужен аргумент"
                keep="$2"
                shift 2
                ;;
            -p|--pattern)
                (( $# >= 2 )) || die_usage "опции $1 нужен аргумент"
                pattern="$2"
                shift 2
                ;;
            -n|--dry-run)
                dry_run=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            --)
                shift
                break
                ;;
            -*)
                die_usage "неизвестная опция: $1"
                ;;
            *)
                [[ -z "$ckpt_dir" ]] || die_usage "папка указана дважды: '$ckpt_dir' и '$1'"
                ckpt_dir="$1"
                shift
                ;;
        esac
    done

    # После «--» может идти только папка (на случай имени, начинающегося с дефиса).
    if (( $# > 0 )); then
        [[ -z "$ckpt_dir" && $# -eq 1 ]] || die_usage "лишние аргументы: $*"
        ckpt_dir="$1"
    fi
}

validate_args() {
    [[ -n "$ckpt_dir" ]] || die_usage "не указана папка с чекпоинтами"
    [[ -d "$ckpt_dir" ]] || die_usage "папка не найдена: $ckpt_dir"
    [[ "$keep" =~ ^[1-9][0-9]*$ ]] || die_usage "--keep: ожидается целое число >= 1, получено '$keep'"
    [[ -n "$pattern" ]] || die_usage "--pattern: шаблон не может быть пустым"
}

# Печатает в stdout пути чекпоинтов, по одному в строке, от самого старого к самому свежему.
# Учитывает только обычные файлы непосредственно в $ckpt_dir, имя которых подходит под $pattern.
# Порядок задаёт время изменения файла (mtime), а не имя.
list_checkpoints_oldest_first() {
    # TODO: реализуйте функцию. Имена файлов могут содержать пробелы.
    not_implemented "${FUNCNAME[0]}"
}

# Читает из stdin список путей (от старых к свежим) и печатает в stdout те, что подлежат удалению:
# все, кроме последних $keep. Если файлов не больше $keep, не печатает ничего.
select_for_deletion() {
    # TODO: реализуйте функцию.
    not_implemented "${FUNCNAME[0]}"
}

# Обрабатывает один файл, путь к которому передан первым аргументом.
# Обычный режим: удаляет файл и печатает в stdout строку с его путём.
# Режим --dry-run (dry_run=1): файл не трогает, печатает в stdout строку с его путём и пометкой,
# что удаления не было.
remove_checkpoint() {
    # TODO: реализуйте функцию.
    not_implemented "${FUNCNAME[0]} (файл: $1, dry_run=$dry_run)"
}

main() {
    parse_args "$@"
    validate_args

    # Каркас основной логики: список на удаление собирается конвейером из двух функций,
    # затем каждый файл передаётся в remove_checkpoint.
    # Список сохраняется в переменную, а не читается из <(...): ошибка внутри <(...) не останавливает
    # скрипт даже с set -e, а ошибка в var="$(...)" — останавливает.
    local to_delete
    to_delete="$(list_checkpoints_oldest_first | select_for_deletion)"

    local file
    while IFS= read -r file; do
        [[ -n "$file" ]] || continue
        remove_checkpoint "$file"
    done <<< "$to_delete"

    # TODO: напечатайте итоговую строку: сколько файлов удалено (в режиме --dry-run — сколько было бы
    # удалено) и сколько чекпоинтов осталось в папке.
}

main "$@"
