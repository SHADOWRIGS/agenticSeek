#!/usr/bin/env bash
# Обёртка для долгих запусков обучения: лог с меткой времени, защита от повторного запуска,
# корректная реакция на сигналы, код возврата обёрнутой команды.
#
# ЗАГОТОВКА. Разбор аргументов и подготовка путей готовы. Функции с пометкой TODO нужно написать
# самостоятельно. Требования перечислены в README.md и в уроке 1.4.4.
set -euo pipefail

readonly SCRIPT_NAME="${0##*/}"
readonly EXIT_USAGE=2
readonly EXIT_LOCKED=75
readonly EXIT_NOT_IMPLEMENTED=99

run_name='train'
log_dir='logs'
lock_file=''
log_file=''
train_cmd=()

usage() {
    cat <<EOF
Использование: $SCRIPT_NAME [опции] -- КОМАНДА [АРГУМЕНТЫ...]

Запускает КОМАНДУ, пишет её stdout и stderr в лог-файл с меткой времени и не даёт запустить
второй экземпляр с тем же именем запуска.

Опции:
  -n, --name NAME       имя запуска: попадает в имя лога и файла блокировки (по умолчанию $run_name)
  -l, --log-dir DIR     папка для логов, создаётся при необходимости (по умолчанию $log_dir)
  -L, --lock-file FILE  файл блокировки (по умолчанию \${TMPDIR:-/tmp}/run_training_NAME.lock)
  -h, --help            показать эту справку

Коды возврата: код возврата КОМАНДЫ; $EXIT_USAGE — ошибка в аргументах;
$EXIT_LOCKED — запуск с таким именем уже идёт.

Пример:
  $SCRIPT_NAME --name resnet18 --log-dir logs -- python train.py --config configs/resnet18.yaml
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

# Служебные сообщения обёртки идут в stderr, чтобы не смешиваться с выводом самой команды.
log() {
    printf '%s [%s] %s\n' "$(date '+%F %T')" "$SCRIPT_NAME" "$1" >&2
}

parse_args() {
    while (( $# > 0 )); do
        case "$1" in
            -n|--name)
                (( $# >= 2 )) || die_usage "опции $1 нужен аргумент"
                run_name="$2"
                shift 2
                ;;
            -l|--log-dir)
                (( $# >= 2 )) || die_usage "опции $1 нужен аргумент"
                log_dir="$2"
                shift 2
                ;;
            -L|--lock-file)
                (( $# >= 2 )) || die_usage "опции $1 нужен аргумент"
                lock_file="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            --)
                shift
                train_cmd=("$@")
                break
                ;;
            *)
                die_usage "неизвестный аргумент: $1 (команду запуска указывают после «--»)"
                ;;
        esac
    done

    (( ${#train_cmd[@]} > 0 )) || die_usage "не указана команда запуска (после «--»)"
    # Имя запуска попадает в имена файлов, поэтому набор символов ограничен.
    [[ "$run_name" =~ ^[A-Za-z0-9._-]+$ ]] || die_usage "--name: допустимы только латиница, цифры, '.', '_' и '-'"

    if [[ -z "$lock_file" ]]; then
        lock_file="${TMPDIR:-/tmp}/run_training_${run_name}.lock"
    fi
    log_file="$log_dir/${run_name}_$(date '+%Y%m%d_%H%M%S').log"
}

# Захватывает блокировку $lock_file на всё время работы скрипта.
# Если блокировка уже занята другим запуском — сообщает об этом и завершает скрипт с кодом $EXIT_LOCKED,
# не дожидаясь освобождения.
acquire_lock() {
    # TODO: реализуйте функцию с помощью flock. Блокировка должна сняться сама при любом завершении
    # скрипта, в том числе аварийном.
    not_implemented "${FUNCNAME[0]} (файл блокировки: $lock_file)"
}

# Реакция на SIGINT и SIGTERM: передать сигнал запущенной команде, дождаться её завершения,
# записать в лог, что запуск прерван, и завершить скрипт с кодом 128 + номер сигнала.
# shellcheck disable=SC2329  # функция вызывается только из trap, shellcheck этого не видит
on_signal() {
    # TODO: реализуйте функцию. Имя сигнала (INT или TERM) приходит первым аргументом.
    not_implemented "${FUNCNAME[0]} (сигнал: $1)"
}

# Запускает "${train_cmd[@]}" так, что:
#   - stdout и stderr команды попадают и в $log_file, и на экран;
#   - в начало лога записаны время старта, хост и сама команда;
#   - в конец лога записаны код возврата и длительность запуска в секундах;
#   - функция возвращает код возврата самой команды (а не tee или другой части конвейера).
run_and_log() {
    # TODO: реализуйте функцию.
    not_implemented "${FUNCNAME[0]} (команда: ${train_cmd[*]})"
}

main() {
    parse_args "$@"

    mkdir -p "$log_dir"
    log "имя запуска: $run_name"
    log "лог: $log_file"

    acquire_lock

    trap 'on_signal INT' INT
    trap 'on_signal TERM' TERM

    # Внутри функции, вызванной слева от «||», set -e не действует: ошибки в run_and_log
    # нужно обрабатывать явно.
    local status=0
    run_and_log || status=$?

    if (( status == 0 )); then
        log "запуск завершён успешно"
    else
        log "запуск завершён с ошибкой, код возврата: $status"
    fi
    exit "$status"
}

main "$@"
