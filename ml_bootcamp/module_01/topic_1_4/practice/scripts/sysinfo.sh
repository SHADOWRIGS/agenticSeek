#!/usr/bin/env bash
# Сводка о состоянии ML-сервера: CPU, память, диск, GPU и самые «тяжёлые» процессы.
# Образец стиля для остальных скриптов практики.
set -euo pipefail

readonly SCRIPT_NAME="${0##*/}"

disk_paths=()
top_n=5
disk_warn=90
show_gpu=1
strict=0
warnings=()

usage() {
    cat <<EOF
Использование: $SCRIPT_NAME [опции]

Печатает сводку по CPU, памяти, диску, GPU и процессам.

Опции:
  -p, --path DIR       показать заполненность диска для DIR (можно указать несколько раз);
                       по умолчанию: / и домашняя папка
  -n, --top N          сколько процессов показать в рейтинге по памяти (по умолчанию $top_n)
      --disk-warn PCT  порог предупреждения о заполненности диска в процентах (по умолчанию $disk_warn)
      --no-gpu         не опрашивать GPU
      --strict         вернуть код 1, если есть предупреждения (удобно для cron и мониторинга)
  -h, --help           показать эту справку

Коды возврата: 0 — успех, 1 — есть предупреждения (только с --strict), 2 — ошибка в аргументах.
EOF
}

die() {
    printf '%s: %s\n' "$SCRIPT_NAME" "$1" >&2
    exit 2
}

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

section() {
    printf '\n== %s ==\n' "$1"
}

warn() {
    warnings+=("$1")
}

parse_args() {
    while (( $# > 0 )); do
        case "$1" in
            -p|--path)
                (( $# >= 2 )) || die "опции $1 нужен аргумент"
                disk_paths+=("$2")
                shift 2
                ;;
            -n|--top)
                (( $# >= 2 )) || die "опции $1 нужен аргумент"
                is_uint "$2" || die "$1: ожидается целое неотрицательное число, получено '$2'"
                top_n="$2"
                shift 2
                ;;
            --disk-warn)
                (( $# >= 2 )) || die "опции $1 нужен аргумент"
                if ! is_uint "$2" || (( $2 > 100 )); then
                    die "$1: ожидается число от 0 до 100, получено '$2'"
                fi
                disk_warn="$2"
                shift 2
                ;;
            --no-gpu)
                show_gpu=0
                shift
                ;;
            --strict)
                strict=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "неизвестный аргумент: $1 (см. --help)"
                ;;
        esac
    done

    if (( ${#disk_paths[@]} == 0 )); then
        disk_paths=("/" "$HOME")
    fi
}

report_host() {
    section "Система"
    printf 'Хост:      %s\n' "$(hostname)"
    printf 'Ядро:      %s\n' "$(uname -r)"
    printf 'Время:     %s\n' "$(date '+%F %T %Z')"
    printf 'Работает:  %s\n' "$(uptime -p)"
}

report_cpu() {
    section "CPU"
    local cores model load1 load5 load15 _rest
    cores="$(nproc)"
    # На части архитектур (например, ARM) строки model name в /proc/cpuinfo нет.
    model="$(awk -F': ' '/^model name/ { print $2; exit }' /proc/cpuinfo)"
    read -r load1 load5 load15 _rest < /proc/loadavg

    printf 'Модель:        %s\n' "${model:-неизвестно}"
    printf 'Ядер:          %s\n' "$cores"
    printf 'Load average:  %s (1 мин)  %s (5 мин)  %s (15 мин)\n' "$load1" "$load5" "$load15"

    # bash не умеет сравнивать дробные числа, поэтому сравнение делает awk.
    if awk -v load="$load1" -v cores="$cores" 'BEGIN { exit !(load > cores) }'; then
        warn "load average за минуту ($load1) больше числа ядер ($cores): процессы ждут CPU или диск"
    fi
}

report_memory() {
    section "Память"
    local total_kb avail_kb swap_total_kb swap_free_kb used_pct
    total_kb="$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)"
    avail_kb="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
    swap_total_kb="$(awk '/^SwapTotal:/ { print $2 }' /proc/meminfo)"
    swap_free_kb="$(awk '/^SwapFree:/ { print $2 }' /proc/meminfo)"
    used_pct=$(( (total_kb - avail_kb) * 100 / total_kb ))

    awk -v total="$total_kb" -v avail="$avail_kb" -v pct="$used_pct" 'BEGIN {
        gib = 1024 * 1024
        printf "RAM:   всего %.1f GiB, доступно %.1f GiB, занято %d%%\n", total / gib, avail / gib, pct
    }'
    awk -v total="$swap_total_kb" -v free="$swap_free_kb" 'BEGIN {
        gib = 1024 * 1024
        if (total == 0) { print "Swap:  не настроен"; exit }
        printf "Swap:  всего %.1f GiB, используется %.1f GiB\n", total / gib, (total - free) / gib
    }'

    if (( used_pct >= 90 )); then
        warn "занято ${used_pct}% оперативной памяти: новый процесс может быть убит OOM killer"
    fi
}

report_disk() {
    section "Диск"
    local path target pct
    local -A seen_targets=()

    # Заголовок выровнен вручную: printf считает ширину в байтах, а кириллица в UTF-8 двухбайтовая.
    echo "Точка монтирования             Размер   Занято Свободно Занято%"
    for path in "${disk_paths[@]}"; do
        if [[ ! -e "$path" ]]; then
            warn "путь не существует: $path"
            continue
        fi
        target="$(df --output=target "$path" | tail -n 1)"
        # Несколько путей могут лежать на одной файловой системе: показываем её один раз.
        if [[ -n "${seen_targets[$target]:-}" ]]; then
            continue
        fi
        seen_targets[$target]=1

        df -h --output=target,size,used,avail,pcent "$path" | tail -n 1 |
            awk '{ printf "%-28s %8s %8s %8s %7s\n", $1, $2, $3, $4, $5 }'

        pct="$(df --output=pcent "$path" | tail -n 1 | tr -dc '0-9')"
        if (( pct >= disk_warn )); then
            warn "диск $target заполнен на ${pct}% (порог ${disk_warn}%)"
        fi
    done
}

report_gpu() {
    (( show_gpu )) || return 0
    section "GPU"

    if ! command -v nvidia-smi > /dev/null 2>&1; then
        echo "nvidia-smi не найден: GPU NVIDIA нет или драйвер не установлен"
        return 0
    fi

    local gpus apps
    if ! gpus="$(nvidia-smi \
        --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>&1)"; then
        echo "nvidia-smi завершился с ошибкой:"
        echo "$gpus"
        warn "nvidia-smi установлен, но не смог опросить GPU (проверьте драйвер)"
        return 0
    fi

    awk -F', ' '{
        mem_pct = ($5 > 0) ? 100 * $4 / $5 : 0
        printf "GPU %s  %-26s загрузка %3s%%  память %6s / %6s MiB (%.0f%%)  %s C\n",
               $1, $2, $3, $4, $5, mem_pct, $6
    }' <<< "$gpus"

    apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2> /dev/null || true)"
    if [[ -z "$apps" ]]; then
        echo "Процессов на GPU нет"
    else
        echo "Процессы на GPU (PID, имя, память в MiB):"
        awk -F', ' '{ printf "  %-8s %-40s %6s\n", $1, $2, $3 }' <<< "$apps"
    fi
}

report_top() {
    (( top_n > 0 )) || return 0
    section "Процессы: топ-$top_n по памяти"
    # Строки отбирает awk, а не head: head закрыл бы канал раньше времени, ps получил бы SIGPIPE,
    # и из-за pipefail весь скрипт завершился бы с ошибкой.
    ps -eo pid,user,%cpu,%mem,rss,etime,comm --sort=-%mem | awk -v n="$top_n" 'NR <= n + 1'
}

report_warnings() {
    section "Предупреждения"
    if (( ${#warnings[@]} == 0 )); then
        echo "нет"
        return 0
    fi
    local message
    for message in "${warnings[@]}"; do
        printf -- '- %s\n' "$message"
    done
}

main() {
    parse_args "$@"

    report_host
    report_cpu
    report_memory
    report_disk
    report_gpu
    report_top
    report_warnings

    if (( strict && ${#warnings[@]} > 0 )); then
        exit 1
    fi
}

main "$@"
