# BASH-СКРИПТЫ АВТОМАТИЗАЦИИ БЕЗОПАСНОСТИ

> 25 готовых скриптов для защитника: мониторинг, hardening, реагирование, утилиты.
> Все скрипты — **оборонительные**. Копируйте, адаптируйте, внедряйте.

---

## СОДЕРЖАНИЕ

### Мониторинг (8 скриптов)
1. [system_monitor.sh](#1-system_monitorsh) — Мониторинг системы
2. [login_monitor.sh](#2-login_monitorsh) — Мониторинг входов
3. [file_integrity.sh](#3-file_integritysh) — Контроль целостности файлов
4. [network_baseline.sh](#4-network_baselinesh) — Baseline сетевых сервисов
5. [process_watch.sh](#5-process_watchsh) — Мониторинг процессов
6. [cron_audit.sh](#6-cron_auditsh) — Аудит cron-задач
7. [dns_monitor.sh](#7-dns_monitorsh) — Мониторинг DNS
8. [user_audit.sh](#8-user_auditsh) — Аудит пользователей

### Hardening (8 скриптов)
9. [ssh_hardening.sh](#9-ssh_hardeningsh) — Усиление SSH
10. [firewall_setup.sh](#10-firewall_setupsh) — Настройка файрвола
11. [service_hardening.sh](#11-service_hardeningsh) — Отключение лишних сервисов
12. [password_policy.sh](#12-password_policysh) — Парольная политика
13. [filesystem_hardening.sh](#13-filesystem_hardeningsh) — Hardening ФС
14. [kernel_hardening.sh](#14-kernel_hardeningsh) — Hardening ядра
15. [log_hardening.sh](#15-log_hardeningsh) — Hardening логирования
16. [quick_cis_check.sh](#16-quick_cis_checksh) — Быстрая проверка CIS

### Реагирование (5 скриптов)
17. [incident_collector.sh](#17-incident_collectorsh) — Сбор данных при инциденте
18. [block_ip.sh](#18-block_ipsh) — Блокировка IP
19. [quarantine_file.sh](#19-quarantine_filesh) — Карантин файла
20. [kill_suspicious.sh](#20-kill_suspicioussh) — Остановка процессов
21. [emergency_lockdown.sh](#21-emergency_lockdownsh) — Экстренная блокировка

### Утилиты (4 скрипта)
22. [log_parser.sh](#22-log_parsersh) — Парсер логов
23. [ioc_search.sh](#23-ioc_searchsh) — Поиск IoC
24. [backup_verify.sh](#24-backup_verifysh) — Проверка бэкапов
25. [security_report.sh](#25-security_reportsh) — Ежедневный отчёт

---

# ЧАСТЬ 1: МОНИТОРИНГ

---

## 1. system_monitor.sh

**Назначение:** Мониторинг ключевых метрик системы (CPU, RAM, диск, процессы, сетевые соединения). При превышении порогов — алерт в syslog и на экран. Формат вывода совместим с SIEM.

**Когда использовать:** Запускать по cron каждые 5 минут или как демон.

```bash
#!/bin/bash
#======================================================================
# system_monitor.sh — Мониторинг системных ресурсов для SOC
# Использование: ./system_monitor.sh [-t cpu_threshold] [-m mem_threshold]
#                [-d disk_threshold] [-l logfile] [-s] [-h]
#======================================================================

set -euo pipefail

# --- Цвета ---
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

# --- Значения по умолчанию ---
CPU_THRESHOLD=80
MEM_THRESHOLD=85
DISK_THRESHOLD=90
LOGFILE="/var/log/security/system_monitor.log"
SYSLOG_ENABLE=false
HOSTNAME_TAG=$(hostname -s)

usage() {
    echo "Использование: $0 [опции]"
    echo "  -t NUM   Порог CPU в % (по умолчанию: $CPU_THRESHOLD)"
    echo "  -m NUM   Порог RAM в % (по умолчанию: $MEM_THRESHOLD)"
    echo "  -d NUM   Порог диска в % (по умолчанию: $DISK_THRESHOLD)"
    echo "  -l FILE  Файл лога (по умолчанию: $LOGFILE)"
    echo "  -s       Отправлять алерты в syslog"
    echo "  -h       Показать эту справку"
    exit 0
}

while getopts "t:m:d:l:sh" opt; do
    case $opt in
        t) CPU_THRESHOLD="$OPTARG" ;;
        m) MEM_THRESHOLD="$OPTARG" ;;
        d) DISK_THRESHOLD="$OPTARG" ;;
        l) LOGFILE="$OPTARG" ;;
        s) SYSLOG_ENABLE=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

mkdir -p "$(dirname "$LOGFILE")"

log_event() {
    local level="$1" msg="$2"
    local timestamp
    timestamp=$(date '+%Y-%m-%dT%H:%M:%S%z')
    local line="timestamp=$timestamp host=$HOSTNAME_TAG level=$level msg=\"$msg\""
    echo "$line" >> "$LOGFILE"
    if [[ "$level" == "ALERT" ]]; then
        echo -e "${RED}[ALERT]${NC} $msg"
        if $SYSLOG_ENABLE; then
            logger -t "system_monitor" -p auth.warning "$msg"
        fi
    elif [[ "$level" == "WARN" ]]; then
        echo -e "${YELLOW}[WARN]${NC} $msg"
    else
        echo -e "${GREEN}[OK]${NC} $msg"
    fi
}

# --- CPU ---
check_cpu() {
    local cpu_idle
    cpu_idle=$(top -bn1 | grep "Cpu(s)" | awk '{print $8}' | cut -d'.' -f1)
    local cpu_used=$((100 - cpu_idle))
    if (( cpu_used >= CPU_THRESHOLD )); then
        log_event "ALERT" "CPU usage=${cpu_used}% threshold=${CPU_THRESHOLD}%"
    else
        log_event "INFO" "CPU usage=${cpu_used}% threshold=${CPU_THRESHOLD}%"
    fi
}

# --- RAM ---
check_memory() {
    local mem_total mem_avail mem_used_pct
    mem_total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    mem_avail=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    mem_used_pct=$(( (mem_total - mem_avail) * 100 / mem_total ))
    if (( mem_used_pct >= MEM_THRESHOLD )); then
        log_event "ALERT" "Memory usage=${mem_used_pct}% threshold=${MEM_THRESHOLD}%"
    else
        log_event "INFO" "Memory usage=${mem_used_pct}% threshold=${MEM_THRESHOLD}%"
    fi
}

# --- Диск ---
check_disk() {
    while read -r line; do
        local usage mount
        usage=$(echo "$line" | awk '{print $5}' | tr -d '%')
        mount=$(echo "$line" | awk '{print $6}')
        if (( usage >= DISK_THRESHOLD )); then
            log_event "ALERT" "Disk mount=$mount usage=${usage}% threshold=${DISK_THRESHOLD}%"
        fi
    done < <(df -h --output=pcent,target -x tmpfs -x devtmpfs | tail -n +2)
}

# --- Процессы ---
check_processes() {
    local proc_count
    proc_count=$(ps aux --no-headers | wc -l)
    log_event "INFO" "process_count=$proc_count"
    # Топ-5 по CPU
    local top_cpu
    top_cpu=$(ps aux --sort=-%cpu --no-headers | head -5 | awk '{printf "%s(%s%%CPU) ",$11,$3}')
    log_event "INFO" "top_cpu_processes=$top_cpu"
}

# --- Сетевые соединения ---
check_network() {
    local estab_count listen_count
    estab_count=$(ss -tun state established 2>/dev/null | tail -n +2 | wc -l)
    listen_count=$(ss -tln 2>/dev/null | tail -n +2 | wc -l)
    log_event "INFO" "established_connections=$estab_count listening_ports=$listen_count"

    # Обнаружение подозрительных портов (>1024 и не в типичном списке)
    local suspicious
    suspicious=$(ss -tlnp 2>/dev/null | awk 'NR>1 {print $4}' | grep -vE ':(22|80|443|53|3306|5432|8080|8443|25|587|993|995|111|2049)$' | head -5)
    if [[ -n "$suspicious" ]]; then
        log_event "WARN" "unusual_listening_ports=$suspicious"
    fi
}

echo -e "${CYAN}=== Мониторинг системы: $(date) ===${NC}"
check_cpu
check_memory
check_disk
check_processes
check_network
echo -e "${CYAN}=== Завершено ===${NC}"
```

**Пример вывода:**
```
=== Мониторинг системы: 2026-11-12 10:00:00 ===
[OK] CPU usage=23% threshold=80%
[OK] Memory usage=64% threshold=85%
[ALERT] Disk mount=/var usage=92% threshold=90%
[OK] process_count=187
[OK] established_connections=34 listening_ports=8
[WARN] unusual_listening_ports=0.0.0.0:4444
=== Завершено ===
```

**Интеграция:** Добавьте в cron: `*/5 * * * * /opt/scripts/system_monitor.sh -s >> /dev/null 2>&1`. Логи автоматически собираются SIEM (Wazuh/ELK) из файла.

---

## 2. login_monitor.sh

**Назначение:** Обнаружение brute-force атак, входов root, SSH с неизвестных IP. Парсинг auth.log в реальном времени или по расписанию.

```bash
#!/bin/bash
#======================================================================
# login_monitor.sh — Мониторинг входов и обнаружение brute-force
# Использование: ./login_monitor.sh [-f auth_log] [-n max_fails] [-w minutes] [-h]
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

AUTH_LOG="/var/log/auth.log"
MAX_FAILS=5
WINDOW_MIN=5
KNOWN_IPS_FILE="/etc/security/known_ssh_ips.txt"
LOGFILE="/var/log/security/login_monitor.log"

usage() {
    echo "Использование: $0 [опции]"
    echo "  -f FILE  Путь к auth.log (по умолчанию: $AUTH_LOG)"
    echo "  -n NUM   Макс. неудачных попыток (по умолчанию: $MAX_FAILS)"
    echo "  -w MIN   Окно времени в минутах (по умолчанию: $WINDOW_MIN)"
    echo "  -h       Показать справку"
    exit 0
}

while getopts "f:n:w:h" opt; do
    case $opt in
        f) AUTH_LOG="$OPTARG" ;;
        n) MAX_FAILS="$OPTARG" ;;
        w) WINDOW_MIN="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

mkdir -p "$(dirname "$LOGFILE")"
touch "$KNOWN_IPS_FILE" 2>/dev/null || true

log_alert() {
    local msg="$1"
    local timestamp
    timestamp=$(date '+%Y-%m-%dT%H:%M:%S%z')
    echo "$timestamp ALERT $msg" >> "$LOGFILE"
    echo -e "${RED}[ALERT]${NC} $msg"
    logger -t "login_monitor" -p auth.warning "$msg"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# --- Проверка: файл auth.log существует ---
if [[ ! -f "$AUTH_LOG" ]]; then
    # Пробуем journalctl для systemd
    AUTH_LOG=""
    if command -v journalctl &>/dev/null; then
        log_info "auth.log не найден, используем journalctl"
    else
        echo -e "${RED}[ERROR]${NC} auth.log не найден и journalctl недоступен"
        exit 1
    fi
fi

echo -e "\n=== Мониторинг входов: $(date) ==="

# --- 1. Обнаружение brute-force ---
echo -e "\n${YELLOW}[1] Проверка brute-force (>${MAX_FAILS} неудач за ${WINDOW_MIN} мин)${NC}"

declare -A fail_counts
cutoff=$(date -d "-${WINDOW_MIN} minutes" '+%b %e %H:%M' 2>/dev/null || date '+%b %e %H:%M')

if [[ -n "$AUTH_LOG" ]]; then
    while IFS= read -r line; do
        ip=$(echo "$line" | grep -oP 'from \K[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' || true)
        if [[ -n "$ip" ]]; then
            fail_counts[$ip]=$(( ${fail_counts[$ip]:-0} + 1 ))
        fi
    done < <(grep "Failed password\|authentication failure" "$AUTH_LOG" 2>/dev/null | tail -200)
else
    while IFS= read -r line; do
        ip=$(echo "$line" | grep -oP 'from \K[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' || true)
        if [[ -n "$ip" ]]; then
            fail_counts[$ip]=$(( ${fail_counts[$ip]:-0} + 1 ))
        fi
    done < <(journalctl -u sshd --since "${WINDOW_MIN} min ago" --no-pager 2>/dev/null | grep "Failed password\|authentication failure")
fi

brute_detected=false
for ip in "${!fail_counts[@]}"; do
    count=${fail_counts[$ip]}
    if (( count >= MAX_FAILS )); then
        log_alert "BRUTE_FORCE ip=$ip failed_attempts=$count window=${WINDOW_MIN}min"
        brute_detected=true
    fi
done
if ! $brute_detected; then
    log_info "Brute-force не обнаружен"
fi

# --- 2. Входы root ---
echo -e "\n${YELLOW}[2] Проверка входов root${NC}"

root_logins=0
if [[ -n "$AUTH_LOG" ]]; then
    root_logins=$(grep -c "session opened.*root\|Accepted.*root" "$AUTH_LOG" 2>/dev/null | tail -1 || echo 0)
else
    root_logins=$(journalctl -u sshd --since "1 hour ago" --no-pager 2>/dev/null | grep -c "Accepted.*root" || echo 0)
fi

if (( root_logins > 0 )); then
    log_alert "ROOT_LOGIN count=$root_logins in_last_hour"
else
    log_info "Входов root не обнаружено"
fi

# --- 3. SSH с неизвестных IP ---
echo -e "\n${YELLOW}[3] SSH-входы с неизвестных IP${NC}"

if [[ -n "$AUTH_LOG" ]]; then
    ssh_ips=$(grep "Accepted" "$AUTH_LOG" 2>/dev/null | grep -oP 'from \K[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | sort -u)
else
    ssh_ips=$(journalctl -u sshd --since "1 hour ago" --no-pager 2>/dev/null | grep "Accepted" | grep -oP 'from \K[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | sort -u)
fi

new_found=false
for ip in $ssh_ips; do
    if ! grep -qF "$ip" "$KNOWN_IPS_FILE" 2>/dev/null; then
        log_alert "NEW_SSH_IP ip=$ip (не в known_ssh_ips.txt)"
        new_found=true
    fi
done
if ! $new_found; then
    log_info "Все SSH-IP известны"
fi

# --- 4. Текущие SSH-сессии ---
echo -e "\n${YELLOW}[4] Активные SSH-сессии${NC}"
who_output=$(who 2>/dev/null || true)
if [[ -n "$who_output" ]]; then
    echo "$who_output"
    session_count=$(echo "$who_output" | wc -l)
    log_info "Активных сессий: $session_count"
else
    log_info "Активных сессий нет"
fi

# --- 5. Последние 10 успешных входов ---
echo -e "\n${YELLOW}[5] Последние успешные входы${NC}"
if command -v last &>/dev/null; then
    last -10 -a 2>/dev/null | head -10
fi

echo -e "\n=== Завершено ==="
```

**Интеграция с Wazuh:** Скрипт пишет в syslog через `logger`, Wazuh подхватит автоматически. Для cron: `*/5 * * * * /opt/scripts/login_monitor.sh >/dev/null 2>&1`

---

## 3. file_integrity.sh

**Назначение:** Простой File Integrity Monitor (FIM). Создаёт baseline хешей критических файлов, при каждом запуске сверяет. Обнаруживает модификацию конфигов, бэкдоры в crontab, подмену бинарников.

```bash
#!/bin/bash
#======================================================================
# file_integrity.sh — File Integrity Monitor
# Использование: ./file_integrity.sh [-b] [-c] [-d baseline_dir] [-h]
#   -b  Создать baseline
#   -c  Проверить целостность (по умолчанию)
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

BASELINE_DIR="/var/lib/fim"
BASELINE_FILE="$BASELINE_DIR/baseline.sha256"
LOGFILE="/var/log/security/fim.log"
MODE="check"

# Критические файлы для мониторинга
WATCH_FILES=(
    "/etc/passwd"
    "/etc/shadow"
    "/etc/group"
    "/etc/sudoers"
    "/etc/ssh/sshd_config"
    "/etc/hosts"
    "/etc/hosts.allow"
    "/etc/hosts.deny"
    "/etc/crontab"
    "/etc/resolv.conf"
    "/etc/pam.d/common-auth"
    "/etc/pam.d/sshd"
    "/etc/ld.so.preload"
    "/etc/profile"
    "/etc/bashrc"
    "/etc/environment"
    "/root/.bashrc"
    "/root/.bash_profile"
    "/root/.ssh/authorized_keys"
)

# Критические каталоги (все файлы внутри)
WATCH_DIRS=(
    "/etc/cron.d"
    "/etc/cron.daily"
    "/etc/cron.hourly"
    "/etc/systemd/system"
    "/etc/init.d"
    "/etc/sudoers.d"
)

usage() {
    echo "Использование: $0 [опции]"
    echo "  -b        Создать baseline хешей"
    echo "  -c        Проверить целостность (по умолчанию)"
    echo "  -d DIR    Каталог для baseline (по умолчанию: $BASELINE_DIR)"
    echo "  -h        Показать справку"
    exit 0
}

while getopts "bcd:h" opt; do
    case $opt in
        b) MODE="baseline" ;;
        c) MODE="check" ;;
        d) BASELINE_DIR="$OPTARG"; BASELINE_FILE="$BASELINE_DIR/baseline.sha256" ;;
        h) usage ;;
        *) usage ;;
    esac
done

mkdir -p "$BASELINE_DIR"
mkdir -p "$(dirname "$LOGFILE")"

log_event() {
    local level="$1" msg="$2"
    local timestamp
    timestamp=$(date '+%Y-%m-%dT%H:%M:%S%z')
    echo "$timestamp [$level] $msg" >> "$LOGFILE"
    case "$level" in
        ALERT) echo -e "${RED}[ALERT]${NC} $msg" ;;
        WARN)  echo -e "${YELLOW}[WARN]${NC} $msg" ;;
        INFO)  echo -e "${GREEN}[INFO]${NC} $msg" ;;
    esac
}

# Сбор всех файлов для мониторинга
collect_files() {
    local all_files=()
    for f in "${WATCH_FILES[@]}"; do
        [[ -f "$f" ]] && all_files+=("$f")
    done
    for d in "${WATCH_DIRS[@]}"; do
        if [[ -d "$d" ]]; then
            while IFS= read -r -d '' f; do
                all_files+=("$f")
            done < <(find "$d" -type f -print0 2>/dev/null)
        fi
    done
    # Пользовательские crontab-файлы
    if [[ -d /var/spool/cron/crontabs ]]; then
        while IFS= read -r -d '' f; do
            all_files+=("$f")
        done < <(find /var/spool/cron/crontabs -type f -print0 2>/dev/null)
    fi
    printf '%s\n' "${all_files[@]}" | sort -u
}

# --- Создание baseline ---
create_baseline() {
    echo -e "${CYAN}=== Создание baseline ===${NC}"
    local count=0
    > "$BASELINE_FILE"
    while IFS= read -r file; do
        local hash
        hash=$(sha256sum "$file" 2>/dev/null) || continue
        echo "$hash" >> "$BASELINE_FILE"
        count=$((count + 1))
    done < <(collect_files)
    chmod 600 "$BASELINE_FILE"
    log_event "INFO" "Baseline создан: $count файлов в $BASELINE_FILE"
    echo -e "${GREEN}Baseline создан: $count файлов${NC}"
}

# --- Проверка целостности ---
check_integrity() {
    echo -e "${CYAN}=== Проверка целостности: $(date) ===${NC}"

    if [[ ! -f "$BASELINE_FILE" ]]; then
        echo -e "${RED}[ERROR]${NC} Baseline не найден. Сначала запустите: $0 -b"
        exit 1
    fi

    local changed=0 missing=0 new_files=0 ok=0

    # Проверка существующих файлов из baseline
    while IFS= read -r line; do
        local stored_hash stored_path
        stored_hash=$(echo "$line" | awk '{print $1}')
        stored_path=$(echo "$line" | awk '{print $2}')

        if [[ ! -f "$stored_path" ]]; then
            log_event "ALERT" "FILE_DELETED path=$stored_path"
            missing=$((missing + 1))
            continue
        fi

        local current_hash
        current_hash=$(sha256sum "$stored_path" 2>/dev/null | awk '{print $1}')
        if [[ "$current_hash" != "$stored_hash" ]]; then
            log_event "ALERT" "FILE_MODIFIED path=$stored_path old_hash=${stored_hash:0:16}... new_hash=${current_hash:0:16}..."
            changed=$((changed + 1))
        else
            ok=$((ok + 1))
        fi
    done < "$BASELINE_FILE"

    # Проверка на новые файлы
    declare -A baseline_paths
    while IFS= read -r line; do
        local p
        p=$(echo "$line" | awk '{print $2}')
        baseline_paths["$p"]=1
    done < "$BASELINE_FILE"

    while IFS= read -r file; do
        if [[ -z "${baseline_paths[$file]+x}" ]]; then
            log_event "WARN" "NEW_FILE path=$file"
            new_files=$((new_files + 1))
        fi
    done < <(collect_files)

    # Итог
    echo ""
    echo -e "${CYAN}--- Результат ---${NC}"
    echo -e "Без изменений: ${GREEN}$ok${NC}"
    echo -e "Изменено:      ${RED}$changed${NC}"
    echo -e "Удалено:       ${RED}$missing${NC}"
    echo -e "Новых файлов:  ${YELLOW}$new_files${NC}"

    if (( changed > 0 || missing > 0 )); then
        log_event "ALERT" "INTEGRITY_CHECK changed=$changed missing=$missing new=$new_files"
        logger -t "fim" -p auth.crit "FIM: $changed files modified, $missing deleted"
        return 1
    fi
    log_event "INFO" "INTEGRITY_CHECK ok=$ok changed=0 missing=0 new=$new_files"
    return 0
}

case "$MODE" in
    baseline) create_baseline ;;
    check)    check_integrity ;;
esac
```

**Пример вывода:**
```
=== Проверка целостности: 2026-11-12 10:15:00 ===
[ALERT] FILE_MODIFIED path=/etc/passwd old_hash=a1b2c3d4... new_hash=e5f6g7h8...
[WARN] NEW_FILE path=/etc/cron.d/backdoor_task
--- Результат ---
Без изменений: 42
Изменено:      1
Удалено:       0
Новых файлов:  1
```

---

## 4. network_baseline.sh

**Назначение:** Создание и сравнение baseline открытых портов и сервисов. Обнаруживает появление новых listening сервисов — потенциальных бэкдоров.

```bash
#!/bin/bash
#======================================================================
# network_baseline.sh — Baseline сетевых сервисов
# Использование: ./network_baseline.sh [-b] [-c] [-f baseline_file] [-h]
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

BASELINE_FILE="/var/lib/fim/network_baseline.txt"
LOGFILE="/var/log/security/network_baseline.log"
MODE="check"

usage() {
    echo "Использование: $0 [опции]"
    echo "  -b       Создать baseline"
    echo "  -c       Сравнить с baseline (по умолчанию)"
    echo "  -f FILE  Файл baseline"
    echo "  -h       Справка"
    exit 0
}

while getopts "bcf:h" opt; do
    case $opt in
        b) MODE="baseline" ;;
        c) MODE="check" ;;
        f) BASELINE_FILE="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

mkdir -p "$(dirname "$BASELINE_FILE")" "$(dirname "$LOGFILE")"

get_current_state() {
    # Формат: протокол  адрес:порт  процесс
    ss -tlnp 2>/dev/null | awk 'NR>1 {
        split($4, a, ":");
        port = a[length(a)];
        proc = $6;
        gsub(/.*"/, "", proc); gsub(/".*/, "", proc);
        printf "TCP\t%s\t%s\t%s\n", $4, port, proc
    }' | sort -t$'\t' -k3 -n

    ss -ulnp 2>/dev/null | awk 'NR>1 {
        split($4, a, ":");
        port = a[length(a)];
        proc = $6;
        gsub(/.*"/, "", proc); gsub(/".*/, "", proc);
        printf "UDP\t%s\t%s\t%s\n", $4, port, proc
    }' | sort -t$'\t' -k3 -n
}

create_baseline() {
    echo -e "${CYAN}=== Создание network baseline ===${NC}"
    get_current_state > "$BASELINE_FILE"
    local count
    count=$(wc -l < "$BASELINE_FILE")
    chmod 600 "$BASELINE_FILE"
    echo -e "${GREEN}Baseline сохранён: $count сервисов в $BASELINE_FILE${NC}"
    cat "$BASELINE_FILE"
}

check_baseline() {
    echo -e "${CYAN}=== Сравнение с network baseline: $(date) ===${NC}"

    if [[ ! -f "$BASELINE_FILE" ]]; then
        echo -e "${RED}[ERROR]${NC} Baseline не найден. Запустите: $0 -b"
        exit 1
    fi

    local tmpfile
    tmpfile=$(mktemp)
    get_current_state > "$tmpfile"

    # Новые сервисы (есть сейчас, нет в baseline)
    local new_services
    new_services=$(comm -13 <(sort "$BASELINE_FILE") <(sort "$tmpfile") || true)

    # Исчезнувшие сервисы
    local gone_services
    gone_services=$(comm -23 <(sort "$BASELINE_FILE") <(sort "$tmpfile") || true)

    local alerts=0

    if [[ -n "$new_services" ]]; then
        echo -e "\n${RED}[ALERT] НОВЫЕ сервисы (возможный бэкдор):${NC}"
        echo "$new_services" | while IFS= read -r line; do
            echo -e "  ${RED}+ $line${NC}"
            local timestamp
            timestamp=$(date '+%Y-%m-%dT%H:%M:%S%z')
            echo "$timestamp ALERT NEW_SERVICE $line" >> "$LOGFILE"
        done
        alerts=$((alerts + 1))
        logger -t "network_baseline" -p auth.warning "New listening services detected"
    fi

    if [[ -n "$gone_services" ]]; then
        echo -e "\n${YELLOW}[WARN] Исчезнувшие сервисы:${NC}"
        echo "$gone_services" | while IFS= read -r line; do
            echo -e "  ${YELLOW}- $line${NC}"
        done
    fi

    if [[ -z "$new_services" && -z "$gone_services" ]]; then
        echo -e "${GREEN}Изменений нет. Все сервисы соответствуют baseline.${NC}"
    fi

    rm -f "$tmpfile"

    echo -e "\n${CYAN}--- Текущие listening сервисы ---${NC}"
    ss -tlnp 2>/dev/null | head -20

    return $alerts
}

case "$MODE" in
    baseline) create_baseline ;;
    check)    check_baseline ;;
esac
```

---

## 5. process_watch.sh

**Назначение:** Обнаружение подозрительных процессов: не из whitelist, аномальные parent-child связи (shell от веб-сервера), процессы без бинарника на диске (fileless).

```bash
#!/bin/bash
#======================================================================
# process_watch.sh — Мониторинг подозрительных процессов
# Использование: ./process_watch.sh [-w whitelist] [-h]
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

WHITELIST_FILE="/etc/security/process_whitelist.txt"
LOGFILE="/var/log/security/process_watch.log"

# Подозрительные parent-child комбинации
# Формат: parent_name:child_name
SUSPICIOUS_PAIRS=(
    "apache2:sh"
    "apache2:bash"
    "nginx:sh"
    "nginx:bash"
    "httpd:sh"
    "httpd:bash"
    "java:sh"
    "java:bash"
    "php-fpm:sh"
    "php-fpm:bash"
    "node:sh"
    "python3:sh"
    "mysqld:sh"
    "postgres:sh"
    "sshd:python"
    "sshd:perl"
    "sshd:nc"
    "sshd:ncat"
)

# Подозрительные имена процессов
SUSPICIOUS_NAMES=(
    "nc" "ncat" "netcat" "socat"
    "msfconsole" "msfvenom" "meterpreter"
    "mimikatz" "lazagne" "rubeus"
    "chisel" "ligolo" "frp" "rathole"
    "xmrig" "minerd" "cpuminer"
    "cryptonight" "stratum"
)

usage() {
    echo "Использование: $0 [опции]"
    echo "  -w FILE  Whitelist процессов (по умолчанию: $WHITELIST_FILE)"
    echo "  -h       Справка"
    exit 0
}

while getopts "w:h" opt; do
    case $opt in
        w) WHITELIST_FILE="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

mkdir -p "$(dirname "$LOGFILE")"

log_alert() {
    local msg="$1"
    local timestamp
    timestamp=$(date '+%Y-%m-%dT%H:%M:%S%z')
    echo "$timestamp ALERT $msg" >> "$LOGFILE"
    echo -e "${RED}[ALERT]${NC} $msg"
    logger -t "process_watch" -p auth.warning "$msg"
}

echo -e "${CYAN}=== Мониторинг процессов: $(date) ===${NC}"

alerts=0

# --- 1. Подозрительные имена процессов ---
echo -e "\n${YELLOW}[1] Проверка подозрительных имён процессов${NC}"
for name in "${SUSPICIOUS_NAMES[@]}"; do
    pids=$(pgrep -x "$name" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        for pid in $pids; do
            local cmdline user
            cmdline=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ' || echo "N/A")
            user=$(ps -o user= -p "$pid" 2>/dev/null || echo "?")
            log_alert "SUSPICIOUS_PROCESS name=$name pid=$pid user=$user cmd=$cmdline"
            alerts=$((alerts + 1))
        done
    fi
done
if (( alerts == 0 )); then
    echo -e "${GREEN}Подозрительных процессов не найдено${NC}"
fi

# --- 2. Подозрительные parent-child ---
echo -e "\n${YELLOW}[2] Проверка parent-child связей${NC}"
pc_alerts=0
ps -eo pid,ppid,comm --no-headers 2>/dev/null | while read -r pid ppid comm; do
    parent_comm=$(ps -o comm= -p "$ppid" 2>/dev/null || echo "")
    for pair in "${SUSPICIOUS_PAIRS[@]}"; do
        p_name="${pair%%:*}"
        c_name="${pair##*:}"
        if [[ "$parent_comm" == "$p_name" && "$comm" == "$c_name" ]]; then
            user=$(ps -o user= -p "$pid" 2>/dev/null || echo "?")
            cmdline=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ' || echo "N/A")
            log_alert "SUSPICIOUS_PARENT_CHILD parent=$p_name(ppid=$ppid) child=$c_name(pid=$pid) user=$user"
            pc_alerts=$((pc_alerts + 1))
        fi
    done
done
if (( pc_alerts == 0 )); then
    echo -e "${GREEN}Аномальных parent-child не найдено${NC}"
fi

# --- 3. Процессы без бинарника на диске (fileless / удалённый бинарник) ---
echo -e "\n${YELLOW}[3] Процессы без бинарника на диске${NC}"
fl_alerts=0
for pid_dir in /proc/[0-9]*; do
    pid=$(basename "$pid_dir")
    exe_link=$(readlink "$pid_dir/exe" 2>/dev/null || continue)
    if [[ "$exe_link" == *"(deleted)"* ]]; then
        comm=$(cat "$pid_dir/comm" 2>/dev/null || echo "?")
        user=$(ps -o user= -p "$pid" 2>/dev/null || echo "?")
        log_alert "DELETED_BINARY pid=$pid comm=$comm user=$user exe=$exe_link"
        fl_alerts=$((fl_alerts + 1))
    fi
done
if (( fl_alerts == 0 )); then
    echo -e "${GREEN}Процессов с удалённым бинарником нет${NC}"
fi

# --- 4. Процессы с высоким потреблением (потенциальный майнер) ---
echo -e "\n${YELLOW}[4] Процессы с CPU > 80% (потенциальный майнер)${NC}"
high_cpu=$(ps aux --no-headers | awk '$3 > 80 {print $2, $1, $3, $11}')
if [[ -n "$high_cpu" ]]; then
    echo "$high_cpu" | while read -r pid user cpu cmd; do
        log_alert "HIGH_CPU pid=$pid user=$user cpu=${cpu}% cmd=$cmd"
    done
else
    echo -e "${GREEN}Процессов с CPU > 80% нет${NC}"
fi

echo -e "\n=== Завершено ==="
```

---

## 6. cron_audit.sh

**Назначение:** Аудит всех cron-задач (системных и пользовательских). Обнаружение подозрительных команд: wget, curl, base64, nc, reverse shells.

```bash
#!/bin/bash
#======================================================================
# cron_audit.sh — Аудит cron-задач
# Использование: ./cron_audit.sh [-h]
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

LOGFILE="/var/log/security/cron_audit.log"
mkdir -p "$(dirname "$LOGFILE")"

# Подозрительные паттерны в cron-задачах
SUSPICIOUS_PATTERNS=(
    'wget\s'
    'curl\s.*\|.*sh'
    'curl\s.*\|.*bash'
    'base64\s.*-d'
    '\bnc\b.*-e'
    '\bncat\b.*-e'
    '/dev/tcp/'
    'mkfifo'
    'python.*-c.*import.*socket'
    'perl.*-e.*socket'
    'ruby.*-e.*TCPSocket'
    'bash\s.*-i\s.*>&'
    'chmod\s.*777'
    'chmod\s.*\+s'
    '/tmp/\.'
)

usage() {
    echo "Использование: $0 [-h]"
    echo "  -h  Справка"
    exit 0
}
[[ "${1:-}" == "-h" ]] && usage

log_alert() {
    local timestamp
    timestamp=$(date '+%Y-%m-%dT%H:%M:%S%z')
    echo "$timestamp ALERT $1" >> "$LOGFILE"
    echo -e "${RED}[ALERT]${NC} $1"
}

echo -e "${CYAN}=== Аудит cron-задач: $(date) ===${NC}"

alerts=0
total_jobs=0

# --- 1. Системный crontab ---
echo -e "\n${YELLOW}[1] /etc/crontab${NC}"
if [[ -f /etc/crontab ]]; then
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue
        [[ "$line" =~ ^[A-Z_]+= ]] && continue
        total_jobs=$((total_jobs + 1))
        echo "  $line"
        for pattern in "${SUSPICIOUS_PATTERNS[@]}"; do
            if echo "$line" | grep -qPi "$pattern"; then
                log_alert "SUSPICIOUS_CRON source=/etc/crontab pattern=$pattern job=\"$line\""
                alerts=$((alerts + 1))
                break
            fi
        done
    done < /etc/crontab
fi

# --- 2. /etc/cron.d/ ---
echo -e "\n${YELLOW}[2] /etc/cron.d/${NC}"
if [[ -d /etc/cron.d ]]; then
    for f in /etc/cron.d/*; do
        [[ -f "$f" ]] || continue
        echo -e "  ${CYAN}$f:${NC}"
        while IFS= read -r line; do
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "$line" ]] && continue
            [[ "$line" =~ ^[A-Z_]+= ]] && continue
            total_jobs=$((total_jobs + 1))
            echo "    $line"
            for pattern in "${SUSPICIOUS_PATTERNS[@]}"; do
                if echo "$line" | grep -qPi "$pattern"; then
                    log_alert "SUSPICIOUS_CRON source=$f pattern=$pattern"
                    alerts=$((alerts + 1))
                    break
                fi
            done
        done < "$f"
    done
fi

# --- 3. Пользовательские crontab ---
echo -e "\n${YELLOW}[3] Пользовательские crontab${NC}"
while IFS=: read -r username _ uid _ _ home _; do
    (( uid < 0 || uid > 65534 )) && continue
    crontab_out=$(crontab -u "$username" -l 2>/dev/null || true)
    if [[ -n "$crontab_out" ]]; then
        echo -e "  ${CYAN}Пользователь: $username${NC}"
        while IFS= read -r line; do
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "$line" ]] && continue
            total_jobs=$((total_jobs + 1))
            echo "    $line"
            for pattern in "${SUSPICIOUS_PATTERNS[@]}"; do
                if echo "$line" | grep -qPi "$pattern"; then
                    log_alert "SUSPICIOUS_CRON user=$username pattern=$pattern job=\"$line\""
                    alerts=$((alerts + 1))
                    break
                fi
            done
        done <<< "$crontab_out"
    fi
done < /etc/passwd

# --- 4. Cron-каталоги ---
echo -e "\n${YELLOW}[4] Файлы в cron-каталогах${NC}"
for d in /etc/cron.hourly /etc/cron.daily /etc/cron.weekly /etc/cron.monthly; do
    if [[ -d "$d" ]]; then
        count=$(find "$d" -type f | wc -l)
        echo "  $d: $count файлов"
        # Проверяем права (не должны быть writable для всех)
        find "$d" -type f -perm /002 2>/dev/null | while IFS= read -r f; do
            log_alert "WORLD_WRITABLE_CRON file=$f"
            alerts=$((alerts + 1))
        done
    fi
done

# --- 5. systemd-таймеры ---
echo -e "\n${YELLOW}[5] systemd-таймеры${NC}"
if command -v systemctl &>/dev/null; then
    systemctl list-timers --all --no-pager 2>/dev/null | head -20
fi

# --- Итог ---
echo -e "\n${CYAN}--- Итог ---${NC}"
echo -e "Всего cron-задач: $total_jobs"
if (( alerts > 0 )); then
    echo -e "${RED}Подозрительных: $alerts${NC}"
else
    echo -e "${GREEN}Подозрительных задач не найдено${NC}"
fi
```

---

## 7. dns_monitor.sh

**Назначение:** Мониторинг DNS-запросов для обнаружения DNS tunneling (длинные домены, высокая частота, TXT-запросы).

```bash
#!/bin/bash
#======================================================================
# dns_monitor.sh — Мониторинг DNS-запросов (обнаружение DNS tunneling)
# Использование: ./dns_monitor.sh [-i interface] [-d duration] [-t threshold] [-h]
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

INTERFACE="any"
DURATION=60
DOMAIN_LEN_THRESHOLD=50
TXT_THRESHOLD=10
LOGFILE="/var/log/security/dns_monitor.log"

usage() {
    echo "Использование: $0 [опции]"
    echo "  -i IFACE    Сетевой интерфейс (по умолчанию: any)"
    echo "  -d SEC      Длительность захвата в секундах (по умолчанию: $DURATION)"
    echo "  -t LEN      Порог длины домена (по умолчанию: $DOMAIN_LEN_THRESHOLD)"
    echo "  -h          Справка"
    exit 0
}

while getopts "i:d:t:h" opt; do
    case $opt in
        i) INTERFACE="$OPTARG" ;;
        d) DURATION="$OPTARG" ;;
        t) DOMAIN_LEN_THRESHOLD="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

mkdir -p "$(dirname "$LOGFILE")"

log_alert() {
    local timestamp
    timestamp=$(date '+%Y-%m-%dT%H:%M:%S%z')
    echo "$timestamp ALERT $1" >> "$LOGFILE"
    echo -e "${RED}[ALERT]${NC} $1"
}

echo -e "${CYAN}=== Мониторинг DNS ($DURATION сек, интерфейс: $INTERFACE) ===${NC}"

TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT

# Захват DNS-запросов
if command -v tshark &>/dev/null; then
    timeout "$DURATION" tshark -i "$INTERFACE" -f "udp port 53" \
        -T fields -e dns.qry.name -e dns.qry.type \
        -Y "dns.flags.response == 0" 2>/dev/null > "$TMPFILE" || true
elif command -v tcpdump &>/dev/null; then
    timeout "$DURATION" tcpdump -i "$INTERFACE" -nn "port 53" \
        -l 2>/dev/null | grep -oP '[\w.-]+\.\w{2,}' > "$TMPFILE" || true
else
    # Если нет сниффера — анализ /var/log/syslog (dnsmasq) или systemd-resolved
    echo -e "${YELLOW}tshark/tcpdump не найден, анализируем логи${NC}"
    if [[ -f /var/log/syslog ]]; then
        grep "dnsmasq\|resolved" /var/log/syslog 2>/dev/null | \
            grep -oP 'query\[.*?\]\s+\K[\w.-]+' | tail -1000 > "$TMPFILE"
    fi
fi

total_queries=$(wc -l < "$TMPFILE")
echo -e "Перехвачено запросов: $total_queries"

if (( total_queries == 0 )); then
    echo -e "${YELLOW}Нет данных для анализа${NC}"
    exit 0
fi

alerts=0

# --- 1. Длинные домены (признак туннелирования) ---
echo -e "\n${YELLOW}[1] Домены длиннее $DOMAIN_LEN_THRESHOLD символов${NC}"
while IFS= read -r domain; do
    domain=$(echo "$domain" | awk '{print $1}' | tr -d '[:space:]')
    if (( ${#domain} > DOMAIN_LEN_THRESHOLD )); then
        log_alert "LONG_DOMAIN length=${#domain} domain=$domain"
        alerts=$((alerts + 1))
    fi
done < "$TMPFILE"
if (( alerts == 0 )); then
    echo -e "${GREEN}Длинных доменов не найдено${NC}"
fi

# --- 2. Высокая частота запросов к одному домену ---
echo -e "\n${YELLOW}[2] Топ-10 запрашиваемых доменов${NC}"
awk '{print $1}' "$TMPFILE" | sort | uniq -c | sort -rn | head -10 | while read -r count domain; do
    echo "  $count  $domain"
    if (( count > 100 )); then
        log_alert "HIGH_FREQ_DOMAIN count=$count domain=$domain"
    fi
done

# --- 3. TXT-запросы (часто используются для tunneling) ---
echo -e "\n${YELLOW}[3] TXT-запросы (часто — DNS tunneling)${NC}"
txt_count=$(grep -ci "TXT\|type16" "$TMPFILE" 2>/dev/null || echo 0)
echo "  TXT-запросов: $txt_count"
if (( txt_count > TXT_THRESHOLD )); then
    log_alert "HIGH_TXT_QUERIES count=$txt_count threshold=$TXT_THRESHOLD"
fi

# --- 4. Энтропия доменных имён ---
echo -e "\n${YELLOW}[4] Домены с высокой энтропией (рандомные поддомены)${NC}"
awk '{print $1}' "$TMPFILE" | while IFS= read -r domain; do
    # Берём первый поддомен
    subdomain=$(echo "$domain" | awk -F'.' '{print $1}')
    if (( ${#subdomain} > 15 )); then
        # Простая проверка: много цифр и согласных подряд
        digit_ratio=$(echo "$subdomain" | grep -o '[0-9]' | wc -l)
        if (( digit_ratio * 3 > ${#subdomain} )); then
            log_alert "HIGH_ENTROPY_DOMAIN subdomain=$subdomain full=$domain"
        fi
    fi
done

# --- Итог ---
echo -e "\n${CYAN}--- Итог ---${NC}"
echo "Всего запросов: $total_queries"
echo "Уникальных доменов: $(awk '{print $1}' "$TMPFILE" | sort -u | wc -l)"
if (( alerts > 0 )); then
    echo -e "${RED}Алертов: $alerts${NC}"
else
    echo -e "${GREEN}Аномалий не обнаружено${NC}"
fi
```

---

## 8. user_audit.sh

**Назначение:** Аудит учётных записей: пользователи с UID 0, без пароля, в привилегированных группах, неиспользуемые аккаунты.

```bash
#!/bin/bash
#======================================================================
# user_audit.sh — Аудит пользователей системы
# Использование: ./user_audit.sh [-h]
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

LOGFILE="/var/log/security/user_audit.log"
mkdir -p "$(dirname "$LOGFILE")"

log_alert() {
    local timestamp
    timestamp=$(date '+%Y-%m-%dT%H:%M:%S%z')
    echo "$timestamp ALERT $1" >> "$LOGFILE"
    echo -e "${RED}[ALERT]${NC} $1"
}

[[ "${1:-}" == "-h" ]] && { echo "Использование: $0 [-h]"; exit 0; }

echo -e "${CYAN}=== Аудит пользователей: $(date) ===${NC}"

alerts=0

# --- 1. Пользователи с UID 0 (кроме root) ---
echo -e "\n${YELLOW}[1] Пользователи с UID 0 (суперпользователь)${NC}"
uid0_users=$(awk -F: '$3 == 0 && $1 != "root" {print $1}' /etc/passwd)
if [[ -n "$uid0_users" ]]; then
    for u in $uid0_users; do
        log_alert "UID_ZERO_USER user=$u"
        alerts=$((alerts + 1))
    done
else
    echo -e "${GREEN}Только root имеет UID 0${NC}"
fi

# --- 2. Аккаунты без пароля ---
echo -e "\n${YELLOW}[2] Аккаунты без пароля${NC}"
if [[ -r /etc/shadow ]]; then
    no_pass=$(awk -F: '($2 == "" || $2 == "!!" || $2 == "!") && $1 != "root" {print $1}' /etc/shadow)
    # Фильтруем системных пользователей с nologin
    for u in $no_pass; do
        shell=$(grep "^${u}:" /etc/passwd | cut -d: -f7)
        if [[ "$shell" != */nologin && "$shell" != */false ]]; then
            log_alert "NO_PASSWORD_ACCOUNT user=$u shell=$shell"
            alerts=$((alerts + 1))
        fi
    done
    if (( alerts == 0 )); then
        echo -e "${GREEN}Все активные аккаунты имеют пароль${NC}"
    fi
else
    echo -e "${YELLOW}Нет доступа к /etc/shadow (нужен root)${NC}"
fi

# --- 3. Привилегированные группы ---
echo -e "\n${YELLOW}[3] Пользователи в привилегированных группах${NC}"
for group in sudo wheel adm root docker lxd; do
    members=$(getent group "$group" 2>/dev/null | cut -d: -f4)
    if [[ -n "$members" ]]; then
        echo -e "  ${CYAN}$group:${NC} $members"
    fi
done

# --- 4. Пользователи с login shell ---
echo -e "\n${YELLOW}[4] Пользователи с интерактивным shell${NC}"
echo -e "  ${CYAN}Логин            UID   Shell${NC}"
while IFS=: read -r user _ uid _ _ _ shell; do
    if [[ "$shell" != */nologin && "$shell" != */false && "$shell" != */sync ]]; then
        printf "  %-16s %-5s %s\n" "$user" "$uid" "$shell"
    fi
done < /etc/passwd

# --- 5. Проверка /etc/passwd на аномалии ---
echo -e "\n${YELLOW}[5] Аномалии в /etc/passwd${NC}"

# Дубликаты UID
dup_uids=$(awk -F: '{print $3}' /etc/passwd | sort | uniq -d)
if [[ -n "$dup_uids" ]]; then
    for uid in $dup_uids; do
        users=$(awk -F: -v u="$uid" '$3 == u {print $1}' /etc/passwd | tr '\n' ',')
        log_alert "DUPLICATE_UID uid=$uid users=$users"
        alerts=$((alerts + 1))
    done
else
    echo -e "${GREEN}Дубликатов UID нет${NC}"
fi

# Дубликаты имён
dup_names=$(awk -F: '{print $1}' /etc/passwd | sort | uniq -d)
if [[ -n "$dup_names" ]]; then
    log_alert "DUPLICATE_USERNAMES names=$dup_names"
    alerts=$((alerts + 1))
fi

# --- 6. Последние входы ---
echo -e "\n${YELLOW}[6] Пользователи, не входившие > 90 дней${NC}"
current_epoch=$(date +%s)
while IFS=: read -r user _ _ _ _ _ _; do
    last_login=$(lastlog -u "$user" 2>/dev/null | tail -1 | awk '{
        if ($0 ~ /Never logged in/) print "never";
        else print $4" "$5" "$6" "$9
    }')
    if [[ "$last_login" == "never" ]]; then
        shell=$(grep "^${user}:" /etc/passwd | cut -d: -f7)
        if [[ "$shell" != */nologin && "$shell" != */false ]]; then
            echo -e "  ${YELLOW}$user — никогда не входил (shell: $shell)${NC}"
        fi
    fi
done < /etc/passwd

# --- 7. Файлы authorized_keys ---
echo -e "\n${YELLOW}[7] Файлы authorized_keys${NC}"
while IFS=: read -r user _ _ _ _ home _; do
    ak="$home/.ssh/authorized_keys"
    if [[ -f "$ak" ]]; then
        key_count=$(grep -c '^ssh-' "$ak" 2>/dev/null || echo 0)
        echo -e "  $user: $ak ($key_count ключей)"
        # Проверяем права
        perms=$(stat -c '%a' "$ak" 2>/dev/null || echo "???")
        if [[ "$perms" != "600" && "$perms" != "644" ]]; then
            log_alert "BAD_PERMS_AUTHORIZED_KEYS user=$user perms=$perms"
            alerts=$((alerts + 1))
        fi
    fi
done < /etc/passwd

# --- Итог ---
echo -e "\n${CYAN}--- Итог ---${NC}"
echo "Всего пользователей: $(wc -l < /etc/passwd)"
echo "С интерактивным shell: $(grep -vcE '(nologin|false|sync)$' /etc/passwd)"
if (( alerts > 0 )); then
    echo -e "${RED}Алертов: $alerts${NC}"
else
    echo -e "${GREEN}Аномалий не обнаружено${NC}"
fi
```

---

# ЧАСТЬ 2: HARDENING

---

## 9. ssh_hardening.sh

**Назначение:** Автоматическое усиление конфигурации SSH-сервера по best practices. Создаёт бэкап, применяет изменения, показывает diff.

```bash
#!/bin/bash
#======================================================================
# ssh_hardening.sh — Hardening SSH-сервера
# Использование: ./ssh_hardening.sh [-p port] [-u allowed_users] [-a] [-h]
#   -a  Применить изменения (без флага — только показать)
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

SSHD_CONFIG="/etc/ssh/sshd_config"
SSH_PORT=22
ALLOWED_USERS=""
APPLY=false

usage() {
    echo "Использование: $0 [опции]"
    echo "  -p PORT    SSH-порт (по умолчанию: 22)"
    echo "  -u USERS   Разрешённые пользователи (через запятую)"
    echo "  -a         Применить изменения (по умолчанию: только показать)"
    echo "  -h         Справка"
    exit 0
}

while getopts "p:u:ah" opt; do
    case $opt in
        p) SSH_PORT="$OPTARG" ;;
        u) ALLOWED_USERS="$OPTARG" ;;
        a) APPLY=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}[ERROR] Требуются права root${NC}"
    exit 1
fi

echo -e "${CYAN}=== SSH Hardening ===${NC}"

# --- Текущее состояние ---
echo -e "\n${YELLOW}[ТЕКУЩЕЕ СОСТОЯНИЕ]${NC}"
for param in PermitRootLogin PasswordAuthentication PubkeyAuthentication \
             MaxAuthTries LoginGraceTime X11Forwarding AllowTcpForwarding \
             Port Protocol; do
    value=$(grep -i "^${param}" "$SSHD_CONFIG" 2>/dev/null | tail -1 || echo "$param: (по умолчанию)")
    echo "  $value"
done

# --- Рекомендации ---
declare -A SETTINGS
SETTINGS=(
    ["Port"]="$SSH_PORT"
    ["PermitRootLogin"]="no"
    ["PasswordAuthentication"]="no"
    ["PubkeyAuthentication"]="yes"
    ["AuthorizedKeysFile"]=".ssh/authorized_keys"
    ["MaxAuthTries"]="3"
    ["LoginGraceTime"]="30"
    ["X11Forwarding"]="no"
    ["AllowTcpForwarding"]="no"
    ["AllowAgentForwarding"]="no"
    ["PermitEmptyPasswords"]="no"
    ["ClientAliveInterval"]="300"
    ["ClientAliveCountMax"]="2"
    ["UsePAM"]="yes"
    ["LogLevel"]="VERBOSE"
    ["Banner"]="/etc/ssh/banner"
    ["Ciphers"]="chacha20-poly1305@openssh.com,aes256-gcm@openssh.com,aes128-gcm@openssh.com"
    ["MACs"]="hmac-sha2-512-etm@openssh.com,hmac-sha2-256-etm@openssh.com"
    ["KexAlgorithms"]="curve25519-sha256,curve25519-sha256@libssh.org"
)

echo -e "\n${YELLOW}[РЕКОМЕНДУЕМЫЕ ИЗМЕНЕНИЯ]${NC}"
for key in "${!SETTINGS[@]}"; do
    current=$(grep -i "^${key}\s" "$SSHD_CONFIG" 2>/dev/null | awk '{$1=""; print $0}' | xargs || echo "(не задано)")
    recommended="${SETTINGS[$key]}"
    if [[ "$current" != "$recommended" ]]; then
        echo -e "  ${YELLOW}$key:${NC} $current → ${GREEN}$recommended${NC}"
    else
        echo -e "  ${GREEN}$key:${NC} $current (OK)"
    fi
done

if [[ -n "$ALLOWED_USERS" ]]; then
    echo -e "  ${YELLOW}AllowUsers:${NC} → ${GREEN}${ALLOWED_USERS//,/ }${NC}"
fi

if ! $APPLY; then
    echo -e "\n${CYAN}Для применения запустите с флагом -a${NC}"
    exit 0
fi

# --- Применение ---
echo -e "\n${CYAN}[ПРИМЕНЕНИЕ]${NC}"

# Бэкап
backup="${SSHD_CONFIG}.bak.$(date +%Y%m%d%H%M%S)"
cp "$SSHD_CONFIG" "$backup"
echo -e "${GREEN}Бэкап: $backup${NC}"

for key in "${!SETTINGS[@]}"; do
    value="${SETTINGS[$key]}"
    if grep -qi "^${key}\s" "$SSHD_CONFIG"; then
        sed -i "s/^${key}\s.*/${key} ${value}/" "$SSHD_CONFIG"
    elif grep -qi "^#\s*${key}\s" "$SSHD_CONFIG"; then
        sed -i "s/^#\s*${key}\s.*/${key} ${value}/" "$SSHD_CONFIG"
    else
        echo "${key} ${value}" >> "$SSHD_CONFIG"
    fi
done

if [[ -n "$ALLOWED_USERS" ]]; then
    users_space="${ALLOWED_USERS//,/ }"
    if grep -qi "^AllowUsers" "$SSHD_CONFIG"; then
        sed -i "s/^AllowUsers.*/AllowUsers $users_space/" "$SSHD_CONFIG"
    else
        echo "AllowUsers $users_space" >> "$SSHD_CONFIG"
    fi
fi

# Баннер
cat > /etc/ssh/banner << 'BANNER'
*******************************************************************
*  ВНИМАНИЕ: Несанкционированный доступ запрещён.                 *
*  Все действия логируются и могут быть использованы               *
*  в качестве доказательств.                                       *
*******************************************************************
BANNER

# Проверка конфигурации
echo -e "\n${YELLOW}[ПРОВЕРКА]${NC}"
if sshd -t 2>&1; then
    echo -e "${GREEN}Конфигурация валидна${NC}"
    echo -e "\n${YELLOW}Diff:${NC}"
    diff "$backup" "$SSHD_CONFIG" || true
    echo -e "\n${CYAN}Перезапустите SSH: systemctl restart sshd${NC}"
    echo -e "${RED}ВАЖНО: Убедитесь, что SSH-ключ настроен, иначе потеряете доступ!${NC}"
else
    echo -e "${RED}Ошибка в конфигурации! Восстанавливаю бэкап...${NC}"
    cp "$backup" "$SSHD_CONFIG"
fi
```

---

## 10. firewall_setup.sh

**Назначение:** Настройка iptables с default deny, rate limiting, anti-spoofing, логированием.

```bash
#!/bin/bash
#======================================================================
# firewall_setup.sh — Настройка iptables (базовый hardening)
# Использование: ./firewall_setup.sh [-p ssh_port] [-w web] [-a] [-s] [-h]
#   -w  Разрешить HTTP/HTTPS (80,443)
#   -a  Применить правила
#   -s  Сохранить правила
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

SSH_PORT=22
ALLOW_WEB=false
APPLY=false
SAVE=false

usage() {
    echo "Использование: $0 [опции]"
    echo "  -p PORT  SSH-порт (по умолчанию: 22)"
    echo "  -w       Разрешить HTTP/HTTPS (80,443)"
    echo "  -a       Применить правила"
    echo "  -s       Сохранить правила (iptables-save)"
    echo "  -h       Справка"
    exit 0
}

while getopts "p:wash" opt; do
    case $opt in
        p) SSH_PORT="$OPTARG" ;;
        w) ALLOW_WEB=true ;;
        a) APPLY=true ;;
        s) SAVE=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}[ERROR] Нужен root${NC}"
    exit 1
fi

echo -e "${CYAN}=== Настройка Firewall (iptables) ===${NC}"

# Показываем что будет
echo -e "\n${YELLOW}Планируемые правила:${NC}"
echo "  1. Default policy: DROP (INPUT, FORWARD), ACCEPT (OUTPUT)"
echo "  2. Allow loopback"
echo "  3. Allow established/related"
echo "  4. Allow SSH (порт $SSH_PORT) с rate limiting (3 новых/мин)"
if $ALLOW_WEB; then
    echo "  5. Allow HTTP (80) и HTTPS (443)"
fi
echo "  6. Allow ICMP ping (ограничено 1/сек)"
echo "  7. Anti-spoofing (drop bogons)"
echo "  8. Логирование dropped packets"

if ! $APPLY; then
    echo -e "\n${CYAN}Для применения запустите с -a${NC}"
    exit 0
fi

echo -e "\n${YELLOW}[ПРИМЕНЕНИЕ]${NC}"

# Бэкап текущих правил
iptables-save > "/root/iptables.backup.$(date +%Y%m%d%H%M%S)" 2>/dev/null || true

# Сброс всех правил
iptables -F
iptables -X
iptables -Z

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT
echo -e "${GREEN}[+] Default policies: INPUT=DROP FORWARD=DROP OUTPUT=ACCEPT${NC}"

# Loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Established / related
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
echo -e "${GREEN}[+] Established/related — ACCEPT${NC}"

# Anti-spoofing: drop RFC1918 on external interfaces (если интерфейс не lo)
iptables -A INPUT ! -i lo -s 127.0.0.0/8 -j DROP
echo -e "${GREEN}[+] Anti-spoofing для 127.0.0.0/8${NC}"

# SSH с rate limiting
iptables -A INPUT -p tcp --dport "$SSH_PORT" -m conntrack --ctstate NEW \
    -m recent --set --name SSH
iptables -A INPUT -p tcp --dport "$SSH_PORT" -m conntrack --ctstate NEW \
    -m recent --update --seconds 60 --hitcount 4 --name SSH -j DROP
iptables -A INPUT -p tcp --dport "$SSH_PORT" -m conntrack --ctstate NEW -j ACCEPT
echo -e "${GREEN}[+] SSH (порт $SSH_PORT) с rate limit (3 новых/мин)${NC}"

# HTTP/HTTPS
if $ALLOW_WEB; then
    iptables -A INPUT -p tcp --dport 80 -j ACCEPT
    iptables -A INPUT -p tcp --dport 443 -j ACCEPT
    echo -e "${GREEN}[+] HTTP/HTTPS — ACCEPT${NC}"
fi

# ICMP (ping) — ограниченный
iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 1/second --limit-burst 4 -j ACCEPT
echo -e "${GREEN}[+] ICMP ping (1/сек, burst 4)${NC}"

# Логирование dropped
iptables -N LOG_DROP 2>/dev/null || true
iptables -F LOG_DROP
iptables -A LOG_DROP -m limit --limit 5/min -j LOG --log-prefix "[FW_DROP] " --log-level 4
iptables -A LOG_DROP -j DROP
iptables -A INPUT -j LOG_DROP
echo -e "${GREEN}[+] Логирование dropped (5/мин max)${NC}"

# Сохранение
if $SAVE; then
    if command -v iptables-save &>/dev/null; then
        iptables-save > /etc/iptables.rules
        echo -e "${GREEN}[+] Правила сохранены в /etc/iptables.rules${NC}"
    fi
fi

echo -e "\n${CYAN}--- Текущие правила ---${NC}"
iptables -L -n --line-numbers | head -40

echo -e "\n${GREEN}Firewall настроен${NC}"
```

---

## 11. service_hardening.sh

**Назначение:** Аудит запущенных сервисов. Обнаружение ненужных и потенциально опасных. Рекомендации по отключению.

```bash
#!/bin/bash
#======================================================================
# service_hardening.sh — Аудит и hardening сервисов
# Использование: ./service_hardening.sh [-a] [-h]
#   -a  Отключить рекомендуемые сервисы (интерактивно)
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

APPLY=false
[[ "${1:-}" == "-a" ]] && APPLY=true
[[ "${1:-}" == "-h" ]] && { echo "Использование: $0 [-a] [-h]"; exit 0; }

# Сервисы, рекомендуемые к отключению (если не нужны)
UNNECESSARY_SERVICES=(
    "avahi-daemon"        # mDNS — Bonjour/Zeroconf, не нужен на серверах
    "cups"                # Система печати
    "cups-browsed"
    "bluetooth"           # Bluetooth
    "ModemManager"        # Модемы
    "whoopsie"            # Ubuntu error reporting
    "apport"              # Crash reporting
    "snapd"               # Snap (на серверах часто не нужен)
    "rpcbind"             # NFS/RPC
    "nfs-server"
    "autofs"
    "telnet.socket"       # Telnet (небезопасен)
    "xinetd"
    "vsftpd"              # FTP (если не используете)
    "rsyncd"
    "ypserv"              # NIS
    "ypbind"
)

echo -e "${CYAN}=== Аудит сервисов ===${NC}"

# --- 1. Все запущенные сервисы ---
echo -e "\n${YELLOW}[1] Запущенные сервисы${NC}"
systemctl list-units --type=service --state=running --no-pager 2>/dev/null | \
    grep '\.service' | awk '{print "  "$1, $4}' | head -30

# --- 2. Listening сокеты ---
echo -e "\n${YELLOW}[2] Listening сокеты${NC}"
ss -tlnp 2>/dev/null | awk 'NR>1 {printf "  %-25s %s\n", $4, $6}' | head -20

# --- 3. Проверка ненужных сервисов ---
echo -e "\n${YELLOW}[3] Потенциально ненужные сервисы${NC}"
found_unnecessary=0
for svc in "${UNNECESSARY_SERVICES[@]}"; do
    status=$(systemctl is-active "$svc" 2>/dev/null || echo "inactive")
    enabled=$(systemctl is-enabled "$svc" 2>/dev/null || echo "disabled")
    if [[ "$status" == "active" || "$enabled" == "enabled" ]]; then
        echo -e "  ${YELLOW}$svc${NC} — active=$status enabled=$enabled"
        found_unnecessary=$((found_unnecessary + 1))
        if $APPLY; then
            echo -ne "  Отключить $svc? [y/N] "
            read -r answer
            if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
                systemctl stop "$svc" 2>/dev/null || true
                systemctl disable "$svc" 2>/dev/null || true
                echo -e "  ${GREEN}$svc отключён${NC}"
            fi
        fi
    fi
done
if (( found_unnecessary == 0 )); then
    echo -e "${GREEN}Ненужных сервисов не найдено${NC}"
fi

# --- 4. Проверка сервисов от root ---
echo -e "\n${YELLOW}[4] Сервисы, работающие от root (рассмотреть снижение привилегий)${NC}"
ps aux --no-headers | awk '$1 == "root" && $11 !~ /^\[/' | \
    awk '{print "  "$11}' | sort -u | head -20

echo -e "\n${GREEN}Аудит завершён${NC}"
```

---

## 12. password_policy.sh

**Назначение:** Проверка и настройка парольной политики (PAM, login.defs, account lockout).

```bash
#!/bin/bash
#======================================================================
# password_policy.sh — Проверка и настройка парольной политики
# Использование: ./password_policy.sh [-a] [-h]
#   -a  Применить рекомендованные настройки
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

APPLY=false
[[ "${1:-}" == "-a" ]] && APPLY=true
[[ "${1:-}" == "-h" ]] && { echo "Использование: $0 [-a] [-h]"; exit 0; }

check_setting() {
    local file="$1" param="$2" expected="$3"
    local current
    current=$(grep -i "^${param}" "$file" 2>/dev/null | tail -1 | awk '{print $2}' || echo "(не задано)")
    if [[ "$current" == "$expected" ]]; then
        echo -e "  ${GREEN}$param = $current (OK)${NC}"
    else
        echo -e "  ${YELLOW}$param = $current → рекомендуется: $expected${NC}"
    fi
}

echo -e "${CYAN}=== Проверка парольной политики ===${NC}"

# --- 1. /etc/login.defs ---
echo -e "\n${YELLOW}[1] /etc/login.defs${NC}"
check_setting /etc/login.defs "PASS_MAX_DAYS" "90"
check_setting /etc/login.defs "PASS_MIN_DAYS" "7"
check_setting /etc/login.defs "PASS_MIN_LEN" "12"
check_setting /etc/login.defs "PASS_WARN_AGE" "14"
check_setting /etc/login.defs "LOGIN_RETRIES" "3"
check_setting /etc/login.defs "ENCRYPT_METHOD" "SHA512"

# --- 2. PAM: pam_pwquality ---
echo -e "\n${YELLOW}[2] PAM — pam_pwquality${NC}"
if [[ -f /etc/security/pwquality.conf ]]; then
    echo "  Файл: /etc/security/pwquality.conf"
    check_setting /etc/security/pwquality.conf "minlen" "12"
    check_setting /etc/security/pwquality.conf "dcredit" "-1"
    check_setting /etc/security/pwquality.conf "ucredit" "-1"
    check_setting /etc/security/pwquality.conf "lcredit" "-1"
    check_setting /etc/security/pwquality.conf "ocredit" "-1"
    check_setting /etc/security/pwquality.conf "maxrepeat" "3"
    check_setting /etc/security/pwquality.conf "difok" "5"
else
    echo -e "  ${YELLOW}pam_pwquality не установлен (apt install libpam-pwquality)${NC}"
fi

# --- 3. Account lockout ---
echo -e "\n${YELLOW}[3] Блокировка аккаунтов (faillock/tally2)${NC}"
if grep -rq "pam_faillock\|pam_tally2" /etc/pam.d/ 2>/dev/null; then
    echo -e "  ${GREEN}Механизм блокировки настроен${NC}"
    grep -h "pam_faillock\|pam_tally2" /etc/pam.d/* 2>/dev/null | head -5
else
    echo -e "  ${RED}Блокировка аккаунтов НЕ настроена${NC}"
fi

# --- 4. Пароли, которые нужно сменить ---
echo -e "\n${YELLOW}[4] Пароли, подлежащие смене${NC}"
if [[ -r /etc/shadow ]]; then
    current_days=$(( $(date +%s) / 86400 ))
    while IFS=: read -r user pass lastchange _ maxdays _ _ _ _; do
        [[ "$pass" == "!" || "$pass" == "!!" || "$pass" == "*" || -z "$pass" ]] && continue
        [[ -z "$lastchange" || "$lastchange" == "0" ]] && continue
        [[ -z "$maxdays" || "$maxdays" == "99999" ]] && continue
        expire_day=$(( lastchange + maxdays ))
        if (( current_days > expire_day )); then
            echo -e "  ${RED}$user — пароль просрочен (${maxdays} дней макс)${NC}"
        elif (( expire_day - current_days < 14 )); then
            echo -e "  ${YELLOW}$user — пароль истекает через $((expire_day - current_days)) дней${NC}"
        fi
    done < /etc/shadow
fi

# --- Применение ---
if $APPLY; then
    echo -e "\n${CYAN}[ПРИМЕНЕНИЕ]${NC}"

    # login.defs
    cp /etc/login.defs "/etc/login.defs.bak.$(date +%Y%m%d)"
    sed -i 's/^PASS_MAX_DAYS.*/PASS_MAX_DAYS\t90/' /etc/login.defs
    sed -i 's/^PASS_MIN_DAYS.*/PASS_MIN_DAYS\t7/' /etc/login.defs
    sed -i 's/^PASS_WARN_AGE.*/PASS_WARN_AGE\t14/' /etc/login.defs
    echo -e "${GREEN}login.defs обновлён${NC}"

    # pwquality.conf
    if [[ -f /etc/security/pwquality.conf ]]; then
        cp /etc/security/pwquality.conf "/etc/security/pwquality.conf.bak.$(date +%Y%m%d)"
        cat > /etc/security/pwquality.conf << 'EOF'
minlen = 12
dcredit = -1
ucredit = -1
lcredit = -1
ocredit = -1
maxrepeat = 3
difok = 5
reject_username
enforce_for_root
EOF
        echo -e "${GREEN}pwquality.conf обновлён${NC}"
    fi
fi

echo -e "\n${GREEN}Аудит завершён${NC}"
```

---

## 13. filesystem_hardening.sh

**Назначение:** Hardening файловой системы: монтирование /tmp с noexec, аудит SUID/SGID, world-writable каталоги.

```bash
#!/bin/bash
#======================================================================
# filesystem_hardening.sh — Hardening файловой системы
# Использование: ./filesystem_hardening.sh [-a] [-h]
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

APPLY=false
[[ "${1:-}" == "-a" ]] && APPLY=true
[[ "${1:-}" == "-h" ]] && { echo "Использование: $0 [-a] [-h]"; exit 0; }

echo -e "${CYAN}=== Hardening файловой системы ===${NC}"

# --- 1. /tmp mount options ---
echo -e "\n${YELLOW}[1] Опции монтирования /tmp${NC}"
tmp_opts=$(mount | grep ' /tmp ' | grep -oP '\(.*\)' || echo "(не отдельный раздел)")
echo "  Текущие: $tmp_opts"
if echo "$tmp_opts" | grep -q "noexec"; then
    echo -e "  ${GREEN}noexec — OK${NC}"
else
    echo -e "  ${RED}noexec НЕ установлен — рекомендуется${NC}"
fi

# --- 2. SUID/SGID файлы ---
echo -e "\n${YELLOW}[2] Файлы с SUID-битом${NC}"
echo "  Поиск SUID-файлов (это нормальные системные утилиты, но проверьте необычные):"
find / -perm -4000 -type f 2>/dev/null | while IFS= read -r f; do
    owner=$(stat -c '%U' "$f" 2>/dev/null || echo "?")
    echo "  $f (владелец: $owner)"
done | head -25
suid_count=$(find / -perm -4000 -type f 2>/dev/null | wc -l)
echo -e "  Всего SUID-файлов: ${CYAN}$suid_count${NC}"

echo -e "\n  Файлы с SGID-битом:"
sgid_count=$(find / -perm -2000 -type f 2>/dev/null | wc -l)
echo -e "  Всего SGID-файлов: ${CYAN}$sgid_count${NC}"

# --- 3. World-writable каталоги без sticky bit ---
echo -e "\n${YELLOW}[3] World-writable каталоги без sticky bit${NC}"
ww_dirs=$(find / -type d -perm -0002 ! -perm -1000 2>/dev/null | head -20)
if [[ -n "$ww_dirs" ]]; then
    echo "$ww_dirs" | while IFS= read -r d; do
        echo -e "  ${RED}$d${NC}"
    done
    if $APPLY; then
        echo "$ww_dirs" | while IFS= read -r d; do
            chmod +t "$d" 2>/dev/null && echo -e "  ${GREEN}sticky bit добавлен: $d${NC}"
        done
    fi
else
    echo -e "  ${GREEN}Не найдено${NC}"
fi

# --- 4. World-writable файлы ---
echo -e "\n${YELLOW}[4] World-writable файлы${NC}"
ww_files=$(find / -type f -perm -0002 ! -path '/proc/*' ! -path '/sys/*' ! -path '/dev/*' 2>/dev/null | head -20)
if [[ -n "$ww_files" ]]; then
    echo "$ww_files" | while IFS= read -r f; do
        echo -e "  ${RED}$f${NC}"
    done
else
    echo -e "  ${GREEN}Не найдено${NC}"
fi

# --- 5. Файлы без владельца ---
echo -e "\n${YELLOW}[5] Файлы без владельца (unowned)${NC}"
unowned=$(find / -nouser -o -nogroup 2>/dev/null | head -20)
if [[ -n "$unowned" ]]; then
    echo "$unowned" | while IFS= read -r f; do
        echo -e "  ${YELLOW}$f${NC}"
    done
else
    echo -e "  ${GREEN}Не найдено${NC}"
fi

# --- 6. Права на критические файлы ---
echo -e "\n${YELLOW}[6] Права на критические файлы${NC}"
declare -A EXPECTED_PERMS=(
    ["/etc/passwd"]="644"
    ["/etc/shadow"]="640"
    ["/etc/group"]="644"
    ["/etc/gshadow"]="640"
    ["/etc/ssh/sshd_config"]="600"
    ["/etc/crontab"]="600"
    ["/etc/sudoers"]="440"
)
for file in "${!EXPECTED_PERMS[@]}"; do
    if [[ -f "$file" ]]; then
        actual=$(stat -c '%a' "$file")
        expected="${EXPECTED_PERMS[$file]}"
        if [[ "$actual" == "$expected" ]]; then
            echo -e "  ${GREEN}$file: $actual (OK)${NC}"
        else
            echo -e "  ${RED}$file: $actual → рекомендуется $expected${NC}"
            if $APPLY; then
                chmod "$expected" "$file"
                echo -e "  ${GREEN}Исправлено${NC}"
            fi
        fi
    fi
done

echo -e "\n${GREEN}Аудит завершён${NC}"
```

---

## 14. kernel_hardening.sh

**Назначение:** Hardening параметров ядра через sysctl: сетевая безопасность, защита памяти, ограничение утечек информации.

```bash
#!/bin/bash
#======================================================================
# kernel_hardening.sh — Hardening параметров ядра (sysctl)
# Использование: ./kernel_hardening.sh [-a] [-h]
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

APPLY=false
[[ "${1:-}" == "-a" ]] && APPLY=true
[[ "${1:-}" == "-h" ]] && { echo "Использование: $0 [-a] [-h]"; exit 0; }

# Рекомендуемые параметры
declare -A SYSCTL_PARAMS=(
    # Сетевая безопасность
    ["net.ipv4.ip_forward"]="0"
    ["net.ipv4.conf.all.send_redirects"]="0"
    ["net.ipv4.conf.default.send_redirects"]="0"
    ["net.ipv4.conf.all.accept_redirects"]="0"
    ["net.ipv4.conf.default.accept_redirects"]="0"
    ["net.ipv4.conf.all.accept_source_route"]="0"
    ["net.ipv4.conf.default.accept_source_route"]="0"
    ["net.ipv4.conf.all.log_martians"]="1"
    ["net.ipv4.conf.default.log_martians"]="1"
    ["net.ipv4.conf.all.rp_filter"]="1"
    ["net.ipv4.conf.default.rp_filter"]="1"
    ["net.ipv4.tcp_syncookies"]="1"
    ["net.ipv4.icmp_echo_ignore_broadcasts"]="1"
    ["net.ipv4.icmp_ignore_bogus_error_responses"]="1"
    # IPv6 (отключение если не нужен)
    ["net.ipv6.conf.all.accept_redirects"]="0"
    ["net.ipv6.conf.default.accept_redirects"]="0"
    ["net.ipv6.conf.all.accept_source_route"]="0"
    # Защита памяти
    ["kernel.randomize_va_space"]="2"
    ["kernel.dmesg_restrict"]="1"
    ["kernel.kptr_restrict"]="2"
    ["kernel.yama.ptrace_scope"]="2"
    # Core dumps
    ["fs.suid_dumpable"]="0"
    # Защита от SYN flood
    ["net.ipv4.tcp_max_syn_backlog"]="4096"
    ["net.ipv4.tcp_synack_retries"]="2"
)

echo -e "${CYAN}=== Hardening ядра (sysctl) ===${NC}"

echo -e "\n${YELLOW}Параметр → Текущее → Рекомендуемое${NC}"
changes_needed=0
for param in $(echo "${!SYSCTL_PARAMS[@]}" | tr ' ' '\n' | sort); do
    expected="${SYSCTL_PARAMS[$param]}"
    current=$(sysctl -n "$param" 2>/dev/null || echo "N/A")
    if [[ "$current" == "$expected" ]]; then
        echo -e "  ${GREEN}$param = $current (OK)${NC}"
    else
        echo -e "  ${YELLOW}$param = $current → $expected${NC}"
        changes_needed=$((changes_needed + 1))
    fi
done

echo -e "\nИзменений требуется: $changes_needed"

if $APPLY && (( changes_needed > 0 )); then
    echo -e "\n${CYAN}[ПРИМЕНЕНИЕ]${NC}"
    CONF_FILE="/etc/sysctl.d/99-hardening.conf"
    {
        echo "# Hardening sysctl — $(date)"
        echo "# Сгенерировано kernel_hardening.sh"
        for param in $(echo "${!SYSCTL_PARAMS[@]}" | tr ' ' '\n' | sort); do
            echo "${param} = ${SYSCTL_PARAMS[$param]}"
        done
    } > "$CONF_FILE"
    sysctl -p "$CONF_FILE" 2>/dev/null
    echo -e "${GREEN}Параметры применены и сохранены в $CONF_FILE${NC}"
fi
```

---

## 15. log_hardening.sh

**Назначение:** Настройка логирования: централизация через rsyslog, права на логи, auditd правила.

```bash
#!/bin/bash
#======================================================================
# log_hardening.sh — Hardening логирования
# Использование: ./log_hardening.sh [-r remote_server] [-a] [-h]
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

REMOTE_SERVER=""
APPLY=false

usage() {
    echo "Использование: $0 [опции]"
    echo "  -r HOST  Удалённый сервер для логов (syslog)"
    echo "  -a       Применить изменения"
    echo "  -h       Справка"
    exit 0
}

while getopts "r:ah" opt; do
    case $opt in
        r) REMOTE_SERVER="$OPTARG" ;;
        a) APPLY=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

echo -e "${CYAN}=== Hardening логирования ===${NC}"

# --- 1. Проверка прав на логи ---
echo -e "\n${YELLOW}[1] Права на файлы логов${NC}"
for logfile in /var/log/syslog /var/log/auth.log /var/log/kern.log /var/log/secure; do
    if [[ -f "$logfile" ]]; then
        perms=$(stat -c '%a %U:%G' "$logfile")
        echo "  $logfile: $perms"
    fi
done

# --- 2. Rsyslog ---
echo -e "\n${YELLOW}[2] Rsyslog${NC}"
if systemctl is-active rsyslog &>/dev/null; then
    echo -e "  ${GREEN}rsyslog активен${NC}"
else
    echo -e "  ${RED}rsyslog НЕ активен${NC}"
fi

if [[ -n "$REMOTE_SERVER" ]]; then
    echo "  Удалённый сервер: $REMOTE_SERVER"
fi

# --- 3. Logrotate ---
echo -e "\n${YELLOW}[3] Logrotate${NC}"
if [[ -f /etc/logrotate.conf ]]; then
    rotate=$(grep -i "^rotate" /etc/logrotate.conf | head -1 || echo "не задано")
    echo "  Ротация: $rotate"
    compress=$(grep -i "^compress" /etc/logrotate.conf && echo "да" || echo "нет")
    echo "  Сжатие: $compress"
fi

# --- 4. Auditd ---
echo -e "\n${YELLOW}[4] Auditd${NC}"
if command -v auditctl &>/dev/null; then
    if systemctl is-active auditd &>/dev/null; then
        echo -e "  ${GREEN}auditd активен${NC}"
        rule_count=$(auditctl -l 2>/dev/null | wc -l)
        echo "  Правил: $rule_count"
    else
        echo -e "  ${YELLOW}auditd установлен, но не активен${NC}"
    fi
else
    echo -e "  ${RED}auditd не установлен (apt install auditd)${NC}"
fi

# --- Применение ---
if $APPLY; then
    echo -e "\n${CYAN}[ПРИМЕНЕНИЕ]${NC}"

    # Права на логи
    chmod 640 /var/log/syslog /var/log/auth.log /var/log/kern.log 2>/dev/null || true
    echo -e "${GREEN}Права на логи: 640${NC}"

    # Remote syslog
    if [[ -n "$REMOTE_SERVER" ]]; then
        conf="/etc/rsyslog.d/99-remote.conf"
        echo "*.* @@${REMOTE_SERVER}:514" > "$conf"
        systemctl restart rsyslog 2>/dev/null || true
        echo -e "${GREEN}Remote logging → $REMOTE_SERVER${NC}"
    fi

    # Auditd базовые правила
    if command -v auditctl &>/dev/null; then
        AUDIT_RULES="/etc/audit/rules.d/hardening.rules"
        cat > "$AUDIT_RULES" << 'AUDITRULES'
# Мониторинг критических файлов
-w /etc/passwd -p wa -k identity
-w /etc/shadow -p wa -k identity
-w /etc/group -p wa -k identity
-w /etc/sudoers -p wa -k sudoers
-w /etc/ssh/sshd_config -p wa -k sshd_config

# Мониторинг логинов
-w /var/log/auth.log -p wa -k auth_log
-w /var/log/faillog -p wa -k faillog
-w /var/log/lastlog -p wa -k lastlog

# Мониторинг cron
-w /etc/crontab -p wa -k cron
-w /etc/cron.d/ -p wa -k cron
-w /var/spool/cron/ -p wa -k cron

# Мониторинг kernel modules
-w /sbin/insmod -p x -k modules
-w /sbin/rmmod -p x -k modules
-w /sbin/modprobe -p x -k modules

# Мониторинг сетевых настроек
-w /etc/hosts -p wa -k network
-w /etc/resolv.conf -p wa -k network
-w /etc/sysctl.conf -p wa -k sysctl

# Мониторинг systemd
-w /etc/systemd/ -p wa -k systemd
-w /lib/systemd/ -p wa -k systemd
AUDITRULES
        systemctl restart auditd 2>/dev/null || augenrules --load 2>/dev/null || true
        echo -e "${GREEN}Auditd правила применены ($AUDIT_RULES)${NC}"
    fi
fi

echo -e "\n${GREEN}Аудит логирования завершён${NC}"
```

---

## 16. quick_cis_check.sh

**Назначение:** Быстрая проверка по CIS Benchmark: 30 ключевых контролей. Вывод PASS/FAIL/WARN.

```bash
#!/bin/bash
#======================================================================
# quick_cis_check.sh — Быстрая проверка по CIS Benchmark (Linux)
# Использование: ./quick_cis_check.sh [-v] [-h]
#   -v  Подробный вывод
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

VERBOSE=false
[[ "${1:-}" == "-v" ]] && VERBOSE=true
[[ "${1:-}" == "-h" ]] && { echo "Использование: $0 [-v] [-h]"; exit 0; }

PASS=0
FAIL=0
WARN=0

result() {
    local status="$1" id="$2" desc="$3"
    case "$status" in
        PASS) echo -e "  ${GREEN}[PASS]${NC} $id: $desc"; PASS=$((PASS+1)) ;;
        FAIL) echo -e "  ${RED}[FAIL]${NC} $id: $desc"; FAIL=$((FAIL+1)) ;;
        WARN) echo -e "  ${YELLOW}[WARN]${NC} $id: $desc"; WARN=$((WARN+1)) ;;
    esac
}

echo -e "${CYAN}=== Быстрая проверка CIS Benchmark ===${NC}"
echo -e "Система: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || uname -s)"
echo ""

# === SSH ===
echo -e "${CYAN}[SSH]${NC}"

# CIS 5.2.1 — sshd_config permissions
if [[ -f /etc/ssh/sshd_config ]]; then
    perms=$(stat -c '%a' /etc/ssh/sshd_config)
    [[ "$perms" == "600" ]] && result "PASS" "5.2.1" "sshd_config permissions ($perms)" \
        || result "FAIL" "5.2.1" "sshd_config permissions ($perms → 600)"
fi

# CIS 5.2.6 — SSH MaxAuthTries <= 4
mat=$(grep -i '^MaxAuthTries' /etc/ssh/sshd_config 2>/dev/null | awk '{print $2}' || echo "6")
(( mat <= 4 )) && result "PASS" "5.2.6" "MaxAuthTries ($mat)" \
    || result "FAIL" "5.2.6" "MaxAuthTries ($mat → <=4)"

# CIS 5.2.8 — SSH root login disabled
rl=$(grep -i '^PermitRootLogin' /etc/ssh/sshd_config 2>/dev/null | awk '{print $2}' || echo "yes")
[[ "$rl" == "no" ]] && result "PASS" "5.2.8" "PermitRootLogin ($rl)" \
    || result "FAIL" "5.2.8" "PermitRootLogin ($rl → no)"

# CIS 5.2.10 — SSH PasswordAuthentication
pa=$(grep -i '^PasswordAuthentication' /etc/ssh/sshd_config 2>/dev/null | awk '{print $2}' || echo "yes")
[[ "$pa" == "no" ]] && result "PASS" "5.2.10" "PasswordAuthentication ($pa)" \
    || result "WARN" "5.2.10" "PasswordAuthentication ($pa → no)"

# CIS 5.2.15 — SSH Banner
bn=$(grep -i '^Banner' /etc/ssh/sshd_config 2>/dev/null | awk '{print $2}' || echo "none")
[[ "$bn" != "none" && -n "$bn" ]] && result "PASS" "5.2.15" "SSH Banner ($bn)" \
    || result "WARN" "5.2.15" "SSH Banner не настроен"

# === FILESYSTEM ===
echo -e "\n${CYAN}[FILESYSTEM]${NC}"

# CIS 1.1.3 — /tmp noexec
if mount | grep ' /tmp ' | grep -q noexec; then
    result "PASS" "1.1.3" "/tmp смонтирован с noexec"
else
    result "FAIL" "1.1.3" "/tmp без noexec"
fi

# CIS 6.1.2 — /etc/passwd permissions
perms=$(stat -c '%a' /etc/passwd)
[[ "$perms" == "644" ]] && result "PASS" "6.1.2" "/etc/passwd ($perms)" \
    || result "FAIL" "6.1.2" "/etc/passwd ($perms → 644)"

# CIS 6.1.3 — /etc/shadow permissions
if [[ -f /etc/shadow ]]; then
    perms=$(stat -c '%a' /etc/shadow)
    [[ "$perms" == "640" || "$perms" == "600" ]] && result "PASS" "6.1.3" "/etc/shadow ($perms)" \
        || result "FAIL" "6.1.3" "/etc/shadow ($perms → 640)"
fi

# World-writable files
ww_count=$(find / -xdev -type f -perm -0002 2>/dev/null | wc -l)
(( ww_count == 0 )) && result "PASS" "6.1.10" "World-writable файлов: $ww_count" \
    || result "FAIL" "6.1.10" "World-writable файлов: $ww_count"

# Unowned files
unowned=$(find / -xdev \( -nouser -o -nogroup \) 2>/dev/null | wc -l)
(( unowned == 0 )) && result "PASS" "6.1.11" "Файлов без владельца: $unowned" \
    || result "WARN" "6.1.11" "Файлов без владельца: $unowned"

# === KERNEL ===
echo -e "\n${CYAN}[KERNEL]${NC}"

# CIS 3.1.1 — IP forwarding disabled
ipfwd=$(sysctl -n net.ipv4.ip_forward 2>/dev/null || echo "1")
[[ "$ipfwd" == "0" ]] && result "PASS" "3.1.1" "IP forwarding ($ipfwd)" \
    || result "FAIL" "3.1.1" "IP forwarding ($ipfwd → 0)"

# CIS 3.2.2 — ICMP redirects
redir=$(sysctl -n net.ipv4.conf.all.accept_redirects 2>/dev/null || echo "1")
[[ "$redir" == "0" ]] && result "PASS" "3.2.2" "ICMP redirects ($redir)" \
    || result "FAIL" "3.2.2" "ICMP redirects ($redir → 0)"

# CIS 3.2.8 — TCP SYN cookies
syn=$(sysctl -n net.ipv4.tcp_syncookies 2>/dev/null || echo "0")
[[ "$syn" == "1" ]] && result "PASS" "3.2.8" "SYN cookies ($syn)" \
    || result "FAIL" "3.2.8" "SYN cookies ($syn → 1)"

# ASLR
aslr=$(sysctl -n kernel.randomize_va_space 2>/dev/null || echo "0")
[[ "$aslr" == "2" ]] && result "PASS" "1.5.1" "ASLR ($aslr)" \
    || result "FAIL" "1.5.1" "ASLR ($aslr → 2)"

# dmesg_restrict
dmesg=$(sysctl -n kernel.dmesg_restrict 2>/dev/null || echo "0")
[[ "$dmesg" == "1" ]] && result "PASS" "1.5.2" "dmesg_restrict ($dmesg)" \
    || result "WARN" "1.5.2" "dmesg_restrict ($dmesg → 1)"

# === SERVICES ===
echo -e "\n${CYAN}[SERVICES]${NC}"

for svc in avahi-daemon cups rpcbind; do
    if systemctl is-active "$svc" &>/dev/null; then
        result "WARN" "2.2.x" "$svc активен (рекомендуется отключить если не нужен)"
    else
        result "PASS" "2.2.x" "$svc не активен"
    fi
done

# Firewall
if iptables -L -n &>/dev/null; then
    rules=$(iptables -L -n 2>/dev/null | grep -c "^[A-Z]" || echo 0)
    (( rules > 3 )) && result "PASS" "3.5.1" "Firewall настроен ($rules chains)" \
        || result "FAIL" "3.5.1" "Firewall — мало правил"
else
    result "WARN" "3.5.1" "iptables недоступен"
fi

# === PASSWORDS ===
echo -e "\n${CYAN}[PASSWORDS]${NC}"

max_days=$(grep '^PASS_MAX_DAYS' /etc/login.defs 2>/dev/null | awk '{print $2}' || echo "99999")
(( max_days <= 365 )) && result "PASS" "5.4.1.1" "PASS_MAX_DAYS ($max_days)" \
    || result "FAIL" "5.4.1.1" "PASS_MAX_DAYS ($max_days → <=365)"

min_days=$(grep '^PASS_MIN_DAYS' /etc/login.defs 2>/dev/null | awk '{print $2}' || echo "0")
(( min_days >= 1 )) && result "PASS" "5.4.1.2" "PASS_MIN_DAYS ($min_days)" \
    || result "WARN" "5.4.1.2" "PASS_MIN_DAYS ($min_days → >=1)"

# UID 0 check
uid0_count=$(awk -F: '$3 == 0' /etc/passwd | wc -l)
(( uid0_count <= 1 )) && result "PASS" "6.2.2" "Пользователей с UID 0: $uid0_count" \
    || result "FAIL" "6.2.2" "Пользователей с UID 0: $uid0_count (только root!)"

# === LOGGING ===
echo -e "\n${CYAN}[LOGGING]${NC}"

# rsyslog active
if systemctl is-active rsyslog &>/dev/null; then
    result "PASS" "4.2.1.1" "rsyslog активен"
else
    result "FAIL" "4.2.1.1" "rsyslog НЕ активен"
fi

# auditd active
if systemctl is-active auditd &>/dev/null; then
    result "PASS" "4.1.1.1" "auditd активен"
else
    result "WARN" "4.1.1.1" "auditd НЕ активен"
fi

# === NTP ===
echo -e "\n${CYAN}[TIME SYNC]${NC}"
if systemctl is-active systemd-timesyncd &>/dev/null || systemctl is-active chronyd &>/dev/null || systemctl is-active ntpd &>/dev/null; then
    result "PASS" "2.2.1.2" "NTP синхронизация активна"
else
    result "WARN" "2.2.1.2" "NTP синхронизация НЕ активна"
fi

# === ИТОГ ===
echo -e "\n${CYAN}=== ИТОГ ===${NC}"
total=$((PASS + FAIL + WARN))
echo -e "Всего проверок: $total"
echo -e "${GREEN}PASS: $PASS${NC}  ${RED}FAIL: $FAIL${NC}  ${YELLOW}WARN: $WARN${NC}"
score=$((PASS * 100 / total))
echo -e "Оценка: ${CYAN}${score}%${NC}"

if (( score >= 80 )); then
    echo -e "${GREEN}Хороший уровень hardening${NC}"
elif (( score >= 50 )); then
    echo -e "${YELLOW}Требуется доработка${NC}"
else
    echo -e "${RED}Критически низкий уровень hardening${NC}"
fi
```

---

# ЧАСТЬ 3: РЕАГИРОВАНИЕ

---

## 17. incident_collector.sh

**Назначение:** Сбор всей volatile и non-volatile информации при инциденте. Создаёт пакет для форензики.

```bash
#!/bin/bash
#======================================================================
# incident_collector.sh — Сбор данных при инциденте
# Использование: ./incident_collector.sh [-o output_dir] [-h]
#======================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
HOSTNAME_TAG=$(hostname -s)
OUTPUT_DIR="/tmp/incident_${HOSTNAME_TAG}_${TIMESTAMP}"

usage() {
    echo "Использование: $0 [-o output_dir] [-h]"
    echo "  -o DIR  Каталог для сохранения"
    echo "  -h      Справка"
    exit 0
}

while getopts "o:h" opt; do
    case $opt in
        o) OUTPUT_DIR="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

collect() {
    local name="$1" cmd="$2"
    local outfile="$OUTPUT_DIR/${name}.txt"
    echo -e "${CYAN}[Сбор]${NC} $name"
    echo "=== $name ===" > "$outfile"
    echo "Время сбора: $(date)" >> "$outfile"
    echo "---" >> "$outfile"
    eval "$cmd" >> "$outfile" 2>&1 || echo "(команда завершилась с ошибкой)" >> "$outfile"
}

echo -e "${CYAN}=== Сбор данных для расследования ===${NC}"
echo "Каталог: $OUTPUT_DIR"
echo "Начало: $(date)"
echo ""

# --- Volatile data (собираем первым!) ---
collect "01_date_time"       "date; uptime; who -b"
collect "02_logged_in_users" "who -a; w; last -20"
collect "03_processes"       "ps auxfww"
collect "04_process_tree"    "pstree -palun 2>/dev/null || ps axjf"
collect "05_network_conn"    "ss -tupna; echo '---'; netstat -antup 2>/dev/null || true"
collect "06_listening_ports"  "ss -tlnp"
collect "07_open_files"      "lsof -nP 2>/dev/null | head -500"
collect "08_routing"         "ip route show; echo '---'; ip neigh show"
collect "09_arp_cache"       "ip neigh; arp -a 2>/dev/null || true"
collect "10_dns_config"      "cat /etc/resolv.conf; echo '---'; cat /etc/hosts"
collect "11_env_vars"        "env | sort"
collect "12_loaded_modules"  "lsmod"
collect "13_mount_points"    "mount; df -h"

# --- System config ---
collect "14_hostname"        "hostname -f; uname -a; cat /etc/os-release"
collect "15_interfaces"      "ip addr show; ip link show"
collect "16_iptables"        "iptables -L -n -v 2>/dev/null; echo '---'; iptables -t nat -L -n -v 2>/dev/null"
collect "17_users_groups"    "cat /etc/passwd; echo '---'; cat /etc/group"
collect "18_sudoers"         "cat /etc/sudoers 2>/dev/null; echo '---'; ls -la /etc/sudoers.d/ 2>/dev/null"
collect "19_ssh_config"      "cat /etc/ssh/sshd_config 2>/dev/null"
collect "20_ssh_keys"        "find /root /home -name 'authorized_keys' -exec echo '=== {} ===' \\; -exec cat {} \\; 2>/dev/null"

# --- Persistence ---
collect "21_crontabs"        "for u in \$(cut -d: -f1 /etc/passwd); do echo \"=== \$u ===\"; crontab -u \"\$u\" -l 2>/dev/null || echo 'none'; done"
collect "22_cron_dirs"       "ls -laR /etc/cron* 2>/dev/null"
collect "23_systemd_units"   "systemctl list-units --type=service --all --no-pager"
collect "24_systemd_timers"  "systemctl list-timers --all --no-pager"
collect "25_initd"           "ls -la /etc/init.d/ 2>/dev/null"
collect "26_rc_local"        "cat /etc/rc.local 2>/dev/null || echo 'не существует'"

# --- Logs ---
collect "27_auth_log"        "tail -500 /var/log/auth.log 2>/dev/null || journalctl -u sshd -n 500 --no-pager 2>/dev/null"
collect "28_syslog"          "tail -500 /var/log/syslog 2>/dev/null || journalctl -n 500 --no-pager 2>/dev/null"
collect "29_recent_logins"   "last -50; echo '---'; lastb -20 2>/dev/null || true"
collect "30_failed_logins"   "grep 'Failed password' /var/log/auth.log 2>/dev/null | tail -100"

# --- File system ---
collect "31_tmp_files"       "ls -laR /tmp /var/tmp 2>/dev/null | head -200"
collect "32_recent_modified"  "find / -xdev -mtime -1 -type f 2>/dev/null | head -200"
collect "33_suid_files"      "find / -xdev -perm -4000 -type f 2>/dev/null"
collect "34_hidden_files"    "find / -xdev -name '.*' -type f 2>/dev/null | grep -v '/\.' | head -100"

# --- Hashes ---
collect "35_critical_hashes" "sha256sum /etc/passwd /etc/shadow /etc/sudoers /etc/ssh/sshd_config /etc/crontab 2>/dev/null"

# --- Упаковка ---
echo ""
echo -e "${CYAN}[Упаковка]${NC}"
ARCHIVE="${OUTPUT_DIR}.tar.gz"
tar czf "$ARCHIVE" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")" 2>/dev/null
echo -e "${GREEN}Архив: $ARCHIVE${NC}"
echo -e "Размер: $(du -h "$ARCHIVE" | awk '{print $1}')"
echo -e "SHA256: $(sha256sum "$ARCHIVE" | awk '{print $1}')"

echo -e "\n${GREEN}=== Сбор завершён: $(find "$OUTPUT_DIR" -type f | wc -l) файлов ===${NC}"
```

---

## 18. block_ip.sh

**Назначение:** Блокировка IP-адреса через iptables. С whitelist-проверкой и логированием.

```bash
#!/bin/bash
#======================================================================
# block_ip.sh — Блокировка IP-адресов
# Использование: ./block_ip.sh [-i ip] [-f file] [-u ip] [-l] [-h]
#   -i IP    Заблокировать один IP
#   -f FILE  Заблокировать все IP из файла (по одному на строку)
#   -u IP    Разблокировать IP
#   -l       Показать заблокированные
#======================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

WHITELIST_FILE="/etc/security/ip_whitelist.txt"
LOGFILE="/var/log/security/block_ip.log"

usage() {
    echo "Использование: $0 [опции]"
    echo "  -i IP    Заблокировать IP"
    echo "  -f FILE  Заблокировать IP из файла"
    echo "  -u IP    Разблокировать IP"
    echo "  -l       Показать заблокированные"
    echo "  -h       Справка"
    exit 0
}

mkdir -p "$(dirname "$LOGFILE")"
touch "$WHITELIST_FILE" 2>/dev/null || true

log_action() {
    local timestamp
    timestamp=$(date '+%Y-%m-%dT%H:%M:%S%z')
    echo "$timestamp $1" >> "$LOGFILE"
    echo -e "$2"
}

is_whitelisted() {
    grep -qF "$1" "$WHITELIST_FILE" 2>/dev/null
}

block_ip() {
    local ip="$1"
    # Валидация IP
    if ! [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(/[0-9]+)?$ ]]; then
        echo -e "${RED}[ERROR] Невалидный IP: $ip${NC}"
        return 1
    fi
    if is_whitelisted "$ip"; then
        log_action "BLOCKED_WHITELIST ip=$ip" "${YELLOW}[SKIP] $ip в whitelist — не блокируется${NC}"
        return 1
    fi
    # Проверка: уже заблокирован?
    if iptables -C INPUT -s "$ip" -j DROP 2>/dev/null; then
        echo -e "${YELLOW}[INFO] $ip уже заблокирован${NC}"
        return 0
    fi
    iptables -I INPUT -s "$ip" -j DROP
    # hosts.deny
    if ! grep -qF "$ip" /etc/hosts.deny 2>/dev/null; then
        echo "ALL: $ip" >> /etc/hosts.deny
    fi
    log_action "BLOCKED ip=$ip by=$(whoami)" "${GREEN}[BLOCKED] $ip${NC}"
}

unblock_ip() {
    local ip="$1"
    iptables -D INPUT -s "$ip" -j DROP 2>/dev/null || true
    sed -i "/$ip/d" /etc/hosts.deny 2>/dev/null || true
    log_action "UNBLOCKED ip=$ip by=$(whoami)" "${GREEN}[UNBLOCKED] $ip${NC}"
}

list_blocked() {
    echo -e "${CYAN}=== Заблокированные IP ===${NC}"
    iptables -L INPUT -n --line-numbers 2>/dev/null | grep DROP | grep -oP '\d+\.\d+\.\d+\.\d+(/\d+)?' | sort -u
}

ACTION=""
IP=""
FILE=""

while getopts "i:f:u:lh" opt; do
    case $opt in
        i) ACTION="block"; IP="$OPTARG" ;;
        f) ACTION="block_file"; FILE="$OPTARG" ;;
        u) ACTION="unblock"; IP="$OPTARG" ;;
        l) ACTION="list" ;;
        h) usage ;;
        *) usage ;;
    esac
done

[[ -z "$ACTION" ]] && usage

case "$ACTION" in
    block) block_ip "$IP" ;;
    block_file)
        if [[ ! -f "$FILE" ]]; then
            echo -e "${RED}Файл не найден: $FILE${NC}"
            exit 1
        fi
        count=0
        while IFS= read -r ip; do
            ip=$(echo "$ip" | tr -d '[:space:]')
            [[ -z "$ip" || "$ip" =~ ^# ]] && continue
            block_ip "$ip" && count=$((count+1))
        done < "$FILE"
        echo -e "\n${GREEN}Заблокировано: $count IP${NC}"
        ;;
    unblock) unblock_ip "$IP" ;;
    list) list_blocked ;;
esac
```

---

## 19. quarantine_file.sh

**Назначение:** Безопасный карантин подозрительного файла: перемещение с сохранением метаданных.

```bash
#!/bin/bash
#======================================================================
# quarantine_file.sh — Карантин подозрительного файла
# Использование: ./quarantine_file.sh [-f filepath] [-l] [-h]
#======================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

QUARANTINE_DIR="/var/quarantine"
LOGFILE="/var/log/security/quarantine.log"

usage() {
    echo "Использование: $0 [опции]"
    echo "  -f FILE  Файл для карантина"
    echo "  -l       Показать карантин"
    echo "  -h       Справка"
    exit 0
}

mkdir -p "$QUARANTINE_DIR" "$(dirname "$LOGFILE")"
chmod 700 "$QUARANTINE_DIR"

ACTION=""
FILEPATH=""

while getopts "f:lh" opt; do
    case $opt in
        f) ACTION="quarantine"; FILEPATH="$OPTARG" ;;
        l) ACTION="list" ;;
        h) usage ;;
        *) usage ;;
    esac
done
[[ -z "$ACTION" ]] && usage

quarantine_file() {
    local filepath="$1"
    if [[ ! -e "$filepath" ]]; then
        echo -e "${RED}[ERROR] Файл не найден: $filepath${NC}"
        exit 1
    fi

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local basename
    basename=$(basename "$filepath")
    local qdir="$QUARANTINE_DIR/${timestamp}_${basename}"
    mkdir -p "$qdir"

    # Сохраняем метаданные
    {
        echo "=== Метаданные карантина ==="
        echo "Дата:        $(date)"
        echo "Оператор:    $(whoami)"
        echo "Исходный путь: $(realpath "$filepath")"
        echo "Размер:      $(stat -c '%s' "$filepath") байт"
        echo "Права:       $(stat -c '%a' "$filepath")"
        echo "Владелец:    $(stat -c '%U:%G' "$filepath")"
        echo "Изменён:     $(stat -c '%y' "$filepath")"
        echo "SHA256:      $(sha256sum "$filepath" | awk '{print $1}')"
        echo "MD5:         $(md5sum "$filepath" | awk '{print $1}')"
        echo "Тип:         $(file "$filepath")"
        echo ""
        echo "=== ls -la ==="
        ls -la "$filepath"
    } > "$qdir/metadata.txt"

    # Перемещаем файл
    mv "$filepath" "$qdir/sample"
    chmod 000 "$qdir/sample"
    chmod 700 "$qdir"

    local timestamp_log
    timestamp_log=$(date '+%Y-%m-%dT%H:%M:%S%z')
    echo "$timestamp_log QUARANTINED file=$filepath dest=$qdir operator=$(whoami)" >> "$LOGFILE"
    logger -t "quarantine" -p auth.warning "File quarantined: $filepath → $qdir"

    echo -e "${GREEN}[КАРАНТИН]${NC} $filepath"
    echo "  Метаданные: $qdir/metadata.txt"
    echo "  Файл:       $qdir/sample"
}

list_quarantine() {
    echo -e "${CYAN}=== Карантин ===${NC}"
    if [[ -d "$QUARANTINE_DIR" ]]; then
        for d in "$QUARANTINE_DIR"/*/; do
            [[ -d "$d" ]] || continue
            echo -e "\n${CYAN}$(basename "$d")${NC}"
            if [[ -f "$d/metadata.txt" ]]; then
                grep -E "^(Исходный путь|SHA256|Размер):" "$d/metadata.txt" | sed 's/^/  /'
            fi
        done
    fi
    local total
    total=$(find "$QUARANTINE_DIR" -maxdepth 1 -type d | wc -l)
    echo -e "\nВсего: $((total - 1)) файлов в карантине"
}

case "$ACTION" in
    quarantine) quarantine_file "$FILEPATH" ;;
    list) list_quarantine ;;
esac
```

---

## 20. kill_suspicious.sh

**Назначение:** Остановка подозрительных процессов по имени, сетевому паттерну (reverse shell) или родителю. Сохраняет информацию о процессе перед kill.

```bash
#!/bin/bash
#======================================================================
# kill_suspicious.sh — Остановка подозрительных процессов
# Использование: ./kill_suspicious.sh [-n name] [-r] [-p ppid] [-h]
#   -n NAME  Убить процесс по имени
#   -r       Убить reverse shell паттерны
#   -p PPID  Убить всех потомков процесса
#======================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

LOGFILE="/var/log/security/kill_suspicious.log"
mkdir -p "$(dirname "$LOGFILE")"

usage() {
    echo "Использование: $0 [опции]"
    echo "  -n NAME  Убить по имени процесса"
    echo "  -r       Убить reverse shell паттерны"
    echo "  -p PPID  Убить потомков процесса"
    echo "  -h       Справка"
    exit 0
}

save_and_kill() {
    local pid="$1" reason="$2"
    if [[ ! -d "/proc/$pid" ]]; then
        return
    fi
    local comm user cmdline
    comm=$(cat "/proc/$pid/comm" 2>/dev/null || echo "?")
    user=$(ps -o user= -p "$pid" 2>/dev/null || echo "?")
    cmdline=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ' || echo "?")

    local timestamp
    timestamp=$(date '+%Y-%m-%dT%H:%M:%S%z')
    echo "$timestamp KILLED pid=$pid user=$user comm=$comm reason=$reason cmdline=$cmdline" >> "$LOGFILE"
    echo -e "${RED}[KILL]${NC} pid=$pid user=$user comm=$comm reason=$reason"

    kill -9 "$pid" 2>/dev/null || true
}

ACTION=""
NAME=""
PPID_TARGET=""

while getopts "n:rp:h" opt; do
    case $opt in
        n) ACTION="name"; NAME="$OPTARG" ;;
        r) ACTION="revshell" ;;
        p) ACTION="children"; PPID_TARGET="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done
[[ -z "$ACTION" ]] && usage

case "$ACTION" in
    name)
        pids=$(pgrep -x "$NAME" 2>/dev/null || true)
        if [[ -z "$pids" ]]; then
            echo -e "${GREEN}Процесс '$NAME' не найден${NC}"
            exit 0
        fi
        for pid in $pids; do
            save_and_kill "$pid" "name_match=$NAME"
        done
        ;;
    revshell)
        echo -e "${CYAN}=== Поиск reverse shell паттернов ===${NC}"
        killed=0
        # bash -i >&
        for pid in $(pgrep -f 'bash.*-i.*>&' 2>/dev/null || true); do
            save_and_kill "$pid" "reverse_shell_bash"
            killed=$((killed+1))
        done
        # nc -e
        for pid in $(pgrep -f 'nc.*-e\|ncat.*-e' 2>/dev/null || true); do
            save_and_kill "$pid" "reverse_shell_nc"
            killed=$((killed+1))
        done
        # python socket
        for pid in $(pgrep -f 'python.*import.*socket.*connect' 2>/dev/null || true); do
            save_and_kill "$pid" "reverse_shell_python"
            killed=$((killed+1))
        done
        # /dev/tcp
        for pid in $(pgrep -f '/dev/tcp/' 2>/dev/null || true); do
            save_and_kill "$pid" "reverse_shell_devtcp"
            killed=$((killed+1))
        done
        echo -e "\n${GREEN}Остановлено: $killed процессов${NC}"
        ;;
    children)
        echo -e "${CYAN}=== Убийство потомков PID $PPID_TARGET ===${NC}"
        children=$(pgrep -P "$PPID_TARGET" 2>/dev/null || true)
        if [[ -z "$children" ]]; then
            echo -e "${GREEN}Потомков не найдено${NC}"
            exit 0
        fi
        for pid in $children; do
            save_and_kill "$pid" "child_of=$PPID_TARGET"
        done
        ;;
esac
```

---

## 21. emergency_lockdown.sh

**Назначение:** Экстренная блокировка системы: отключение всех SSH кроме текущей, блокировка inbound, disable logins. С возможностью отката.

```bash
#!/bin/bash
#======================================================================
# emergency_lockdown.sh — Экстренная блокировка
# Использование: ./emergency_lockdown.sh [-a] [-r] [-h]
#   -a  Активировать lockdown
#   -r  Откатить lockdown
#======================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

LOCKDOWN_STATE="/var/lib/lockdown_state"
LOGFILE="/var/log/security/lockdown.log"

usage() {
    echo "Использование: $0 [опции]"
    echo "  -a           Активировать экстренную блокировку"
    echo "  -r           Откатить блокировку"
    echo "  -h           Справка"
    echo ""
    echo "  ВНИМАНИЕ: lockdown блокирует ВСЕ входящие соединения"
    echo "  кроме текущей SSH-сессии!"
    exit 0
}

mkdir -p "$(dirname "$LOCKDOWN_STATE")" "$(dirname "$LOGFILE")"

log_action() {
    local timestamp
    timestamp=$(date '+%Y-%m-%dT%H:%M:%S%z')
    echo "$timestamp $1" >> "$LOGFILE"
    logger -t "lockdown" -p auth.crit "$1"
}

activate_lockdown() {
    echo -e "${RED}!!! ЭКСТРЕННАЯ БЛОКИРОВКА !!!${NC}"
    echo ""

    # Сохраняем состояние для отката
    iptables-save > "$LOCKDOWN_STATE.iptables" 2>/dev/null || true

    # Определяем текущую SSH-сессию
    my_ip=$(who am i 2>/dev/null | grep -oP '\(\K[0-9.]+' || echo "")
    my_tty=$(tty 2>/dev/null || echo "")
    echo "my_ip=$my_ip" > "$LOCKDOWN_STATE"
    echo "my_tty=$my_tty" >> "$LOCKDOWN_STATE"
    echo "timestamp=$(date)" >> "$LOCKDOWN_STATE"

    echo -e "${YELLOW}Текущая сессия: IP=$my_ip TTY=$my_tty${NC}"

    # 1. Блокируем все входящие кроме нашего IP
    echo -e "\n${CYAN}[1] Блокировка входящих соединений${NC}"
    iptables -F INPUT
    iptables -P INPUT DROP
    iptables -A INPUT -i lo -j ACCEPT
    iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
    if [[ -n "$my_ip" ]]; then
        iptables -A INPUT -s "$my_ip" -p tcp --dport 22 -j ACCEPT
        echo -e "${GREEN}SSH разрешён только с $my_ip${NC}"
    fi
    log_action "LOCKDOWN_ACTIVATED by=$(whoami) from=$my_ip"

    # 2. Убиваем чужие SSH-сессии
    echo -e "\n${CYAN}[2] Завершение чужих SSH-сессий${NC}"
    who | while IFS= read -r line; do
        sess_ip=$(echo "$line" | grep -oP '\(\K[0-9.]+' || echo "")
        sess_tty=$(echo "$line" | awk '{print $2}')
        if [[ "$sess_ip" != "$my_ip" || "/dev/$sess_tty" != "$my_tty" ]]; then
            sess_user=$(echo "$line" | awk '{print $1}')
            sess_pid=$(ps aux | grep "sshd:.*$sess_user" | grep -v grep | awk '{print $2}' | head -1)
            if [[ -n "$sess_pid" ]]; then
                echo -e "${RED}  Завершение: $sess_user ($sess_ip) PID=$sess_pid${NC}"
                kill "$sess_pid" 2>/dev/null || true
                log_action "SESSION_KILLED user=$sess_user ip=$sess_ip pid=$sess_pid"
            fi
        fi
    done

    # 3. Запрет новых логинов
    echo -e "\n${CYAN}[3] Запрет новых логинов${NC}"
    touch /etc/nologin
    echo "Система в режиме экстренной блокировки. Вход запрещён." > /etc/nologin
    echo -e "${GREEN}/etc/nologin создан${NC}"

    echo -e "\n${RED}=== LOCKDOWN АКТИВЕН ===${NC}"
    echo -e "Для отката: $0 -r"
}

rollback_lockdown() {
    echo -e "${CYAN}=== Откат блокировки ===${NC}"

    # Восстанавливаем iptables
    if [[ -f "$LOCKDOWN_STATE.iptables" ]]; then
        iptables-restore < "$LOCKDOWN_STATE.iptables"
        echo -e "${GREEN}iptables восстановлены${NC}"
    else
        iptables -F INPUT
        iptables -P INPUT ACCEPT
        echo -e "${YELLOW}iptables сброшены в ACCEPT (бэкап не найден)${NC}"
    fi

    # Убираем nologin
    rm -f /etc/nologin
    echo -e "${GREEN}/etc/nologin удалён${NC}"

    log_action "LOCKDOWN_DEACTIVATED by=$(whoami)"
    rm -f "$LOCKDOWN_STATE" "$LOCKDOWN_STATE.iptables"

    echo -e "\n${GREEN}=== LOCKDOWN СНЯТ ===${NC}"
}

ACTION=""
while getopts "arh" opt; do
    case $opt in
        a) ACTION="activate" ;;
        r) ACTION="rollback" ;;
        h) usage ;;
        *) usage ;;
    esac
done
[[ -z "$ACTION" ]] && usage

case "$ACTION" in
    activate) activate_lockdown ;;
    rollback) rollback_lockdown ;;
esac
```

---

# ЧАСТЬ 4: УТИЛИТЫ

---

## 22. log_parser.sh

**Назначение:** Универсальный парсер логов: поиск IP, email, URL, хешей. Фильтрация по дате, статистика.

```bash
#!/bin/bash
#======================================================================
# log_parser.sh — Универсальный парсер логов
# Использование: ./log_parser.sh [-f logfile] [-p pattern] [-t type] [-n top_n] [-h]
#   -t  Тип паттерна: ip, email, url, hash, custom
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

LOGFILE=""
PATTERN_TYPE="ip"
CUSTOM_PATTERN=""
TOP_N=20
DATE_FROM=""
DATE_TO=""

usage() {
    echo "Использование: $0 [опции]"
    echo "  -f FILE     Файл лога"
    echo "  -t TYPE     Тип: ip, email, url, hash, custom"
    echo "  -p PATTERN  Кастомный regex (для -t custom)"
    echo "  -n NUM      Top-N результатов (по умолчанию: $TOP_N)"
    echo "  -h          Справка"
    exit 0
}

while getopts "f:t:p:n:h" opt; do
    case $opt in
        f) LOGFILE="$OPTARG" ;;
        t) PATTERN_TYPE="$OPTARG" ;;
        p) CUSTOM_PATTERN="$OPTARG" ;;
        n) TOP_N="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

[[ -z "$LOGFILE" ]] && { echo -e "${RED}Укажите файл: -f FILE${NC}"; exit 1; }
[[ ! -f "$LOGFILE" ]] && { echo -e "${RED}Файл не найден: $LOGFILE${NC}"; exit 1; }

# Регулярные выражения
declare -A PATTERNS=(
    ["ip"]='[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'
    ["email"]='[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    ["url"]='https?://[a-zA-Z0-9./?=&_%-]+'
    ["hash"]='[a-fA-F0-9]{32,64}'
)

if [[ "$PATTERN_TYPE" == "custom" ]]; then
    regex="$CUSTOM_PATTERN"
else
    regex="${PATTERNS[$PATTERN_TYPE]:-${PATTERNS[ip]}}"
fi

echo -e "${CYAN}=== Парсинг логов ===${NC}"
echo "Файл: $LOGFILE ($(wc -l < "$LOGFILE") строк)"
echo "Паттерн: $PATTERN_TYPE ($regex)"

# Извлечение
matches=$(grep -oP "$regex" "$LOGFILE" 2>/dev/null || true)
total=$(echo "$matches" | grep -c '.' || echo 0)
unique=$(echo "$matches" | sort -u | grep -c '.' || echo 0)

echo -e "\nВсего совпадений: $total"
echo -e "Уникальных: $unique"

# Top-N
echo -e "\n${YELLOW}Top-$TOP_N:${NC}"
echo "$matches" | sort | uniq -c | sort -rn | head -"$TOP_N" | while read -r count value; do
    printf "  %6d  %s\n" "$count" "$value"
done

# Для IP — дополнительная инфо
if [[ "$PATTERN_TYPE" == "ip" ]]; then
    echo -e "\n${YELLOW}Приватные vs публичные IP:${NC}"
    private=$(echo "$matches" | grep -cE '^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.|127\.)' || echo 0)
    public=$((total - private))
    echo "  Приватные: $private"
    echo "  Публичные: $public"
fi
```

---

## 23. ioc_search.sh

**Назначение:** Поиск Indicators of Compromise на хосте: по IP, хешу, домену, имени файла.

```bash
#!/bin/bash
#======================================================================
# ioc_search.sh — Поиск IoC на хосте
# Использование: ./ioc_search.sh [-i ip] [-d domain] [-s hash] [-n filename] [-f ioc_file] [-h]
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

usage() {
    echo "Использование: $0 [опции]"
    echo "  -i IP       Поиск IP в логах и соединениях"
    echo "  -d DOMAIN   Поиск домена"
    echo "  -s HASH     Поиск файла по SHA256/MD5"
    echo "  -n NAME     Поиск файла по имени"
    echo "  -f FILE     Файл с IoC (по одному на строку)"
    echo "  -h          Справка"
    exit 0
}

search_ip() {
    local ip="$1"
    echo -e "${CYAN}=== Поиск IP: $ip ===${NC}"
    local found=0

    # Активные соединения
    echo -e "\n${YELLOW}[1] Активные соединения:${NC}"
    ss_result=$(ss -tupna 2>/dev/null | grep "$ip" || true)
    if [[ -n "$ss_result" ]]; then
        echo "$ss_result"
        found=$((found+1))
    else
        echo "  Не найден"
    fi

    # Логи
    echo -e "\n${YELLOW}[2] В логах:${NC}"
    for logfile in /var/log/auth.log /var/log/syslog /var/log/apache2/access.log /var/log/nginx/access.log; do
        if [[ -f "$logfile" ]]; then
            count=$(grep -c "$ip" "$logfile" 2>/dev/null || echo 0)
            if (( count > 0 )); then
                echo -e "  ${RED}$logfile: $count вхождений${NC}"
                grep "$ip" "$logfile" | tail -3 | sed 's/^/    /'
                found=$((found+1))
            fi
        fi
    done

    # iptables
    echo -e "\n${YELLOW}[3] В iptables:${NC}"
    ipt=$(iptables -L -n 2>/dev/null | grep "$ip" || true)
    if [[ -n "$ipt" ]]; then
        echo "$ipt" | sed 's/^/  /'
    else
        echo "  Не найден"
    fi

    echo -e "\n${CYAN}Итог: найден в $found источниках${NC}"
}

search_domain() {
    local domain="$1"
    echo -e "${CYAN}=== Поиск домена: $domain ===${NC}"

    # DNS cache (systemd-resolved)
    echo -e "\n${YELLOW}[1] DNS cache:${NC}"
    resolvectl query "$domain" 2>/dev/null || echo "  Не найден в кеше"

    # Логи
    echo -e "\n${YELLOW}[2] В логах:${NC}"
    grep -rh "$domain" /var/log/ 2>/dev/null | tail -10 | sed 's/^/  /' || echo "  Не найден"

    # /etc/hosts
    echo -e "\n${YELLOW}[3] В /etc/hosts:${NC}"
    grep "$domain" /etc/hosts 2>/dev/null || echo "  Не найден"
}

search_hash() {
    local hash="$1"
    local hash_len=${#hash}
    echo -e "${CYAN}=== Поиск хеша: ${hash:0:16}... ===${NC}"

    local cmd="sha256sum"
    if (( hash_len == 32 )); then
        cmd="md5sum"
    fi

    echo -e "${YELLOW}Поиск в ключевых каталогах...${NC}"
    for dir in /tmp /var/tmp /home /opt /usr/local/bin /etc/cron.d; do
        if [[ -d "$dir" ]]; then
            find "$dir" -type f -size +0 2>/dev/null | head -1000 | while IFS= read -r f; do
                file_hash=$($cmd "$f" 2>/dev/null | awk '{print $1}')
                if [[ "$file_hash" == "$hash" ]]; then
                    echo -e "${RED}[НАЙДЕН] $f${NC}"
                    ls -la "$f"
                fi
            done
        fi
    done
    echo -e "${GREEN}Поиск завершён${NC}"
}

search_filename() {
    local name="$1"
    echo -e "${CYAN}=== Поиск файла: $name ===${NC}"
    find / -name "$name" -type f 2>/dev/null | while IFS= read -r f; do
        echo -e "${RED}[НАЙДЕН]${NC} $f"
        ls -la "$f" | sed 's/^/  /'
        echo "  SHA256: $(sha256sum "$f" 2>/dev/null | awk '{print $1}')"
    done
}

IOC_IP="" IOC_DOMAIN="" IOC_HASH="" IOC_NAME="" IOC_FILE=""

while getopts "i:d:s:n:f:h" opt; do
    case $opt in
        i) IOC_IP="$OPTARG" ;;
        d) IOC_DOMAIN="$OPTARG" ;;
        s) IOC_HASH="$OPTARG" ;;
        n) IOC_NAME="$OPTARG" ;;
        f) IOC_FILE="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

[[ -n "$IOC_IP" ]] && search_ip "$IOC_IP"
[[ -n "$IOC_DOMAIN" ]] && search_domain "$IOC_DOMAIN"
[[ -n "$IOC_HASH" ]] && search_hash "$IOC_HASH"
[[ -n "$IOC_NAME" ]] && search_filename "$IOC_NAME"

if [[ -n "$IOC_FILE" ]]; then
    echo -e "${CYAN}=== Массовый поиск IoC из файла: $IOC_FILE ===${NC}"
    while IFS= read -r ioc; do
        ioc=$(echo "$ioc" | tr -d '[:space:]')
        [[ -z "$ioc" || "$ioc" =~ ^# ]] && continue
        if [[ "$ioc" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            search_ip "$ioc"
        elif [[ "$ioc" =~ ^[a-fA-F0-9]{32,64}$ ]]; then
            search_hash "$ioc"
        else
            search_domain "$ioc"
        fi
        echo "---"
    done < "$IOC_FILE"
fi

[[ -z "$IOC_IP$IOC_DOMAIN$IOC_HASH$IOC_NAME$IOC_FILE" ]] && usage
```

---

## 24. backup_verify.sh

**Назначение:** Проверка бэкапов: существование, возраст, размер, hash verification. Алерт если бэкап устарел.

```bash
#!/bin/bash
#======================================================================
# backup_verify.sh — Проверка состояния бэкапов
# Использование: ./backup_verify.sh [-d backup_dir] [-m max_age_days] [-h]
#======================================================================

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

BACKUP_DIR="/var/backups"
MAX_AGE_DAYS=7
MIN_SIZE_BYTES=1024
LOGFILE="/var/log/security/backup_verify.log"

usage() {
    echo "Использование: $0 [опции]"
    echo "  -d DIR   Каталог бэкапов (по умолчанию: $BACKUP_DIR)"
    echo "  -m DAYS  Макс. возраст в днях (по умолчанию: $MAX_AGE_DAYS)"
    echo "  -h       Справка"
    exit 0
}

while getopts "d:m:h" opt; do
    case $opt in
        d) BACKUP_DIR="$OPTARG" ;;
        m) MAX_AGE_DAYS="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

mkdir -p "$(dirname "$LOGFILE")"

echo -e "${CYAN}=== Проверка бэкапов ===${NC}"
echo "Каталог: $BACKUP_DIR"
echo "Макс. возраст: $MAX_AGE_DAYS дней"

if [[ ! -d "$BACKUP_DIR" ]]; then
    echo -e "${RED}[ALERT] Каталог бэкапов не существует: $BACKUP_DIR${NC}"
    exit 1
fi

alerts=0
ok=0
total=0

find "$BACKUP_DIR" -maxdepth 2 -type f \( -name "*.tar.gz" -o -name "*.gz" -o -name "*.bak" -o -name "*.sql" -o -name "*.dump" -o -name "*.zip" \) 2>/dev/null | sort | while IFS= read -r backup; do
    total=$((total+1))
    name=$(basename "$backup")
    size=$(stat -c '%s' "$backup" 2>/dev/null || echo 0)
    age_sec=$(( $(date +%s) - $(stat -c '%Y' "$backup" 2>/dev/null || echo 0) ))
    age_days=$(( age_sec / 86400 ))

    size_human=$(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B")

    echo -e "\n${CYAN}$name${NC}"
    echo "  Путь:    $backup"
    echo "  Размер:  $size_human"
    echo "  Возраст: $age_days дней"

    # Проверка размера
    if (( size < MIN_SIZE_BYTES )); then
        echo -e "  ${RED}[FAIL] Подозрительно маленький (<1KB) — возможно пустой!${NC}"
        alerts=$((alerts+1))
    else
        echo -e "  ${GREEN}[OK] Размер${NC}"
    fi

    # Проверка возраста
    if (( age_days > MAX_AGE_DAYS )); then
        echo -e "  ${RED}[FAIL] Устарел (>${MAX_AGE_DAYS} дней)${NC}"
        alerts=$((alerts+1))
    else
        echo -e "  ${GREEN}[OK] Свежий${NC}"
    fi

    # Проверка целостности (для .gz)
    if [[ "$backup" == *.gz || "$backup" == *.tar.gz ]]; then
        if gzip -t "$backup" 2>/dev/null; then
            echo -e "  ${GREEN}[OK] Целостность (gzip test)${NC}"
        else
            echo -e "  ${RED}[FAIL] Повреждён (gzip test failed)!${NC}"
            alerts=$((alerts+1))
        fi
    fi
done

echo -e "\n${CYAN}--- Итог ---${NC}"
backup_count=$(find "$BACKUP_DIR" -maxdepth 2 -type f \( -name "*.tar.gz" -o -name "*.gz" -o -name "*.bak" -o -name "*.sql" -o -name "*.dump" -o -name "*.zip" \) 2>/dev/null | wc -l)
echo "Бэкапов найдено: $backup_count"

if (( backup_count == 0 )); then
    echo -e "${RED}[CRITICAL] Бэкапы не найдены!${NC}"
    logger -t "backup_verify" -p auth.crit "No backups found in $BACKUP_DIR"
fi
```

---

## 25. security_report.sh

**Назначение:** Ежедневный отчёт безопасности в HTML-формате. Неудачные входы, изменения, обновления, порты.

```bash
#!/bin/bash
#======================================================================
# security_report.sh — Ежедневный отчёт безопасности
# Использование: ./security_report.sh [-o output] [-e email] [-h]
#======================================================================

set -euo pipefail

CYAN='\033[0;36m'
NC='\033[0m'

OUTPUT="/var/log/security/daily_report_$(date +%Y%m%d).html"
EMAIL=""
HOSTNAME_TAG=$(hostname -f 2>/dev/null || hostname)
REPORT_DATE=$(date '+%Y-%m-%d %H:%M')

usage() {
    echo "Использование: $0 [опции]"
    echo "  -o FILE  Файл отчёта (HTML)"
    echo "  -e ADDR  Отправить на email (требует mailutils)"
    echo "  -h       Справка"
    exit 0
}

while getopts "o:e:h" opt; do
    case $opt in
        o) OUTPUT="$OPTARG" ;;
        e) EMAIL="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

mkdir -p "$(dirname "$OUTPUT")"

echo -e "${CYAN}=== Генерация отчёта безопасности ===${NC}"

# Сбор данных
FAILED_LOGINS=$(grep -c "Failed password" /var/log/auth.log 2>/dev/null || echo 0)
FAILED_IPS=$(grep "Failed password" /var/log/auth.log 2>/dev/null | grep -oP 'from \K[0-9.]+' | sort | uniq -c | sort -rn | head -10)
SUCCESS_LOGINS=$(grep -c "Accepted" /var/log/auth.log 2>/dev/null || echo 0)
ROOT_LOGINS=$(grep -c "session opened.*root" /var/log/auth.log 2>/dev/null || echo 0)
CURRENT_USERS=$(who 2>/dev/null | wc -l)
LISTENING_PORTS=$(ss -tlnp 2>/dev/null | tail -n +2 | wc -l)
PROCESS_COUNT=$(ps aux --no-headers 2>/dev/null | wc -l)
DISK_USAGE=$(df -h --output=target,pcent 2>/dev/null | tail -n +2 | awk '{printf "<tr><td>%s</td><td>%s</td></tr>", $1, $2}')
UPTIME=$(uptime -p 2>/dev/null || uptime)
MODIFIED_FILES=$(find /etc -mtime -1 -type f 2>/dev/null | wc -l)
NEW_USERS=$(grep -c "new user" /var/log/auth.log 2>/dev/null || echo 0)
UPDATES_AVAILABLE=$(apt list --upgradable 2>/dev/null | grep -c "upgradable" || echo "N/A")
SUID_COUNT=$(find / -xdev -perm -4000 -type f 2>/dev/null | wc -l)
CPU_LOAD=$(cat /proc/loadavg | awk '{print $1, $2, $3}')
MEM_USED=$(free -h | awk '/Mem:/ {print $3 "/" $2}')

# Генерация HTML
cat > "$OUTPUT" << HTMLEOF
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Отчёт безопасности — $HOSTNAME_TAG</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
.container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #34495e; margin-top: 25px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background: #3498db; color: white; }
tr:nth-child(even) { background: #f2f2f2; }
.alert { color: #e74c3c; font-weight: bold; }
.ok { color: #27ae60; }
.warn { color: #f39c12; }
.metric { display: inline-block; background: #ecf0f1; padding: 10px 20px; margin: 5px; border-radius: 5px; text-align: center; }
.metric .value { font-size: 24px; font-weight: bold; color: #2c3e50; }
.metric .label { font-size: 12px; color: #7f8c8d; }
pre { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }
</style>
</head>
<body>
<div class="container">
<h1>Отчёт безопасности</h1>
<p><strong>Хост:</strong> $HOSTNAME_TAG &nbsp; | &nbsp; <strong>Дата:</strong> $REPORT_DATE &nbsp; | &nbsp; <strong>Uptime:</strong> $UPTIME</p>

<h2>Ключевые метрики</h2>
<div>
<div class="metric"><div class="value">$FAILED_LOGINS</div><div class="label">Неудачных входов</div></div>
<div class="metric"><div class="value">$SUCCESS_LOGINS</div><div class="label">Успешных входов</div></div>
<div class="metric"><div class="value">$ROOT_LOGINS</div><div class="label">Входов root</div></div>
<div class="metric"><div class="value">$CURRENT_USERS</div><div class="label">Сейчас онлайн</div></div>
<div class="metric"><div class="value">$LISTENING_PORTS</div><div class="label">Открытых портов</div></div>
<div class="metric"><div class="value">$MODIFIED_FILES</div><div class="label">Изменённых в /etc</div></div>
<div class="metric"><div class="value">$NEW_USERS</div><div class="label">Новых пользователей</div></div>
<div class="metric"><div class="value">$SUID_COUNT</div><div class="label">SUID-файлов</div></div>
</div>

<h2>Неудачные входы — Top-10 IP</h2>
<table>
<tr><th>Попыток</th><th>IP-адрес</th></tr>
$(echo "$FAILED_IPS" | awk '{printf "<tr><td>%s</td><td>%s</td></tr>\n", $1, $2}')
</table>

<h2>Использование дисков</h2>
<table>
<tr><th>Точка монтирования</th><th>Использование</th></tr>
$DISK_USAGE
</table>

<h2>Система</h2>
<table>
<tr><td>CPU Load</td><td>$CPU_LOAD</td></tr>
<tr><td>RAM</td><td>$MEM_USED</td></tr>
<tr><td>Процессов</td><td>$PROCESS_COUNT</td></tr>
<tr><td>Обновлений ожидает</td><td>$UPDATES_AVAILABLE</td></tr>
</table>

<h2>Listening сервисы</h2>
<pre>$(ss -tlnp 2>/dev/null | head -20)</pre>

<h2>Последние входы</h2>
<pre>$(last -10 2>/dev/null)</pre>

<h2>Изменённые файлы в /etc (за 24 часа)</h2>
<pre>$(find /etc -mtime -1 -type f 2>/dev/null | head -30)</pre>

<hr>
<p style="color: #7f8c8d; font-size: 12px;">Сгенерировано security_report.sh · $(date)</p>
</div>
</body>
</html>
HTMLEOF

echo -e "${CYAN}Отчёт сохранён: $OUTPUT${NC}"

# Отправка email
if [[ -n "$EMAIL" ]]; then
    if command -v mail &>/dev/null; then
        mail -s "Security Report: $HOSTNAME_TAG ($REPORT_DATE)" -a "Content-Type: text/html" "$EMAIL" < "$OUTPUT"
        echo -e "${CYAN}Отправлен на: $EMAIL${NC}"
    else
        echo -e "${YELLOW}mailutils не установлен — email не отправлен${NC}"
    fi
fi

# Краткий вывод в терминал
echo -e "\n${CYAN}Краткий итог:${NC}"
echo "  Неудачных входов:    $FAILED_LOGINS"
echo "  Успешных входов:     $SUCCESS_LOGINS"
echo "  Входов root:         $ROOT_LOGINS"
echo "  Открытых портов:     $LISTENING_PORTS"
echo "  Изменённых в /etc:   $MODIFIED_FILES"
echo "  SUID-файлов:         $SUID_COUNT"
```

**Cron для ежедневного отчёта:**
```bash
0 7 * * * /opt/scripts/security_report.sh -e admin@company.ru >/dev/null 2>&1
```

---

# СПРАВОЧНИК: БЫСТРЫЙ СТАРТ

## Установка всех скриптов

```bash
# Создаём каталог
sudo mkdir -p /opt/security-scripts
sudo chmod 750 /opt/security-scripts

# Копируем скрипты
sudo cp *.sh /opt/security-scripts/
sudo chmod +x /opt/security-scripts/*.sh

# Создаём каталоги для логов и данных
sudo mkdir -p /var/log/security
sudo mkdir -p /var/lib/fim
sudo mkdir -p /var/quarantine
sudo chmod 700 /var/quarantine

# Создаём whitelist-файлы
sudo touch /etc/security/known_ssh_ips.txt
sudo touch /etc/security/ip_whitelist.txt
sudo touch /etc/security/process_whitelist.txt
```

## Рекомендуемый crontab

```bash
# Мониторинг (каждые 5 минут)
*/5 * * * * /opt/security-scripts/system_monitor.sh -s >/dev/null 2>&1
*/5 * * * * /opt/security-scripts/login_monitor.sh >/dev/null 2>&1

# Целостность (каждый час)
0 * * * * /opt/security-scripts/file_integrity.sh -c >/dev/null 2>&1

# Аудит (ежедневно)
0 6 * * * /opt/security-scripts/user_audit.sh >/dev/null 2>&1
0 6 * * * /opt/security-scripts/cron_audit.sh >/dev/null 2>&1
30 6 * * * /opt/security-scripts/network_baseline.sh -c >/dev/null 2>&1
0 3 * * * /opt/security-scripts/process_watch.sh >/dev/null 2>&1

# CIS-проверка (еженедельно)
0 2 * * 1 /opt/security-scripts/quick_cis_check.sh > /var/log/security/cis_weekly.log 2>&1

# Отчёт (ежедневно в 7:00)
0 7 * * * /opt/security-scripts/security_report.sh >/dev/null 2>&1

# Проверка бэкапов (ежедневно)
0 8 * * * /opt/security-scripts/backup_verify.sh >/dev/null 2>&1
```

## Первоначальная настройка

```bash
# 1. Создать baseline для FIM
sudo /opt/security-scripts/file_integrity.sh -b

# 2. Создать network baseline
sudo /opt/security-scripts/network_baseline.sh -b

# 3. Добавить известные SSH IP
echo "10.0.0.1" | sudo tee /etc/security/known_ssh_ips.txt
echo "192.168.1.100" | sudo tee -a /etc/security/known_ssh_ips.txt

# 4. Добавить IP в whitelist (не блокировать)
echo "10.0.0.1" | sudo tee /etc/security/ip_whitelist.txt

# 5. Запустить CIS-проверку
sudo /opt/security-scripts/quick_cis_check.sh

# 6. Применить hardening (осторожно!)
sudo /opt/security-scripts/kernel_hardening.sh -a
sudo /opt/security-scripts/ssh_hardening.sh -a
sudo /opt/security-scripts/password_policy.sh -a
```

---

*25 скриптов · Мониторинг · Hardening · Реагирование · Утилиты*
*Все скрипты — оборонительные. Для курса «Кибер-Щит».*
