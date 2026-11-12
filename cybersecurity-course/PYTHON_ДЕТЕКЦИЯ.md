# PYTHON-СКРИПТЫ ДЕТЕКЦИИ И АНАЛИЗА

> 15 Python-инструментов для SOC-аналитика: сетевой анализ, корреляция логов,
> threat intelligence, анализ артефактов, автоматизация SIEM.
> Все скрипты — **оборонительные**. Только обнаружение и защита.

---

## СОДЕРЖАНИЕ

1. [Сетевой анализатор (Scapy)](#1-сетевой-анализатор)
2. [Коррелятор логов](#2-коррелятор-логов)
3. [YARA-сканер](#3-yara-сканер)
4. [Threat Intel агрегатор](#4-threat-intel-агрегатор)
5. [Email-анализатор (антифишинг)](#5-email-анализатор)
6. [Анализатор PE-файлов](#6-анализатор-pe-файлов)
7. [Детектор аномалий трафика](#7-детектор-аномалий-трафика)
8. [Web-сканер безопасности](#8-web-сканер-безопасности)
9. [Коррелятор событий SIEM](#9-коррелятор-событий-siem)
10. [Автоматизация Wazuh API](#10-автоматизация-wazuh-api)
11. [Парсер Windows Event Log](#11-парсер-windows-event-log)
12. [Сканер уязвимостей зависимостей](#12-сканер-уязвимостей-зависимостей)
13. [Генератор отчётов безопасности](#13-генератор-отчётов-безопасности)
14. [Honeypot-детектор](#14-honeypot-детектор)
15. [Мониторинг DNS-запросов](#15-мониторинг-dns-запросов)

---

## 1. Сетевой анализатор

**Назначение:** Пассивный захват и анализ сетевого трафика. Обнаружение port scan, ARP spoofing, DNS tunneling. Использует scapy.

**Зависимости:** `pip install scapy`

```python
#!/usr/bin/env python3
"""
network_analyzer.py — Сетевой анализатор для SOC
Обнаружение: port scan, ARP spoofing, DNS tunneling, подозрительные пакеты.
Запуск: sudo python3 network_analyzer.py -i eth0 -d 300
"""

import argparse
import sys
import time
import json
from collections import defaultdict
from datetime import datetime

try:
    from scapy.all import (
        sniff, IP, TCP, UDP, DNS, DNSQR, ARP, Ether,
        conf, get_if_list
    )
except ImportError:
    print("[!] pip install scapy")
    sys.exit(1)


class NetworkAnalyzer:
    """Пассивный сетевой анализатор для обнаружения аномалий."""

    def __init__(self, interface="eth0", duration=300, verbose=False):
        self.interface = interface
        self.duration = duration
        self.verbose = verbose

        # Счётчики для обнаружения
        self.syn_count = defaultdict(lambda: defaultdict(int))
        self.arp_table = {}
        self.dns_queries = defaultdict(list)
        self.connection_count = defaultdict(int)
        self.alerts = []
        self.packet_count = 0
        self.start_time = None

        # Пороги обнаружения
        self.PORT_SCAN_THRESHOLD = 15      # портов за окно
        self.PORT_SCAN_WINDOW = 60         # секунд
        self.DNS_TUNNEL_LEN = 50           # длина домена
        self.DNS_TUNNEL_FREQ = 20          # запросов к одному домену за мин
        self.CONN_FLOOD_THRESHOLD = 100    # соединений от одного IP

    def alert(self, category, severity, message, details=None):
        """Формирование алерта в формате, совместимом с SIEM."""
        alert_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "category": category,
            "severity": severity,
            "message": message,
            "details": details or {},
            "analyzer": "network_analyzer",
            "interface": self.interface,
        }
        self.alerts.append(alert_obj)

        severity_colors = {
            "critical": "\033[91m",
            "high": "\033[91m",
            "medium": "\033[93m",
            "low": "\033[94m",
        }
        color = severity_colors.get(severity, "\033[0m")
        print(f"{color}[{severity.upper()}] {category}: {message}\033[0m")
        if self.verbose and details:
            print(f"  Детали: {json.dumps(details, ensure_ascii=False)}")

    def detect_port_scan(self, pkt):
        """Обнаружение сканирования портов (SYN scan)."""
        if TCP in pkt and pkt[TCP].flags == 0x02:  # SYN flag
            src = pkt[IP].src
            dst_port = pkt[TCP].dport
            self.syn_count[src][dst_port] = int(time.time())

            unique_ports = len(self.syn_count[src])
            if unique_ports >= self.PORT_SCAN_THRESHOLD:
                self.alert(
                    "PORT_SCAN",
                    "high",
                    f"Обнаружено сканирование портов с {src}",
                    {
                        "source_ip": src,
                        "ports_scanned": unique_ports,
                        "destination": pkt[IP].dst,
                        "sample_ports": list(self.syn_count[src].keys())[:20],
                    },
                )
                self.syn_count[src].clear()

    def detect_arp_spoof(self, pkt):
        """Обнаружение ARP spoofing (смена MAC для известного IP)."""
        if ARP in pkt and pkt[ARP].op == 2:  # is-at (reply)
            ip = pkt[ARP].psrc
            mac = pkt[ARP].hwsrc

            if ip in self.arp_table:
                if self.arp_table[ip] != mac:
                    self.alert(
                        "ARP_SPOOFING",
                        "critical",
                        f"ARP spoof: IP {ip} сменил MAC {self.arp_table[ip]} → {mac}",
                        {
                            "ip": ip,
                            "old_mac": self.arp_table[ip],
                            "new_mac": mac,
                        },
                    )
            self.arp_table[ip] = mac

    def detect_dns_tunnel(self, pkt):
        """Обнаружение DNS tunneling по длине домена и частоте запросов."""
        if DNS in pkt and pkt[DNS].qr == 0 and DNSQR in pkt:
            query_name = pkt[DNSQR].qname.decode("utf-8", errors="ignore").rstrip(".")
            src = pkt[IP].src if IP in pkt else "unknown"

            # Длина домена
            if len(query_name) > self.DNS_TUNNEL_LEN:
                self.alert(
                    "DNS_TUNNELING",
                    "high",
                    f"Подозрительно длинный DNS-запрос от {src}",
                    {
                        "source": src,
                        "domain": query_name,
                        "length": len(query_name),
                    },
                )

            # Частота запросов к одному базовому домену
            parts = query_name.split(".")
            base_domain = ".".join(parts[-2:]) if len(parts) >= 2 else query_name
            self.dns_queries[base_domain].append(time.time())

            # Чистим старые записи
            cutoff = time.time() - 60
            self.dns_queries[base_domain] = [
                t for t in self.dns_queries[base_domain] if t > cutoff
            ]

            freq = len(self.dns_queries[base_domain])
            if freq > self.DNS_TUNNEL_FREQ:
                self.alert(
                    "DNS_TUNNELING",
                    "medium",
                    f"Высокая частота DNS к {base_domain}: {freq} запросов/мин",
                    {
                        "source": src,
                        "domain": base_domain,
                        "frequency": freq,
                    },
                )

    def detect_connection_flood(self, pkt):
        """Обнаружение flood'а соединений от одного IP."""
        if IP in pkt:
            src = pkt[IP].src
            self.connection_count[src] += 1

            if self.connection_count[src] == self.CONN_FLOOD_THRESHOLD:
                self.alert(
                    "CONNECTION_FLOOD",
                    "medium",
                    f"Много пакетов от {src}: {self.connection_count[src]}",
                    {"source_ip": src, "packet_count": self.connection_count[src]},
                )

    def process_packet(self, pkt):
        """Обработка каждого пакета всеми детекторами."""
        self.packet_count += 1
        self.detect_port_scan(pkt)
        self.detect_arp_spoof(pkt)
        self.detect_dns_tunnel(pkt)
        self.detect_connection_flood(pkt)

    def run(self):
        """Запуск захвата и анализа."""
        print(f"[*] Запуск на интерфейсе {self.interface} ({self.duration} сек)")
        print(f"[*] Пороги: port_scan={self.PORT_SCAN_THRESHOLD}, "
              f"dns_tunnel_len={self.DNS_TUNNEL_LEN}")
        self.start_time = time.time()

        try:
            sniff(
                iface=self.interface,
                prn=self.process_packet,
                timeout=self.duration,
                store=False,
            )
        except PermissionError:
            print("[!] Нужен root: sudo python3 network_analyzer.py")
            return
        except KeyboardInterrupt:
            print("\n[*] Остановлено пользователем")

        self.report()

    def report(self):
        """Итоговый отчёт."""
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"ИТОГ: {self.packet_count} пакетов за {elapsed:.0f} сек")
        print(f"Алертов: {len(self.alerts)}")
        for a in self.alerts:
            print(f"  [{a['severity']}] {a['category']}: {a['message']}")

        # Сохранение в JSON для SIEM
        report_file = f"/var/log/security/net_analyzer_{int(time.time())}.json"
        try:
            with open(report_file, "w") as f:
                json.dump(
                    {
                        "start": self.start_time,
                        "duration": elapsed,
                        "packets": self.packet_count,
                        "alerts": self.alerts,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"Отчёт: {report_file}")
        except PermissionError:
            print("(не удалось сохранить — нет доступа к /var/log/security)")


def main():
    parser = argparse.ArgumentParser(description="Сетевой анализатор SOC")
    parser.add_argument("-i", "--interface", default="eth0", help="Интерфейс")
    parser.add_argument("-d", "--duration", type=int, default=300, help="Секунды")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    analyzer = NetworkAnalyzer(args.interface, args.duration, args.verbose)
    analyzer.run()


if __name__ == "__main__":
    main()
```

**Интеграция в SOC:** JSON-отчёт подхватывается Filebeat → ELK. Алерты формируют дашборд «Network Anomalies».

---

## 2. Коррелятор логов

**Назначение:** Парсинг и корреляция событий из нескольких источников (auth.log, syslog, apache). Обнаружение атакующих цепочек.

**Зависимости:** `pip install python-dateutil`

```python
#!/usr/bin/env python3
"""
log_correlator.py — Коррелятор логов для SOC
Объединяет события из auth.log, syslog, apache access.log.
Обнаруживает: brute-force → login → lateral movement цепочки.
Запуск: python3 log_correlator.py -a /var/log/auth.log -s /var/log/syslog
"""

import re
import argparse
import json
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path


class LogEvent:
    """Нормализованное событие."""

    def __init__(self, timestamp, source, event_type, severity, ip=None,
                 user=None, details=None):
        self.timestamp = timestamp
        self.source = source
        self.event_type = event_type
        self.severity = severity
        self.ip = ip
        self.user = user
        self.details = details or {}

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "type": self.event_type,
            "severity": self.severity,
            "ip": self.ip,
            "user": self.user,
            "details": self.details,
        }


class LogCorrelator:
    """Движок корреляции логов."""

    # Регулярные выражения для парсинга
    AUTH_FAILED = re.compile(
        r"(\w+\s+\d+\s+[\d:]+)\s+\S+\s+sshd.*Failed password for "
        r"(?:invalid user )?(\S+) from ([\d.]+)"
    )
    AUTH_ACCEPTED = re.compile(
        r"(\w+\s+\d+\s+[\d:]+)\s+\S+\s+sshd.*Accepted \S+ for "
        r"(\S+) from ([\d.]+)"
    )
    AUTH_SUDO = re.compile(
        r"(\w+\s+\d+\s+[\d:]+)\s+\S+\s+sudo:\s+(\S+)\s.*COMMAND=(.*)"
    )
    SYSLOG_SERVICE = re.compile(
        r"(\w+\s+\d+\s+[\d:]+)\s+\S+\s+systemd.*: (Started|Stopped) (.*)"
    )
    APACHE_ACCESS = re.compile(
        r"([\d.]+) - - \[([^\]]+)\] \"(\S+) (\S+) \S+\" (\d+) (\d+)"
    )

    def __init__(self):
        self.events = []
        self.correlations = []
        self.ip_timeline = defaultdict(list)
        self.user_timeline = defaultdict(list)

    def parse_syslog_time(self, time_str, year=None):
        """Парсинг времени syslog формата."""
        if year is None:
            year = datetime.now().year
        try:
            return datetime.strptime(f"{year} {time_str}", "%Y %b %d %H:%M:%S")
        except ValueError:
            return datetime.now()

    def parse_auth_log(self, filepath):
        """Парсинг auth.log."""
        if not Path(filepath).exists():
            print(f"[!] Файл не найден: {filepath}")
            return

        with open(filepath, "r", errors="ignore") as f:
            for line in f:
                # Неудачный вход
                m = self.AUTH_FAILED.search(line)
                if m:
                    ts = self.parse_syslog_time(m.group(1))
                    evt = LogEvent(
                        timestamp=ts,
                        source="auth.log",
                        event_type="LOGIN_FAILED",
                        severity="medium",
                        ip=m.group(3),
                        user=m.group(2),
                    )
                    self.events.append(evt)
                    self.ip_timeline[m.group(3)].append(evt)
                    continue

                # Успешный вход
                m = self.AUTH_ACCEPTED.search(line)
                if m:
                    ts = self.parse_syslog_time(m.group(1))
                    evt = LogEvent(
                        timestamp=ts,
                        source="auth.log",
                        event_type="LOGIN_SUCCESS",
                        severity="info",
                        ip=m.group(3),
                        user=m.group(2),
                    )
                    self.events.append(evt)
                    self.ip_timeline[m.group(3)].append(evt)
                    self.user_timeline[m.group(2)].append(evt)
                    continue

                # sudo
                m = self.AUTH_SUDO.search(line)
                if m:
                    ts = self.parse_syslog_time(m.group(1))
                    evt = LogEvent(
                        timestamp=ts,
                        source="auth.log",
                        event_type="SUDO_COMMAND",
                        severity="medium",
                        user=m.group(2),
                        details={"command": m.group(3).strip()},
                    )
                    self.events.append(evt)
                    self.user_timeline[m.group(2)].append(evt)

    def parse_apache_log(self, filepath):
        """Парсинг Apache access.log."""
        if not Path(filepath).exists():
            return

        with open(filepath, "r", errors="ignore") as f:
            for line in f:
                m = self.APACHE_ACCESS.search(line)
                if m:
                    ip = m.group(1)
                    method = m.group(3)
                    path = m.group(4)
                    status = int(m.group(5))

                    severity = "info"
                    event_type = "HTTP_REQUEST"

                    # Обнаружение подозрительных запросов
                    suspicious_patterns = [
                        r"\.\./", r"etc/passwd", r"etc/shadow",
                        r"<script", r"union\s+select", r";\s*DROP",
                        r"\.php\?", r"wp-admin", r"\.env",
                    ]
                    for pat in suspicious_patterns:
                        if re.search(pat, path, re.IGNORECASE):
                            event_type = "HTTP_SUSPICIOUS"
                            severity = "high"
                            break

                    if status in (401, 403):
                        event_type = "HTTP_FORBIDDEN"
                        severity = "medium"

                    if severity != "info":
                        evt = LogEvent(
                            timestamp=datetime.now(),  # Упрощённо
                            source="apache",
                            event_type=event_type,
                            severity=severity,
                            ip=ip,
                            details={
                                "method": method,
                                "path": path,
                                "status": status,
                            },
                        )
                        self.events.append(evt)
                        self.ip_timeline[ip].append(evt)

    def correlate_brute_force_success(self):
        """Корреляция: brute-force → успешный вход (с того же IP)."""
        for ip, events in self.ip_timeline.items():
            failed = [e for e in events if e.event_type == "LOGIN_FAILED"]
            success = [e for e in events if e.event_type == "LOGIN_SUCCESS"]

            if len(failed) >= 5 and success:
                for s in success:
                    recent_fails = [
                        f for f in failed
                        if 0 < (s.timestamp - f.timestamp).total_seconds() < 3600
                    ]
                    if len(recent_fails) >= 5:
                        self.correlations.append({
                            "rule": "BRUTE_FORCE_SUCCESS",
                            "severity": "critical",
                            "description": (
                                f"IP {ip}: {len(recent_fails)} неудач → "
                                f"успешный вход как {s.user}"
                            ),
                            "ip": ip,
                            "user": s.user,
                            "failed_count": len(recent_fails),
                            "success_time": s.timestamp.isoformat(),
                        })

    def correlate_privilege_escalation(self):
        """Корреляция: login → sudo (подозрительные команды)."""
        suspicious_cmds = [
            "chmod 777", "chmod +s", "useradd", "usermod",
            "passwd", "visudo", "crontab", "nc ", "ncat",
            "curl.*|.*sh", "wget.*|.*sh", "python.*-c",
        ]

        for user, events in self.user_timeline.items():
            login = [e for e in events if e.event_type == "LOGIN_SUCCESS"]
            sudo = [e for e in events if e.event_type == "SUDO_COMMAND"]

            for s in sudo:
                cmd = s.details.get("command", "")
                for pattern in suspicious_cmds:
                    if re.search(pattern, cmd, re.IGNORECASE):
                        self.correlations.append({
                            "rule": "SUSPICIOUS_PRIVILEGE_ESCALATION",
                            "severity": "high",
                            "description": (
                                f"Пользователь {user}: "
                                f"подозрительная sudo-команда: {cmd}"
                            ),
                            "user": user,
                            "command": cmd,
                            "timestamp": s.timestamp.isoformat(),
                        })
                        break

    def correlate_web_to_ssh(self):
        """Корреляция: web-атака → SSH-вход (с того же IP)."""
        for ip, events in self.ip_timeline.items():
            web_attacks = [
                e for e in events if e.event_type == "HTTP_SUSPICIOUS"
            ]
            ssh_logins = [
                e for e in events if e.event_type == "LOGIN_SUCCESS"
            ]

            if web_attacks and ssh_logins:
                self.correlations.append({
                    "rule": "WEB_ATTACK_THEN_SSH",
                    "severity": "critical",
                    "description": (
                        f"IP {ip}: {len(web_attacks)} подозрительных "
                        f"web-запросов + SSH-вход"
                    ),
                    "ip": ip,
                    "web_attack_count": len(web_attacks),
                    "ssh_user": ssh_logins[0].user,
                })

    def run_all_correlations(self):
        """Запуск всех правил корреляции."""
        self.correlate_brute_force_success()
        self.correlate_privilege_escalation()
        self.correlate_web_to_ssh()

    def report(self):
        """Генерация отчёта."""
        print(f"\n{'='*60}")
        print(f"ОТЧЁТ КОРРЕЛЯЦИИ ЛОГОВ")
        print(f"{'='*60}")
        print(f"Событий обработано: {len(self.events)}")
        print(f"Уникальных IP: {len(self.ip_timeline)}")
        print(f"Уникальных пользователей: {len(self.user_timeline)}")
        print(f"Корреляций найдено: {len(self.correlations)}")

        if self.correlations:
            print(f"\n{'='*60}")
            print("ОБНАРУЖЕННЫЕ ЦЕПОЧКИ АТАК:")
            for i, corr in enumerate(self.correlations, 1):
                sev = corr["severity"]
                color = "\033[91m" if sev == "critical" else "\033[93m"
                print(f"\n{color}[{i}] [{sev.upper()}] {corr['rule']}\033[0m")
                print(f"    {corr['description']}")

        # JSON-выход для SIEM
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "events_total": len(self.events),
            "unique_ips": len(self.ip_timeline),
            "correlations": self.correlations,
        }
        output_file = f"/tmp/log_correlation_{int(datetime.now().timestamp())}.json"
        with open(output_file, "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nJSON-отчёт: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Коррелятор логов")
    parser.add_argument("-a", "--auth", default="/var/log/auth.log")
    parser.add_argument("-s", "--syslog", default="/var/log/syslog")
    parser.add_argument("-w", "--web", default="/var/log/apache2/access.log")
    args = parser.parse_args()

    correlator = LogCorrelator()
    print("[*] Парсинг логов...")
    correlator.parse_auth_log(args.auth)
    correlator.parse_apache_log(args.web)
    print(f"[*] Загружено событий: {len(correlator.events)}")

    print("[*] Запуск корреляции...")
    correlator.run_all_correlations()
    correlator.report()


if __name__ == "__main__":
    main()
```

**Интеграция:** Вывод в JSON подхватывается SIEM. Запускайте по cron каждый час.

---

## 3. YARA-сканер

**Назначение:** Сканирование файловой системы с помощью YARA-правил. 10 готовых правил для обнаружения типичных угроз.

**Зависимости:** `pip install yara-python`

```python
#!/usr/bin/env python3
"""
yara_scanner.py — YARA-сканер файловой системы
Запуск: python3 yara_scanner.py -r rules/ -t /tmp -t /home
"""

import argparse
import os
import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path

try:
    import yara
except ImportError:
    print("[!] pip install yara-python")
    sys.exit(1)


# ==================== ВСТРОЕННЫЕ YARA-ПРАВИЛА ====================
BUILTIN_RULES = """
rule Webshell_Generic {
    meta:
        description = "Обнаружение типичных webshell"
        severity = "critical"
    strings:
        $php1 = "eval($_POST" nocase
        $php2 = "eval($_GET" nocase
        $php3 = "eval($_REQUEST" nocase
        $php4 = "system($_" nocase
        $php5 = "passthru($_" nocase
        $php6 = "shell_exec($_" nocase
        $php7 = "base64_decode($_" nocase
        $asp1 = "eval(Request" nocase
        $jsp1 = "Runtime.getRuntime().exec" nocase
    condition:
        any of them
}

rule Suspicious_Script_Download {
    meta:
        description = "Скрипт, скачивающий и выполняющий код"
        severity = "high"
    strings:
        $s1 = "curl" nocase
        $s2 = "wget" nocase
        $s3 = "|sh" nocase
        $s4 = "|bash" nocase
        $s5 = "python -c" nocase
    condition:
        ($s1 or $s2) and ($s3 or $s4 or $s5)
}

rule Reverse_Shell_Indicator {
    meta:
        description = "Признаки reverse shell"
        severity = "critical"
    strings:
        $s1 = "/dev/tcp/" nocase
        $s2 = "mkfifo" nocase
        $s3 = "nc -e" nocase
        $s4 = "ncat -e" nocase
        $s5 = "bash -i >& /dev/tcp" nocase
        $s6 = "import socket" nocase
        $s7 = "subprocess.call" nocase
        $s8 = "TCPSocket.open" nocase
    condition:
        ($s1 and $s2) or $s3 or $s4 or $s5 or (($s6 and $s7) and filesize < 5KB)
}

rule Cryptominer_Indicator {
    meta:
        description = "Признаки криптомайнера"
        severity = "high"
    strings:
        $s1 = "stratum+tcp" nocase
        $s2 = "xmrig" nocase
        $s3 = "cryptonight" nocase
        $s4 = "minerd" nocase
        $s5 = "cpuminer" nocase
        $s6 = "hashrate" nocase
        $s7 = "pool.minergate" nocase
        $s8 = "nicehash" nocase
    condition:
        2 of them
}

rule SSH_Key_In_Unusual_Location {
    meta:
        description = "SSH-ключ в подозрительном месте"
        severity = "medium"
    strings:
        $s1 = "-----BEGIN RSA PRIVATE KEY-----"
        $s2 = "-----BEGIN OPENSSH PRIVATE KEY-----"
        $s3 = "-----BEGIN EC PRIVATE KEY-----"
    condition:
        any of them
}

rule Base64_Encoded_Command {
    meta:
        description = "Base64-кодированная команда"
        severity = "medium"
    strings:
        $s1 = "base64 -d" nocase
        $s2 = "base64 --decode" nocase
        $s3 = /echo\s+[A-Za-z0-9+\/]{50,}={0,2}\s*\|\s*base64/
    condition:
        any of them
}

rule Persistence_Cron_Backdoor {
    meta:
        description = "Подозрительная cron-задача (бэкдор)"
        severity = "high"
    strings:
        $s1 = "*/1 * * * *" nocase
        $s2 = "curl" nocase
        $s3 = "wget" nocase
        $s4 = "/dev/null" nocase
        $s5 = "2>&1" nocase
    condition:
        $s1 and ($s2 or $s3) and ($s4 or $s5)
}

rule Packed_ELF_Binary {
    meta:
        description = "UPX-упакованный ELF (потенциальный малварь)"
        severity = "medium"
    strings:
        $elf = { 7f 45 4c 46 }
        $upx = "UPX!" ascii
    condition:
        $elf at 0 and $upx
}

rule Data_Exfiltration_Script {
    meta:
        description = "Скрипт возможной эксфильтрации данных"
        severity = "high"
    strings:
        $tar = "tar " nocase
        $zip = "zip " nocase
        $find = "find " nocase
        $scp = "scp " nocase
        $curl = "curl " nocase
        $ftp = "ftp " nocase
        $passwd = "/etc/passwd"
        $shadow = "/etc/shadow"
        $ssh_dir = ".ssh/"
    condition:
        ($tar or $zip) and ($scp or $curl or $ftp) and ($passwd or $shadow or $ssh_dir)
}

rule Suspicious_Python_Script {
    meta:
        description = "Подозрительный Python-скрипт"
        severity = "medium"
    strings:
        $import_socket = "import socket" nocase
        $import_subprocess = "import subprocess" nocase
        $import_os = "import os" nocase
        $connect = ".connect((" nocase
        $popen = "Popen(" nocase
        $b64 = "b64decode" nocase
    condition:
        $import_socket and $connect and ($import_subprocess or $popen or $b64)
        and filesize < 10KB
}
"""


class YaraScanner:
    """Сканер файлов с помощью YARA-правил."""

    def __init__(self, rules_dir=None, use_builtin=True):
        self.rules = None
        self.findings = []
        self.scanned = 0
        self.errors = 0

        sources = {}
        if use_builtin:
            sources["builtin"] = BUILTIN_RULES

        if rules_dir and Path(rules_dir).is_dir():
            for rule_file in Path(rules_dir).glob("*.yar"):
                sources[rule_file.stem] = rule_file.read_text()

        if not sources:
            print("[!] Нет YARA-правил")
            sys.exit(1)

        try:
            self.rules = yara.compile(sources=sources)
            print(f"[*] Загружено источников правил: {len(sources)}")
        except yara.SyntaxError as e:
            print(f"[!] Ошибка компиляции YARA: {e}")
            sys.exit(1)

    def scan_file(self, filepath):
        """Сканирование одного файла."""
        try:
            matches = self.rules.match(filepath, timeout=30)
            self.scanned += 1

            if matches:
                file_hash = self._sha256(filepath)
                file_size = os.path.getsize(filepath)

                for match in matches:
                    severity = "medium"
                    if match.meta:
                        severity = match.meta.get("severity", "medium")

                    finding = {
                        "file": str(filepath),
                        "rule": match.rule,
                        "severity": severity,
                        "description": match.meta.get("description", ""),
                        "strings_matched": len(match.strings),
                        "sha256": file_hash,
                        "size": file_size,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    self.findings.append(finding)

                    sev_color = {
                        "critical": "\033[91m",
                        "high": "\033[91m",
                        "medium": "\033[93m",
                    }
                    color = sev_color.get(severity, "\033[0m")
                    print(
                        f"{color}[MATCH] {match.rule} → {filepath}\033[0m"
                    )
        except yara.TimeoutError:
            self.errors += 1
        except (PermissionError, OSError):
            self.errors += 1

    def scan_directory(self, target_dir, max_size_mb=50):
        """Рекурсивное сканирование каталога."""
        max_size = max_size_mb * 1024 * 1024
        target = Path(target_dir)

        if not target.exists():
            print(f"[!] Каталог не найден: {target_dir}")
            return

        print(f"[*] Сканирование: {target_dir}")
        for root, dirs, files in os.walk(target):
            dirs[:] = [
                d for d in dirs
                if d not in (".git", "node_modules", "__pycache__", ".venv")
            ]
            for filename in files:
                filepath = os.path.join(root, filename)
                try:
                    if os.path.getsize(filepath) > max_size:
                        continue
                    if os.path.islink(filepath):
                        continue
                    self.scan_file(filepath)
                except OSError:
                    continue

    @staticmethod
    def _sha256(filepath):
        h = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except (PermissionError, OSError):
            return "N/A"

    def report(self):
        """Вывод итогового отчёта."""
        print(f"\n{'='*60}")
        print(f"РЕЗУЛЬТАТЫ YARA-СКАНИРОВАНИЯ")
        print(f"Просканировано файлов: {self.scanned}")
        print(f"Ошибок: {self.errors}")
        print(f"Совпадений: {len(self.findings)}")

        if self.findings:
            print(f"\n{'='*60}")
            by_severity = {}
            for f in self.findings:
                by_severity.setdefault(f["severity"], []).append(f)
            for sev in ["critical", "high", "medium", "low"]:
                items = by_severity.get(sev, [])
                if items:
                    print(f"\n[{sev.upper()}] — {len(items)} совпадений:")
                    for item in items:
                        print(f"  {item['rule']}: {item['file']}")
                        print(f"    SHA256: {item['sha256'][:32]}...")

        output_file = f"/tmp/yara_scan_{int(datetime.now().timestamp())}.json"
        with open(output_file, "w") as f:
            json.dump(self.findings, f, ensure_ascii=False, indent=2)
        print(f"\nJSON: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="YARA-сканер")
    parser.add_argument("-r", "--rules", help="Каталог с .yar правилами")
    parser.add_argument("-t", "--target", action="append", required=True,
                        help="Каталоги для сканирования")
    parser.add_argument("--no-builtin", action="store_true",
                        help="Не использовать встроенные правила")
    parser.add_argument("-m", "--max-size", type=int, default=50,
                        help="Макс. размер файла в МБ")
    args = parser.parse_args()

    scanner = YaraScanner(
        rules_dir=args.rules, use_builtin=not args.no_builtin
    )
    for target in args.target:
        scanner.scan_directory(target, args.max_size)
    scanner.report()


if __name__ == "__main__":
    main()
```

---

## 4. Threat Intel агрегатор

**Назначение:** Сбор IoC из открытых источников (AbuseIPDB, OTX, VirusTotal), проверка IP/домена/хеша.

**Зависимости:** `pip install requests`

```python
#!/usr/bin/env python3
"""
threat_intel.py — Агрегатор Threat Intelligence
Проверка IP/доменов/хешей по открытым TI-источникам.
Запуск: python3 threat_intel.py --ip 1.2.3.4
        python3 threat_intel.py --hash abc123...
        python3 threat_intel.py --file iocs.txt
"""

import argparse
import json
import sys
import time
from datetime import datetime

try:
    import requests
except ImportError:
    print("[!] pip install requests")
    sys.exit(1)


class ThreatIntelAggregator:
    """Агрегатор TI из открытых источников."""

    def __init__(self, config=None):
        self.config = config or {}
        self.results = []
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "SOC-TI-Aggregator/1.0"

    def check_abuseipdb(self, ip):
        """Проверка IP в AbuseIPDB."""
        api_key = self.config.get("abuseipdb_key", "")
        if not api_key:
            return {"source": "AbuseIPDB", "status": "no_api_key"}

        url = "https://api.abuseipdb.com/api/v2/check"
        headers = {"Key": api_key, "Accept": "application/json"}
        params = {"ipAddress": ip, "maxAgeInDays": 90}

        try:
            resp = self.session.get(url, headers=headers, params=params,
                                    timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                return {
                    "source": "AbuseIPDB",
                    "ip": ip,
                    "abuse_score": data.get("abuseConfidenceScore", 0),
                    "total_reports": data.get("totalReports", 0),
                    "country": data.get("countryCode", ""),
                    "isp": data.get("isp", ""),
                    "is_tor": data.get("isTor", False),
                    "last_reported": data.get("lastReportedAt", ""),
                }
        except requests.RequestException as e:
            return {"source": "AbuseIPDB", "error": str(e)}

    def check_otx(self, indicator, indicator_type="IPv4"):
        """Проверка в AlienVault OTX."""
        api_key = self.config.get("otx_key", "")
        type_map = {
            "IPv4": f"indicators/IPv4/{indicator}/general",
            "domain": f"indicators/domain/{indicator}/general",
            "hash": f"indicators/file/{indicator}/general",
        }
        url = f"https://otx.alienvault.com/api/v1/{type_map.get(indicator_type, '')}"
        headers = {}
        if api_key:
            headers["X-OTX-API-KEY"] = api_key

        try:
            resp = self.session.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                pulse_count = data.get("pulse_info", {}).get("count", 0)
                return {
                    "source": "OTX",
                    "indicator": indicator,
                    "type": indicator_type,
                    "pulse_count": pulse_count,
                    "reputation": data.get("reputation", 0),
                    "malicious": pulse_count > 0,
                }
        except requests.RequestException as e:
            return {"source": "OTX", "error": str(e)}

    def check_virustotal(self, indicator, indicator_type="ip"):
        """Проверка в VirusTotal."""
        api_key = self.config.get("virustotal_key", "")
        if not api_key:
            return {"source": "VirusTotal", "status": "no_api_key"}

        type_map = {
            "ip": f"ip_addresses/{indicator}",
            "domain": f"domains/{indicator}",
            "hash": f"files/{indicator}",
        }
        url = f"https://www.virustotal.com/api/v3/{type_map.get(indicator_type, '')}"
        headers = {"x-apikey": api_key}

        try:
            resp = self.session.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("data", {}).get("attributes", {})
                stats = data.get("last_analysis_stats", {})
                return {
                    "source": "VirusTotal",
                    "indicator": indicator,
                    "malicious": stats.get("malicious", 0),
                    "suspicious": stats.get("suspicious", 0),
                    "harmless": stats.get("harmless", 0),
                    "undetected": stats.get("undetected", 0),
                }
        except requests.RequestException as e:
            return {"source": "VirusTotal", "error": str(e)}

    def check_ip(self, ip):
        """Комплексная проверка IP."""
        print(f"\n[*] Проверка IP: {ip}")
        results = {
            "indicator": ip,
            "type": "ip",
            "timestamp": datetime.utcnow().isoformat(),
            "sources": [],
        }

        results["sources"].append(self.check_abuseipdb(ip))
        time.sleep(0.5)
        results["sources"].append(self.check_otx(ip, "IPv4"))
        time.sleep(0.5)
        results["sources"].append(self.check_virustotal(ip, "ip"))

        # Итоговый вердикт
        malicious_count = sum(
            1 for s in results["sources"]
            if s.get("malicious") or s.get("abuse_score", 0) > 50
        )
        results["verdict"] = (
            "MALICIOUS" if malicious_count >= 2
            else "SUSPICIOUS" if malicious_count == 1
            else "CLEAN"
        )

        self._print_result(results)
        self.results.append(results)
        return results

    def check_domain(self, domain):
        """Комплексная проверка домена."""
        print(f"\n[*] Проверка домена: {domain}")
        results = {
            "indicator": domain,
            "type": "domain",
            "timestamp": datetime.utcnow().isoformat(),
            "sources": [],
        }
        results["sources"].append(self.check_otx(domain, "domain"))
        time.sleep(0.5)
        results["sources"].append(self.check_virustotal(domain, "domain"))

        malicious_count = sum(
            1 for s in results["sources"] if s.get("malicious")
        )
        results["verdict"] = (
            "MALICIOUS" if malicious_count >= 1 else "CLEAN"
        )

        self._print_result(results)
        self.results.append(results)
        return results

    def check_hash(self, file_hash):
        """Комплексная проверка хеша файла."""
        print(f"\n[*] Проверка хеша: {file_hash[:32]}...")
        results = {
            "indicator": file_hash,
            "type": "hash",
            "timestamp": datetime.utcnow().isoformat(),
            "sources": [],
        }
        results["sources"].append(self.check_otx(file_hash, "hash"))
        time.sleep(0.5)
        results["sources"].append(self.check_virustotal(file_hash, "hash"))

        malicious_count = sum(
            1 for s in results["sources"]
            if s.get("malicious") and (
                isinstance(s["malicious"], bool)
                or (isinstance(s["malicious"], int) and s["malicious"] > 3)
            )
        )
        results["verdict"] = (
            "MALICIOUS" if malicious_count >= 1 else "CLEAN"
        )

        self._print_result(results)
        self.results.append(results)
        return results

    def check_file(self, filepath):
        """Проверка IoC из файла (по одному на строку)."""
        import re
        ip_pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
        hash_pattern = re.compile(r"^[a-fA-F0-9]{32,64}$")

        with open(filepath) as f:
            for line in f:
                ioc = line.strip()
                if not ioc or ioc.startswith("#"):
                    continue
                if ip_pattern.match(ioc):
                    self.check_ip(ioc)
                elif hash_pattern.match(ioc):
                    self.check_hash(ioc)
                else:
                    self.check_domain(ioc)
                time.sleep(1)

    @staticmethod
    def _print_result(result):
        verdict = result.get("verdict", "UNKNOWN")
        colors = {
            "MALICIOUS": "\033[91m",
            "SUSPICIOUS": "\033[93m",
            "CLEAN": "\033[92m",
        }
        color = colors.get(verdict, "\033[0m")
        print(f"  {color}Вердикт: {verdict}\033[0m")
        for src in result.get("sources", []):
            name = src.get("source", "?")
            if "error" in src:
                print(f"    {name}: ошибка — {src['error']}")
            elif "no_api_key" == src.get("status"):
                print(f"    {name}: нет API-ключа")
            else:
                details = {
                    k: v for k, v in src.items()
                    if k not in ("source", "indicator", "type")
                }
                print(f"    {name}: {json.dumps(details, ensure_ascii=False)}")

    def save_report(self, output_path="/tmp/ti_report.json"):
        with open(output_path, "w") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n[*] Отчёт: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Threat Intel агрегатор")
    parser.add_argument("--ip", help="Проверить IP")
    parser.add_argument("--domain", help="Проверить домен")
    parser.add_argument("--hash", help="Проверить хеш файла")
    parser.add_argument("--file", help="Файл с IoC (по одному на строку)")
    parser.add_argument("--config", help="JSON-файл с API-ключами")
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    ti = ThreatIntelAggregator(config)

    if args.ip:
        ti.check_ip(args.ip)
    if args.domain:
        ti.check_domain(args.domain)
    if args.hash:
        ti.check_hash(args.hash)
    if args.file:
        ti.check_file(args.file)

    if not any([args.ip, args.domain, args.hash, args.file]):
        parser.print_help()
        return

    ti.save_report()


if __name__ == "__main__":
    main()
```

**Конфигурация API-ключей** (`ti_config.json`):
```json
{
    "abuseipdb_key": "YOUR_KEY",
    "otx_key": "YOUR_KEY",
    "virustotal_key": "YOUR_KEY"
}
```

---

## 5. Email-анализатор

**Назначение:** Анализ email-заголовков и тела для обнаружения фишинга: SPF/DKIM/DMARC, подозрительные ссылки, вложения.

```python
#!/usr/bin/env python3
"""
email_analyzer.py — Анализатор email для обнаружения фишинга
Запуск: python3 email_analyzer.py -f suspicious_email.eml
"""

import argparse
import email
import re
import sys
import json
from email import policy
from email.parser import BytesParser
from urllib.parse import urlparse
from datetime import datetime


class EmailAnalyzer:
    """Анализатор email на фишинг-индикаторы."""

    SUSPICIOUS_EXTENSIONS = {
        ".exe", ".scr", ".bat", ".cmd", ".vbs", ".js", ".wsf",
        ".ps1", ".msi", ".dll", ".hta", ".cpl", ".jar", ".lnk",
    }
    SUSPICIOUS_TLDS = {
        ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
        ".click", ".loan", ".work", ".date", ".racing",
    }

    def __init__(self, eml_path):
        self.eml_path = eml_path
        self.msg = None
        self.findings = []
        self.score = 0  # Чем выше — тем подозрительнее

    def load(self):
        with open(self.eml_path, "rb") as f:
            self.msg = BytesParser(policy=policy.default).parse(f)

    def add_finding(self, category, severity, description, details=None):
        score_map = {"critical": 30, "high": 20, "medium": 10, "low": 5}
        self.score += score_map.get(severity, 5)
        self.findings.append({
            "category": category,
            "severity": severity,
            "description": description,
            "details": details or {},
        })

    def check_headers(self):
        """Анализ заголовков."""
        # From vs Return-Path
        from_addr = self.msg.get("From", "")
        return_path = self.msg.get("Return-Path", "")
        reply_to = self.msg.get("Reply-To", "")

        from_domain = self._extract_domain(from_addr)
        rp_domain = self._extract_domain(return_path)

        if from_domain and rp_domain and from_domain != rp_domain:
            self.add_finding(
                "HEADER_MISMATCH", "high",
                f"From домен ({from_domain}) ≠ Return-Path ({rp_domain})",
                {"from": from_addr, "return_path": return_path},
            )

        if reply_to:
            reply_domain = self._extract_domain(reply_to)
            if reply_domain and from_domain and reply_domain != from_domain:
                self.add_finding(
                    "REPLY_TO_MISMATCH", "medium",
                    f"Reply-To ({reply_domain}) ≠ From ({from_domain})",
                )

        # SPF
        received_spf = self.msg.get("Received-SPF", "").lower()
        auth_results = self.msg.get("Authentication-Results", "").lower()

        if "fail" in received_spf:
            self.add_finding("SPF_FAIL", "high", "SPF проверка не пройдена")
        elif "softfail" in received_spf:
            self.add_finding("SPF_SOFTFAIL", "medium", "SPF softfail")

        # DKIM
        if auth_results:
            if "dkim=fail" in auth_results:
                self.add_finding("DKIM_FAIL", "high", "DKIM не пройден")
            if "dmarc=fail" in auth_results:
                self.add_finding("DMARC_FAIL", "high", "DMARC не пройден")

        # Количество Received headers (слишком много — подозрительно)
        received_count = len(self.msg.get_all("Received", []))
        if received_count > 10:
            self.add_finding(
                "MANY_HOPS", "low",
                f"Много промежуточных серверов: {received_count}",
            )

    def check_body(self):
        """Анализ тела письма."""
        body = ""
        if self.msg.is_multipart():
            for part in self.msg.walk():
                ct = part.get_content_type()
                if ct in ("text/plain", "text/html"):
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode("utf-8", errors="ignore")
        else:
            payload = self.msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="ignore")

        # Извлечение ссылок
        urls = re.findall(r'https?://[^\s<>"\']+', body)
        for url in urls:
            self._check_url(url)

        # Подозрительные фразы
        urgency_phrases = [
            r"срочно", r"немедленно", r"ваш аккаунт.*заблокирован",
            r"подтвердите.*данные", r"urgent", r"verify your account",
            r"click here immediately", r"your account.*suspended",
            r"act now", r"limited time", r"confirm your identity",
        ]
        for phrase in urgency_phrases:
            if re.search(phrase, body, re.IGNORECASE):
                self.add_finding(
                    "URGENCY_LANGUAGE", "medium",
                    f"Фраза давления: «{phrase}»",
                )
                break

        # Формы ввода в HTML
        if "<form" in body.lower() and ("password" in body.lower()
                                        or "login" in body.lower()):
            self.add_finding(
                "EMBEDDED_FORM", "critical",
                "HTML-форма с полями пароля/логина в теле письма",
            )

    def check_attachments(self):
        """Анализ вложений."""
        if not self.msg.is_multipart():
            return

        for part in self.msg.walk():
            filename = part.get_filename()
            if not filename:
                continue

            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

            if ext in self.SUSPICIOUS_EXTENSIONS:
                self.add_finding(
                    "DANGEROUS_ATTACHMENT", "critical",
                    f"Опасное вложение: {filename}",
                    {"filename": filename, "extension": ext},
                )

            # Двойное расширение
            parts = filename.split(".")
            if len(parts) >= 3:
                self.add_finding(
                    "DOUBLE_EXTENSION", "high",
                    f"Двойное расширение: {filename}",
                )

            # Размер
            payload = part.get_payload(decode=True)
            if payload:
                size_kb = len(payload) / 1024
                if ext == ".zip" and size_kb < 5:
                    self.add_finding(
                        "TINY_ARCHIVE", "medium",
                        f"Подозрительно маленький архив: {filename} ({size_kb:.1f} KB)",
                    )

    def _check_url(self, url):
        """Проверка отдельной ссылки."""
        parsed = urlparse(url)
        domain = parsed.hostname or ""

        # IP вместо домена
        if re.match(r"^\d+\.\d+\.\d+\.\d+$", domain):
            self.add_finding(
                "IP_IN_URL", "high",
                f"Ссылка с IP вместо домена: {url[:80]}",
            )

        # Подозрительный TLD
        for tld in self.SUSPICIOUS_TLDS:
            if domain.endswith(tld):
                self.add_finding(
                    "SUSPICIOUS_TLD", "medium",
                    f"Подозрительный TLD в ссылке: {domain}",
                )
                break

        # URL shortener
        shorteners = [
            "bit.ly", "tinyurl.com", "goo.gl", "t.co",
            "is.gd", "cli.gs", "ow.ly", "cutt.ly",
        ]
        if domain in shorteners:
            self.add_finding(
                "URL_SHORTENER", "medium",
                f"Сокращённая ссылка: {url[:80]}",
            )

        # @ в URL (user@host маскировка)
        if "@" in parsed.netloc:
            self.add_finding(
                "AT_IN_URL", "high",
                f"Символ @ в URL (маскировка): {url[:80]}",
            )

    @staticmethod
    def _extract_domain(addr):
        m = re.search(r"@([\w.-]+)", addr)
        return m.group(1).lower() if m else ""

    def analyze(self):
        """Полный анализ."""
        self.load()
        print(f"\n{'='*60}")
        print(f"АНАЛИЗ EMAIL: {self.eml_path}")
        print(f"{'='*60}")
        print(f"From:    {self.msg.get('From', 'N/A')}")
        print(f"To:      {self.msg.get('To', 'N/A')}")
        print(f"Subject: {self.msg.get('Subject', 'N/A')}")
        print(f"Date:    {self.msg.get('Date', 'N/A')}")

        self.check_headers()
        self.check_body()
        self.check_attachments()

        print(f"\n{'='*60}")
        print(f"РЕЗУЛЬТАТ: score={self.score}")

        verdict = (
            "ФИШИНГ" if self.score >= 50
            else "ПОДОЗРИТЕЛЬНО" if self.score >= 20
            else "ЧИСТО"
        )
        colors = {"ФИШИНГ": "\033[91m", "ПОДОЗРИТЕЛЬНО": "\033[93m", "ЧИСТО": "\033[92m"}
        print(f"{colors[verdict]}Вердикт: {verdict}\033[0m")

        if self.findings:
            print(f"\nНаходки ({len(self.findings)}):")
            for f in self.findings:
                sev = f["severity"]
                print(f"  [{sev.upper()}] {f['category']}: {f['description']}")

        return {"verdict": verdict, "score": self.score, "findings": self.findings}


def main():
    parser = argparse.ArgumentParser(description="Email-анализатор (антифишинг)")
    parser.add_argument("-f", "--file", required=True, help="Путь к .eml файлу")
    args = parser.parse_args()

    analyzer = EmailAnalyzer(args.file)
    result = analyzer.analyze()

    output = f"/tmp/email_analysis_{int(datetime.now().timestamp())}.json"
    with open(output, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nОтчёт: {output}")


if __name__ == "__main__":
    main()
```

---

## 6. Анализатор PE-файлов

**Назначение:** Статический анализ PE-файлов (Windows executables): энтропия, импорты, секции, пакер, строки.

**Зависимости:** `pip install pefile`

```python
#!/usr/bin/env python3
"""
pe_analyzer.py — Статический анализ PE-файлов
Запуск: python3 pe_analyzer.py -f sample.exe
"""

import argparse
import hashlib
import math
import sys
import json
from datetime import datetime
from collections import Counter

try:
    import pefile
except ImportError:
    print("[!] pip install pefile")
    sys.exit(1)


class PEAnalyzer:
    """Анализатор PE-файлов для SOC."""

    SUSPICIOUS_IMPORTS = {
        "VirtualAlloc", "VirtualAllocEx", "VirtualProtect",
        "WriteProcessMemory", "CreateRemoteThread",
        "NtUnmapViewOfSection", "RtlMoveMemory",
        "IsDebuggerPresent", "CheckRemoteDebuggerPresent",
        "GetProcAddress", "LoadLibraryA", "LoadLibraryW",
        "URLDownloadToFileA", "InternetOpenA",
        "WinExec", "ShellExecuteA", "CreateProcessA",
        "RegSetValueExA", "RegCreateKeyExA",
        "CryptEncrypt", "CryptDecrypt",
        "AdjustTokenPrivileges", "OpenProcessToken",
    }

    PACKER_SIGNATURES = {
        "UPX": [b"UPX0", b"UPX1", b"UPX!"],
        "ASPack": [b".aspack", b".adata"],
        "PECompact": [b"PEC2", b"PECompact"],
        "Themida": [b".themida"],
        "VMProtect": [b".vmp0", b".vmp1"],
    }

    def __init__(self, filepath):
        self.filepath = filepath
        self.pe = None
        self.findings = []
        self.score = 0

    def load(self):
        try:
            self.pe = pefile.PE(self.filepath)
        except pefile.PEFormatError as e:
            print(f"[!] Не валидный PE: {e}")
            sys.exit(1)

    def add_finding(self, category, severity, description):
        score_map = {"critical": 25, "high": 15, "medium": 10, "low": 5}
        self.score += score_map.get(severity, 5)
        self.findings.append({
            "category": category, "severity": severity,
            "description": description,
        })

    def file_hashes(self):
        """Вычисление хешей файла."""
        with open(self.filepath, "rb") as f:
            data = f.read()
        return {
            "md5": hashlib.md5(data).hexdigest(),
            "sha1": hashlib.sha1(data).hexdigest(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }

    @staticmethod
    def entropy(data):
        """Энтропия Шеннона."""
        if not data:
            return 0.0
        counter = Counter(data)
        length = len(data)
        return -sum(
            (c / length) * math.log2(c / length)
            for c in counter.values()
        )

    def check_sections(self):
        """Анализ секций PE."""
        sections = []
        for section in self.pe.sections:
            name = section.Name.decode("utf-8", errors="ignore").rstrip("\x00")
            raw_size = section.SizeOfRawData
            virtual_size = section.Misc_VirtualSize
            ent = self.entropy(section.get_data())

            sec_info = {
                "name": name,
                "virtual_size": virtual_size,
                "raw_size": raw_size,
                "entropy": round(ent, 2),
                "characteristics": hex(section.Characteristics),
            }
            sections.append(sec_info)

            # Высокая энтропия — возможно упакован/зашифрован
            if ent > 7.0:
                self.add_finding(
                    "HIGH_ENTROPY_SECTION", "high",
                    f"Секция {name}: энтропия {ent:.2f} (возможна упаковка/шифрование)",
                )

            # Исполняемая и записываемая секция
            is_exec = section.Characteristics & 0x20000000
            is_write = section.Characteristics & 0x80000000
            if is_exec and is_write:
                self.add_finding(
                    "RWX_SECTION", "high",
                    f"Секция {name}: RWX (Write+Execute) — подозрительно",
                )

        return sections

    def check_imports(self):
        """Анализ импортов."""
        suspicious_found = []
        all_imports = []

        if hasattr(self.pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in self.pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode("utf-8", errors="ignore")
                for imp in entry.imports:
                    if imp.name:
                        func_name = imp.name.decode("utf-8", errors="ignore")
                        all_imports.append(f"{dll_name}:{func_name}")
                        if func_name in self.SUSPICIOUS_IMPORTS:
                            suspicious_found.append(func_name)

        if suspicious_found:
            self.add_finding(
                "SUSPICIOUS_IMPORTS", "medium",
                f"Подозрительные импорты: {', '.join(suspicious_found[:10])}",
            )

        if len(all_imports) < 5:
            self.add_finding(
                "FEW_IMPORTS", "medium",
                f"Мало импортов ({len(all_imports)}) — возможно динамическая загрузка",
            )

        return all_imports

    def check_packer(self):
        """Обнаружение пакера."""
        with open(self.filepath, "rb") as f:
            data = f.read()

        for packer, sigs in self.PACKER_SIGNATURES.items():
            for sig in sigs:
                if sig in data:
                    self.add_finding(
                        "PACKER_DETECTED", "high",
                        f"Обнаружен пакер: {packer}",
                    )
                    return packer
        return None

    def check_compilation_time(self):
        """Проверка времени компиляции."""
        ts = self.pe.FILE_HEADER.TimeDateStamp
        comp_time = datetime.utcfromtimestamp(ts)

        if comp_time.year < 2000 or comp_time > datetime.utcnow():
            self.add_finding(
                "SUSPICIOUS_TIMESTAMP", "medium",
                f"Подозрительное время компиляции: {comp_time}",
            )
        return comp_time.isoformat()

    def check_resources(self):
        """Анализ ресурсов PE."""
        if not hasattr(self.pe, "DIRECTORY_ENTRY_RESOURCE"):
            return []

        resources = []
        for entry in self.pe.DIRECTORY_ENTRY_RESOURCE.entries:
            res_type = pefile.RESOURCE_TYPE.get(entry.id, str(entry.id))
            if hasattr(entry, "directory"):
                for sub in entry.directory.entries:
                    if hasattr(sub, "directory"):
                        for res in sub.directory.entries:
                            data = self.pe.get_data(
                                res.data.struct.OffsetToData,
                                res.data.struct.Size,
                            )
                            ent = self.entropy(data)
                            resources.append({
                                "type": str(res_type),
                                "size": res.data.struct.Size,
                                "entropy": round(ent, 2),
                            })
                            if ent > 7.0 and res.data.struct.Size > 10000:
                                self.add_finding(
                                    "HIGH_ENTROPY_RESOURCE", "medium",
                                    f"Ресурс {res_type}: высокая энтропия ({ent:.2f}), размер {res.data.struct.Size}",
                                )
        return resources

    def analyze(self):
        """Полный анализ."""
        self.load()
        hashes = self.file_hashes()
        sections = self.check_sections()
        imports = self.check_imports()
        packer = self.check_packer()
        comp_time = self.check_compilation_time()
        resources = self.check_resources()

        verdict = (
            "ВРЕДОНОСНЫЙ" if self.score >= 50
            else "ПОДОЗРИТЕЛЬНЫЙ" if self.score >= 20
            else "ЧИСТЫЙ"
        )

        result = {
            "file": self.filepath,
            "hashes": hashes,
            "compilation_time": comp_time,
            "sections": sections,
            "import_count": len(imports),
            "packer": packer,
            "resources_count": len(resources),
            "findings": self.findings,
            "score": self.score,
            "verdict": verdict,
        }

        # Вывод
        print(f"\n{'='*60}")
        print(f"PE-АНАЛИЗ: {self.filepath}")
        print(f"{'='*60}")
        print(f"SHA256:      {hashes['sha256']}")
        print(f"Размер:      {hashes['size']} байт")
        print(f"Компиляция:  {comp_time}")
        print(f"Секций:      {len(sections)}")
        print(f"Импортов:    {len(imports)}")
        print(f"Пакер:       {packer or 'не обнаружен'}")
        print(f"Score:       {self.score}")
        colors = {"ВРЕДОНОСНЫЙ": "\033[91m", "ПОДОЗРИТЕЛЬНЫЙ": "\033[93m", "ЧИСТЫЙ": "\033[92m"}
        print(f"{colors[verdict]}Вердикт:     {verdict}\033[0m")

        if self.findings:
            print(f"\nНаходки:")
            for f in self.findings:
                print(f"  [{f['severity'].upper()}] {f['description']}")

        return result


def main():
    parser = argparse.ArgumentParser(description="PE-анализатор")
    parser.add_argument("-f", "--file", required=True, help="PE-файл")
    args = parser.parse_args()

    analyzer = PEAnalyzer(args.file)
    result = analyzer.analyze()

    output = f"/tmp/pe_analysis_{int(datetime.now().timestamp())}.json"
    with open(output, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nОтчёт: {output}")


if __name__ == "__main__":
    main()
```

---

## 7. Детектор аномалий трафика

**Назначение:** Статистический анализ сетевого трафика: baseline, обнаружение отклонений, спайков.

**Зависимости:** `pip install pandas numpy`

```python
#!/usr/bin/env python3
"""
traffic_anomaly.py — Детектор аномалий трафика
Анализирует netflow-данные или ss/netstat вывод, строит baseline,
обнаруживает отклонения.
Запуск: python3 traffic_anomaly.py -f netflow.csv
        python3 traffic_anomaly.py --live -d 60
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

try:
    import numpy as np
except ImportError:
    print("[!] pip install numpy")
    sys.exit(1)


class TrafficAnomalyDetector:
    """Детектор аномалий сетевого трафика."""

    def __init__(self, std_multiplier=3.0):
        self.std_multiplier = std_multiplier
        self.baseline = {}
        self.alerts = []

    def collect_live_sample(self):
        """Сбор текущего состояния соединений."""
        result = subprocess.run(
            ["ss", "-tun", "--no-header"],
            capture_output=True, text=True,
        )
        connections = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 5:
                state = parts[0]
                local = parts[3]
                remote = parts[4]
                connections.append({
                    "state": state,
                    "local": local,
                    "remote": remote,
                })
        return connections

    def build_baseline(self, samples):
        """Построение baseline из серии samples."""
        metrics_over_time = defaultdict(list)

        for sample in samples:
            stats = self._compute_stats(sample)
            for key, value in stats.items():
                metrics_over_time[key].append(value)

        for key, values in metrics_over_time.items():
            arr = np.array(values, dtype=float)
            self.baseline[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        return self.baseline

    def detect_anomalies(self, current_sample):
        """Обнаружение аномалий в текущем sample."""
        if not self.baseline:
            return []

        stats = self._compute_stats(current_sample)
        anomalies = []

        for key, value in stats.items():
            if key not in self.baseline:
                continue
            bl = self.baseline[key]
            mean = bl["mean"]
            std = bl["std"]
            threshold = mean + self.std_multiplier * max(std, 1)

            if value > threshold:
                deviation = (value - mean) / max(std, 1)
                anomalies.append({
                    "metric": key,
                    "current": value,
                    "baseline_mean": round(mean, 2),
                    "baseline_std": round(std, 2),
                    "threshold": round(threshold, 2),
                    "deviation_sigma": round(deviation, 2),
                    "timestamp": datetime.utcnow().isoformat(),
                })

        return anomalies

    @staticmethod
    def _compute_stats(connections):
        """Вычисление метрик из списка соединений."""
        total = len(connections)
        states = defaultdict(int)
        remote_ips = set()
        remote_ports = defaultdict(int)

        for conn in connections:
            states[conn.get("state", "UNKNOWN")] += 1
            remote = conn.get("remote", "")
            if ":" in remote:
                ip, port = remote.rsplit(":", 1)
                remote_ips.add(ip)
                remote_ports[port] += 1

        return {
            "total_connections": total,
            "established": states.get("ESTAB", 0),
            "time_wait": states.get("TIME-WAIT", 0),
            "syn_sent": states.get("SYN-SENT", 0),
            "unique_remote_ips": len(remote_ips),
            "unique_remote_ports": len(remote_ports),
        }

    def live_monitor(self, duration=300, interval=10, baseline_samples=10):
        """Мониторинг в реальном времени."""
        print(f"[*] Сбор baseline ({baseline_samples} samples)...")
        samples = []
        for i in range(baseline_samples):
            s = self.collect_live_sample()
            samples.append(s)
            print(f"  Sample {i+1}/{baseline_samples}: {len(s)} соединений")
            time.sleep(interval)

        self.build_baseline(samples)
        print(f"[*] Baseline построен: {json.dumps(self.baseline, indent=2)}")

        print(f"\n[*] Мониторинг ({duration} сек, интервал {interval} сек)...")
        end_time = time.time() + duration

        while time.time() < end_time:
            current = self.collect_live_sample()
            anomalies = self.detect_anomalies(current)

            if anomalies:
                for a in anomalies:
                    print(
                        f"\033[91m[ANOMALY] {a['metric']}: "
                        f"{a['current']} (baseline: {a['baseline_mean']}±"
                        f"{a['baseline_std']}, deviation: "
                        f"{a['deviation_sigma']}σ)\033[0m"
                    )
                    self.alerts.append(a)
            else:
                stats = self._compute_stats(current)
                print(f"[OK] {stats['total_connections']} conn, "
                      f"{stats['unique_remote_ips']} IPs")

            time.sleep(interval)

        print(f"\n[*] Завершено. Алертов: {len(self.alerts)}")

    def analyze_csv(self, filepath):
        """Анализ из CSV-файла (timestamp, src, dst, port, bytes)."""
        rows = []
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        # Группируем по минутам
        by_minute = defaultdict(list)
        for row in rows:
            ts = row.get("timestamp", "")[:16]
            by_minute[ts].append({
                "state": "ESTAB",
                "local": row.get("src", ""),
                "remote": f"{row.get('dst', '')}:{row.get('port', '')}",
            })

        minutes = sorted(by_minute.keys())
        if len(minutes) < 5:
            print("[!] Слишком мало данных для baseline")
            return

        # Первые 70% — baseline, остальное — проверка
        split = int(len(minutes) * 0.7)
        baseline_mins = minutes[:split]
        check_mins = minutes[split:]

        baseline_samples = [by_minute[m] for m in baseline_mins]
        self.build_baseline(baseline_samples)

        print(f"[*] Baseline из {len(baseline_mins)} минут")
        print(f"[*] Проверка {len(check_mins)} минут")

        for m in check_mins:
            anomalies = self.detect_anomalies(by_minute[m])
            if anomalies:
                for a in anomalies:
                    print(f"[ANOMALY] {m} — {a['metric']}: {a['current']} "
                          f"(baseline: {a['baseline_mean']})")
                    self.alerts.append(a)


def main():
    parser = argparse.ArgumentParser(description="Детектор аномалий трафика")
    parser.add_argument("-f", "--file", help="CSV-файл с netflow")
    parser.add_argument("--live", action="store_true", help="Живой мониторинг")
    parser.add_argument("-d", "--duration", type=int, default=300)
    parser.add_argument("-i", "--interval", type=int, default=10)
    parser.add_argument("-s", "--sigma", type=float, default=3.0,
                        help="Множитель σ для порога")
    args = parser.parse_args()

    detector = TrafficAnomalyDetector(std_multiplier=args.sigma)

    if args.live:
        detector.live_monitor(args.duration, args.interval)
    elif args.file:
        detector.analyze_csv(args.file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

---

## 8. Web-сканер безопасности

**Назначение:** Проверка собственных веб-ресурсов: security headers, SSL/TLS, cookie flags.

**Зависимости:** `pip install requests`

```python
#!/usr/bin/env python3
"""
web_security_scanner.py — Сканер безопасности веб-сервера
Проверка security headers, SSL, cookies для СОБСТВЕННЫХ ресурсов.
Запуск: python3 web_security_scanner.py -u https://example.com
"""

import argparse
import json
import ssl
import socket
import sys
from datetime import datetime

try:
    import requests
except ImportError:
    print("[!] pip install requests")
    sys.exit(1)


class WebSecurityScanner:
    """Сканер безопасности веб-ресурсов."""

    REQUIRED_HEADERS = {
        "Strict-Transport-Security": {
            "severity": "high",
            "description": "HSTS не настроен — возможен downgrade атака",
        },
        "Content-Security-Policy": {
            "severity": "high",
            "description": "CSP не настроен — возможен XSS",
        },
        "X-Content-Type-Options": {
            "severity": "medium",
            "expected": "nosniff",
            "description": "Нет nosniff — возможен MIME sniffing",
        },
        "X-Frame-Options": {
            "severity": "medium",
            "description": "Нет защиты от clickjacking",
        },
        "X-XSS-Protection": {
            "severity": "low",
            "description": "XSS-Protection заголовок отсутствует",
        },
        "Referrer-Policy": {
            "severity": "low",
            "description": "Referrer-Policy не настроен",
        },
        "Permissions-Policy": {
            "severity": "low",
            "description": "Permissions-Policy не настроен",
        },
    }

    DANGEROUS_HEADERS = [
        "Server", "X-Powered-By", "X-AspNet-Version",
        "X-AspNetMvc-Version",
    ]

    def __init__(self, url):
        self.url = url
        self.findings = []
        self.score = 100  # Стартуем с 100, вычитаем за каждую проблему

    def add_finding(self, category, severity, description, details=None):
        deduct = {"critical": 20, "high": 15, "medium": 10, "low": 5}
        self.score = max(0, self.score - deduct.get(severity, 5))
        self.findings.append({
            "category": category,
            "severity": severity,
            "description": description,
            "details": details or {},
        })

    def check_headers(self):
        """Проверка security headers."""
        print(f"\n[1] Проверка Security Headers...")
        try:
            resp = requests.get(self.url, timeout=10, allow_redirects=True)
        except requests.RequestException as e:
            print(f"[!] Ошибка: {e}")
            return

        headers = resp.headers

        for header, info in self.REQUIRED_HEADERS.items():
            if header.lower() not in {k.lower(): k for k in headers}:
                self.add_finding(
                    "MISSING_HEADER", info["severity"],
                    f"Отсутствует: {header} — {info['description']}",
                )
            else:
                actual = headers.get(header, "")
                expected = info.get("expected")
                if expected and expected.lower() not in actual.lower():
                    self.add_finding(
                        "WRONG_HEADER_VALUE", info["severity"],
                        f"{header}: ожидается '{expected}', получено '{actual}'",
                    )

        # Утечка информации через заголовки
        for header in self.DANGEROUS_HEADERS:
            val = headers.get(header)
            if val:
                self.add_finding(
                    "INFO_DISCLOSURE", "medium",
                    f"Утечка информации: {header}: {val}",
                )

    def check_ssl(self):
        """Проверка SSL/TLS."""
        print(f"\n[2] Проверка SSL/TLS...")
        from urllib.parse import urlparse
        parsed = urlparse(self.url)
        if parsed.scheme != "https":
            self.add_finding("NO_HTTPS", "critical", "Сайт не использует HTTPS")
            return

        hostname = parsed.hostname
        port = parsed.port or 443

        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    protocol = ssock.version()
                    cipher = ssock.cipher()

                    print(f"  Протокол: {protocol}")
                    print(f"  Шифр:     {cipher[0]} ({cipher[2]} бит)")

                    # Проверка протокола
                    if protocol in ("TLSv1", "TLSv1.1", "SSLv3"):
                        self.add_finding(
                            "WEAK_TLS", "critical",
                            f"Устаревший протокол: {protocol}",
                        )

                    # Срок действия
                    not_after = cert.get("notAfter", "")
                    if not_after:
                        exp = datetime.strptime(
                            not_after, "%b %d %H:%M:%S %Y %Z"
                        )
                        days_left = (exp - datetime.utcnow()).days
                        print(f"  Сертификат истекает: {exp} ({days_left} дней)")
                        if days_left < 0:
                            self.add_finding(
                                "CERT_EXPIRED", "critical",
                                "SSL-сертификат истёк!",
                            )
                        elif days_left < 30:
                            self.add_finding(
                                "CERT_EXPIRING", "high",
                                f"Сертификат истекает через {days_left} дней",
                            )

        except ssl.SSLError as e:
            self.add_finding("SSL_ERROR", "critical", f"SSL ошибка: {e}")
        except socket.error as e:
            self.add_finding("CONNECTION_ERROR", "high", f"Ошибка подключения: {e}")

    def check_cookies(self):
        """Проверка cookie flags."""
        print(f"\n[3] Проверка Cookies...")
        try:
            resp = requests.get(self.url, timeout=10)
        except requests.RequestException:
            return

        for cookie in resp.cookies:
            issues = []
            if not cookie.secure:
                issues.append("нет Secure")
            if not cookie.has_nonstandard_attr("HttpOnly"):
                issues.append("нет HttpOnly")
            samesite = cookie.get_nonstandard_attr("SameSite")
            if not samesite:
                issues.append("нет SameSite")

            if issues:
                self.add_finding(
                    "INSECURE_COOKIE", "medium",
                    f"Cookie '{cookie.name}': {', '.join(issues)}",
                )

    def check_redirect(self):
        """Проверка HTTP → HTTPS redirect."""
        print(f"\n[4] Проверка HTTP→HTTPS redirect...")
        http_url = self.url.replace("https://", "http://")
        try:
            resp = requests.get(http_url, timeout=10, allow_redirects=False)
            if resp.status_code in (301, 302, 307, 308):
                location = resp.headers.get("Location", "")
                if location.startswith("https://"):
                    print(f"  [OK] Redirect на HTTPS")
                else:
                    self.add_finding(
                        "NO_HTTPS_REDIRECT", "high",
                        "HTTP не перенаправляет на HTTPS",
                    )
            else:
                self.add_finding(
                    "NO_HTTPS_REDIRECT", "high",
                    f"HTTP отвечает {resp.status_code} без redirect",
                )
        except requests.RequestException:
            pass

    def scan(self):
        """Полное сканирование."""
        print(f"{'='*60}")
        print(f"WEB SECURITY SCAN: {self.url}")
        print(f"{'='*60}")

        self.check_headers()
        self.check_ssl()
        self.check_cookies()
        self.check_redirect()

        # Итог
        print(f"\n{'='*60}")
        print(f"РЕЗУЛЬТАТ")
        grade = (
            "A" if self.score >= 90
            else "B" if self.score >= 75
            else "C" if self.score >= 50
            else "D" if self.score >= 30
            else "F"
        )
        colors = {"A": "\033[92m", "B": "\033[92m", "C": "\033[93m",
                  "D": "\033[91m", "F": "\033[91m"}
        print(f"Оценка: {colors[grade]}{grade} ({self.score}/100)\033[0m")
        print(f"Проблем: {len(self.findings)}")

        for f in self.findings:
            print(f"  [{f['severity'].upper()}] {f['description']}")

        return {
            "url": self.url, "score": self.score, "grade": grade,
            "findings": self.findings,
        }


def main():
    parser = argparse.ArgumentParser(description="Web Security Scanner")
    parser.add_argument("-u", "--url", required=True, help="URL для проверки")
    args = parser.parse_args()

    scanner = WebSecurityScanner(args.url)
    result = scanner.scan()

    output = f"/tmp/web_scan_{int(datetime.now().timestamp())}.json"
    with open(output, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nОтчёт: {output}")


if __name__ == "__main__":
    main()
```

---

## 9. Коррелятор событий SIEM

**Назначение:** Движок корреляции событий: цепочки по времени, группировка алертов, уменьшение шума.

```python
#!/usr/bin/env python3
"""
siem_correlator.py — Простой движок корреляции SIEM
Группирует алерты в инциденты, обнаруживает цепочки атак.
Запуск: python3 siem_correlator.py -f alerts.json
"""

import argparse
import json
from datetime import datetime, timedelta
from collections import defaultdict


class CorrelationRule:
    """Базовый класс правила корреляции."""

    def __init__(self, name, severity, window_seconds=300):
        self.name = name
        self.severity = severity
        self.window = timedelta(seconds=window_seconds)

    def evaluate(self, events):
        raise NotImplementedError


class ThresholdRule(CorrelationRule):
    """Правило: N событий одного типа за окно времени."""

    def __init__(self, name, event_type, threshold, group_by="src_ip",
                 window_seconds=300, severity="high"):
        super().__init__(name, severity, window_seconds)
        self.event_type = event_type
        self.threshold = threshold
        self.group_by = group_by

    def evaluate(self, events):
        matches = []
        filtered = [e for e in events if e.get("type") == self.event_type]
        groups = defaultdict(list)

        for e in filtered:
            key = e.get(self.group_by, "unknown")
            groups[key].append(e)

        for key, group in groups.items():
            group.sort(key=lambda x: x.get("timestamp", ""))
            for i, event in enumerate(group):
                ts = datetime.fromisoformat(
                    event.get("timestamp", "2000-01-01T00:00:00")
                )
                window_events = [
                    e for e in group
                    if abs((datetime.fromisoformat(
                        e.get("timestamp", "2000-01-01T00:00:00")
                    ) - ts).total_seconds()) <= self.window.total_seconds()
                ]
                if len(window_events) >= self.threshold:
                    matches.append({
                        "rule": self.name,
                        "severity": self.severity,
                        self.group_by: key,
                        "count": len(window_events),
                        "first_event": window_events[0].get("timestamp"),
                        "last_event": window_events[-1].get("timestamp"),
                    })
                    break
        return matches


class SequenceRule(CorrelationRule):
    """Правило: последовательность типов событий за окно."""

    def __init__(self, name, sequence, group_by="src_ip",
                 window_seconds=600, severity="critical"):
        super().__init__(name, severity, window_seconds)
        self.sequence = sequence
        self.group_by = group_by

    def evaluate(self, events):
        matches = []
        groups = defaultdict(list)

        for e in events:
            if e.get("type") in self.sequence:
                key = e.get(self.group_by, "unknown")
                groups[key].append(e)

        for key, group in groups.items():
            group.sort(key=lambda x: x.get("timestamp", ""))
            seq_idx = 0
            first_ts = None

            for event in group:
                if event.get("type") == self.sequence[seq_idx]:
                    if seq_idx == 0:
                        first_ts = datetime.fromisoformat(
                            event.get("timestamp", "2000-01-01T00:00:00")
                        )
                    else:
                        current_ts = datetime.fromisoformat(
                            event.get("timestamp", "2000-01-01T00:00:00")
                        )
                        if (current_ts - first_ts) > self.window:
                            seq_idx = 0
                            continue

                    seq_idx += 1
                    if seq_idx == len(self.sequence):
                        matches.append({
                            "rule": self.name,
                            "severity": self.severity,
                            self.group_by: key,
                            "sequence": self.sequence,
                            "completed": True,
                        })
                        seq_idx = 0
        return matches


class SIEMCorrelator:
    """Движок корреляции."""

    def __init__(self):
        self.rules = []
        self.incidents = []
        self._setup_default_rules()

    def _setup_default_rules(self):
        self.rules = [
            ThresholdRule(
                "BRUTE_FORCE_SSH",
                event_type="auth_failure",
                threshold=10,
                group_by="src_ip",
                window_seconds=300,
                severity="high",
            ),
            ThresholdRule(
                "PORT_SCAN",
                event_type="connection_attempt",
                threshold=50,
                group_by="src_ip",
                window_seconds=60,
                severity="high",
            ),
            ThresholdRule(
                "WEB_ATTACK_BURST",
                event_type="waf_alert",
                threshold=20,
                group_by="src_ip",
                window_seconds=120,
                severity="high",
            ),
            SequenceRule(
                "BRUTE_FORCE_THEN_LOGIN",
                sequence=["auth_failure", "auth_failure", "auth_success"],
                group_by="src_ip",
                window_seconds=600,
                severity="critical",
            ),
            SequenceRule(
                "RECON_THEN_EXPLOIT",
                sequence=["port_scan", "vulnerability_scan", "exploit_attempt"],
                group_by="src_ip",
                window_seconds=1800,
                severity="critical",
            ),
            SequenceRule(
                "LOGIN_THEN_EXFILTRATION",
                sequence=["auth_success", "data_access", "large_upload"],
                group_by="user",
                window_seconds=3600,
                severity="critical",
            ),
        ]

    def correlate(self, events):
        """Запуск всех правил корреляции."""
        print(f"[*] Корреляция {len(events)} событий по {len(self.rules)} правилам")

        for rule in self.rules:
            matches = rule.evaluate(events)
            if matches:
                for m in matches:
                    self.incidents.append(m)
                    sev = m["severity"]
                    color = "\033[91m" if sev == "critical" else "\033[93m"
                    print(f"{color}[{sev.upper()}] {m['rule']}: "
                          f"{json.dumps({k: v for k, v in m.items() if k != 'rule'}, ensure_ascii=False)}\033[0m")

        # Дедупликация
        seen = set()
        unique = []
        for inc in self.incidents:
            key = f"{inc['rule']}_{inc.get('src_ip', inc.get('user', ''))}"
            if key not in seen:
                seen.add(key)
                unique.append(inc)
        self.incidents = unique

        print(f"\n[*] Уникальных инцидентов: {len(self.incidents)}")
        return self.incidents


def main():
    parser = argparse.ArgumentParser(description="SIEM Correlator")
    parser.add_argument("-f", "--file", required=True,
                        help="JSON-файл с алертами")
    args = parser.parse_args()

    with open(args.file) as f:
        events = json.load(f)

    correlator = SIEMCorrelator()
    incidents = correlator.correlate(events)

    output = f"/tmp/siem_correlation_{int(datetime.now().timestamp())}.json"
    with open(output, "w") as f:
        json.dump(incidents, f, ensure_ascii=False, indent=2)
    print(f"Отчёт: {output}")


if __name__ == "__main__":
    main()
```

**Пример входных данных** (`alerts.json`):
```json
[
    {"timestamp": "2026-11-12T10:00:01", "type": "auth_failure", "src_ip": "10.0.0.50", "user": "admin"},
    {"timestamp": "2026-11-12T10:00:05", "type": "auth_failure", "src_ip": "10.0.0.50", "user": "admin"},
    {"timestamp": "2026-11-12T10:02:30", "type": "auth_success", "src_ip": "10.0.0.50", "user": "admin"},
    {"timestamp": "2026-11-12T10:05:00", "type": "data_access", "src_ip": "10.0.0.50", "user": "admin"},
    {"timestamp": "2026-11-12T10:10:00", "type": "large_upload", "src_ip": "10.0.0.50", "user": "admin"}
]
```

---

## 10. Автоматизация Wazuh API

**Назначение:** Работа с Wazuh SIEM через API: статус агентов, алерты, управление правилами.

**Зависимости:** `pip install requests`

```python
#!/usr/bin/env python3
"""
wazuh_automation.py — Автоматизация Wazuh через API
Запуск: python3 wazuh_automation.py --status
        python3 wazuh_automation.py --alerts --level 10
        python3 wazuh_automation.py --agents
"""

import argparse
import json
import sys

try:
    import requests
    from requests.auth import HTTPBasicAuth
except ImportError:
    print("[!] pip install requests")
    sys.exit(1)


class WazuhAPI:
    """Клиент Wazuh REST API."""

    def __init__(self, host="https://localhost", port=55000,
                 user="wazuh", password="wazuh"):
        self.base_url = f"{host}:{port}"
        self.auth = HTTPBasicAuth(user, password)
        self.token = None
        self.session = requests.Session()
        self.session.verify = False  # Для самоподписанных сертификатов

    def authenticate(self):
        """Получение JWT-токена."""
        resp = self.session.post(
            f"{self.base_url}/security/user/authenticate",
            auth=self.auth,
        )
        if resp.status_code == 200:
            self.token = resp.json()["data"]["token"]
            self.session.headers["Authorization"] = f"Bearer {self.token}"
            return True
        print(f"[!] Ошибка аутентификации: {resp.status_code}")
        return False

    def get_status(self):
        """Статус Wazuh Manager."""
        resp = self.session.get(f"{self.base_url}/manager/status")
        if resp.status_code == 200:
            return resp.json()["data"]["affected_items"][0]
        return None

    def get_agents(self, limit=500):
        """Список агентов."""
        resp = self.session.get(
            f"{self.base_url}/agents",
            params={"limit": limit, "sort": "-lastKeepAlive"},
        )
        if resp.status_code == 200:
            return resp.json()["data"]["affected_items"]
        return []

    def get_alerts(self, level=None, limit=20):
        """Последние алерты."""
        params = {"limit": limit, "sort": "-timestamp"}
        if level:
            params["level"] = level
        resp = self.session.get(
            f"{self.base_url}/alerts", params=params
        )
        if resp.status_code == 200:
            return resp.json()["data"]["affected_items"]
        return []

    def get_agent_vulnerabilities(self, agent_id):
        """Уязвимости агента."""
        resp = self.session.get(
            f"{self.base_url}/vulnerability/{agent_id}"
        )
        if resp.status_code == 200:
            return resp.json()["data"]["affected_items"]
        return []

    def restart_agent(self, agent_id):
        """Перезапуск агента."""
        resp = self.session.put(
            f"{self.base_url}/agents/{agent_id}/restart"
        )
        return resp.status_code == 200


def main():
    parser = argparse.ArgumentParser(description="Wazuh API Automation")
    parser.add_argument("--host", default="https://localhost")
    parser.add_argument("--port", type=int, default=55000)
    parser.add_argument("--user", default="wazuh")
    parser.add_argument("--password", default="wazuh")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--agents", action="store_true")
    parser.add_argument("--alerts", action="store_true")
    parser.add_argument("--level", type=int, help="Мин. уровень алертов")
    parser.add_argument("--vulns", help="ID агента для проверки уязвимостей")
    args = parser.parse_args()

    api = WazuhAPI(args.host, args.port, args.user, args.password)
    if not api.authenticate():
        sys.exit(1)

    if args.status:
        status = api.get_status()
        if status:
            print(json.dumps(status, indent=2, ensure_ascii=False))

    if args.agents:
        agents = api.get_agents()
        print(f"\nАгенты ({len(agents)}):")
        for a in agents:
            status_color = "\033[92m" if a.get("status") == "active" else "\033[91m"
            print(f"  {status_color}{a.get('id')} | {a.get('name')} | "
                  f"{a.get('status')} | {a.get('os', {}).get('name', 'N/A')} | "
                  f"IP: {a.get('ip', 'N/A')}\033[0m")

    if args.alerts:
        alerts = api.get_alerts(level=args.level)
        print(f"\nАлерты ({len(alerts)}):")
        for a in alerts:
            level = a.get("rule", {}).get("level", 0)
            desc = a.get("rule", {}).get("description", "")
            agent = a.get("agent", {}).get("name", "")
            print(f"  [L{level}] {desc} (агент: {agent})")

    if args.vulns:
        vulns = api.get_agent_vulnerabilities(args.vulns)
        print(f"\nУязвимости агента {args.vulns} ({len(vulns)}):")
        for v in vulns:
            print(f"  {v.get('cve')} | {v.get('severity')} | {v.get('name')}")


if __name__ == "__main__":
    main()
```

---

## 11. Парсер Windows Event Log

**Назначение:** Парсинг EVTX-файлов. Извлечение подозрительных Event ID (логины, службы, процессы).

**Зависимости:** `pip install python-evtx`

```python
#!/usr/bin/env python3
"""
evtx_parser.py — Парсер Windows Event Log (EVTX)
Запуск: python3 evtx_parser.py -f Security.evtx
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime

try:
    import Evtx.Evtx as evtx
except ImportError:
    print("[!] pip install python-evtx")
    sys.exit(1)


# Подозрительные Event ID
INTERESTING_EVENTS = {
    # Security
    4624: {"desc": "Успешный вход", "severity": "info"},
    4625: {"desc": "Неудачный вход", "severity": "medium"},
    4634: {"desc": "Выход", "severity": "info"},
    4648: {"desc": "Вход с explicit credentials", "severity": "medium"},
    4672: {"desc": "Special privileges assigned", "severity": "medium"},
    4688: {"desc": "Создание процесса", "severity": "info"},
    4689: {"desc": "Завершение процесса", "severity": "info"},
    4697: {"desc": "Установка службы", "severity": "high"},
    4698: {"desc": "Создание scheduled task", "severity": "high"},
    4720: {"desc": "Создание пользователя", "severity": "high"},
    4722: {"desc": "Аккаунт включён", "severity": "medium"},
    4723: {"desc": "Смена пароля", "severity": "medium"},
    4724: {"desc": "Сброс пароля", "severity": "high"},
    4728: {"desc": "Добавление в global group", "severity": "medium"},
    4732: {"desc": "Добавление в local group", "severity": "high"},
    4738: {"desc": "Изменение пользователя", "severity": "medium"},
    4776: {"desc": "NTLM аутентификация", "severity": "info"},
    # System
    7034: {"desc": "Служба неожиданно завершилась", "severity": "medium"},
    7036: {"desc": "Служба изменила состояние", "severity": "info"},
    7040: {"desc": "Изменение типа запуска службы", "severity": "medium"},
    7045: {"desc": "Новая служба установлена", "severity": "high"},
    # Sysmon
    1: {"desc": "Создание процесса (Sysmon)", "severity": "info"},
    3: {"desc": "Сетевое соединение (Sysmon)", "severity": "info"},
    7: {"desc": "Загрузка модуля (Sysmon)", "severity": "info"},
    8: {"desc": "CreateRemoteThread (Sysmon)", "severity": "high"},
    10: {"desc": "ProcessAccess (Sysmon)", "severity": "medium"},
    11: {"desc": "Создание файла (Sysmon)", "severity": "info"},
    12: {"desc": "Изменение реестра (Sysmon)", "severity": "medium"},
    13: {"desc": "Значение реестра установлено (Sysmon)", "severity": "medium"},
}

SUSPICIOUS_LOGON_TYPES = {
    3: "Network (потенциально lateral movement)",
    4: "Batch",
    5: "Service",
    8: "NetworkCleartext (пароль открытым текстом!)",
    9: "NewCredentials",
    10: "RemoteInteractive (RDP)",
}


def parse_evtx(filepath, interesting_only=True):
    """Парсинг EVTX-файла."""
    events = []
    event_counter = Counter()
    ns = "{http://schemas.microsoft.com/win/2004/08/events/event}"

    with evtx.Evtx(filepath) as log:
        for record in log.records():
            try:
                root = ET.fromstring(record.xml())
                system = root.find(f"{ns}System")
                event_id = int(system.find(f"{ns}EventID").text)
                event_counter[event_id] += 1

                if interesting_only and event_id not in INTERESTING_EVENTS:
                    continue

                timestamp = system.find(f"{ns}TimeCreated").attrib.get(
                    "SystemTime", ""
                )
                computer = system.find(f"{ns}Computer").text

                # Извлечение EventData
                event_data = {}
                data_elem = root.find(f"{ns}EventData")
                if data_elem is not None:
                    for data in data_elem.findall(f"{ns}Data"):
                        name = data.attrib.get("Name", "")
                        value = data.text or ""
                        if name:
                            event_data[name] = value

                event_info = INTERESTING_EVENTS.get(event_id, {})
                events.append({
                    "event_id": event_id,
                    "timestamp": timestamp,
                    "computer": computer,
                    "description": event_info.get("desc", ""),
                    "severity": event_info.get("severity", "info"),
                    "data": event_data,
                })
            except (ET.ParseError, AttributeError):
                continue

    return events, event_counter


def analyze_events(events):
    """Анализ извлечённых событий."""
    findings = []

    # Brute-force: много 4625
    failed_by_ip = Counter()
    for e in events:
        if e["event_id"] == 4625:
            ip = e["data"].get("IpAddress", "N/A")
            failed_by_ip[ip] += 1
    for ip, count in failed_by_ip.most_common(10):
        if count >= 10:
            findings.append({
                "type": "BRUTE_FORCE",
                "severity": "high",
                "message": f"IP {ip}: {count} неудачных входов",
            })

    # Подозрительные Logon Type
    for e in events:
        if e["event_id"] == 4624:
            logon_type = int(e["data"].get("LogonType", 0))
            if logon_type in SUSPICIOUS_LOGON_TYPES:
                user = e["data"].get("TargetUserName", "?")
                findings.append({
                    "type": "SUSPICIOUS_LOGON",
                    "severity": "medium",
                    "message": (
                        f"Logon Type {logon_type} "
                        f"({SUSPICIOUS_LOGON_TYPES[logon_type]}) "
                        f"для {user}"
                    ),
                })

    # Новые службы (7045)
    for e in events:
        if e["event_id"] == 7045:
            svc_name = e["data"].get("ServiceName", "?")
            svc_file = e["data"].get("ImagePath", "?")
            findings.append({
                "type": "NEW_SERVICE",
                "severity": "high",
                "message": f"Новая служба: {svc_name} → {svc_file}",
            })

    return findings


def main():
    parser = argparse.ArgumentParser(description="EVTX Parser")
    parser.add_argument("-f", "--file", required=True, help="EVTX-файл")
    parser.add_argument("--all", action="store_true",
                        help="Все события (не только интересные)")
    args = parser.parse_args()

    print(f"[*] Парсинг: {args.file}")
    events, counter = parse_evtx(args.file, not args.all)
    print(f"[*] Извлечено: {len(events)} событий")

    print(f"\nTop-10 Event ID:")
    for eid, count in counter.most_common(10):
        desc = INTERESTING_EVENTS.get(eid, {}).get("desc", "")
        print(f"  {eid}: {count:>6}  {desc}")

    findings = analyze_events(events)
    if findings:
        print(f"\n[!] Находки ({len(findings)}):")
        for f in findings:
            color = "\033[91m" if f["severity"] == "high" else "\033[93m"
            print(f"  {color}[{f['severity'].upper()}] {f['message']}\033[0m")

    output = f"/tmp/evtx_analysis_{int(datetime.now().timestamp())}.json"
    with open(output, "w") as f_out:
        json.dump({"events": events[:100], "findings": findings}, f_out,
                  ensure_ascii=False, indent=2)
    print(f"\nОтчёт: {output}")


if __name__ == "__main__":
    main()
```

---

## 12. Сканер уязвимостей зависимостей

**Назначение:** Проверка Python (pip) и Node.js (npm) зависимостей на известные CVE.

```python
#!/usr/bin/env python3
"""
dependency_scanner.py — Сканер уязвимостей зависимостей
Проверяет pip/npm пакеты по API с CVE-базами.
Запуск: python3 dependency_scanner.py --pip requirements.txt
        python3 dependency_scanner.py --npm package-lock.json
"""

import argparse
import json
import re
import sys

try:
    import requests
except ImportError:
    print("[!] pip install requests")
    sys.exit(1)


class DependencyScanner:
    """Сканер уязвимостей зависимостей."""

    OSV_API = "https://api.osv.dev/v1/query"

    def scan_pip(self, requirements_file):
        """Сканирование Python-зависимостей."""
        packages = []
        with open(requirements_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                m = re.match(r"([a-zA-Z0-9_.-]+)\s*[=<>!~]+\s*([0-9.]+)", line)
                if m:
                    packages.append({
                        "name": m.group(1),
                        "version": m.group(2),
                        "ecosystem": "PyPI",
                    })
                else:
                    name = re.match(r"([a-zA-Z0-9_.-]+)", line)
                    if name:
                        packages.append({
                            "name": name.group(1),
                            "version": "latest",
                            "ecosystem": "PyPI",
                        })
        return self._scan_packages(packages)

    def scan_npm(self, lockfile):
        """Сканирование npm-зависимостей."""
        with open(lockfile) as f:
            data = json.load(f)

        packages = []
        deps = data.get("dependencies", data.get("packages", {}))
        for name, info in deps.items():
            if name.startswith("node_modules/"):
                name = name[len("node_modules/"):]
            if not name:
                continue
            version = info if isinstance(info, str) else info.get("version", "")
            if version:
                packages.append({
                    "name": name,
                    "version": version,
                    "ecosystem": "npm",
                })
        return self._scan_packages(packages)

    def _scan_packages(self, packages):
        """Проверка пакетов через OSV API."""
        findings = []
        total = len(packages)
        print(f"[*] Проверка {total} пакетов...")

        for i, pkg in enumerate(packages):
            if pkg["version"] == "latest":
                continue
            try:
                resp = requests.post(
                    self.OSV_API,
                    json={
                        "package": {
                            "name": pkg["name"],
                            "ecosystem": pkg["ecosystem"],
                        },
                        "version": pkg["version"],
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    vulns = resp.json().get("vulns", [])
                    if vulns:
                        for v in vulns:
                            severity = "unknown"
                            for s in v.get("severity", []):
                                severity = s.get("type", severity)
                            findings.append({
                                "package": pkg["name"],
                                "version": pkg["version"],
                                "vuln_id": v.get("id", ""),
                                "summary": v.get("summary", ""),
                                "severity": severity,
                                "aliases": v.get("aliases", []),
                            })
                            print(
                                f"\033[91m  [VULN] {pkg['name']}=={pkg['version']}: "
                                f"{v.get('id', '')} — {v.get('summary', '')[:60]}\033[0m"
                            )
            except requests.RequestException:
                pass

            if (i + 1) % 10 == 0:
                print(f"  Проверено: {i + 1}/{total}")

        print(f"\n[*] Уязвимостей найдено: {len(findings)}")
        return findings


def main():
    parser = argparse.ArgumentParser(description="Dependency Vulnerability Scanner")
    parser.add_argument("--pip", help="requirements.txt")
    parser.add_argument("--npm", help="package-lock.json")
    args = parser.parse_args()

    scanner = DependencyScanner()
    all_findings = []

    if args.pip:
        all_findings.extend(scanner.scan_pip(args.pip))
    if args.npm:
        all_findings.extend(scanner.scan_npm(args.npm))

    if not args.pip and not args.npm:
        parser.print_help()
        return

    output = f"/tmp/dep_scan_{int(json.loads(json.dumps(0)))}.json"
    with open(output, "w") as f:
        json.dump(all_findings, f, ensure_ascii=False, indent=2)
    print(f"Отчёт: {output}")


if __name__ == "__main__":
    main()
```

---

## 13. Генератор отчётов безопасности

**Назначение:** Сбор данных из всех инструментов, генерация HTML-отчёта.

```python
#!/usr/bin/env python3
"""
report_generator.py — Генератор HTML-отчёта безопасности
Запуск: python3 report_generator.py -o report.html
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


class SecurityReportGenerator:
    """Генератор сводного отчёта безопасности."""

    def __init__(self):
        self.sections = []
        self.summary = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    def add_section(self, title, content_html, findings=None):
        self.sections.append({
            "title": title,
            "content": content_html,
        })
        if findings:
            for f in findings:
                sev = f.get("severity", "low")
                if sev in self.summary:
                    self.summary[sev] += 1

    def collect_system_info(self):
        """Сбор системной информации."""

        def run(cmd):
            try:
                return subprocess.run(
                    cmd, shell=True, capture_output=True,
                    text=True, timeout=10,
                ).stdout.strip()
            except (subprocess.TimeoutExpired, Exception):
                return "N/A"

        info = {
            "hostname": run("hostname -f"),
            "os": run("cat /etc/os-release | grep PRETTY_NAME | cut -d'\"' -f2"),
            "kernel": run("uname -r"),
            "uptime": run("uptime -p"),
            "cpu_load": run("cat /proc/loadavg"),
            "memory": run("free -h | awk '/Mem:/{print $3\"/\"$2}'"),
            "disk": run("df -h / | awk 'NR==2{print $3\"/\"$2\" (\"$5\")\"}'"),
        }

        html = "<table>"
        for k, v in info.items():
            html += f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>"
        html += "</table>"
        self.add_section("Информация о системе", html)

    def collect_network_info(self):
        """Сетевая информация."""

        def run(cmd):
            try:
                return subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=10,
                ).stdout.strip()
            except Exception:
                return "N/A"

        ports = run("ss -tlnp 2>/dev/null | head -20")
        connections = run("ss -tun state established 2>/dev/null | wc -l")

        html = f"<p>Активных соединений: <strong>{connections}</strong></p>"
        html += f"<pre>{ports}</pre>"
        self.add_section("Сеть", html)

    def collect_user_info(self):
        """Информация о пользователях."""

        def run(cmd):
            try:
                return subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=10,
                ).stdout.strip()
            except Exception:
                return "N/A"

        users = run("who 2>/dev/null || echo 'N/A'")
        last_logins = run("last -5 2>/dev/null || echo 'N/A'")
        failed = run("grep -c 'Failed password' /var/log/auth.log 2>/dev/null || echo 0")

        html = f"<p>Неудачных входов (auth.log): <strong>{failed}</strong></p>"
        html += f"<h4>Текущие сессии</h4><pre>{users}</pre>"
        html += f"<h4>Последние входы</h4><pre>{last_logins}</pre>"
        self.add_section("Пользователи", html)

    def import_json_findings(self, filepath, section_title):
        """Импорт находок из JSON-файла."""
        if not Path(filepath).exists():
            return
        with open(filepath) as f:
            data = json.load(f)

        findings = data if isinstance(data, list) else data.get("findings", [])
        if not findings:
            return

        html = "<table><tr><th>Severity</th><th>Описание</th></tr>"
        for f in findings:
            sev = f.get("severity", "info")
            desc = f.get("description", f.get("message", f.get("rule", "")))
            color = {
                "critical": "#dc3545", "high": "#dc3545",
                "medium": "#ffc107", "low": "#17a2b8",
            }.get(sev, "#6c757d")
            html += f'<tr><td style="color:{color}">{sev.upper()}</td><td>{desc}</td></tr>'
        html += "</table>"
        self.add_section(section_title, html, findings)

    def generate_html(self, output_path):
        """Генерация HTML-отчёта."""
        total_issues = sum(self.summary.values())
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Отчёт безопасности</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
.container {{ max-width: 900px; margin: 0 auto; }}
.header {{ background: linear-gradient(135deg, #1a237e, #0d47a1); color: white; padding: 30px; border-radius: 12px 12px 0 0; }}
.summary {{ display: flex; gap: 15px; margin: 20px 0; }}
.card {{ flex: 1; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.card .number {{ font-size: 32px; font-weight: bold; }}
.critical {{ color: #dc3545; }}
.high {{ color: #e65100; }}
.medium {{ color: #f9a825; }}
.low {{ color: #0288d1; }}
.section {{ background: white; margin: 15px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.section h2 {{ color: #1a237e; border-bottom: 2px solid #e3f2fd; padding-bottom: 8px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #e0e0e0; padding: 8px; text-align: left; }}
th {{ background: #e3f2fd; }}
pre {{ background: #263238; color: #e0e0e0; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 13px; }}
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>Отчёт безопасности</h1>
<p>{report_time}</p>
</div>

<div class="summary">
<div class="card"><div class="number critical">{self.summary['critical']}</div><div>Critical</div></div>
<div class="card"><div class="number high">{self.summary['high']}</div><div>High</div></div>
<div class="card"><div class="number medium">{self.summary['medium']}</div><div>Medium</div></div>
<div class="card"><div class="number low">{self.summary['low']}</div><div>Low</div></div>
<div class="card"><div class="number">{total_issues}</div><div>Всего</div></div>
</div>
"""
        for section in self.sections:
            html += f"""
<div class="section">
<h2>{section['title']}</h2>
{section['content']}
</div>
"""
        html += """
<p style="text-align:center; color:#9e9e9e; margin-top:20px;">
Сгенерировано report_generator.py</p>
</div></body></html>
"""
        with open(output_path, "w") as f:
            f.write(html)
        print(f"[*] Отчёт сохранён: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Security Report Generator")
    parser.add_argument("-o", "--output", default="/tmp/security_report.html")
    parser.add_argument("--findings", nargs="*", help="JSON-файлы с находками")
    args = parser.parse_args()

    gen = SecurityReportGenerator()
    gen.collect_system_info()
    gen.collect_network_info()
    gen.collect_user_info()

    if args.findings:
        for f in args.findings:
            gen.import_json_findings(f, f"Находки: {Path(f).stem}")

    gen.generate_html(args.output)


if __name__ == "__main__":
    main()
```

---

## 14. Honeypot-детектор

**Назначение:** Простой сервисный honeypot (SSH/HTTP), логирует все попытки подключения.

```python
#!/usr/bin/env python3
"""
honeypot.py — Простой honeypot для обнаружения разведки
Имитирует SSH/HTTP-сервис, логирует все попытки подключения.
Запуск: sudo python3 honeypot.py --ssh 2222 --http 8080
"""

import argparse
import json
import socket
import threading
import logging
from datetime import datetime


logging.basicConfig(
    filename="/var/log/security/honeypot.log",
    level=logging.INFO,
    format="%(message)s",
)


class HoneypotServer:
    """Базовый honeypot-сервер."""

    def __init__(self, port, service_name, banner=""):
        self.port = port
        self.service_name = service_name
        self.banner = banner
        self.connections = []

    def log_connection(self, client_ip, client_port, data=""):
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": self.service_name,
            "port": self.port,
            "source_ip": client_ip,
            "source_port": client_port,
            "data": data[:500],
        }
        self.connections.append(entry)
        logging.info(json.dumps(entry, ensure_ascii=False))
        print(f"\033[91m[HONEYPOT] {self.service_name}:{self.port} ← "
              f"{client_ip}:{client_port}\033[0m")

    def start(self):
        raise NotImplementedError


class SSHHoneypot(HoneypotServer):
    """SSH-honeypot."""

    SSH_BANNER = b"SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.1\r\n"

    def __init__(self, port=2222):
        super().__init__(port, "SSH", "")

    def handle_client(self, conn, addr):
        try:
            conn.settimeout(30)
            conn.send(self.SSH_BANNER)
            data = conn.recv(1024)
            self.log_connection(addr[0], addr[1], data.decode("utf-8", errors="ignore"))
            # Симулируем задержку и отказ
            conn.send(b"Permission denied (publickey,password).\r\n")
        except (socket.timeout, ConnectionResetError, BrokenPipeError):
            self.log_connection(addr[0], addr[1], "(timeout/reset)")
        finally:
            conn.close()

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.port))
        sock.listen(5)
        print(f"[*] SSH Honeypot listening on :{self.port}")

        while True:
            conn, addr = sock.accept()
            t = threading.Thread(target=self.handle_client, args=(conn, addr))
            t.daemon = True
            t.start()


class HTTPHoneypot(HoneypotServer):
    """HTTP-honeypot."""

    RESPONSE = (
        b"HTTP/1.1 200 OK\r\n"
        b"Server: Apache/2.4.54 (Ubuntu)\r\n"
        b"Content-Type: text/html\r\n"
        b"\r\n"
        b"<html><body><h1>Welcome</h1></body></html>\r\n"
    )

    def __init__(self, port=8080):
        super().__init__(port, "HTTP", "")

    def handle_client(self, conn, addr):
        try:
            conn.settimeout(10)
            data = conn.recv(4096)
            request = data.decode("utf-8", errors="ignore")
            self.log_connection(addr[0], addr[1], request.split("\r\n")[0])
            conn.send(self.RESPONSE)
        except (socket.timeout, ConnectionResetError):
            pass
        finally:
            conn.close()

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.port))
        sock.listen(5)
        print(f"[*] HTTP Honeypot listening on :{self.port}")

        while True:
            conn, addr = sock.accept()
            t = threading.Thread(target=self.handle_client, args=(conn, addr))
            t.daemon = True
            t.start()


def main():
    parser = argparse.ArgumentParser(description="Honeypot")
    parser.add_argument("--ssh", type=int, help="SSH honeypot порт")
    parser.add_argument("--http", type=int, help="HTTP honeypot порт")
    args = parser.parse_args()

    threads = []
    if args.ssh:
        hp = SSHHoneypot(args.ssh)
        t = threading.Thread(target=hp.start)
        t.daemon = True
        t.start()
        threads.append(t)

    if args.http:
        hp = HTTPHoneypot(args.http)
        t = threading.Thread(target=hp.start)
        t.daemon = True
        t.start()
        threads.append(t)

    if not threads:
        parser.print_help()
        return

    print(f"[*] Honeypot запущен. Ctrl+C для остановки.")
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n[*] Остановлен")


if __name__ == "__main__":
    main()
```

---

## 15. Мониторинг DNS-запросов

**Назначение:** Анализ DNS-кеша и запросов в реальном времени через scapy. Обнаружение DGA, tunneling.

```python
#!/usr/bin/env python3
"""
dns_monitor.py — Мониторинг DNS-запросов (Python-версия)
Обнаружение DGA-доменов, tunneling, подозрительных TLD.
Запуск: sudo python3 dns_monitor.py -i eth0 -d 120
"""

import argparse
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

try:
    from scapy.all import sniff, DNS, DNSQR, IP, conf
except ImportError:
    print("[!] pip install scapy")
    sys.exit(1)


class DNSMonitor:
    """Мониторинг и анализ DNS-запросов."""

    SUSPICIOUS_TLDS = {
        ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
        ".click", ".loan", ".work", ".date", ".racing", ".bid",
        ".win", ".stream", ".download",
    }

    def __init__(self, interface="eth0", duration=300):
        self.interface = interface
        self.duration = duration
        self.queries = []
        self.domain_counter = Counter()
        self.ip_queries = defaultdict(list)
        self.alerts = []

    @staticmethod
    def entropy(s):
        """Энтропия строки (для обнаружения DGA)."""
        if not s:
            return 0.0
        counter = Counter(s.lower())
        length = len(s)
        return -sum(
            (c / length) * math.log2(c / length)
            for c in counter.values()
        )

    def process_packet(self, pkt):
        """Обработка DNS-пакета."""
        if DNS in pkt and pkt[DNS].qr == 0 and DNSQR in pkt:
            query_name = pkt[DNSQR].qname.decode("utf-8", errors="ignore").rstrip(".")
            src_ip = pkt[IP].src if IP in pkt else "?"
            query_type = pkt[DNSQR].qtype

            self.queries.append({
                "time": time.time(),
                "domain": query_name,
                "src_ip": src_ip,
                "type": query_type,
            })

            self.domain_counter[query_name] += 1
            self.ip_queries[src_ip].append(query_name)

            # Проверка длины (DNS tunneling)
            if len(query_name) > 60:
                self._alert("DNS_TUNNEL_LENGTH", "high",
                            f"Длинный домен ({len(query_name)} символов): {query_name[:80]}",
                            src_ip)

            # Проверка энтропии (DGA)
            parts = query_name.split(".")
            if len(parts) >= 2:
                subdomain = parts[0]
                ent = self.entropy(subdomain)
                if ent > 4.0 and len(subdomain) > 10:
                    self._alert("DGA_DOMAIN", "high",
                                f"Высокая энтропия ({ent:.2f}): {query_name}",
                                src_ip)

            # Подозрительный TLD
            for tld in self.SUSPICIOUS_TLDS:
                if query_name.endswith(tld):
                    self._alert("SUSPICIOUS_TLD", "medium",
                                f"Подозрительный TLD: {query_name}", src_ip)
                    break

            # TXT-запросы (часто используются для tunneling)
            if query_type == 16:  # TXT
                self._alert("TXT_QUERY", "medium",
                            f"TXT-запрос: {query_name}", src_ip)

    def _alert(self, category, severity, message, src_ip=""):
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "category": category,
            "severity": severity,
            "message": message,
            "src_ip": src_ip,
        }
        self.alerts.append(alert)
        color = "\033[91m" if severity in ("critical", "high") else "\033[93m"
        print(f"{color}[{severity.upper()}] {category}: {message}\033[0m")

    def run(self):
        print(f"[*] DNS мониторинг на {self.interface} ({self.duration} сек)")
        try:
            sniff(
                iface=self.interface,
                filter="udp port 53",
                prn=self.process_packet,
                timeout=self.duration,
                store=False,
            )
        except PermissionError:
            print("[!] Нужен root")
            return

        self.report()

    def report(self):
        print(f"\n{'='*60}")
        print(f"DNS МОНИТОРИНГ — ИТОГ")
        print(f"Запросов: {len(self.queries)}")
        print(f"Уникальных доменов: {len(self.domain_counter)}")
        print(f"Алертов: {len(self.alerts)}")

        print(f"\nTop-15 доменов:")
        for domain, count in self.domain_counter.most_common(15):
            print(f"  {count:>5}  {domain}")

        print(f"\nTop-10 клиентов:")
        top_clients = sorted(
            self.ip_queries.items(),
            key=lambda x: len(x[1]), reverse=True,
        )[:10]
        for ip, queries in top_clients:
            print(f"  {len(queries):>5}  {ip}")


def main():
    parser = argparse.ArgumentParser(description="DNS Monitor")
    parser.add_argument("-i", "--interface", default="eth0")
    parser.add_argument("-d", "--duration", type=int, default=300)
    args = parser.parse_args()

    monitor = DNSMonitor(args.interface, args.duration)
    monitor.run()


if __name__ == "__main__":
    main()
```

---

# СПРАВОЧНИК: УСТАНОВКА ЗАВИСИМОСТЕЙ

```bash
# Базовые зависимости для всех скриптов
pip install scapy requests yara-python pefile python-evtx \
    python-dateutil pandas numpy

# Для Wazuh API
pip install requests

# Для генератора отчётов (опционально)
pip install jinja2 weasyprint  # Для PDF-генерации
```

## Быстрый запуск

```bash
# 1. Сетевой анализатор (нужен root)
sudo python3 network_analyzer.py -i eth0 -d 300 -v

# 2. Коррелятор логов
python3 log_correlator.py -a /var/log/auth.log -w /var/log/apache2/access.log

# 3. YARA-сканирование
python3 yara_scanner.py -t /tmp -t /var/tmp -t /home

# 4. Threat Intelligence проверка
python3 threat_intel.py --ip 185.220.101.1 --config ti_config.json

# 5. Анализ email
python3 email_analyzer.py -f phishing_sample.eml

# 6. PE-анализ
python3 pe_analyzer.py -f suspicious.exe

# 7. Аномалии трафика (живой мониторинг)
python3 traffic_anomaly.py --live -d 600

# 8. Web security scan (своих ресурсов!)
python3 web_security_scanner.py -u https://your-site.com

# 9. SIEM-корреляция
python3 siem_correlator.py -f alerts.json

# 10. Wazuh автоматизация
python3 wazuh_automation.py --agents --alerts --level 10

# 11. EVTX-парсинг
python3 evtx_parser.py -f Security.evtx

# 12. Сканер зависимостей
python3 dependency_scanner.py --pip requirements.txt

# 13. Отчёт
python3 report_generator.py -o report.html --findings *.json

# 14. Honeypot
sudo python3 honeypot.py --ssh 2222 --http 8080

# 15. DNS-мониторинг
sudo python3 dns_monitor.py -i eth0 -d 120
```

---

*15 Python-инструментов · Детекция · Анализ · Threat Intel · Автоматизация SIEM*
*Все скрипты — оборонительные. Для курса «Кибер-Щит».*
