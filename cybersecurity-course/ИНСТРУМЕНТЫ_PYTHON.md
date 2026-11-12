# ИНСТРУМЕНТЫ PYTHON ДЛЯ SOC-АНАЛИТИКА

> 30 готовых инструментов на Python для мониторинга, детекции и реагирования.
> Каждый скрипт — рабочий, документированный, готовый к использованию.
> Зависимости: стандартная библиотека Python 3.8+ и распространённые пакеты (requests, scapy, yara-python, python-evtx, dpkt, jinja2).

---

## ЧАСТЬ 1: СЕТЕВОЙ АНАЛИЗ

---

## Инструмент 1: Network Asset Scanner

### Назначение
Сканирование подсети для инвентаризации хостов. SOC-аналитику необходимо знать, какие устройства находятся в сети.
Скрипт выполняет ARP-сканирование локальной подсети (или TCP-пинг для удалённых), сравнивает результат с предыдущим
baseline-файлом и выводит **новые** и **исчезнувшие** хосты.

### Как работает
1. Принимает подсеть в формате CIDR (например, `192.168.1.0/24`).
2. Для каждого адреса в подсети пробует TCP-подключение к портам 22, 80, 443, 445 (timeout 0.3 с).
3. Для отвечающих хостов определяет имя (reverse DNS) и открытые порты.
4. Загружает предыдущий baseline (JSON) и сравнивает: новые хосты, исчезнувшие хосты, изменившиеся порты.
5. Сохраняет текущее состояние как новый baseline.

### Код

```python
#!/usr/bin/env python3
"""network_scanner.py — Сканер подсети для инвентаризации активов."""

import socket
import json
import sys
import os
from datetime import datetime
from ipaddress import IPv4Network
from concurrent.futures import ThreadPoolExecutor, as_completed

COMMON_PORTS = [22, 80, 443, 445, 3389, 8080, 8443]
TIMEOUT = 0.5
BASELINE_FILE = "network_baseline.json"


def tcp_probe(ip: str, port: int, timeout: float = TIMEOUT) -> bool:
    """Проверка доступности порта TCP-подключением."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((ip, port)) == 0
    except OSError:
        return False


def reverse_dns(ip: str) -> str:
    """Обратный DNS-запрос."""
    try:
        return socket.gethostbyaddr(ip)[0]
    except (socket.herror, socket.gaierror, OSError):
        return ""


def scan_host(ip: str) -> dict | None:
    """Сканирование одного хоста: открытые порты + reverse DNS."""
    open_ports = []
    for port in COMMON_PORTS:
        if tcp_probe(ip, port):
            open_ports.append(port)
    if not open_ports:
        return None
    hostname = reverse_dns(ip)
    return {
        "ip": ip,
        "hostname": hostname,
        "open_ports": open_ports,
        "last_seen": datetime.utcnow().isoformat()
    }


def scan_network(cidr: str, max_workers: int = 100) -> list[dict]:
    """Параллельное сканирование подсети."""
    network = IPv4Network(cidr, strict=False)
    results = []
    hosts = [str(ip) for ip in network.hosts()]
    print(f"[*] Сканирование {len(hosts)} хостов в {cidr}...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(scan_host, ip): ip for ip in hosts}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
                print(f"  [+] {result['ip']:16s} порты: {result['open_ports']}"
                      f"  {result['hostname']}")
    return results


def load_baseline(path: str) -> dict:
    """Загрузка предыдущего baseline."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return {entry["ip"]: entry for entry in data}


def save_baseline(path: str, hosts: list[dict]):
    """Сохранение текущего baseline."""
    with open(path, "w") as f:
        json.dump(hosts, f, indent=2, ensure_ascii=False)
    print(f"[*] Baseline сохранён: {path} ({len(hosts)} хостов)")


def compare_baselines(old: dict, new: dict):
    """Сравнение baseline: новые хосты, исчезнувшие, изменившиеся порты."""
    old_ips = set(old.keys())
    new_ips = set(new.keys())

    added = new_ips - old_ips
    removed = old_ips - new_ips
    common = old_ips & new_ips

    if added:
        print(f"\n[!] НОВЫЕ ХОСТЫ ({len(added)}):")
        for ip in sorted(added):
            h = new[ip]
            print(f"    {ip:16s} порты: {h['open_ports']}  {h['hostname']}")

    if removed:
        print(f"\n[!] ИСЧЕЗНУВШИЕ ХОСТЫ ({len(removed)}):")
        for ip in sorted(removed):
            h = old[ip]
            print(f"    {ip:16s} порты: {h['open_ports']}  {h['hostname']}")

    changed = []
    for ip in common:
        if set(old[ip]["open_ports"]) != set(new[ip]["open_ports"]):
            changed.append(ip)
    if changed:
        print(f"\n[!] ИЗМЕНИЛИСЬ ПОРТЫ ({len(changed)}):")
        for ip in sorted(changed):
            print(f"    {ip:16s}  было: {old[ip]['open_ports']}"
                  f"  стало: {new[ip]['open_ports']}")

    if not added and not removed and not changed:
        print("\n[✓] Изменений не обнаружено.")


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 network_scanner.py <CIDR>")
        print("Пример:        python3 network_scanner.py 192.168.1.0/24")
        sys.exit(1)

    cidr = sys.argv[1]
    baseline_old = load_baseline(BASELINE_FILE)
    hosts = scan_network(cidr)
    baseline_new = {h["ip"]: h for h in hosts}

    if baseline_old:
        compare_baselines(baseline_old, baseline_new)
    else:
        print(f"\n[*] Первый запуск — baseline создан ({len(hosts)} хостов).")

    save_baseline(BASELINE_FILE, hosts)


if __name__ == "__main__":
    main()
```

### Использование

```bash
# Первый запуск — создание baseline
python3 network_scanner.py 192.168.1.0/24

# Повторный запуск — сравнение с baseline
python3 network_scanner.py 192.168.1.0/24
```

### Пример вывода

```
[*] Сканирование 254 хостов в 192.168.1.0/24...
  [+] 192.168.1.1      порты: [80, 443]  router.local
  [+] 192.168.1.10     порты: [22, 80]  webserver.local
  [+] 192.168.1.50     порты: [445, 3389]  win-pc01.corp.local

[!] НОВЫЕ ХОСТЫ (1):
    192.168.1.50     порты: [445, 3389]  win-pc01.corp.local

[✓] Baseline сохранён: network_baseline.json (3 хостов)
```

### Интеграция с SOC
- Запускайте по cron каждые 15 минут, результаты различий отправляйте в Wazuh через `ossec-log`.
- JSON-baseline можно подгружать в SIEM как asset inventory.
- Алерт «новый хост в сети» — один из первых признаков компрометации (rogue device).

---

## Инструмент 2: DNS Entropy Analyzer

### Назначение
Анализ DNS-запросов для обнаружения DGA-доменов (Domain Generation Algorithm), DNS-туннелирования
и аномальной активности. Высокая энтропия доменного имени — признак автоматически сгенерированного домена,
а длинные TXT/CNAME запросы могут указывать на DNS-туннель (iodine, dnscat2).

### Как работает
1. Читает DNS-логи (формат Zeek dns.log или plain text: `timestamp query type`).
2. Для каждого домена вычисляет энтропию Шеннона.
3. Помечает домены с энтропией > порога (по умолчанию 3.5) как подозрительные (DGA).
4. Ищет длинные поддомены (> 50 символов) — признак DNS-туннеля.
5. Считает частоту запросов по доменам — аномально высокая частота = beaconing или tunneling.
6. Выводит отчёт с категоризацией угроз.

### Код

```python
#!/usr/bin/env python3
"""dns_analyzer.py — Анализ DNS-логов: DGA, туннели, аномалии."""

import math
import sys
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime

ENTROPY_THRESHOLD = 3.5
SUBDOMAIN_LEN_THRESHOLD = 50
FREQ_THRESHOLD = 100  # запросов за период


def shannon_entropy(s: str) -> float:
    """Вычисление энтропии Шеннона для строки."""
    if not s:
        return 0.0
    freq = Counter(s)
    length = len(s)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in freq.values()
    )


def extract_domain_parts(fqdn: str) -> tuple:
    """Разбор FQDN на поддомен и базовый домен."""
    parts = fqdn.rstrip(".").split(".")
    if len(parts) <= 2:
        return "", fqdn.rstrip(".")
    base = ".".join(parts[-2:])
    subdomain = ".".join(parts[:-2])
    return subdomain, base


def parse_dns_log(filepath: str) -> list[dict]:
    """Парсинг DNS-лога (поддержка Zeek и простого формата)."""
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) >= 3:
                records.append({
                    "timestamp": parts[0],
                    "query": parts[1] if len(parts) > 1 else "",
                    "qtype": parts[2] if len(parts) > 2 else "A",
                    "raw": line
                })
    return records


def analyze_dga(records: list[dict]) -> list[dict]:
    """Обнаружение DGA-доменов по энтропии."""
    suspicious = []
    for rec in records:
        subdomain, base = extract_domain_parts(rec["query"])
        target = subdomain if subdomain else base
        ent = shannon_entropy(target)
        if ent > ENTROPY_THRESHOLD:
            suspicious.append({
                "query": rec["query"],
                "entropy": round(ent, 3),
                "subdomain": subdomain,
                "base_domain": base,
                "type": "DGA",
                "timestamp": rec["timestamp"]
            })
    return suspicious


def analyze_tunneling(records: list[dict]) -> list[dict]:
    """Обнаружение DNS-туннелирования по длине поддоменов и типу записей."""
    suspicious = []
    for rec in records:
        subdomain, base = extract_domain_parts(rec["query"])
        if len(subdomain) > SUBDOMAIN_LEN_THRESHOLD:
            suspicious.append({
                "query": rec["query"],
                "subdomain_length": len(subdomain),
                "base_domain": base,
                "qtype": rec["qtype"],
                "type": "TUNNEL",
                "timestamp": rec["timestamp"]
            })
        if rec["qtype"] in ("TXT", "NULL", "CNAME") and len(subdomain) > 30:
            suspicious.append({
                "query": rec["query"],
                "subdomain_length": len(subdomain),
                "qtype": rec["qtype"],
                "type": "TUNNEL_TXT",
                "timestamp": rec["timestamp"]
            })
    return suspicious


def analyze_frequency(records: list[dict]) -> list[dict]:
    """Обнаружение аномальной частоты DNS-запросов (beaconing)."""
    domain_counts = Counter()
    for rec in records:
        _, base = extract_domain_parts(rec["query"])
        domain_counts[base] += 1

    suspicious = []
    for domain, count in domain_counts.most_common():
        if count >= FREQ_THRESHOLD:
            suspicious.append({
                "domain": domain,
                "count": count,
                "type": "HIGH_FREQ"
            })
    return suspicious


def print_report(dga: list, tunnels: list, freq: list):
    """Вывод отчёта по результатам анализа."""
    print("=" * 70)
    print("ОТЧЁТ: АНАЛИЗ DNS-АКТИВНОСТИ")
    print("=" * 70)

    print(f"\n--- DGA-подозрения (энтропия > {ENTROPY_THRESHOLD}) ---")
    if dga:
        unique_bases = set()
        for item in dga:
            unique_bases.add(item.get("base_domain", ""))
            print(f"  [{item['entropy']:.2f}] {item['query']}")
        print(f"  Итого: {len(dga)} запросов, {len(unique_bases)} уникальных баз-доменов")
    else:
        print("  Не обнаружено.")

    print(f"\n--- DNS-туннелирование (поддомен > {SUBDOMAIN_LEN_THRESHOLD} символов) ---")
    if tunnels:
        for item in tunnels:
            print(f"  [{item['type']}] длина={item.get('subdomain_length', '?')} "
                  f" {item['query'][:80]}...")
        print(f"  Итого: {len(tunnels)} подозрительных запросов")
    else:
        print("  Не обнаружено.")

    print(f"\n--- Аномальная частота (> {FREQ_THRESHOLD} запросов) ---")
    if freq:
        for item in freq:
            print(f"  [{item['count']:>6}] {item['domain']}")
    else:
        print("  Не обнаружено.")

    total = len(dga) + len(tunnels) + len(freq)
    print(f"\n{'=' * 70}")
    print(f"ВСЕГО АНОМАЛИЙ: {total}")
    if total > 0:
        print("[!] Рекомендуется проверить выявленные домены в Threat Intelligence.")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 dns_analyzer.py <dns_log_file>")
        print("Формат лога: timestamp<TAB>query<TAB>qtype (одна запись на строку)")
        sys.exit(1)

    records = parse_dns_log(sys.argv[1])
    print(f"[*] Загружено {len(records)} DNS-записей")

    dga = analyze_dga(records)
    tunnels = analyze_tunneling(records)
    freq = analyze_frequency(records)
    print_report(dga, tunnels, freq)

    report = {"dga": dga, "tunnels": tunnels, "high_freq": freq}
    with open("dns_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[*] JSON-отчёт сохранён: dns_report.json")


if __name__ == "__main__":
    main()
```

### Использование

```bash
python3 dns_analyzer.py /var/log/zeek/dns.log
python3 dns_analyzer.py dns_queries.txt
```

### Пример вывода

```
[*] Загружено 14523 DNS-записей
======================================================================
ОТЧЁТ: АНАЛИЗ DNS-АКТИВНОСТИ
======================================================================

--- DGA-подозрения (энтропия > 3.5) ---
  [4.12] xk7gq2m9p4.evil.com
  [3.89] adf8h3kqpw.botnet.xyz
  Итого: 2 запросов, 2 уникальных баз-доменов

--- DNS-туннелирование (поддомен > 50 символов) ---
  [TUNNEL] длина=64  dGhpcyBpcyBhIGhpZGRlbiBtZXNzYWdl.t.example.com...
  Итого: 1 подозрительных запросов

--- Аномальная частота (> 100 запросов) ---
  [  347] suspicious-cdn.net
  [  204] update-check.xyz

ВСЕГО АНОМАЛИЙ: 5
[!] Рекомендуется проверить выявленные домены в Threat Intelligence.
======================================================================
```

### Интеграция с SOC
- Подключите к Zeek: скрипт читает нативный `dns.log`.
- Выходной `dns_report.json` можно загрузить в ELK через Filebeat.
- Настройте cron для периодического запуска — алерты отправляйте в Slack/Telegram через webhook.

---

## Инструмент 3: Beacon Detector

### Назначение
Обнаружение C2-beaconing (периодических callback-вызовов малвари к серверу управления) по временным
интервалам между соединениями. Малварь обычно «стучит» через равные промежутки (±jitter).
Если интервалы между соединениями к одному IP стабильны — это вероятный beacon.

### Как работает
1. Читает лог соединений (Zeek conn.log или CSV: `timestamp, src_ip, dst_ip, dst_port, duration, bytes`).
2. Группирует соединения по паре `src_ip → dst_ip:port`.
3. Для каждой пары вычисляет интервалы между соединениями.
4. Анализирует стабильность интервалов: среднее, стандартное отклонение, коэффициент вариации (CV).
5. Если CV < порога (по умолчанию 0.3) и количество соединений достаточно — помечает как beacon.
6. Jitter analysis: определяет период и допустимый jitter.

### Код

```python
#!/usr/bin/env python3
"""beacon_detector.py — Обнаружение C2-beaconing по временным паттернам."""

import sys
import json
import statistics
from collections import defaultdict
from datetime import datetime

CV_THRESHOLD = 0.35       # коэффициент вариации (чем ниже — тем точнее beacon)
MIN_CONNECTIONS = 10      # минимум соединений для анализа
MIN_INTERVAL = 5          # минимальный средний интервал (сек) — игнорировать мусор
MAX_INTERVAL = 7200       # максимальный средний интервал (2 часа)


def parse_conn_log(filepath: str) -> list[dict]:
    """Парсинг лога соединений (Zeek conn.log или CSV)."""
    connections = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split(",")
            if len(parts) < 4:
                continue
            try:
                ts = float(parts[0]) if "." in parts[0] else \
                    datetime.fromisoformat(parts[0]).timestamp()
            except (ValueError, OSError):
                continue
            connections.append({
                "ts": ts,
                "src": parts[1].strip(),
                "dst": parts[2].strip(),
                "port": parts[3].strip(),
            })
    return connections


def group_connections(connections: list[dict]) -> dict:
    """Группировка соединений по паре src → dst:port."""
    groups = defaultdict(list)
    for conn in connections:
        key = f"{conn['src']} → {conn['dst']}:{conn['port']}"
        groups[key].append(conn["ts"])
    # Сортировка timestamps
    for key in groups:
        groups[key].sort()
    return groups


def analyze_beacon(timestamps: list[float]) -> dict | None:
    """Анализ временных интервалов: поиск периодичности."""
    if len(timestamps) < MIN_CONNECTIONS:
        return None

    intervals = [
        timestamps[i + 1] - timestamps[i]
        for i in range(len(timestamps) - 1)
    ]
    # Фильтрация нулевых интервалов
    intervals = [iv for iv in intervals if iv > 0]
    if len(intervals) < MIN_CONNECTIONS - 1:
        return None

    mean_iv = statistics.mean(intervals)
    if mean_iv < MIN_INTERVAL or mean_iv > MAX_INTERVAL:
        return None

    stdev_iv = statistics.stdev(intervals) if len(intervals) > 1 else 0
    cv = stdev_iv / mean_iv if mean_iv > 0 else 999

    if cv > CV_THRESHOLD:
        return None

    return {
        "count": len(timestamps),
        "mean_interval_sec": round(mean_iv, 1),
        "stdev_sec": round(stdev_iv, 1),
        "cv": round(cv, 4),
        "jitter_pct": round(cv * 100, 1),
        "period_min": round(mean_iv / 60, 1),
        "first_seen": datetime.utcfromtimestamp(timestamps[0]).isoformat(),
        "last_seen": datetime.utcfromtimestamp(timestamps[-1]).isoformat(),
        "score": round((1 - cv) * 100, 1),  # 100 = идеальный beacon
    }


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 beacon_detector.py <conn_log>")
        print("Формат: timestamp<TAB>src_ip<TAB>dst_ip<TAB>dst_port")
        sys.exit(1)

    connections = parse_conn_log(sys.argv[1])
    print(f"[*] Загружено {len(connections)} соединений")

    groups = group_connections(connections)
    print(f"[*] Уникальных пар src→dst:port: {len(groups)}")

    beacons = []
    for pair, timestamps in groups.items():
        result = analyze_beacon(timestamps)
        if result:
            result["pair"] = pair
            beacons.append(result)

    beacons.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'=' * 70}")
    print(f"ОБНАРУЖЕНО ВЕРОЯТНЫХ BEACON-КАНАЛОВ: {len(beacons)}")
    print(f"{'=' * 70}")

    for b in beacons:
        print(f"\n  [SCORE: {b['score']}%] {b['pair']}")
        print(f"    Соединений: {b['count']}")
        print(f"    Период: ~{b['mean_interval_sec']} сек ({b['period_min']} мин)")
        print(f"    Jitter: {b['jitter_pct']}%")
        print(f"    Первое: {b['first_seen']}  Последнее: {b['last_seen']}")

    if not beacons:
        print("\n  [✓] Beacon-активность не обнаружена.")

    with open("beacon_report.json", "w") as f:
        json.dump(beacons, f, indent=2)
    print(f"\n[*] Отчёт: beacon_report.json")


if __name__ == "__main__":
    main()
```

### Использование

```bash
python3 beacon_detector.py /var/log/zeek/conn.log
python3 beacon_detector.py connections.csv
```

### Пример вывода

```
[*] Загружено 48291 соединений
[*] Уникальных пар src→dst:port: 3241

======================================================================
ОБНАРУЖЕНО ВЕРОЯТНЫХ BEACON-КАНАЛОВ: 2
======================================================================

  [SCORE: 92.3%] 10.0.1.55 → 185.234.72.11:443
    Соединений: 147
    Период: ~300.2 сек (5.0 мин)
    Jitter: 7.7%
    Первое: 2024-01-15T08:12:33  Последнее: 2024-01-15T20:28:15

  [SCORE: 78.1%] 10.0.1.102 → 91.215.44.6:8080
    Соединений: 53
    Период: ~900.5 сек (15.0 мин)
    Jitter: 21.9%
    Первое: 2024-01-15T09:00:01  Последнее: 2024-01-15T21:17:44
```

### Интеграция с SOC
- Запускайте ежечасно на свежем `conn.log` от Zeek.
- Score > 80% — почти наверняка beacon, требует немедленного расследования.
- Выходные данные легко загрузить в ELK / Wazuh для корреляции с другими алертами.

---

## Инструмент 4: PCAP Traffic Analyzer

### Назначение
Парсинг PCAP-файлов: статистика по протоколам, top talkers (IP с наибольшим трафиком), извлечение HTTP-URL, обнаружение аномалий (большие передачи, нестандартные порты, подозрительные User-Agent).

### Как работает
1. Читает PCAP через `dpkt` (или `scapy` как fallback).
2. Для каждого пакета извлекает: IP-адреса, порты, протокол, размер, HTTP-данные.
3. Агрегирует статистику: top talkers, протоколы, порты.
4. Извлекает HTTP-URL и User-Agent заголовки.
5. Обнаруживает аномалии: большие передачи на нестандартные порты.

### Код

```python
#!/usr/bin/env python3
"""pcap_analyzer.py — Анализ PCAP: протоколы, top talkers, HTTP-URL, аномалии."""

import sys
import struct
import json
from collections import Counter, defaultdict
from datetime import datetime

try:
    import dpkt
    HAS_DPKT = True
except ImportError:
    HAS_DPKT = False
    print("[!] dpkt не установлен: pip install dpkt")

KNOWN_PORTS = {
    22: "SSH", 53: "DNS", 80: "HTTP", 443: "HTTPS",
    445: "SMB", 3389: "RDP", 8080: "HTTP-ALT", 8443: "HTTPS-ALT",
    25: "SMTP", 110: "POP3", 143: "IMAP", 993: "IMAPS"
}
LARGE_TRANSFER_THRESHOLD = 10_000_000  # 10 MB


def analyze_pcap(filepath: str) -> dict:
    """Полный анализ PCAP-файла."""
    stats = {
        "total_packets": 0,
        "total_bytes": 0,
        "ip_src": Counter(),
        "ip_dst": Counter(),
        "protocols": Counter(),
        "dst_ports": Counter(),
        "http_urls": [],
        "user_agents": Counter(),
        "large_transfers": [],
        "conversations": defaultdict(int),
    }

    with open(filepath, "rb") as f:
        try:
            pcap = dpkt.pcap.Reader(f)
        except Exception as e:
            print(f"[!] Ошибка чтения PCAP: {e}")
            return stats

        for ts, buf in pcap:
            stats["total_packets"] += 1
            stats["total_bytes"] += len(buf)

            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue

            if not isinstance(eth.data, dpkt.ip.IP):
                continue

            ip = eth.data
            src_ip = ".".join(str(b) for b in ip.src) if len(ip.src) == 4 \
                else _inet_ntoa(ip.src)
            dst_ip = ".".join(str(b) for b in ip.dst) if len(ip.dst) == 4 \
                else _inet_ntoa(ip.dst)

            stats["ip_src"][src_ip] += len(buf)
            stats["ip_dst"][dst_ip] += len(buf)

            conv_key = f"{src_ip} → {dst_ip}"
            stats["conversations"][conv_key] += len(buf)

            if isinstance(ip.data, dpkt.tcp.TCP):
                tcp = ip.data
                stats["protocols"]["TCP"] += 1
                stats["dst_ports"][tcp.dport] += 1

                if tcp.dport == 80 or tcp.sport == 80:
                    try:
                        http = dpkt.http.Request(tcp.data)
                        url = f"http://{http.headers.get('host', dst_ip)}{http.uri}"
                        stats["http_urls"].append({
                            "ts": datetime.utcfromtimestamp(ts).isoformat(),
                            "src": src_ip,
                            "url": url,
                            "method": http.method
                        })
                        ua = http.headers.get("user-agent", "")
                        if ua:
                            stats["user_agents"][ua] += 1
                    except Exception:
                        pass

            elif isinstance(ip.data, dpkt.udp.UDP):
                stats["protocols"]["UDP"] += 1
                stats["dst_ports"][ip.data.dport] += 1

            elif isinstance(ip.data, dpkt.icmp.ICMP):
                stats["protocols"]["ICMP"] += 1

    # Обнаружение крупных передач
    for conv, total_bytes in stats["conversations"].items():
        if total_bytes > LARGE_TRANSFER_THRESHOLD:
            stats["large_transfers"].append({
                "conversation": conv,
                "bytes": total_bytes,
                "megabytes": round(total_bytes / 1_000_000, 2)
            })

    return stats


def _inet_ntoa(packed: bytes) -> str:
    """Конвертация packed IP в строку."""
    try:
        import socket
        return socket.inet_ntoa(packed)
    except Exception:
        return ".".join(str(b) for b in packed[:4])


def print_report(stats: dict):
    """Вывод отчёта по PCAP."""
    print("=" * 70)
    print("АНАЛИЗ PCAP-ФАЙЛА")
    print("=" * 70)
    print(f"  Пакетов: {stats['total_packets']:,}")
    print(f"  Байт:    {stats['total_bytes']:,} "
          f"({stats['total_bytes'] / 1_000_000:.1f} MB)")

    print(f"\n--- Протоколы ---")
    for proto, count in stats["protocols"].most_common(10):
        print(f"  {proto:8s} {count:>8,}")

    print(f"\n--- Top-10 источников (по объёму) ---")
    for ip, bts in stats["ip_src"].most_common(10):
        print(f"  {ip:18s} {bts:>12,} байт ({bts / 1_000_000:.1f} MB)")

    print(f"\n--- Top-10 назначений (по объёму) ---")
    for ip, bts in stats["ip_dst"].most_common(10):
        print(f"  {ip:18s} {bts:>12,} байт ({bts / 1_000_000:.1f} MB)")

    print(f"\n--- Top-10 портов ---")
    for port, count in stats["dst_ports"].most_common(10):
        svc = KNOWN_PORTS.get(port, "?")
        print(f"  {port:>6} ({svc:10s}) {count:>8,}")

    if stats["http_urls"]:
        print(f"\n--- HTTP URL (первые 20) ---")
        for entry in stats["http_urls"][:20]:
            print(f"  [{entry['method']}] {entry['url']}")

    if stats["user_agents"]:
        print(f"\n--- User-Agent (топ-5) ---")
        for ua, count in stats["user_agents"].most_common(5):
            print(f"  [{count:>4}] {ua[:80]}")

    if stats["large_transfers"]:
        print(f"\n--- [!] КРУПНЫЕ ПЕРЕДАЧИ (>{LARGE_TRANSFER_THRESHOLD // 1_000_000} MB) ---")
        for lt in stats["large_transfers"]:
            print(f"  {lt['conversation']:40s} {lt['megabytes']} MB")


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 pcap_analyzer.py <file.pcap>")
        sys.exit(1)
    if not HAS_DPKT:
        sys.exit(1)

    stats = analyze_pcap(sys.argv[1])
    print_report(stats)

    report = {
        "total_packets": stats["total_packets"],
        "total_bytes": stats["total_bytes"],
        "top_sources": stats["ip_src"].most_common(20),
        "top_destinations": stats["ip_dst"].most_common(20),
        "protocols": dict(stats["protocols"]),
        "http_urls": stats["http_urls"][:100],
        "large_transfers": stats["large_transfers"],
    }
    with open("pcap_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[*] JSON-отчёт: pcap_report.json")


if __name__ == "__main__":
    main()
```

### Использование

```bash
pip install dpkt
python3 pcap_analyzer.py capture.pcap
```

### Пример вывода

```
======================================================================
АНАЛИЗ PCAP-ФАЙЛА
======================================================================
  Пакетов: 148,291
  Байт:    87,234,102 (87.2 MB)

--- Протоколы ---
  TCP      124,821
  UDP       22,103
  ICMP       1,367

--- Top-10 источников (по объёму) ---
  10.0.1.55          42,123,456 байт (42.1 MB)
  10.0.1.102          8,234,567 байт (8.2 MB)

--- [!] КРУПНЫЕ ПЕРЕДАЧИ (>10 MB) ---
  10.0.1.55 → 185.234.72.11              42.1 MB
```

### Интеграция с SOC
- Запускайте на подозрительных PCAP, захваченных по алерту.
- Крупные передачи на нестандартные порты — возможная эксфильтрация.
- JSON-отчёт легко интегрируется с тикет-системой (Jira, TheHive).

---

## Инструмент 5: SSL/TLS Certificate Checker

### Назначение
Проверка SSL/TLS-сертификатов сервисов на: истечение срока, self-signed, слабые алгоритмы,
несовпадение имён. SOC-аналитик использует для аудита инфраструктуры и обнаружения подозрительных
сертификатов (C2-серверы часто используют самоподписанные сертификаты).

### Код

```python
#!/usr/bin/env python3
"""ssl_checker.py — Проверка SSL/TLS-сертификатов: expiry, self-signed, weak ciphers."""

import ssl
import socket
import sys
import json
from datetime import datetime, timezone

WEAK_SIGNATURES = ["sha1", "md5", "md2"]
WEAK_KEY_SIZES = {"RSA": 1024, "DSA": 1024, "EC": 128}


def check_certificate(host: str, port: int = 443, timeout: float = 5.0) -> dict:
    """Проверка сертификата хоста."""
    result = {
        "host": host, "port": port, "status": "OK", "issues": [],
        "subject": "", "issuer": "", "not_before": "", "not_after": "",
        "days_until_expiry": 0, "serial": "", "version": 0,
        "signature_algorithm": "", "self_signed": False,
    }

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert_bin = ssock.getpeercert(binary_form=True)
                cert = ssock.getpeercert()

                if not cert:
                    # Декодируем бинарный сертификат
                    result["status"] = "WARNING"
                    result["issues"].append("Не удалось получить расшифрованный сертификат")
                    return result

                # Subject и Issuer
                subject_parts = []
                for rdn in cert.get("subject", ()):
                    for attr_name, attr_val in rdn:
                        subject_parts.append(f"{attr_name}={attr_val}")
                result["subject"] = ", ".join(subject_parts)

                issuer_parts = []
                for rdn in cert.get("issuer", ()):
                    for attr_name, attr_val in rdn:
                        issuer_parts.append(f"{attr_name}={attr_val}")
                result["issuer"] = ", ".join(issuer_parts)

                # Self-signed
                if result["subject"] == result["issuer"]:
                    result["self_signed"] = True
                    result["issues"].append("SELF-SIGNED: сертификат самоподписанный")

                # Срок действия
                not_after_str = cert.get("notAfter", "")
                not_before_str = cert.get("notBefore", "")
                result["not_before"] = not_before_str
                result["not_after"] = not_after_str

                if not_after_str:
                    expiry = datetime.strptime(not_after_str, "%b %d %H:%M:%S %Y %Z")
                    now = datetime.utcnow()
                    delta = expiry - now
                    result["days_until_expiry"] = delta.days

                    if delta.days < 0:
                        result["issues"].append(
                            f"EXPIRED: сертификат истёк {abs(delta.days)} дней назад")
                    elif delta.days < 30:
                        result["issues"].append(
                            f"EXPIRING SOON: до истечения {delta.days} дней")

                # Serial
                result["serial"] = cert.get("serialNumber", "")
                result["version"] = cert.get("version", 0)

                # SAN (Subject Alternative Names)
                san = cert.get("subjectAltName", ())
                san_list = [v for _, v in san]
                if host not in san_list and f"*.{'.'.join(host.split('.')[1:])}" not in san_list:
                    result["issues"].append(
                        f"NAME MISMATCH: {host} не в SAN: {san_list[:5]}")

    except ssl.SSLError as e:
        result["status"] = "ERROR"
        result["issues"].append(f"SSL Error: {e}")
    except (socket.timeout, ConnectionRefusedError, OSError) as e:
        result["status"] = "UNREACHABLE"
        result["issues"].append(f"Недоступен: {e}")

    if result["issues"]:
        result["status"] = "WARNING" if result["status"] == "OK" else result["status"]

    return result


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 ssl_checker.py <host1> [host2] ...")
        print("       или:    python3 ssl_checker.py -f hosts.txt")
        sys.exit(1)

    hosts = []
    if sys.argv[1] == "-f":
        with open(sys.argv[2]) as f:
            hosts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        hosts = sys.argv[1:]

    results = []
    print(f"[*] Проверка {len(hosts)} хостов...\n")

    for host in hosts:
        port = 443
        if ":" in host:
            host, port = host.rsplit(":", 1)
            port = int(port)

        result = check_certificate(host, port)
        results.append(result)

        icon = "✓" if result["status"] == "OK" else "!" if result["status"] == "WARNING" else "✗"
        print(f"  [{icon}] {host}:{port}")
        print(f"      Subject: {result['subject'][:60]}")
        print(f"      Issuer:  {result['issuer'][:60]}")
        print(f"      Expiry:  {result['not_after']} ({result['days_until_expiry']} дней)")
        if result["self_signed"]:
            print(f"      [!] САМОПОДПИСАННЫЙ")
        for issue in result["issues"]:
            print(f"      [!] {issue}")
        print()

    # Сводка
    ok = sum(1 for r in results if r["status"] == "OK")
    warn = sum(1 for r in results if r["status"] == "WARNING")
    err = sum(1 for r in results if r["status"] in ("ERROR", "UNREACHABLE"))
    print(f"ИТОГО: {ok} OK / {warn} предупреждений / {err} ошибок")

    with open("ssl_report.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[*] Отчёт: ssl_report.json")


if __name__ == "__main__":
    main()
```

### Использование

```bash
python3 ssl_checker.py google.com github.com example.com:8443
python3 ssl_checker.py -f servers.txt
```

### Интеграция с SOC
- Запускайте еженедельно для аудита всех публичных сервисов.
- Self-signed сертификаты на неожиданных хостах — признак C2.
- Алерт «сертификат истекает через 7 дней» → предотвращение инцидента.

---

## Инструмент 6: Port Baseline Monitor

### Назначение
Мониторинг открытых портов на серверах. Сравнение текущего состояния с baseline — обнаружение
новых открытых портов (бэкдор, нежелательный сервис) или закрытых (DoS, сбой).

### Код

```python
#!/usr/bin/env python3
"""port_monitor.py — Мониторинг портов: baseline comparison, обнаружение изменений."""

import socket
import sys
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

BASELINE_FILE = "port_baseline.json"
COMMON_PORTS = list(range(1, 1025)) + [
    3306, 3389, 5432, 5900, 6379, 8080, 8443, 8888, 9090, 9200, 27017
]


def scan_ports(host: str, ports: list[int], timeout: float = 0.5,
               workers: int = 200) -> list[int]:
    """Быстрое сканирование портов хоста."""
    open_ports = []

    def check(port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                if s.connect_ex((host, port)) == 0:
                    return port
        except OSError:
            pass
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check, p): p for p in ports}
        for f in as_completed(futures):
            result = f.result()
            if result is not None:
                open_ports.append(result)

    return sorted(open_ports)


def load_baseline(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_baseline(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 port_monitor.py <host1> [host2] ...")
        sys.exit(1)

    hosts = sys.argv[1:]
    baseline = load_baseline(BASELINE_FILE)
    current = {}
    alerts = []

    for host in hosts:
        print(f"[*] Сканирование {host}...")
        open_ports = scan_ports(host, COMMON_PORTS)
        current[host] = {
            "ports": open_ports,
            "timestamp": datetime.utcnow().isoformat()
        }
        print(f"    Открытых портов: {len(open_ports)}")

        if host in baseline:
            old_ports = set(baseline[host]["ports"])
            new_ports = set(open_ports)
            added = new_ports - old_ports
            removed = old_ports - new_ports

            if added:
                msg = f"[!] {host}: НОВЫЕ порты: {sorted(added)}"
                print(f"    {msg}")
                alerts.append({"host": host, "event": "new_ports",
                               "ports": sorted(added)})
            if removed:
                msg = f"[!] {host}: ЗАКРЫТЫЕ порты: {sorted(removed)}"
                print(f"    {msg}")
                alerts.append({"host": host, "event": "closed_ports",
                               "ports": sorted(removed)})
            if not added and not removed:
                print(f"    [✓] Без изменений")
        else:
            print(f"    [*] Первый скан — baseline создан")

    save_baseline(BASELINE_FILE, current)

    if alerts:
        print(f"\n{'=' * 50}")
        print(f"АЛЕРТОВ: {len(alerts)}")
        for a in alerts:
            print(f"  [{a['event']}] {a['host']}: {a['ports']}")

        with open("port_alerts.json", "w") as f:
            json.dump(alerts, f, indent=2)
        print(f"[*] Алерты сохранены: port_alerts.json")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Запускайте каждые 15 минут по cron на критичных серверах.
- Новый порт 4444 на сервере → вероятный reverse shell → немедленное расследование.
- JSON-алерты отправляйте через webhook в Slack/Telegram/Wazuh.

---

## ЧАСТЬ 2: АНАЛИЗ ЛОГОВ

---

## Инструмент 7: Windows Event Log Parser

### Назначение
Парсинг .evtx файлов Windows Event Log: фильтрация по EventID, временному диапазону, экспорт
в CSV/JSON. Незаменим при расследовании инцидентов на Windows-системах.

### Код

```python
#!/usr/bin/env python3
"""evtx_parser.py — Парсинг Windows Event Log (.evtx): фильтрация, экспорт."""

import sys
import json
import csv
import argparse
from datetime import datetime
from xml.etree import ElementTree as ET

try:
    import Evtx.Evtx as evtx
    HAS_EVTX = True
except ImportError:
    HAS_EVTX = False
    print("[!] python-evtx не установлен: pip install python-evtx")

SECURITY_EVENTS = {
    4624: "Успешный вход",
    4625: "Неудачный вход",
    4634: "Выход из системы",
    4648: "Вход с явными учётными данными",
    4672: "Назначение привилегий",
    4688: "Создание процесса",
    4689: "Завершение процесса",
    4720: "Создание учётной записи",
    4722: "Включение учётной записи",
    4724: "Сброс пароля",
    4728: "Добавление в группу безопасности",
    4732: "Добавление в локальную группу",
    4768: "Запрос TGT (Kerberos)",
    4769: "Запрос TGS (Kerberos)",
    4771: "Предварительная аутентификация Kerberos провалена",
    4776: "Проверка учётных данных",
    1102: "Очистка журнала аудита",
}

NS = {"ns": "http://schemas.microsoft.com/win/2004/08/events/event"}


def parse_evtx(filepath: str, event_ids: list[int] = None,
               start_time: str = None, end_time: str = None) -> list[dict]:
    """Парсинг EVTX-файла с фильтрацией."""
    records = []

    start_dt = datetime.fromisoformat(start_time) if start_time else None
    end_dt = datetime.fromisoformat(end_time) if end_time else None

    with evtx.Evtx(filepath) as log:
        for record in log.records():
            try:
                xml = record.xml()
                root = ET.fromstring(xml)

                system = root.find("ns:System", NS)
                if system is None:
                    continue

                eid_el = system.find("ns:EventID", NS)
                eid = int(eid_el.text) if eid_el is not None else 0

                if event_ids and eid not in event_ids:
                    continue

                time_el = system.find("ns:TimeCreated", NS)
                timestamp = time_el.get("SystemTime", "") if time_el is not None else ""

                if start_dt or end_dt:
                    try:
                        ts = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00").split(".")[0])
                        ts = ts.replace(tzinfo=None)
                        if start_dt and ts < start_dt:
                            continue
                        if end_dt and ts > end_dt:
                            continue
                    except (ValueError, AttributeError):
                        pass

                computer_el = system.find("ns:Computer", NS)
                computer = computer_el.text if computer_el is not None else ""

                # EventData
                event_data = {}
                ed = root.find("ns:EventData", NS)
                if ed is not None:
                    for data_el in ed.findall("ns:Data", NS):
                        name = data_el.get("Name", "")
                        value = data_el.text or ""
                        if name:
                            event_data[name] = value

                records.append({
                    "EventID": eid,
                    "Description": SECURITY_EVENTS.get(eid, ""),
                    "Timestamp": timestamp,
                    "Computer": computer,
                    "Data": event_data,
                })

            except Exception:
                continue

    return records


def export_csv(records: list[dict], filepath: str):
    """Экспорт в CSV."""
    if not records:
        return
    fieldnames = ["EventID", "Description", "Timestamp", "Computer"]
    # Собираем все ключи из Data
    data_keys = set()
    for r in records:
        data_keys.update(r.get("Data", {}).keys())
    fieldnames.extend(sorted(data_keys))

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = {k: r[k] for k in ["EventID", "Description", "Timestamp", "Computer"]}
            row.update(r.get("Data", {}))
            writer.writerow(row)
    print(f"[*] CSV сохранён: {filepath} ({len(records)} записей)")


def export_json(records: list[dict], filepath: str):
    """Экспорт в JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[*] JSON сохранён: {filepath} ({len(records)} записей)")


def main():
    parser = argparse.ArgumentParser(description="Windows EVTX Parser")
    parser.add_argument("file", help="Путь к .evtx файлу")
    parser.add_argument("-e", "--events", nargs="+", type=int,
                        help="Фильтр по EventID (например: 4624 4625)")
    parser.add_argument("--start", help="Начало периода (ISO: 2024-01-15T00:00:00)")
    parser.add_argument("--end", help="Конец периода (ISO: 2024-01-16T00:00:00)")
    parser.add_argument("-o", "--output", default="evtx_export",
                        help="Имя выходного файла (без расширения)")
    parser.add_argument("-f", "--format", choices=["csv", "json", "both"],
                        default="both", help="Формат экспорта")
    args = parser.parse_args()

    if not HAS_EVTX:
        sys.exit(1)

    print(f"[*] Парсинг {args.file}...")
    records = parse_evtx(args.file, args.events, args.start, args.end)
    print(f"[*] Найдено записей: {len(records)}")

    # Краткая статистика
    eid_counts = {}
    for r in records:
        eid = r["EventID"]
        eid_counts[eid] = eid_counts.get(eid, 0) + 1

    print("\n--- Статистика по EventID ---")
    for eid, count in sorted(eid_counts.items(), key=lambda x: -x[1]):
        desc = SECURITY_EVENTS.get(eid, "")
        print(f"  {eid:>5}  {count:>6}  {desc}")

    if args.format in ("csv", "both"):
        export_csv(records, f"{args.output}.csv")
    if args.format in ("json", "both"):
        export_json(records, f"{args.output}.json")


if __name__ == "__main__":
    main()
```

### Использование

```bash
pip install python-evtx
python3 evtx_parser.py Security.evtx -e 4624 4625 4648
python3 evtx_parser.py Security.evtx --start 2024-01-15T00:00:00 --end 2024-01-16T00:00:00
python3 evtx_parser.py Security.evtx -f csv -o security_export
```

### Интеграция с SOC
- Используйте при расследовании: скопируйте .evtx с заражённой машины, парсите локально.
- EventID 4625 (массовые) → brute force. EventID 1102 → кто-то очистил логи.
- CSV-экспорт открывается в Excel/LibreOffice для ручного анализа.

---

## Инструмент 8: Sysmon Process Tree Analyzer

### Назначение
Парсинг Sysmon-логов (XML/EVTX) и построение дерева процессов (parent→child).
Обнаружение подозрительных цепочек: `winword.exe → cmd.exe → powershell.exe` — признак эксплуатации
через документ.

### Код

```python
#!/usr/bin/env python3
"""sysmon_analyzer.py — Построение дерева процессов из Sysmon, обнаружение аномалий."""

import sys
import json
from collections import defaultdict
from xml.etree import ElementTree as ET

# Подозрительные parent→child пары
SUSPICIOUS_CHAINS = [
    ("winword.exe", "cmd.exe"),
    ("winword.exe", "powershell.exe"),
    ("winword.exe", "wscript.exe"),
    ("excel.exe", "cmd.exe"),
    ("excel.exe", "powershell.exe"),
    ("outlook.exe", "powershell.exe"),
    ("explorer.exe", "mshta.exe"),
    ("svchost.exe", "cmd.exe"),
    ("wmiprvse.exe", "powershell.exe"),
    ("services.exe", "cmd.exe"),
    ("lsass.exe", "cmd.exe"),
    ("spoolsv.exe", "cmd.exe"),
    ("w3wp.exe", "cmd.exe"),
    ("w3wp.exe", "powershell.exe"),
    ("httpd.exe", "cmd.exe"),
    ("nginx.exe", "cmd.exe"),
    ("rundll32.exe", "cmd.exe"),
    ("regsvr32.exe", "cmd.exe"),
    ("mshta.exe", "powershell.exe"),
    ("certutil.exe", "cmd.exe"),
]

# Подозрительные командные строки (паттерны)
SUSPICIOUS_CMDLINE = [
    "-enc ", "-encodedcommand ", "downloadstring", "invoke-expression",
    "iex(", "bypass", "-nop ", "-w hidden", "certutil -urlcache",
    "bitsadmin /transfer", "regsvr32 /s /n /u",
    "mshta vbscript:", "mshta javascript:",
]


def parse_sysmon_xml(filepath: str) -> list[dict]:
    """Парсинг Sysmon XML-лога (Event ID 1 — Process Create)."""
    processes = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        for event in root.iter("Event"):
            event_data = {}
            for data in event.iter("Data"):
                name = data.get("Name", "")
                event_data[name] = data.text or ""

            if event_data.get("EventID") == "1" or "Image" in event_data:
                processes.append({
                    "pid": event_data.get("ProcessId", ""),
                    "ppid": event_data.get("ParentProcessId", ""),
                    "image": event_data.get("Image", "").lower(),
                    "parent_image": event_data.get("ParentImage", "").lower(),
                    "cmdline": event_data.get("CommandLine", ""),
                    "user": event_data.get("User", ""),
                    "timestamp": event_data.get("UtcTime", ""),
                    "hash": event_data.get("Hashes", ""),
                })
    except ET.ParseError:
        # Попробуем построчный парсинг (каждый Event — отдельный XML)
        with open(filepath, "r") as f:
            content = f.read()
        for block in content.split("</Event>"):
            block = block.strip()
            if not block:
                continue
            block += "</Event>"
            try:
                root = ET.fromstring(block)
                event_data = {}
                for data in root.iter("Data"):
                    name = data.get("Name", "")
                    event_data[name] = data.text or ""
                if "Image" in event_data:
                    processes.append({
                        "pid": event_data.get("ProcessId", ""),
                        "ppid": event_data.get("ParentProcessId", ""),
                        "image": event_data.get("Image", "").lower(),
                        "parent_image": event_data.get("ParentImage", "").lower(),
                        "cmdline": event_data.get("CommandLine", ""),
                        "user": event_data.get("User", ""),
                        "timestamp": event_data.get("UtcTime", ""),
                        "hash": event_data.get("Hashes", ""),
                    })
            except ET.ParseError:
                continue
    return processes


def parse_jsonl(filepath: str) -> list[dict]:
    """Парсинг Sysmon-логов в формате JSON Lines."""
    processes = []
    with open(filepath, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get("EventID") == 1 or "Image" in data:
                    processes.append({
                        "pid": str(data.get("ProcessId", "")),
                        "ppid": str(data.get("ParentProcessId", "")),
                        "image": data.get("Image", "").lower(),
                        "parent_image": data.get("ParentImage", "").lower(),
                        "cmdline": data.get("CommandLine", ""),
                        "user": data.get("User", ""),
                        "timestamp": data.get("UtcTime", ""),
                        "hash": data.get("Hashes", ""),
                    })
            except json.JSONDecodeError:
                continue
    return processes


def build_tree(processes: list[dict]) -> dict:
    """Построение дерева процессов по PID→PPID."""
    children = defaultdict(list)
    by_pid = {}
    for proc in processes:
        by_pid[proc["pid"]] = proc
        children[proc["ppid"]].append(proc["pid"])
    return children, by_pid


def detect_anomalies(processes: list[dict]) -> list[dict]:
    """Обнаружение подозрительных parent→child и командных строк."""
    alerts = []

    for proc in processes:
        parent_name = proc["parent_image"].split("\\")[-1]
        child_name = proc["image"].split("\\")[-1]

        # Проверка parent→child
        for sus_parent, sus_child in SUSPICIOUS_CHAINS:
            if parent_name == sus_parent and child_name == sus_child:
                alerts.append({
                    "type": "SUSPICIOUS_PARENT_CHILD",
                    "severity": "HIGH",
                    "parent": proc["parent_image"],
                    "child": proc["image"],
                    "cmdline": proc["cmdline"][:200],
                    "user": proc["user"],
                    "timestamp": proc["timestamp"],
                    "pid": proc["pid"],
                })

        # Проверка командной строки
        cmdline_lower = proc["cmdline"].lower()
        for pattern in SUSPICIOUS_CMDLINE:
            if pattern in cmdline_lower:
                alerts.append({
                    "type": "SUSPICIOUS_CMDLINE",
                    "severity": "HIGH",
                    "pattern": pattern,
                    "image": proc["image"],
                    "cmdline": proc["cmdline"][:300],
                    "user": proc["user"],
                    "timestamp": proc["timestamp"],
                    "pid": proc["pid"],
                })
                break  # Одного паттерна достаточно

    return alerts


def print_tree(children: dict, by_pid: dict, pid: str, depth: int = 0,
               max_depth: int = 5):
    """Рекурсивный вывод дерева процессов."""
    if depth > max_depth or pid not in by_pid:
        return
    proc = by_pid[pid]
    indent = "  " * depth + ("└─ " if depth > 0 else "")
    name = proc["image"].split("\\")[-1]
    print(f"{indent}[{proc['pid']}] {name}  ({proc['user']})")
    for child_pid in children.get(pid, []):
        print_tree(children, by_pid, child_pid, depth + 1, max_depth)


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 sysmon_analyzer.py <sysmon_log>")
        print("Поддерживаемые форматы: XML, JSONL")
        sys.exit(1)

    filepath = sys.argv[1]

    if filepath.endswith(".json") or filepath.endswith(".jsonl"):
        processes = parse_jsonl(filepath)
    else:
        processes = parse_sysmon_xml(filepath)

    print(f"[*] Загружено процессов: {len(processes)}")

    # Обнаружение аномалий
    alerts = detect_anomalies(processes)
    print(f"\n{'=' * 60}")
    print(f"АНОМАЛИЙ ОБНАРУЖЕНО: {len(alerts)}")
    print(f"{'=' * 60}")

    for alert in alerts:
        print(f"\n  [{alert['severity']}] {alert['type']}")
        if alert['type'] == "SUSPICIOUS_PARENT_CHILD":
            print(f"    Parent: {alert['parent']}")
            print(f"    Child:  {alert['child']}")
        else:
            print(f"    Image:   {alert['image']}")
            print(f"    Pattern: {alert['pattern']}")
        print(f"    CmdLine: {alert['cmdline'][:100]}")
        print(f"    User:    {alert['user']}")
        print(f"    Time:    {alert['timestamp']}")

    # Сохранение
    if alerts:
        with open("sysmon_alerts.json", "w") as f:
            json.dump(alerts, f, indent=2, ensure_ascii=False)
        print(f"\n[*] Алерты: sysmon_alerts.json")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Экспортируйте Sysmon в JSON через Wazuh/Winlogbeat, анализируйте централизованно.
- `winword.exe → cmd.exe → powershell.exe` → немедленная изоляция хоста.
- Дерево процессов — ключевой артефакт для IR-отчёта.

---

## Инструмент 9: Auth Log Analyzer (Brute Force Detection)

### Назначение
Анализ `/var/log/auth.log` (Linux) или `/var/log/secure`: обнаружение brute force атак,
password spraying, аномальных входов по времени и геолокации.

### Код

```python
#!/usr/bin/env python3
"""auth_analyzer.py — Анализ auth.log: brute force, password spraying, аномалии."""

import re
import sys
import json
from collections import Counter, defaultdict
from datetime import datetime

BRUTE_FORCE_THRESHOLD = 5   # неудачных попыток за окно
SPRAY_THRESHOLD = 3          # разных пользователей с одного IP
TIME_WINDOW_MIN = 10         # временное окно (минуты)

# Регулярные выражения для auth.log
RE_FAILED = re.compile(
    r"(\w+\s+\d+\s+\d+:\d+:\d+)\s+\S+\s+sshd\[\d+\]:\s+"
    r"Failed password for (?:invalid user )?(\S+)\s+from\s+(\S+)"
)
RE_SUCCESS = re.compile(
    r"(\w+\s+\d+\s+\d+:\d+:\d+)\s+\S+\s+sshd\[\d+\]:\s+"
    r"Accepted (?:password|publickey) for (\S+)\s+from\s+(\S+)"
)
RE_INVALID_USER = re.compile(
    r"(\w+\s+\d+\s+\d+:\d+:\d+)\s+\S+\s+sshd\[\d+\]:\s+"
    r"Invalid user (\S+)\s+from\s+(\S+)"
)


def parse_auth_log(filepath: str) -> tuple:
    """Парсинг auth.log: неудачные входы, успешные входы, невалидные пользователи."""
    failed = []
    success = []
    invalid_users = []

    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            m = RE_FAILED.search(line)
            if m:
                failed.append({
                    "timestamp": m.group(1),
                    "user": m.group(2),
                    "ip": m.group(3)
                })
                continue

            m = RE_SUCCESS.search(line)
            if m:
                success.append({
                    "timestamp": m.group(1),
                    "user": m.group(2),
                    "ip": m.group(3)
                })
                continue

            m = RE_INVALID_USER.search(line)
            if m:
                invalid_users.append({
                    "timestamp": m.group(1),
                    "user": m.group(2),
                    "ip": m.group(3)
                })

    return failed, success, invalid_users


def detect_brute_force(failed: list[dict]) -> list[dict]:
    """Обнаружение brute force: много неудачных попыток с одного IP к одному user."""
    ip_user_counts = defaultdict(int)
    for entry in failed:
        key = f"{entry['ip']}→{entry['user']}"
        ip_user_counts[key] += 1

    alerts = []
    for key, count in ip_user_counts.items():
        if count >= BRUTE_FORCE_THRESHOLD:
            ip, user = key.split("→")
            alerts.append({
                "type": "BRUTE_FORCE",
                "ip": ip,
                "target_user": user,
                "attempts": count,
                "severity": "HIGH" if count > 20 else "MEDIUM"
            })
    return sorted(alerts, key=lambda x: -x["attempts"])


def detect_spray(failed: list[dict]) -> list[dict]:
    """Обнаружение password spraying: один IP пробует много разных пользователей."""
    ip_users = defaultdict(set)
    for entry in failed:
        ip_users[entry["ip"]].add(entry["user"])

    alerts = []
    for ip, users in ip_users.items():
        if len(users) >= SPRAY_THRESHOLD:
            alerts.append({
                "type": "PASSWORD_SPRAY",
                "ip": ip,
                "unique_users": len(users),
                "users": sorted(users)[:20],
                "severity": "HIGH" if len(users) > 10 else "MEDIUM"
            })
    return sorted(alerts, key=lambda x: -x["unique_users"])


def detect_success_after_failure(failed: list, success: list) -> list[dict]:
    """Обнаружение успешного входа после серии неудачных (скомпрометирован пароль)."""
    failed_ips = set()
    for entry in failed:
        failed_ips.add(entry["ip"])

    alerts = []
    for entry in success:
        if entry["ip"] in failed_ips:
            alerts.append({
                "type": "SUCCESS_AFTER_BRUTE",
                "ip": entry["ip"],
                "user": entry["user"],
                "timestamp": entry["timestamp"],
                "severity": "CRITICAL"
            })
    return alerts


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 auth_analyzer.py /var/log/auth.log")
        sys.exit(1)

    failed, success, invalid_users = parse_auth_log(sys.argv[1])
    print(f"[*] Неудачных входов: {len(failed)}")
    print(f"[*] Успешных входов:  {len(success)}")
    print(f"[*] Невалидных user:  {len(invalid_users)}")

    brute = detect_brute_force(failed)
    spray = detect_spray(failed)
    compromised = detect_success_after_failure(failed, success)

    print(f"\n{'=' * 60}")

    if compromised:
        print(f"\n[!!!] ВОЗМОЖНАЯ КОМПРОМЕТАЦИЯ ({len(compromised)}):")
        for a in compromised:
            print(f"  {a['ip']} → {a['user']} в {a['timestamp']}")

    if brute:
        print(f"\n[!] BRUTE FORCE ({len(brute)}):")
        for a in brute[:10]:
            print(f"  {a['ip']} → {a['target_user']}  ({a['attempts']} попыток)")

    if spray:
        print(f"\n[!] PASSWORD SPRAY ({len(spray)}):")
        for a in spray[:10]:
            print(f"  {a['ip']} → {a['unique_users']} пользователей: "
                  f"{', '.join(a['users'][:5])}")

    # Топ IP по неудачным
    top_ips = Counter(e["ip"] for e in failed).most_common(10)
    print(f"\n--- Топ-10 IP по неудачным входам ---")
    for ip, count in top_ips:
        print(f"  {ip:18s} {count:>6}")

    all_alerts = brute + spray + compromised
    with open("auth_alerts.json", "w") as f:
        json.dump(all_alerts, f, indent=2, ensure_ascii=False)
    print(f"\n[*] Алерты: auth_alerts.json ({len(all_alerts)} шт.)")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Запускайте ежечасно; `SUCCESS_AFTER_BRUTE` = критический алерт (пароль подобран!).
- Выходной JSON отправляйте в SIEM через syslog / filebeat.
- Автоматическая блокировка IP через fail2ban или Wazuh active response.

---

## Инструмент 10: Web Log Analyzer (Attack Detection)

### Назначение
Анализ access.log (Apache/Nginx) на предмет веб-атак: SQL injection, XSS, path traversal,
сканеры уязвимостей, подозрительные User-Agent, аномальные коды ответов.

### Код

```python
#!/usr/bin/env python3
"""web_log_analyzer.py — Обнаружение веб-атак в access.log."""

import re
import sys
import json
from collections import Counter
from urllib.parse import unquote

# Паттерны атак (в URL / query string)
ATTACK_PATTERNS = {
    "SQL_INJECTION": [
        r"(?:union\s+select|select\s+.*from|insert\s+into|update\s+.*set|delete\s+from)",
        r"(?:or\s+1\s*=\s*1|and\s+1\s*=\s*1|'\s+or\s+'|'\s+and\s+')",
        r"(?:;\s*drop\s+|;\s*alter\s+|;\s*create\s+)",
        r"(?:benchmark\s*\(|sleep\s*\(|waitfor\s+delay)",
    ],
    "XSS": [
        r"(?:<script|javascript:|on\w+\s*=|<img\s+.*onerror)",
        r"(?:alert\s*\(|confirm\s*\(|prompt\s*\()",
        r"(?:document\.cookie|document\.write|\.innerHTML)",
    ],
    "PATH_TRAVERSAL": [
        r"(?:\.\./|\.\.\\|%2e%2e/|%2e%2e\\)",
        r"(?:/etc/passwd|/etc/shadow|/proc/self|c:\\windows)",
    ],
    "COMMAND_INJECTION": [
        r"(?:;\s*(?:ls|cat|id|whoami|uname|wget|curl|nc|bash|sh)\b)",
        r"(?:\|\s*(?:ls|cat|id|whoami|uname))",
        r"(?:`[^`]+`|\$\([^)]+\))",
    ],
    "SCANNER": [
        r"(?:nikto|sqlmap|nmap|masscan|nuclei|gobuster|dirbuster|wpscan)",
        r"(?:acunetix|nessus|openvas|burp|zap)",
    ],
}

# Apache/Nginx combined log format
LOG_RE = re.compile(
    r'(\S+)\s+\S+\s+(\S+)\s+\[([^\]]+)\]\s+"(\S+)\s+(\S+)\s+(\S+)"\s+(\d+)\s+(\d+)\s+"([^"]*)"\s+"([^"]*)"'
)


def parse_access_log(filepath: str) -> list[dict]:
    """Парсинг access.log (combined format)."""
    entries = []
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            m = LOG_RE.match(line)
            if m:
                entries.append({
                    "ip": m.group(1),
                    "user": m.group(2),
                    "timestamp": m.group(3),
                    "method": m.group(4),
                    "url": m.group(5),
                    "protocol": m.group(6),
                    "status": int(m.group(7)),
                    "size": int(m.group(8)) if m.group(8) != "-" else 0,
                    "referer": m.group(9),
                    "user_agent": m.group(10),
                })
    return entries


def detect_attacks(entries: list[dict]) -> list[dict]:
    """Обнаружение веб-атак по паттернам."""
    alerts = []
    compiled = {}
    for attack_type, patterns in ATTACK_PATTERNS.items():
        compiled[attack_type] = [re.compile(p, re.IGNORECASE) for p in patterns]

    for entry in entries:
        url_decoded = unquote(entry["url"]).lower()
        ua_lower = entry["user_agent"].lower()
        check_string = url_decoded + " " + ua_lower

        for attack_type, regexes in compiled.items():
            for regex in regexes:
                if regex.search(check_string):
                    alerts.append({
                        "type": attack_type,
                        "ip": entry["ip"],
                        "url": entry["url"][:200],
                        "method": entry["method"],
                        "status": entry["status"],
                        "user_agent": entry["user_agent"][:100],
                        "timestamp": entry["timestamp"],
                    })
                    break  # Одного совпадения достаточно для этого типа

    return alerts


def print_report(entries: list, alerts: list):
    """Вывод отчёта."""
    print(f"{'=' * 60}")
    print(f"АНАЛИЗ WEB-ЛОГОВ")
    print(f"{'=' * 60}")
    print(f"  Записей:  {len(entries):,}")
    print(f"  Алертов:  {len(alerts)}")

    # Статистика по типам атак
    attack_types = Counter(a["type"] for a in alerts)
    print(f"\n--- Типы атак ---")
    for atype, count in attack_types.most_common():
        print(f"  {atype:25s} {count:>6}")

    # Топ IP-атакующих
    attacker_ips = Counter(a["ip"] for a in alerts)
    print(f"\n--- Топ-10 атакующих IP ---")
    for ip, count in attacker_ips.most_common(10):
        print(f"  {ip:18s} {count:>6} атак")

    # Статус-коды
    status_codes = Counter(e["status"] for e in entries)
    print(f"\n--- Коды ответов ---")
    for code, count in sorted(status_codes.items()):
        print(f"  {code:>3}  {count:>8,}")

    # Примеры атак
    print(f"\n--- Примеры атак (первые 10) ---")
    for a in alerts[:10]:
        print(f"  [{a['type']}] {a['ip']} {a['method']} {a['url'][:80]}")


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 web_log_analyzer.py access.log")
        sys.exit(1)

    entries = parse_access_log(sys.argv[1])
    print(f"[*] Загружено {len(entries)} записей")

    alerts = detect_attacks(entries)
    print_report(entries, alerts)

    with open("web_attacks.json", "w") as f:
        json.dump(alerts, f, indent=2, ensure_ascii=False)
    print(f"\n[*] Алерты: web_attacks.json ({len(alerts)} шт.)")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Подключите к ModSecurity WAF для корреляции с блокировками.
- Топ атакующих IP → автоматическая блокировка через Wazuh active response.
- JSON-отчёт для TheHive / SOAR-систем.

---

## Инструмент 11: Firewall Log Parser

### Назначение
Парсинг логов iptables и Windows Firewall: статистика заблокированных соединений,
top blocked IPs, обнаружение сканирования и DDoS-паттернов.

### Код

```python
#!/usr/bin/env python3
"""firewall_parser.py — Анализ логов iptables / Windows Firewall."""

import re
import sys
import json
from collections import Counter, defaultdict

# iptables log pattern
IPTABLES_RE = re.compile(
    r"(\w+\s+\d+\s+\d+:\d+:\d+).*?(IN=\S*)\s+(OUT=\S*)\s+.*?"
    r"SRC=(\S+)\s+DST=(\S+)\s+.*?PROTO=(\S+)(?:\s+SPT=(\d+))?\s*(?:DPT=(\d+))?"
)


def parse_iptables(filepath: str) -> list[dict]:
    """Парсинг iptables log."""
    entries = []
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            m = IPTABLES_RE.search(line)
            if m:
                entries.append({
                    "timestamp": m.group(1),
                    "interface_in": m.group(2).replace("IN=", ""),
                    "interface_out": m.group(3).replace("OUT=", ""),
                    "src_ip": m.group(4),
                    "dst_ip": m.group(5),
                    "proto": m.group(6),
                    "src_port": int(m.group(7)) if m.group(7) else 0,
                    "dst_port": int(m.group(8)) if m.group(8) else 0,
                })
    return entries


def analyze(entries: list[dict]) -> dict:
    """Анализ firewall-логов."""
    src_ips = Counter(e["src_ip"] for e in entries)
    dst_ports = Counter(e["dst_port"] for e in entries if e["dst_port"])
    protos = Counter(e["proto"] for e in entries)

    # Обнаружение сканирования: один IP → много портов
    ip_ports = defaultdict(set)
    for e in entries:
        if e["dst_port"]:
            ip_ports[e["src_ip"]].add(e["dst_port"])

    scanners = {ip: len(ports) for ip, ports in ip_ports.items() if len(ports) > 20}

    # DDoS pattern: один порт от множества IP
    port_ips = defaultdict(set)
    for e in entries:
        if e["dst_port"]:
            port_ips[e["dst_port"]].add(e["src_ip"])

    ddos_targets = {port: len(ips) for port, ips in port_ips.items() if len(ips) > 50}

    return {
        "total": len(entries),
        "top_sources": src_ips.most_common(20),
        "top_ports": dst_ports.most_common(20),
        "protocols": dict(protos),
        "scanners": scanners,
        "ddos_targets": ddos_targets,
    }


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 firewall_parser.py <firewall.log>")
        sys.exit(1)

    entries = parse_iptables(sys.argv[1])
    print(f"[*] Записей: {len(entries)}")

    result = analyze(entries)

    print(f"\n--- Протоколы ---")
    for proto, count in result["protocols"].items():
        print(f"  {proto:8s} {count:>8,}")

    print(f"\n--- Топ-10 заблокированных IP ---")
    for ip, count in result["top_sources"][:10]:
        scanner_tag = " [SCANNER]" if ip in result["scanners"] else ""
        print(f"  {ip:18s} {count:>8,}{scanner_tag}")

    print(f"\n--- Топ-10 целевых портов ---")
    for port, count in result["top_ports"][:10]:
        ddos_tag = " [DDoS?]" if port in result["ddos_targets"] else ""
        print(f"  {port:>6} {count:>8,}{ddos_tag}")

    if result["scanners"]:
        print(f"\n[!] СКАНЕРЫ ({len(result['scanners'])}):")
        for ip, ports in sorted(result["scanners"].items(), key=lambda x: -x[1]):
            print(f"  {ip:18s} → {ports} уникальных портов")

    with open("firewall_report.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[*] Отчёт: firewall_report.json")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Логи iptables направляйте в Wazuh/ELK, этот скрипт — для offline-анализа больших файлов.
- `SCANNER` + top-blocked IP → автоматическая блокировка через fail2ban / ipset.

---

## Инструмент 12: Multi-Source Log Correlator

### Назначение
Корреляция событий из разных источников в единый timeline. Объединяет auth.log, access.log,
firewall log и Sysmon в хронологический поток для расследования инцидентов.

### Код

```python
#!/usr/bin/env python3
"""log_correlator.py — Корреляция событий из разных источников в timeline."""

import re
import sys
import json
import csv
from datetime import datetime
from pathlib import Path

TIMESTAMP_FORMATS = [
    "%b %d %H:%M:%S",           # syslog: Jan 15 14:23:45
    "%d/%b/%Y:%H:%M:%S %z",     # Apache: 15/Jan/2024:14:23:45 +0000
    "%Y-%m-%dT%H:%M:%S",        # ISO: 2024-01-15T14:23:45
    "%Y-%m-%d %H:%M:%S",        # Generic: 2024-01-15 14:23:45
    "%m/%d/%Y %H:%M:%S",        # US: 01/15/2024 14:23:45
]


def parse_timestamp(ts_str: str) -> datetime | None:
    """Попытка распарсить timestamp разными форматами."""
    ts_str = ts_str.strip()
    for fmt in TIMESTAMP_FORMATS:
        try:
            dt = datetime.strptime(ts_str[:len(datetime.now().strftime(fmt))], fmt)
            if dt.year == 1900:
                dt = dt.replace(year=datetime.now().year)
            return dt
        except (ValueError, OverflowError):
            continue
    return None


def ingest_generic_log(filepath: str, source_name: str) -> list[dict]:
    """Импорт произвольного лога: первый токен — timestamp, остальное — message."""
    events = []
    with open(filepath, "r", errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Попытка извлечь timestamp из начала строки
            ts = None
            for fmt in TIMESTAMP_FORMATS:
                try:
                    sample_len = len(datetime.now().strftime(fmt))
                    ts = datetime.strptime(line[:sample_len], fmt)
                    if ts.year == 1900:
                        ts = ts.replace(year=datetime.now().year)
                    message = line[sample_len:].strip()
                    break
                except (ValueError, OverflowError):
                    continue

            if ts is None:
                ts = datetime.min
                message = line

            # Извлечение IP-адресов из сообщения
            ips = re.findall(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', message)

            events.append({
                "timestamp": ts.isoformat(),
                "source": source_name,
                "message": message[:500],
                "ips": ips[:5],
                "file": filepath,
                "line": line_num,
            })
    return events


def correlate_by_ip(events: list[dict]) -> dict:
    """Группировка событий по IP-адресам — корреляция между источниками."""
    ip_events = {}
    for event in events:
        for ip in event.get("ips", []):
            if ip not in ip_events:
                ip_events[ip] = []
            ip_events[ip].append(event)

    # Оставляем только IP, которые появляются в нескольких источниках
    correlated = {}
    for ip, evts in ip_events.items():
        sources = set(e["source"] for e in evts)
        if len(sources) > 1:
            correlated[ip] = {
                "sources": sorted(sources),
                "event_count": len(evts),
                "events": sorted(evts, key=lambda x: x["timestamp"]),
            }
    return correlated


def export_timeline(events: list[dict], filepath: str):
    """Экспорт отсортированного timeline в CSV."""
    events_sorted = sorted(events, key=lambda x: x["timestamp"])
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Source", "IPs", "Message"])
        for e in events_sorted:
            writer.writerow([
                e["timestamp"],
                e["source"],
                "; ".join(e.get("ips", [])),
                e["message"][:200]
            ])
    print(f"[*] Timeline: {filepath} ({len(events_sorted)} событий)")


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 log_correlator.py <log1> [log2] [log3] ...")
        print("Пример: python3 log_correlator.py auth.log access.log firewall.log")
        sys.exit(1)

    all_events = []
    for filepath in sys.argv[1:]:
        source = Path(filepath).stem
        events = ingest_generic_log(filepath, source)
        all_events.extend(events)
        print(f"[*] {source}: {len(events)} событий")

    print(f"[*] Всего событий: {len(all_events)}")

    correlated = correlate_by_ip(all_events)
    print(f"\n{'=' * 60}")
    print(f"КОРРЕЛЯЦИЯ ПО IP ({len(correlated)} адресов в нескольких источниках)")
    print(f"{'=' * 60}")

    for ip, data in sorted(correlated.items(), key=lambda x: -x[1]["event_count"]):
        print(f"\n  {ip} — {data['event_count']} событий из {data['sources']}")
        for e in data["events"][:5]:
            print(f"    [{e['source']}] {e['timestamp']} {e['message'][:80]}")

    export_timeline(all_events, "correlated_timeline.csv")

    with open("correlation_report.json", "w") as f:
        json.dump(correlated, f, indent=2, ensure_ascii=False, default=str)
    print(f"[*] Отчёт: correlation_report.json")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Замена «ручной» корреляции при расследовании: загрузите все логи, получите единый timeline.
- IP, появившийся в firewall log + auth.log + access.log = подтверждённая атака.
- CSV timeline открывается в Excel для визуального анализа.

---

## ЧАСТЬ 3: ДЕТЕКЦИЯ УГРОЗ

---

## Инструмент 13: IOC Checker (Threat Intelligence)

### Назначение
Проверка индикаторов компрометации (IP, домены, хеши) по открытым TI-источникам:
AbuseIPDB, VirusTotal, OTX AlienVault. Кеширование результатов для экономии API-запросов.

### Код

```python
#!/usr/bin/env python3
"""ioc_checker.py — Проверка IoC по AbuseIPDB, VirusTotal, OTX."""

import sys
import json
import os
import hashlib
import re
import time
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

CACHE_DIR = Path(".ioc_cache")
CACHE_TTL = 86400  # 24 часа

IOC_PATTERNS = {
    "ip": re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"),
    "domain": re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}$"),
    "md5": re.compile(r"^[a-fA-F0-9]{32}$"),
    "sha256": re.compile(r"^[a-fA-F0-9]{64}$"),
    "sha1": re.compile(r"^[a-fA-F0-9]{40}$"),
}


def detect_ioc_type(ioc: str) -> str:
    """Определение типа IoC."""
    for ioc_type, pattern in IOC_PATTERNS.items():
        if pattern.match(ioc):
            return ioc_type
    return "unknown"


def get_cache(ioc: str) -> dict | None:
    """Чтение кешированного результата."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{hashlib.md5(ioc.encode()).hexdigest()}.json"
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < CACHE_TTL:
            with open(cache_file, "r") as f:
                return json.load(f)
    return None


def set_cache(ioc: str, data: dict):
    """Запись результата в кеш."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{hashlib.md5(ioc.encode()).hexdigest()}.json"
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def check_abuseipdb(ip: str, api_key: str) -> dict:
    """Проверка IP в AbuseIPDB."""
    url = "https://api.abuseipdb.com/api/v2/check"
    headers = {"Key": api_key, "Accept": "application/json"}
    params = {"ipAddress": ip, "maxAgeInDays": 90}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            return {
                "source": "AbuseIPDB",
                "score": data.get("abuseConfidenceScore", 0),
                "reports": data.get("totalReports", 0),
                "country": data.get("countryCode", ""),
                "isp": data.get("isp", ""),
                "malicious": data.get("abuseConfidenceScore", 0) > 50,
            }
    except Exception as e:
        return {"source": "AbuseIPDB", "error": str(e)}
    return {"source": "AbuseIPDB", "error": f"HTTP {resp.status_code}"}


def check_virustotal(ioc: str, ioc_type: str, api_key: str) -> dict:
    """Проверка IoC в VirusTotal."""
    base_url = "https://www.virustotal.com/api/v3"
    headers = {"x-apikey": api_key}

    if ioc_type == "ip":
        url = f"{base_url}/ip_addresses/{ioc}"
    elif ioc_type == "domain":
        url = f"{base_url}/domains/{ioc}"
    elif ioc_type in ("md5", "sha1", "sha256"):
        url = f"{base_url}/files/{ioc}"
    else:
        return {"source": "VirusTotal", "error": "Unsupported type"}

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            attrs = resp.json().get("data", {}).get("attributes", {})
            stats = attrs.get("last_analysis_stats", {})
            malicious = stats.get("malicious", 0)
            total = sum(stats.values()) if stats else 0
            return {
                "source": "VirusTotal",
                "malicious_detections": malicious,
                "total_engines": total,
                "detection_rate": f"{malicious}/{total}",
                "malicious": malicious > 3,
            }
        elif resp.status_code == 404:
            return {"source": "VirusTotal", "found": False}
    except Exception as e:
        return {"source": "VirusTotal", "error": str(e)}
    return {"source": "VirusTotal", "error": f"HTTP {resp.status_code}"}


def check_ioc(ioc: str, api_keys: dict) -> dict:
    """Комплексная проверка одного IoC."""
    cached = get_cache(ioc)
    if cached:
        cached["from_cache"] = True
        return cached

    ioc_type = detect_ioc_type(ioc)
    result = {"ioc": ioc, "type": ioc_type, "checks": [], "verdict": "CLEAN"}

    if ioc_type == "ip" and api_keys.get("abuseipdb"):
        check = check_abuseipdb(ioc, api_keys["abuseipdb"])
        result["checks"].append(check)

    if api_keys.get("virustotal"):
        check = check_virustotal(ioc, ioc_type, api_keys["virustotal"])
        result["checks"].append(check)

    # Вердикт
    for check in result["checks"]:
        if check.get("malicious"):
            result["verdict"] = "MALICIOUS"
            break

    set_cache(ioc, result)
    return result


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 ioc_checker.py <ioc1> [ioc2] ...")
        print("       или:    python3 ioc_checker.py -f iocs.txt")
        print("\nAPI-ключи (переменные окружения):")
        print("  ABUSEIPDB_KEY, VIRUSTOTAL_KEY")
        sys.exit(1)

    api_keys = {
        "abuseipdb": os.environ.get("ABUSEIPDB_KEY", ""),
        "virustotal": os.environ.get("VIRUSTOTAL_KEY", ""),
    }

    if not any(api_keys.values()):
        print("[!] Нет API-ключей. Установите ABUSEIPDB_KEY и/или VIRUSTOTAL_KEY.")
        print("[*] Работаю без API — только определение типа IoC.\n")

    iocs = []
    if sys.argv[1] == "-f":
        with open(sys.argv[2]) as f:
            iocs = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        iocs = sys.argv[1:]

    results = []
    for ioc in iocs:
        result = check_ioc(ioc, api_keys)
        results.append(result)
        icon = "✗" if result["verdict"] == "MALICIOUS" else "✓"
        cache_tag = " (cache)" if result.get("from_cache") else ""
        print(f"  [{icon}] {ioc:40s} [{result['type']}] {result['verdict']}{cache_tag}")
        for check in result.get("checks", []):
            src = check.get("source", "?")
            if "error" in check:
                print(f"       {src}: ошибка — {check['error']}")
            elif check.get("malicious"):
                print(f"       {src}: ВРЕДОНОСНЫЙ")
            else:
                print(f"       {src}: чисто")

    # Сводка
    malicious = [r for r in results if r["verdict"] == "MALICIOUS"]
    print(f"\n{'=' * 50}")
    print(f"Проверено: {len(results)} | Вредоносных: {len(malicious)}")

    with open("ioc_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[*] Отчёт: ioc_results.json")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Встройте в Wazuh active response: при алерте автоматически проверяйте src_ip.
- Кеш экономит API-запросы при повторных проверках.
- `MALICIOUS` вердикт → автоматическая блокировка на firewall.

---

## Инструмент 14: File Integrity Monitor (Hash Scanner)

### Назначение
Рекурсивное хеширование файлов, сравнение с baseline — обнаружение изменённых, добавленных
и удалённых файлов. Базовый FIM (File Integrity Monitoring).

### Код

```python
#!/usr/bin/env python3
"""file_hash_scanner.py — Мониторинг целостности файлов (FIM)."""

import hashlib
import os
import sys
import json
from datetime import datetime
from pathlib import Path

BASELINE_FILE = "fim_baseline.json"
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".cache", ".venv"}
SKIP_EXT = {".pyc", ".pyo", ".swp", ".tmp"}


def hash_file(filepath: str, algo: str = "sha256") -> str:
    """Вычисление хеша файла."""
    h = hashlib.new(algo)
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except (PermissionError, OSError):
        return "ERROR"


def scan_directory(directory: str) -> dict:
    """Рекурсивное сканирование директории."""
    files = {}
    for root, dirs, filenames in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in filenames:
            if Path(fname).suffix in SKIP_EXT:
                continue
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, directory)
            file_hash = hash_file(fpath)
            try:
                stat = os.stat(fpath)
                files[rel_path] = {
                    "hash": file_hash,
                    "size": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            except OSError:
                files[rel_path] = {"hash": file_hash, "size": 0, "mtime": ""}
    return files


def compare(old: dict, new: dict) -> dict:
    """Сравнение baseline с текущим состоянием."""
    old_files = set(old.keys())
    new_files = set(new.keys())

    added = new_files - old_files
    removed = old_files - new_files
    common = old_files & new_files

    modified = []
    for f in common:
        if old[f]["hash"] != new[f]["hash"]:
            modified.append(f)

    return {
        "added": sorted(added),
        "removed": sorted(removed),
        "modified": sorted(modified),
        "unchanged": len(common) - len(modified),
    }


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 file_hash_scanner.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    print(f"[*] Сканирование {directory}...")
    current = scan_directory(directory)
    print(f"[*] Файлов: {len(current)}")

    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r") as f:
            baseline = json.load(f)
        diff = compare(baseline, current)

        print(f"\n{'=' * 50}")
        if diff["added"]:
            print(f"\n[+] ДОБАВЛЕНО ({len(diff['added'])}):")
            for f_path in diff["added"][:20]:
                print(f"    {f_path}")
        if diff["removed"]:
            print(f"\n[-] УДАЛЕНО ({len(diff['removed'])}):")
            for f_path in diff["removed"][:20]:
                print(f"    {f_path}")
        if diff["modified"]:
            print(f"\n[~] ИЗМЕНЕНО ({len(diff['modified'])}):")
            for f_path in diff["modified"][:20]:
                print(f"    {f_path}")
                print(f"      было: {baseline[f_path]['hash'][:16]}...")
                print(f"      стало: {current[f_path]['hash'][:16]}...")
        if not diff["added"] and not diff["removed"] and not diff["modified"]:
            print("\n[✓] Изменений не обнаружено.")
        print(f"\nБез изменений: {diff['unchanged']}")
    else:
        print("[*] Первый запуск — baseline создан.")

    with open(BASELINE_FILE, "w") as f:
        json.dump(current, f, indent=2)
    print(f"[*] Baseline: {BASELINE_FILE}")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Запускайте по cron на critical-серверах (web root, /etc, /usr/bin).
- Изменение файлов в `/usr/bin` → возможный rootkit → P1 алерт.
- Замена Wazuh syscheck для оффлайн-проверок.

---

## Инструмент 15: Process Monitor

### Назначение
Мониторинг процессов Linux: обнаружение новых процессов, подозрительных parent→child цепочек,
скрытых процессов (различие /proc и ps).

### Код

```python
#!/usr/bin/env python3
"""process_monitor.py — Мониторинг процессов: новые, подозрительные, скрытые."""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

SUSPICIOUS_NAMES = [
    "nc", "ncat", "netcat", "socat", "nmap", "masscan",
    "msfconsole", "meterpreter", "reverse", "backdoor",
    "xmrig", "minerd", "cpuminer", "cryptonight",
]

SUSPICIOUS_PARENTS = {
    "apache2": ["bash", "sh", "python", "perl", "ruby"],
    "nginx": ["bash", "sh", "python"],
    "httpd": ["bash", "sh"],
    "sshd": ["bash"],  # это нормально, но если bash → python → nc — нет
}

BASELINE_FILE = "process_baseline.json"


def get_processes() -> dict:
    """Получение списка процессов из /proc."""
    procs = {}
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        pid = int(pid_dir.name)
        try:
            cmdline = (pid_dir / "cmdline").read_text().replace("\x00", " ").strip()
            stat = (pid_dir / "stat").read_text().split()
            comm = stat[1].strip("()")
            ppid = int(stat[3])
            status_text = (pid_dir / "status").read_text()
            uid_line = [l for l in status_text.split("\n") if l.startswith("Uid:")][0]
            uid = int(uid_line.split()[1])

            procs[pid] = {
                "pid": pid,
                "ppid": ppid,
                "name": comm,
                "cmdline": cmdline[:500],
                "uid": uid,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except (FileNotFoundError, IndexError, PermissionError, ValueError):
            continue
    return procs


def detect_suspicious(procs: dict) -> list[dict]:
    """Обнаружение подозрительных процессов."""
    alerts = []

    for pid, proc in procs.items():
        name_lower = proc["name"].lower()

        # Подозрительные имена
        for sus in SUSPICIOUS_NAMES:
            if sus in name_lower:
                alerts.append({
                    "type": "SUSPICIOUS_NAME",
                    "pid": pid,
                    "name": proc["name"],
                    "cmdline": proc["cmdline"][:200],
                    "severity": "HIGH"
                })
                break

        # Подозрительные parent→child
        parent = procs.get(proc["ppid"])
        if parent:
            parent_name = parent["name"].lower()
            for sus_parent, sus_children in SUSPICIOUS_PARENTS.items():
                if sus_parent in parent_name and name_lower in sus_children:
                    alerts.append({
                        "type": "SUSPICIOUS_PARENT",
                        "pid": pid,
                        "parent": f"{parent['name']} (PID {proc['ppid']})",
                        "child": proc["name"],
                        "cmdline": proc["cmdline"][:200],
                        "severity": "MEDIUM"
                    })

        # Процессы от root на нестандартных путях
        if proc["uid"] == 0 and "/tmp/" in proc["cmdline"]:
            alerts.append({
                "type": "ROOT_IN_TMP",
                "pid": pid,
                "cmdline": proc["cmdline"][:200],
                "severity": "HIGH"
            })

    return alerts


def compare_with_baseline(current: dict) -> list[dict]:
    """Сравнение с baseline: новые процессы."""
    if not os.path.exists(BASELINE_FILE):
        return []

    with open(BASELINE_FILE, "r") as f:
        baseline = json.load(f)

    baseline_names = set(p["name"] for p in baseline.values())
    new_names = set()
    new_procs = []

    for pid, proc in current.items():
        if proc["name"] not in baseline_names and proc["name"] not in new_names:
            new_names.add(proc["name"])
            new_procs.append({
                "type": "NEW_PROCESS",
                "name": proc["name"],
                "pid": pid,
                "cmdline": proc["cmdline"][:200],
                "severity": "LOW"
            })

    return new_procs


def main():
    current = get_processes()
    print(f"[*] Процессов: {len(current)}")

    suspicious = detect_suspicious(current)
    new_procs = compare_with_baseline(current)
    all_alerts = suspicious + new_procs

    if all_alerts:
        print(f"\n[!] АЛЕРТОВ: {len(all_alerts)}")
        for a in all_alerts:
            print(f"  [{a['severity']}] {a['type']}: {a.get('name', '')} "
                  f"PID={a.get('pid', '')}")
            if a.get("cmdline"):
                print(f"    cmdline: {a['cmdline'][:80]}")
    else:
        print("\n[✓] Подозрительных процессов не обнаружено.")

    with open(BASELINE_FILE, "w") as f:
        json.dump({str(k): v for k, v in current.items()}, f, indent=2)

    if all_alerts:
        with open("process_alerts.json", "w") as f:
            json.dump(all_alerts, f, indent=2)
        print(f"[*] Алерты: process_alerts.json")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- По cron каждую минуту; алерты `ROOT_IN_TMP` или `SUSPICIOUS_NAME` → немедленная изоляция.
- Сравнение с baseline помогает обнаружить новое ПО, установленное атакующим.

---

## Инструмент 16: Registry Persistence Monitor (Windows)

### Назначение
Мониторинг ключевых точек автозапуска (persistence) в реестре Windows. Скрипт сравнивает
текущее состояние с baseline и выявляет новые/изменённые записи.

### Код

```python
#!/usr/bin/env python3
"""registry_monitor.py — Мониторинг persistence-точек реестра Windows."""

import sys
import json
import os
from datetime import datetime

# Для работы на Windows
try:
    import winreg
    ON_WINDOWS = True
except ImportError:
    ON_WINDOWS = False

PERSISTENCE_KEYS = [
    (winreg.HKEY_CURRENT_USER if ON_WINDOWS else 0,
     r"Software\Microsoft\Windows\CurrentVersion\Run"),
    (winreg.HKEY_CURRENT_USER if ON_WINDOWS else 0,
     r"Software\Microsoft\Windows\CurrentVersion\RunOnce"),
    (winreg.HKEY_LOCAL_MACHINE if ON_WINDOWS else 0,
     r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"),
    (winreg.HKEY_LOCAL_MACHINE if ON_WINDOWS else 0,
     r"SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnce"),
    (winreg.HKEY_LOCAL_MACHINE if ON_WINDOWS else 0,
     r"SYSTEM\CurrentControlSet\Services"),
    (winreg.HKEY_CURRENT_USER if ON_WINDOWS else 0,
     r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"),
    (winreg.HKEY_LOCAL_MACHINE if ON_WINDOWS else 0,
     r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon"),
] if ON_WINDOWS else []

BASELINE_FILE = "registry_baseline.json"


def read_registry_key(hive, key_path: str) -> dict:
    """Чтение всех значений ключа реестра."""
    values = {}
    try:
        key = winreg.OpenKey(hive, key_path, 0, winreg.KEY_READ)
        i = 0
        while True:
            try:
                name, data, dtype = winreg.EnumValue(key, i)
                values[name] = {"data": str(data), "type": dtype}
                i += 1
            except WindowsError:
                break
        winreg.CloseKey(key)
    except (WindowsError, FileNotFoundError):
        pass
    return values


def scan_persistence() -> dict:
    """Сканирование всех persistence-точек."""
    result = {}
    hive_names = {
        winreg.HKEY_CURRENT_USER: "HKCU",
        winreg.HKEY_LOCAL_MACHINE: "HKLM",
    }
    for hive, key_path in PERSISTENCE_KEYS:
        full_path = f"{hive_names.get(hive, '?')}\\{key_path}"
        result[full_path] = read_registry_key(hive, key_path)
    return result


def compare_baselines(old: dict, new: dict) -> list[dict]:
    """Сравнение baseline: новые, удалённые, изменённые значения."""
    alerts = []
    all_keys = set(old.keys()) | set(new.keys())

    for key_path in all_keys:
        old_vals = old.get(key_path, {})
        new_vals = new.get(key_path, {})

        for name in set(new_vals.keys()) - set(old_vals.keys()):
            alerts.append({
                "type": "NEW_VALUE",
                "key": key_path,
                "name": name,
                "data": new_vals[name]["data"],
                "severity": "HIGH"
            })

        for name in set(old_vals.keys()) - set(new_vals.keys()):
            alerts.append({
                "type": "REMOVED_VALUE",
                "key": key_path,
                "name": name,
                "severity": "MEDIUM"
            })

        for name in set(old_vals.keys()) & set(new_vals.keys()):
            if old_vals[name]["data"] != new_vals[name]["data"]:
                alerts.append({
                    "type": "MODIFIED_VALUE",
                    "key": key_path,
                    "name": name,
                    "old_data": old_vals[name]["data"],
                    "new_data": new_vals[name]["data"],
                    "severity": "HIGH"
                })

    return alerts


def main():
    if not ON_WINDOWS:
        print("[!] Этот скрипт работает только на Windows.")
        print("[*] Для демонстрации: загрузите экспорт реестра в JSON.")
        sys.exit(0)

    current = scan_persistence()
    total_values = sum(len(v) for v in current.values())
    print(f"[*] Просканировано ключей: {len(current)}, значений: {total_values}")

    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r") as f:
            baseline = json.load(f)
        alerts = compare_baselines(baseline, current)
        if alerts:
            print(f"\n[!] ИЗМЕНЕНИЙ: {len(alerts)}")
            for a in alerts:
                print(f"  [{a['severity']}] {a['type']}")
                print(f"    Ключ:     {a['key']}")
                print(f"    Имя:      {a.get('name', '')}")
                if a.get("data"):
                    print(f"    Данные:   {a['data'][:100]}")
        else:
            print("\n[✓] Изменений не обнаружено.")
    else:
        print("[*] Первый запуск — baseline создан.")

    with open(BASELINE_FILE, "w") as f:
        json.dump(current, f, indent=2, ensure_ascii=False)

    print(f"[*] Baseline: {BASELINE_FILE}")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Запускайте через Task Scheduler каждые 5 минут; алерт `NEW_VALUE` в Run → немедленное расследование.
- Дополняет Sysmon EID 12/13 оффлайн-проверкой.

---

## Инструмент 17: YARA Scanner Wrapper

### Назначение
Обёртка для yara-python: сканирование директорий набором YARA-правил с выводом
результатов в JSON. Удобнее, чем командная строка `yara`.

### Код

```python
#!/usr/bin/env python3
"""yara_scanner.py — Сканирование файлов набором YARA-правил."""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

try:
    import yara
    HAS_YARA = True
except ImportError:
    HAS_YARA = False
    print("[!] yara-python не установлен: pip install yara-python")


def compile_rules(rules_path: str) -> yara.Rules:
    """Компиляция YARA-правил (файл или директория)."""
    if os.path.isdir(rules_path):
        rule_files = {}
        for f in Path(rules_path).glob("*.yar"):
            rule_files[f.stem] = str(f)
        for f in Path(rules_path).glob("*.yara"):
            rule_files[f.stem] = str(f)
        return yara.compile(filepaths=rule_files)
    else:
        return yara.compile(filepath=rules_path)


def scan_file(rules: yara.Rules, filepath: str) -> list[dict]:
    """Сканирование одного файла."""
    matches = []
    try:
        results = rules.match(filepath, timeout=30)
        for match in results:
            matches.append({
                "rule": match.rule,
                "tags": match.tags,
                "strings": [
                    {"offset": s[0], "identifier": s[1],
                     "data": s[2][:50].hex() if isinstance(s[2], bytes) else str(s[2])[:50]}
                    for s in match.strings[:10]
                ],
                "meta": match.meta,
            })
    except yara.TimeoutError:
        matches.append({"rule": "TIMEOUT", "tags": [], "strings": []})
    except yara.Error as e:
        matches.append({"rule": "ERROR", "error": str(e)})
    return matches


def scan_directory(rules: yara.Rules, directory: str,
                   max_size: int = 50_000_000) -> list[dict]:
    """Рекурсивное сканирование директории."""
    results = []
    scanned = 0
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__"}]
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                if os.path.getsize(fpath) > max_size:
                    continue
            except OSError:
                continue

            matches = scan_file(rules, fpath)
            scanned += 1
            if matches:
                results.append({
                    "file": fpath,
                    "matches": matches,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                for m in matches:
                    print(f"  [!] {m['rule']:30s} → {fpath}")

    print(f"\n[*] Просканировано файлов: {scanned}")
    return results


def main():
    if len(sys.argv) < 3:
        print("Использование: python3 yara_scanner.py <rules> <target>")
        print("  rules:  файл .yar или директория с правилами")
        print("  target: файл или директория для сканирования")
        sys.exit(1)

    if not HAS_YARA:
        sys.exit(1)

    rules_path = sys.argv[1]
    target = sys.argv[2]

    print(f"[*] Компиляция правил: {rules_path}")
    rules = compile_rules(rules_path)

    if os.path.isdir(target):
        results = scan_directory(rules, target)
    else:
        matches = scan_file(rules, target)
        results = [{"file": target, "matches": matches}] if matches else []

    print(f"\n{'=' * 50}")
    print(f"ОБНАРУЖЕНО СОВПАДЕНИЙ: {len(results)}")

    with open("yara_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[*] Отчёт: yara_results.json")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Запускайте при подозрении на малварь: `python3 yara_scanner.py community_rules/ /tmp/suspicious/`
- Набор YARA-правил: скачайте из `https://github.com/Yara-Rules/rules`.
- JSON-результаты → TheHive / SOAR для автоматического кейса.

---

## Инструмент 18: Sigma Rule Tester

### Назначение
Загрузка Sigma-правил и проверка их срабатывания на JSON-логах.
Полезно для тестирования правил перед загрузкой в SIEM.

### Код

```python
#!/usr/bin/env python3
"""sigma_tester.py — Тестирование Sigma-правил на JSON-логах."""

import sys
import json
import re
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("[!] pyyaml не установлен: pip install pyyaml")


def load_sigma_rule(filepath: str) -> dict:
    """Загрузка Sigma-правила из YAML."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def match_value(log_value, sigma_value) -> bool:
    """Сравнение значения лога с условием Sigma."""
    if log_value is None:
        return False
    log_str = str(log_value).lower()
    sigma_str = str(sigma_value).lower()

    # Wildcard matching
    if "*" in sigma_str:
        pattern = re.escape(sigma_str).replace(r"\*", ".*")
        return bool(re.match(f"^{pattern}$", log_str))

    return log_str == sigma_str


def evaluate_selection(log_entry: dict, selection: dict) -> bool:
    """Проверка одного selection-блока против записи лога."""
    for field, value in selection.items():
        # Sigma использует | для модификаторов (например: TargetFilename|endswith)
        field_parts = field.split("|")
        field_name = field_parts[0]
        modifiers = field_parts[1:] if len(field_parts) > 1 else []

        log_value = log_entry.get(field_name)
        if log_value is None:
            # Попробуем case-insensitive
            for k, v in log_entry.items():
                if k.lower() == field_name.lower():
                    log_value = v
                    break

        if log_value is None:
            return False

        log_str = str(log_value).lower()

        if isinstance(value, list):
            # OR: любое значение из списка
            matched = False
            for v in value:
                if _apply_modifiers(log_str, str(v).lower(), modifiers):
                    matched = True
                    break
            if not matched:
                return False
        else:
            if not _apply_modifiers(log_str, str(value).lower(), modifiers):
                return False

    return True


def _apply_modifiers(log_str: str, sigma_str: str, modifiers: list) -> bool:
    """Применение Sigma-модификаторов."""
    if "contains" in modifiers:
        return sigma_str in log_str
    elif "endswith" in modifiers:
        return log_str.endswith(sigma_str)
    elif "startswith" in modifiers:
        return log_str.startswith(sigma_str)
    elif "re" in modifiers:
        return bool(re.search(sigma_str, log_str))
    else:
        return match_value(log_str, sigma_str)


def test_rule(rule: dict, logs: list[dict]) -> list[dict]:
    """Тестирование Sigma-правила на наборе логов."""
    detection = rule.get("detection", {})
    condition = detection.get("condition", "selection")
    matches = []

    for i, log_entry in enumerate(logs):
        # Простая реализация: поддержка 'selection', 'selection and not filter'
        if "selection" in detection:
            sel = detection["selection"]
            if evaluate_selection(log_entry, sel):
                # Проверка фильтра
                if "filter" in detection and "not" in condition:
                    if evaluate_selection(log_entry, detection["filter"]):
                        continue  # Отфильтровано
                matches.append({
                    "log_index": i,
                    "log_entry": {k: str(v)[:100] for k, v in log_entry.items()},
                })

    return matches


def main():
    if len(sys.argv) < 3:
        print("Использование: python3 sigma_tester.py <rule.yml> <logs.json>")
        print("  logs.json — массив JSON-объектов (записи логов)")
        sys.exit(1)

    if not HAS_YAML:
        sys.exit(1)

    rule = load_sigma_rule(sys.argv[1])
    print(f"[*] Правило: {rule.get('title', '?')}")
    print(f"    Уровень: {rule.get('level', '?')}")
    print(f"    Описание: {rule.get('description', '')[:100]}")

    with open(sys.argv[2], "r") as f:
        logs = json.load(f)
    print(f"[*] Записей логов: {len(logs)}")

    matches = test_rule(rule, logs)
    print(f"\n{'=' * 50}")
    print(f"СОВПАДЕНИЙ: {len(matches)} из {len(logs)}")

    for m in matches[:10]:
        print(f"\n  Запись #{m['log_index']}:")
        for k, v in m["log_entry"].items():
            print(f"    {k}: {v}")

    with open("sigma_test_results.json", "w") as f:
        json.dump({
            "rule": rule.get("title"),
            "total_logs": len(logs),
            "matches": len(matches),
            "details": matches[:100],
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[*] Результаты: sigma_test_results.json")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Тестируйте новые Sigma-правила на исторических логах перед production-деплоем.
- CI/CD для detection engineering: коммит правила → автотест на тестовых логах.

---

## ЧАСТЬ 4: ФОРЕНЗИКА

---

## Инструмент 19: Universal Timeline Builder

### Назначение
Построение единого хронологического timeline из разных источников (evtx, syslog, CSV, JSON).
Ключевой инструмент расследования — позволяет увидеть полную картину инцидента.

### Код

```python
#!/usr/bin/env python3
"""timeline_builder.py — Построение единого timeline из разных источников."""

import sys
import csv
import json
from datetime import datetime
from pathlib import Path


def parse_auto(filepath: str) -> list[dict]:
    """Автоматическое определение формата и парсинг."""
    ext = Path(filepath).suffix.lower()
    source = Path(filepath).stem

    if ext == ".json" or ext == ".jsonl":
        return parse_json_log(filepath, source)
    elif ext == ".csv":
        return parse_csv_log(filepath, source)
    else:
        return parse_text_log(filepath, source)


def parse_json_log(filepath: str, source: str) -> list[dict]:
    """Парсинг JSON/JSONL лога."""
    events = []
    with open(filepath, "r", errors="ignore") as f:
        content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.split("\n") if line.strip()]

    for entry in data:
        ts = entry.get("timestamp") or entry.get("Timestamp") or \
             entry.get("@timestamp") or entry.get("UtcTime") or ""
        msg = entry.get("message") or entry.get("Message") or json.dumps(entry)[:200]
        events.append({
            "timestamp": str(ts),
            "source": source,
            "event": str(msg)[:500],
            "raw": entry,
        })
    return events


def parse_csv_log(filepath: str, source: str) -> list[dict]:
    """Парсинг CSV лога (первая колонка — timestamp)."""
    events = []
    with open(filepath, "r", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = ""
            for key in ["timestamp", "Timestamp", "time", "Time", "date", "Date"]:
                if key in row:
                    ts = row[key]
                    break
            if not ts:
                values = list(row.values())
                ts = values[0] if values else ""
            events.append({
                "timestamp": ts,
                "source": source,
                "event": " | ".join(f"{k}={v}" for k, v in row.items())[:500],
            })
    return events


def parse_text_log(filepath: str, source: str) -> list[dict]:
    """Парсинг текстового лога."""
    import re
    events = []
    ts_re = re.compile(
        r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"
        r"|(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"
    )

    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = ts_re.search(line)
            ts = m.group(0) if m else ""
            events.append({
                "timestamp": ts,
                "source": source,
                "event": line[:500],
            })
    return events


def normalize_timestamp(ts_str: str) -> str:
    """Нормализация timestamp в ISO формат для сортировки."""
    formats = [
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
        "%b %d %H:%M:%S", "%m/%d/%Y %H:%M:%S",
        "%d/%b/%Y:%H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(ts_str[:len(datetime.now().strftime(fmt))], fmt)
            if dt.year == 1900:
                dt = dt.replace(year=datetime.now().year)
            return dt.isoformat()
        except (ValueError, OverflowError):
            continue
    return ts_str


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 timeline_builder.py <file1> [file2] ...")
        sys.exit(1)

    all_events = []
    for filepath in sys.argv[1:]:
        events = parse_auto(filepath)
        all_events.extend(events)
        print(f"[*] {Path(filepath).name}: {len(events)} событий")

    # Нормализация и сортировка
    for event in all_events:
        event["sort_key"] = normalize_timestamp(event["timestamp"])

    all_events.sort(key=lambda x: x["sort_key"])
    print(f"\n[*] Всего событий: {len(all_events)}")

    # Экспорт CSV
    output = "timeline.csv"
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Source", "Event"])
        for event in all_events:
            writer.writerow([event["timestamp"], event["source"],
                           event["event"][:300]])

    print(f"[*] Timeline: {output}")

    # Краткая статистика
    sources = {}
    for e in all_events:
        sources[e["source"]] = sources.get(e["source"], 0) + 1
    print(f"\n--- Источники ---")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src:20s} {count:>8,}")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- При расследовании: соберите логи со всех машин, запустите timeline — увидите хронологию атаки.
- CSV timeline открывается в Timeline Explorer / Excel.

---

## Инструмент 20: Email Header Analyzer

### Назначение
Парсинг email-заголовков для расследования фишинга: проверка SPF/DKIM/DMARC, анализ цепочки серверов,
определение реального отправителя, обнаружение подделки заголовков.

### Код

```python
#!/usr/bin/env python3
"""email_analyzer.py — Анализ email-заголовков: SPF, DKIM, DMARC, hops."""

import sys
import re
import json
from email import policy
from email.parser import BytesParser
from datetime import datetime


def parse_email_headers(filepath: str) -> dict:
    """Парсинг email-файла (.eml) или raw headers."""
    try:
        with open(filepath, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
    except Exception:
        with open(filepath, "r", errors="ignore") as f:
            content = f.read()
        msg = None
        return parse_raw_headers(content)

    headers = {}
    for key in msg.keys():
        values = msg.get_all(key, [])
        if len(values) == 1:
            headers[key] = str(values[0])
        else:
            headers[key] = [str(v) for v in values]

    return headers


def parse_raw_headers(text: str) -> dict:
    """Парсинг raw text заголовков."""
    headers = {}
    current_key = None
    current_value = []

    for line in text.split("\n"):
        if re.match(r"^[A-Za-z\-]+:", line):
            if current_key:
                val = " ".join(current_value).strip()
                if current_key in headers:
                    if isinstance(headers[current_key], list):
                        headers[current_key].append(val)
                    else:
                        headers[current_key] = [headers[current_key], val]
                else:
                    headers[current_key] = val
            parts = line.split(":", 1)
            current_key = parts[0].strip()
            current_value = [parts[1].strip() if len(parts) > 1 else ""]
        elif line.startswith((" ", "\t")) and current_key:
            current_value.append(line.strip())

    if current_key:
        headers[current_key] = " ".join(current_value).strip()

    return headers


def analyze_hops(headers: dict) -> list[dict]:
    """Анализ Received-заголовков (путь письма)."""
    received = headers.get("Received", [])
    if isinstance(received, str):
        received = [received]

    hops = []
    for i, r in enumerate(reversed(received)):
        hop = {"hop": i + 1, "raw": r[:200]}

        from_match = re.search(r"from\s+(\S+)", r)
        by_match = re.search(r"by\s+(\S+)", r)
        ip_match = re.findall(r"\[(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\]", r)

        if from_match:
            hop["from"] = from_match.group(1)
        if by_match:
            hop["by"] = by_match.group(1)
        if ip_match:
            hop["ips"] = ip_match

        hops.append(hop)
    return hops


def check_authentication(headers: dict) -> dict:
    """Проверка SPF, DKIM, DMARC по заголовкам."""
    auth_results = headers.get("Authentication-Results", "")
    if isinstance(auth_results, list):
        auth_results = " ".join(auth_results)

    spf = "unknown"
    dkim = "unknown"
    dmarc = "unknown"

    if "spf=pass" in auth_results.lower():
        spf = "PASS"
    elif "spf=fail" in auth_results.lower():
        spf = "FAIL"
    elif "spf=softfail" in auth_results.lower():
        spf = "SOFTFAIL"

    if "dkim=pass" in auth_results.lower():
        dkim = "PASS"
    elif "dkim=fail" in auth_results.lower():
        dkim = "FAIL"

    if "dmarc=pass" in auth_results.lower():
        dmarc = "PASS"
    elif "dmarc=fail" in auth_results.lower():
        dmarc = "FAIL"

    return {"spf": spf, "dkim": dkim, "dmarc": dmarc}


def analyze_email(filepath: str) -> dict:
    """Полный анализ email."""
    headers = parse_email_headers(filepath)

    result = {
        "from": headers.get("From", ""),
        "to": headers.get("To", ""),
        "subject": headers.get("Subject", ""),
        "date": headers.get("Date", ""),
        "message_id": headers.get("Message-ID", ""),
        "return_path": headers.get("Return-Path", ""),
        "reply_to": headers.get("Reply-To", ""),
        "x_mailer": headers.get("X-Mailer", ""),
        "authentication": check_authentication(headers),
        "hops": analyze_hops(headers),
        "suspicious": [],
    }

    # Проверки на подозрительность
    from_addr = result["from"].lower()
    reply_to = result["reply_to"].lower()
    return_path = result["return_path"].lower()

    if reply_to and reply_to != from_addr and reply_to not in from_addr:
        result["suspicious"].append("Reply-To отличается от From (возможный фишинг)")

    if return_path and return_path not in from_addr and "@" in return_path:
        result["suspicious"].append("Return-Path отличается от From")

    if result["authentication"]["spf"] == "FAIL":
        result["suspicious"].append("SPF FAIL — домен отправителя не авторизовал сервер")

    if result["authentication"]["dkim"] == "FAIL":
        result["suspicious"].append("DKIM FAIL — подпись не прошла проверку")

    if result["authentication"]["dmarc"] == "FAIL":
        result["suspicious"].append("DMARC FAIL — письмо не прошло политику домена")

    return result


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 email_analyzer.py <email.eml>")
        print("       или:    python3 email_analyzer.py <headers.txt>")
        sys.exit(1)

    result = analyze_email(sys.argv[1])

    print(f"{'=' * 60}")
    print(f"АНАЛИЗ EMAIL")
    print(f"{'=' * 60}")
    print(f"  From:       {result['from']}")
    print(f"  To:         {result['to']}")
    print(f"  Subject:    {result['subject']}")
    print(f"  Date:       {result['date']}")
    print(f"  Reply-To:   {result['reply_to']}")
    print(f"  Return-Path:{result['return_path']}")

    auth = result["authentication"]
    print(f"\n--- Аутентификация ---")
    print(f"  SPF:   {auth['spf']}")
    print(f"  DKIM:  {auth['dkim']}")
    print(f"  DMARC: {auth['dmarc']}")

    print(f"\n--- Маршрут ({len(result['hops'])} хопов) ---")
    for hop in result["hops"]:
        print(f"  Hop {hop['hop']}: {hop.get('from', '?')} → {hop.get('by', '?')}"
              f"  IPs: {hop.get('ips', [])}")

    if result["suspicious"]:
        print(f"\n--- [!] ПОДОЗРИТЕЛЬНО ---")
        for s in result["suspicious"]:
            print(f"  • {s}")
    else:
        print(f"\n[✓] Подозрений не обнаружено.")

    with open("email_analysis.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[*] Отчёт: email_analysis.json")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Первый шаг при анализе фишинга: сохраните .eml, запустите скрипт.
- SPF/DKIM/DMARC FAIL + отличающийся Reply-To = подтверждённый фишинг.
- JSON-отчёт для тикета в TheHive.

---

## ЧАСТЬ 5: АВТОМАТИЗАЦИЯ SOC

---

## Инструмент 21: Alert Enrichment Engine

### Назначение
Автоматическое обогащение алертов: GeoIP, reverse DNS, WHOIS, проверка в TI-базах.
На вход — JSON-алерт (IP, домен), на выход — обогащённый алерт с контекстом.

### Код

```python
#!/usr/bin/env python3
"""alert_enricher.py — Обогащение алертов: GeoIP, rDNS, WHOIS, TI."""

import socket
import json
import sys
from datetime import datetime

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def reverse_dns(ip: str) -> str:
    try:
        return socket.gethostbyaddr(ip)[0]
    except (socket.herror, socket.gaierror, OSError):
        return ""


def geoip_lookup(ip: str) -> dict:
    """GeoIP через бесплатный API ip-api.com."""
    if not HAS_REQUESTS:
        return {}
    try:
        resp = requests.get(f"http://ip-api.com/json/{ip}", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "country": data.get("country", ""),
                "city": data.get("city", ""),
                "isp": data.get("isp", ""),
                "org": data.get("org", ""),
                "as": data.get("as", ""),
            }
    except Exception:
        pass
    return {}


def enrich_alert(alert: dict) -> dict:
    """Обогащение одного алерта."""
    enriched = dict(alert)
    enriched["enrichment"] = {"timestamp": datetime.utcnow().isoformat()}

    # IP enrichment
    for field in ["src_ip", "dst_ip", "ip", "source_ip", "dest_ip"]:
        ip = alert.get(field)
        if ip:
            enriched["enrichment"][f"{field}_rdns"] = reverse_dns(ip)
            enriched["enrichment"][f"{field}_geo"] = geoip_lookup(ip)

    return enriched


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 alert_enricher.py <alerts.json>")
        print("  alerts.json — массив JSON-объектов с полями src_ip/dst_ip")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        alerts = json.load(f)

    if isinstance(alerts, dict):
        alerts = [alerts]

    print(f"[*] Обогащение {len(alerts)} алертов...")
    enriched = []
    for i, alert in enumerate(alerts):
        result = enrich_alert(alert)
        enriched.append(result)
        ip = alert.get("src_ip") or alert.get("ip", "?")
        geo = result["enrichment"].get("src_ip_geo") or \
              result["enrichment"].get("ip_geo", {})
        print(f"  [{i+1}] {ip} → {geo.get('country', '?')} / "
              f"{geo.get('isp', '?')}")

    with open("enriched_alerts.json", "w") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    print(f"\n[*] Результат: enriched_alerts.json")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Вызывайте из Wazuh active response или SOAR (Shuffle, TheHive).
- Обогащённые алерты содержат контекст, необходимый для решения L1-аналитика.

---

## Инструмент 22: IR Report Generator

### Назначение
Автоматическая генерация Incident Response отчёта по шаблону из структурированных данных.

### Код

```python
#!/usr/bin/env python3
"""ir_report_generator.py — Генерация IR-отчёта из JSON-данных."""

import sys
import json
from datetime import datetime

TEMPLATE = """
# ОТЧЁТ ОБ ИНЦИДЕНТЕ

**ID инцидента:** {incident_id}
**Дата обнаружения:** {detected_at}
**Дата отчёта:** {report_date}
**Приоритет:** {priority}
**Статус:** {status}
**Аналитик:** {analyst}

---

## 1. КРАТКОЕ ОПИСАНИЕ

{summary}

## 2. ХРОНОЛОГИЯ (TIMELINE)

| Время | Событие | Источник |
|-------|---------|----------|
{timeline_rows}

## 3. ЗАТРОНУТЫЕ СИСТЕМЫ

{affected_systems}

## 4. ИНДИКАТОРЫ КОМПРОМЕТАЦИИ (IoC)

| Тип | Значение | Описание |
|-----|----------|----------|
{ioc_rows}

## 5. ДЕЙСТВИЯ ПО РЕАГИРОВАНИЮ

{response_actions}

## 6. КОРНЕВАЯ ПРИЧИНА

{root_cause}

## 7. РЕКОМЕНДАЦИИ

{recommendations}

## 8. LESSONS LEARNED

{lessons_learned}

---

*Отчёт сгенерирован автоматически: {report_date}*
"""


def generate_report(data: dict) -> str:
    """Генерация отчёта из структурированных данных."""
    # Timeline
    timeline_rows = ""
    for event in data.get("timeline", []):
        timeline_rows += f"| {event.get('time', '')} | {event.get('event', '')} | {event.get('source', '')} |\n"

    # Affected systems
    affected = ""
    for sys_item in data.get("affected_systems", []):
        affected += f"- **{sys_item.get('hostname', '')}** ({sys_item.get('ip', '')}) — {sys_item.get('role', '')}\n"

    # IoC
    ioc_rows = ""
    for ioc in data.get("iocs", []):
        ioc_rows += f"| {ioc.get('type', '')} | `{ioc.get('value', '')}` | {ioc.get('description', '')} |\n"

    # Response actions
    actions = ""
    for i, action in enumerate(data.get("response_actions", []), 1):
        status_icon = "✅" if action.get("status") == "done" else "🔲"
        actions += f"{i}. {status_icon} {action.get('action', '')} ({action.get('time', '')})\n"

    # Recommendations
    recs = ""
    for rec in data.get("recommendations", []):
        recs += f"- {rec}\n"

    return TEMPLATE.format(
        incident_id=data.get("incident_id", "INC-???"),
        detected_at=data.get("detected_at", ""),
        report_date=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        priority=data.get("priority", "P2"),
        status=data.get("status", "Расследуется"),
        analyst=data.get("analyst", "SOC Team"),
        summary=data.get("summary", ""),
        timeline_rows=timeline_rows,
        affected_systems=affected,
        ioc_rows=ioc_rows,
        response_actions=actions,
        root_cause=data.get("root_cause", "Определяется"),
        recommendations=recs,
        lessons_learned=data.get("lessons_learned", "Будет дополнено после закрытия."),
    )


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 ir_report_generator.py <incident.json>")
        print("\nФормат JSON: см. шаблон incident_template.json")
        # Генерируем шаблон
        template = {
            "incident_id": "INC-2024-001",
            "detected_at": "2024-01-15 14:23 UTC",
            "priority": "P1 — Critical",
            "status": "Contained",
            "analyst": "Иванов И.И.",
            "summary": "Обнаружена ransomware-атака на файловый сервер FS01...",
            "timeline": [
                {"time": "14:23", "event": "Алерт SIEM: массовое переименование файлов", "source": "Wazuh"},
                {"time": "14:25", "event": "Подтверждение: шифровальщик активен", "source": "SOC L1"},
                {"time": "14:28", "event": "Изоляция FS01 от сети", "source": "SOC L2"},
            ],
            "affected_systems": [
                {"hostname": "FS01", "ip": "10.0.1.50", "role": "Файловый сервер"},
            ],
            "iocs": [
                {"type": "SHA256", "value": "a1b2c3d4...", "description": "Ransomware binary"},
                {"type": "IP", "value": "185.234.72.11", "description": "C2 сервер"},
            ],
            "response_actions": [
                {"action": "Изоляция заражённого хоста", "time": "14:28", "status": "done"},
                {"action": "Сброс учётных записей", "time": "14:45", "status": "done"},
                {"action": "Восстановление из бэкапа", "time": "в процессе", "status": "pending"},
            ],
            "root_cause": "Фишинговое письмо → macro → загрузка ransomware.",
            "recommendations": [
                "Включить блокировку макросов в Office (GPO)",
                "Настроить email-фильтрацию (DMARC strict)",
                "Внедрить EDR на все рабочие станции",
            ],
            "lessons_learned": "Время реагирования 5 минут — в рамках SLA. Нужно автоматизировать изоляцию."
        }
        with open("incident_template.json", "w") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        print("[*] Шаблон сохранён: incident_template.json")
        sys.exit(0)

    with open(sys.argv[1], "r") as f:
        data = json.load(f)

    report = generate_report(data)
    output = f"IR_Report_{data.get('incident_id', 'INC')}.md"
    with open(output, "w") as f:
        f.write(report)
    print(f"[*] Отчёт сгенерирован: {output}")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Заполняйте JSON в процессе расследования, генерируйте отчёт одной командой.
- Шаблон стандартизирует отчёты по NIST SP 800-61.

---

## Инструмент 23: CIS Benchmark Compliance Checker (Linux)

### Назначение
Проверка Linux-сервера по ключевым пунктам CIS Benchmark. Генерирует отчёт
соответствия с рекомендациями по исправлению.

### Код

```python
#!/usr/bin/env python3
"""compliance_checker.py — Проверка Linux по CIS Benchmark (ключевые пункты)."""

import subprocess
import os
import json
from pathlib import Path


def run_cmd(cmd: str) -> str:
    """Выполнение команды, возврат stdout."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True,
                               text=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return ""


def check(name: str, cmd: str, expected: str, fix: str) -> dict:
    """Проверка одного пункта CIS Benchmark."""
    actual = run_cmd(cmd)
    passed = expected.lower() in actual.lower() if expected else bool(actual)
    return {
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "expected": expected,
        "actual": actual[:200],
        "fix": fix,
    }


CHECKS = [
    ("SSH: запрет root-входа",
     "grep -i '^PermitRootLogin' /etc/ssh/sshd_config 2>/dev/null || echo 'not set'",
     "no",
     "Установить PermitRootLogin no в /etc/ssh/sshd_config"),

    ("SSH: только протокол 2",
     "grep -i '^Protocol' /etc/ssh/sshd_config 2>/dev/null || echo '2'",
     "2",
     "Установить Protocol 2 в /etc/ssh/sshd_config"),

    ("Пароли: минимальная длина",
     "grep -i '^PASS_MIN_LEN' /etc/login.defs 2>/dev/null || echo 'not set'",
     "8",
     "Установить PASS_MIN_LEN 12 в /etc/login.defs"),

    ("Файловая система: noexec на /tmp",
     "mount | grep '/tmp' | grep 'noexec' || echo 'not set'",
     "noexec",
     "Добавить noexec в опции монтирования /tmp в /etc/fstab"),

    ("Firewall: iptables/nftables активен",
     "systemctl is-active iptables nftables ufw 2>/dev/null | grep -c active",
     "1",
     "Включить firewall: systemctl enable --now ufw"),

    ("Auditd: служба активна",
     "systemctl is-active auditd 2>/dev/null || echo 'inactive'",
     "active",
     "Включить auditd: apt install auditd && systemctl enable --now auditd"),

    ("Cron: доступ ограничен",
     "test -f /etc/cron.allow && echo 'exists' || echo 'missing'",
     "exists",
     "Создать /etc/cron.allow с разрешёнными пользователями"),

    ("Permissions: /etc/passwd",
     "stat -c '%a' /etc/passwd 2>/dev/null",
     "644",
     "chmod 644 /etc/passwd"),

    ("Permissions: /etc/shadow",
     "stat -c '%a' /etc/shadow 2>/dev/null",
     "640",
     "chmod 640 /etc/shadow"),

    ("SUID: подозрительные файлы",
     "find / -perm -4000 -type f 2>/dev/null | wc -l",
     "",
     "Проверить список SUID-файлов: find / -perm -4000 -type f"),

    ("Сервисы: отключён telnet",
     "systemctl is-active telnet.socket xinetd 2>/dev/null | grep -c active || echo '0'",
     "0",
     "Отключить telnet: systemctl disable --now telnet.socket"),

    ("NTP: время синхронизировано",
     "timedatectl status 2>/dev/null | grep -i 'synchronized' || echo 'unknown'",
     "yes",
     "Настроить NTP: systemctl enable --now systemd-timesyncd"),

    ("Логирование: rsyslog активен",
     "systemctl is-active rsyslog 2>/dev/null || echo 'inactive'",
     "active",
     "Включить rsyslog: systemctl enable --now rsyslog"),

    ("Kernel: ASLR включён",
     "cat /proc/sys/kernel/randomize_va_space 2>/dev/null",
     "2",
     "echo 2 > /proc/sys/kernel/randomize_va_space"),

    ("Kernel: core dumps отключены",
     "grep -r 'hard core' /etc/security/limits.conf 2>/dev/null || echo 'not set'",
     "0",
     "Добавить '* hard core 0' в /etc/security/limits.conf"),
]


def main():
    print(f"{'=' * 60}")
    print(f"CIS BENCHMARK — ПРОВЕРКА LINUX")
    print(f"{'=' * 60}\n")

    results = []
    passed = 0
    failed = 0

    for name, cmd, expected, fix in CHECKS:
        result = check(name, cmd, expected, fix)
        results.append(result)

        icon = "✓" if result["status"] == "PASS" else "✗"
        print(f"  [{icon}] {name}")
        if result["status"] == "PASS":
            passed += 1
        else:
            failed += 1
            print(f"      Ожидалось: {expected}")
            print(f"      Найдено:   {result['actual'][:60]}")
            print(f"      Исправить: {fix}")

    score = (passed / len(CHECKS)) * 100 if CHECKS else 0
    print(f"\n{'=' * 60}")
    print(f"РЕЗУЛЬТАТ: {passed}/{len(CHECKS)} ({score:.0f}%)")
    print(f"  PASS: {passed}  |  FAIL: {failed}")
    print(f"{'=' * 60}")

    with open("compliance_report.json", "w") as f:
        json.dump({
            "date": str(Path("/etc/hostname").read_text().strip()
                       if Path("/etc/hostname").exists() else "unknown"),
            "score": f"{score:.0f}%",
            "passed": passed,
            "failed": failed,
            "checks": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"[*] Отчёт: compliance_report.json")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Запускайте после hardening, перед аудитом.
- JSON-отчёт для комплаенс-документации (152-ФЗ, PCI DSS).

---

## Инструмент 24: Canary File Monitor

### Назначение
Создание и мониторинг canary-файлов для раннего обнаружения ransomware.
При изменении/удалении canary-файла — немедленный алерт.

### Код

```python
#!/usr/bin/env python3
"""canary_monitor.py — Canary-файлы для обнаружения ransomware."""

import os
import sys
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path

CANARY_DIR = ".canary_files"
CANARY_CONTENT = "This is a canary file for ransomware detection. DO NOT MODIFY."
STATE_FILE = ".canary_state.json"


def create_canaries(directories: list[str], count: int = 3) -> dict:
    """Создание canary-файлов в указанных директориях."""
    state = {}
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue

        canary_dir = dir_path / CANARY_DIR
        canary_dir.mkdir(exist_ok=True)

        for i in range(count):
            name = f"_canary_{i:03d}.txt"
            fpath = canary_dir / name
            fpath.write_text(CANARY_CONTENT)

            file_hash = hashlib.sha256(CANARY_CONTENT.encode()).hexdigest()
            state[str(fpath)] = {
                "hash": file_hash,
                "size": len(CANARY_CONTENT),
                "created": datetime.utcnow().isoformat(),
            }
            print(f"  [+] Создан: {fpath}")

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"\n[*] Canary-файлов создано: {len(state)}")
    print(f"[*] Состояние сохранено: {STATE_FILE}")
    return state


def check_canaries() -> list[dict]:
    """Проверка целостности canary-файлов."""
    if not os.path.exists(STATE_FILE):
        print("[!] State-файл не найден. Сначала создайте canary: --create")
        return []

    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    alerts = []
    for fpath, expected in state.items():
        path = Path(fpath)

        if not path.exists():
            alerts.append({
                "type": "DELETED",
                "file": fpath,
                "severity": "CRITICAL",
                "message": "Canary-файл УДАЛЁН — возможная ransomware-активность!",
                "timestamp": datetime.utcnow().isoformat(),
            })
            continue

        content = path.read_text()
        current_hash = hashlib.sha256(content.encode()).hexdigest()

        if current_hash != expected["hash"]:
            alerts.append({
                "type": "MODIFIED",
                "file": fpath,
                "severity": "CRITICAL",
                "message": "Canary-файл ИЗМЕНЁН — возможное шифрование!",
                "expected_hash": expected["hash"],
                "current_hash": current_hash,
                "timestamp": datetime.utcnow().isoformat(),
            })

    return alerts


def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python3 canary_monitor.py --create /path/dir1 /path/dir2")
        print("  python3 canary_monitor.py --check")
        print("  python3 canary_monitor.py --watch  (непрерывный мониторинг)")
        sys.exit(1)

    if sys.argv[1] == "--create":
        dirs = sys.argv[2:] if len(sys.argv) > 2 else ["."]
        create_canaries(dirs)

    elif sys.argv[1] == "--check":
        alerts = check_canaries()
        if alerts:
            print(f"\n[!!!] АЛЕРТ: {len(alerts)} canary-файлов скомпрометированы!")
            for a in alerts:
                print(f"  [{a['severity']}] {a['type']}: {a['file']}")
                print(f"    {a['message']}")
            with open("canary_alerts.json", "w") as f:
                json.dump(alerts, f, indent=2)
        else:
            print("[✓] Все canary-файлы в порядке.")

    elif sys.argv[1] == "--watch":
        print("[*] Мониторинг canary-файлов (Ctrl+C для остановки)...")
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        try:
            while True:
                alerts = check_canaries()
                if alerts:
                    print(f"\n[!!!] {datetime.utcnow()} — АЛЕРТ!")
                    for a in alerts:
                        print(f"  {a['type']}: {a['file']}")
                    with open("canary_alerts.json", "w") as f:
                        json.dump(alerts, f, indent=2)
                    break  # В реальной системе: отправка алерта + изоляция
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[*] Мониторинг остановлен.")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Разместите canary в ключевых директориях: SMB-шары, /home, рабочие столы.
- Мониторинг через cron / systemd timer каждые 5 секунд.
- Алерт DELETED/MODIFIED → немедленная изоляция хоста от сети через EDR/Wazuh.

---

## Инструмент 25: SOC Metrics Dashboard Collector

### Назначение
Сбор метрик для SOC-дашборда: количество алертов, MTTD/MTTR, false positive rate,
алерты по категориям. Выходные данные — JSON для Grafana/ELK.

### Код

```python
#!/usr/bin/env python3
"""soc_metrics.py — Сбор метрик SOC: алерты, MTTR, FP rate."""

import json
import sys
from datetime import datetime, timedelta
from collections import Counter


def calculate_metrics(alerts: list[dict]) -> dict:
    """Расчёт SOC-метрик из списка алертов."""
    total = len(alerts)
    if total == 0:
        return {"total_alerts": 0, "message": "Нет данных"}

    # Категории
    categories = Counter(a.get("category", "unknown") for a in alerts)
    severities = Counter(a.get("severity", "unknown") for a in alerts)
    statuses = Counter(a.get("status", "unknown") for a in alerts)

    # True/False Positive
    tp = sum(1 for a in alerts if a.get("verdict") == "true_positive")
    fp = sum(1 for a in alerts if a.get("verdict") == "false_positive")
    fp_rate = (fp / (tp + fp) * 100) if (tp + fp) > 0 else 0

    # MTTD (Mean Time to Detect) и MTTR (Mean Time to Respond)
    mttd_values = []
    mttr_values = []
    for a in alerts:
        if a.get("attack_time") and a.get("detect_time"):
            try:
                t1 = datetime.fromisoformat(a["attack_time"])
                t2 = datetime.fromisoformat(a["detect_time"])
                mttd_values.append((t2 - t1).total_seconds() / 60)
            except (ValueError, TypeError):
                pass
        if a.get("detect_time") and a.get("resolve_time"):
            try:
                t1 = datetime.fromisoformat(a["detect_time"])
                t2 = datetime.fromisoformat(a["resolve_time"])
                mttr_values.append((t2 - t1).total_seconds() / 60)
            except (ValueError, TypeError):
                pass

    mttd = sum(mttd_values) / len(mttd_values) if mttd_values else None
    mttr = sum(mttr_values) / len(mttr_values) if mttr_values else None

    return {
        "period": datetime.utcnow().strftime("%Y-%m-%d"),
        "total_alerts": total,
        "by_category": dict(categories.most_common()),
        "by_severity": dict(severities),
        "by_status": dict(statuses),
        "true_positive": tp,
        "false_positive": fp,
        "fp_rate_pct": round(fp_rate, 1),
        "mttd_min": round(mttd, 1) if mttd else None,
        "mttr_min": round(mttr, 1) if mttr else None,
    }


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 soc_metrics.py <alerts.json>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        alerts = json.load(f)

    metrics = calculate_metrics(alerts)

    print(f"{'=' * 50}")
    print(f"SOC МЕТРИКИ — {metrics['period']}")
    print(f"{'=' * 50}")
    print(f"  Всего алертов:   {metrics['total_alerts']}")
    print(f"  True Positive:   {metrics['true_positive']}")
    print(f"  False Positive:  {metrics['false_positive']}")
    print(f"  FP Rate:         {metrics['fp_rate_pct']}%")
    if metrics["mttd_min"]:
        print(f"  MTTD:            {metrics['mttd_min']} мин")
    if metrics["mttr_min"]:
        print(f"  MTTR:            {metrics['mttr_min']} мин")

    print(f"\n--- По категориям ---")
    for cat, count in metrics["by_category"].items():
        print(f"  {cat:25s} {count:>5}")

    print(f"\n--- По severity ---")
    for sev, count in metrics["by_severity"].items():
        print(f"  {sev:25s} {count:>5}")

    with open("soc_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[*] Метрики: soc_metrics.json")


if __name__ == "__main__":
    main()
```

### Интеграция с SOC
- Запускайте ежедневно, выходной JSON подгружайте в Grafana.
- FP Rate > 50% → пересмотреть правила детекции.
- MTTR > SLA → улучшить процесс реагирования.

---

## СПРАВОЧНАЯ ТАБЛИЦА

| # | Инструмент | Категория | Входные данные | Ключевые зависимости |
|---|-----------|-----------|----------------|---------------------|
| 1 | Network Scanner | Сеть | CIDR | stdlib |
| 2 | DNS Analyzer | Сеть | dns.log | stdlib |
| 3 | Beacon Detector | Сеть | conn.log | stdlib |
| 4 | PCAP Analyzer | Сеть | .pcap | dpkt |
| 5 | SSL Checker | Сеть | хосты | stdlib |
| 6 | Port Monitor | Сеть | хосты | stdlib |
| 7 | EVTX Parser | Логи | .evtx | python-evtx |
| 8 | Sysmon Analyzer | Логи | XML/JSON | stdlib |
| 9 | Auth Analyzer | Логи | auth.log | stdlib |
| 10 | Web Log Analyzer | Логи | access.log | stdlib |
| 11 | Firewall Parser | Логи | iptables.log | stdlib |
| 12 | Log Correlator | Логи | any logs | stdlib |
| 13 | IOC Checker | Детекция | IP/domain/hash | requests |
| 14 | File Hash Scanner | Детекция | директория | stdlib |
| 15 | Process Monitor | Детекция | /proc | stdlib |
| 16 | Registry Monitor | Детекция | реестр Windows | winreg |
| 17 | YARA Scanner | Детекция | файлы + правила | yara-python |
| 18 | Sigma Tester | Детекция | правило + логи | pyyaml |
| 19 | Timeline Builder | Форензика | разные логи | stdlib |
| 20 | Email Analyzer | Форензика | .eml / headers | stdlib |
| 21 | Alert Enricher | Автоматизация | JSON алерты | requests |
| 22 | IR Report Gen | Автоматизация | JSON данные | stdlib |
| 23 | CIS Checker | Автоматизация | система | stdlib |
| 24 | Canary Monitor | Автоматизация | файловая система | stdlib |
| 25 | SOC Metrics | Автоматизация | JSON алерты | stdlib |

---

## УСТАНОВКА ЗАВИСИМОСТЕЙ

```bash
# Основные (большинство скриптов работает без внешних зависимостей)
pip install requests

# Для работы с PCAP
pip install dpkt

# Для YARA
pip install yara-python

# Для Windows Event Logs
pip install python-evtx

# Для Sigma
pip install pyyaml

# Все сразу
pip install requests dpkt yara-python python-evtx pyyaml
```

---

*25 инструментов · Все скрипты работают на Python 3.8+ · Только защитные инструменты*
*Курс «Кибер-Щит» · Практическая кибербезопасность (Blue Team)*
