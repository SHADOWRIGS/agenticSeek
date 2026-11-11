# ТРЕК: SOC-АНАЛИТИК С НУЛЯ ЗА 4–6 МЕСЯЦЕВ

> Короткий карьерный маршрут: от IT-новичка до первой работы в SOC.
> Минимум теории — максимум практики на реальных инструментах.
> Этот трек можно пройти самостоятельно или как дополнение к полному курсу «Кибер-Щит».

---

## ОГЛАВЛЕНИЕ

1. [Для кого](#для-кого)
2. [Результат](#результат)
3. [Требования](#требования)
4. [Месяц 1: Фундамент](#месяц-1-фундамент)
5. [Месяц 2: Инструменты SOC](#месяц-2-инструменты-soc)
6. [Месяц 3: Детекция атак](#месяц-3-детекция-атак)
7. [Месяц 4: Incident Response](#месяц-4-incident-response)
8. [Месяц 5: Портфолио и трудоустройство](#месяц-5-портфолио-и-трудоустройство)
9. [Месяц 6: Сертификация (опционально)](#месяц-6-сертификация-опционально)
10. [Топ-50 вопросов собеседования SOC L1](#топ-50-вопросов-собеседования-soc-l1)
11. [Шаблон резюме](#шаблон-резюме-junior-soc-аналитика)
12. [Чеклист готовности к работе](#чеклист-готовности-к-работе)
13. [Ресурсы](#ресурсы-для-самообучения)

---

## ДЛЯ КОГО

- IT-специалисты, переходящие в кибербезопасность
- Студенты ИБ/ИТ-направлений
- Сисадмины, хелпдеск, DevOps — все, кто хочет в SOC
- Те, кому не нужен полный 12-недельный курс, а нужен конкретный путь к работе
- Люди после 30, меняющие профессию — SOC принимает без опыта в ИБ

**Не подойдёт, если:**
- Вы хотите стать пентестером / red team (это другой трек)
- У вас нет базового понимания, что такое компьютер и интернет
- Вы не готовы уделять 10–15 часов в неделю

---

## РЕЗУЛЬТАТ

После прохождения вы:

- Умеете работать с SIEM (Wazuh / ELK) — триаж алертов, написание правил
- Знаете 20 ключевых техник атак и как их обнаружить
- Можете провести первичное расследование инцидента
- Владеете Sigma, Sysmon, Suricata на базовом уровне
- Имеете портфолио: 5 разобранных инцидентов + набор Sigma-правил
- Готовы к собеседованию на Junior SOC Analyst (L1)

**Средняя зарплата Junior SOC Analyst (L1) в России (2024–2025):**

| Город | Зарплата (руб/мес) |
|-------|-------------------|
| Москва | 80 000 – 140 000 |
| Санкт-Петербург | 70 000 – 120 000 |
| Регионы | 55 000 – 90 000 |
| Удалёнка | 70 000 – 130 000 |

---

## ТРЕБОВАНИЯ

### Железо
- Ноутбук/ПК с 8+ ГБ ОЗУ (16 ГБ рекомендуется)
- 50+ ГБ свободного места
- Возможность запускать виртуальные машины (VirtualBox / VMware)

### Софт (всё бесплатное)
- VirtualBox или VMware Workstation Player
- Ubuntu Server 22.04 LTS
- Windows 10/11 (можно evaluation)
- Wazuh (open source SIEM)
- Sysmon (бесплатный от Microsoft)
- Suricata (open source IDS/IPS)
- Wireshark
- Python 3.10+

### Время
- 10–15 часов в неделю
- 4 месяца на основной трек
- +1–2 месяца на портфолио и сертификацию

---

## МЕСЯЦ 1: ФУНДАМЕНТ

> **Цель:** понять, как работают сети и операционные системы глазами защитника.
> **Время:** 10–15 часов/неделю

---

### Неделя 1–2: Сети и протоколы

#### Что изучить

**Модель OSI и TCP/IP:**
- 7 уровней OSI — что происходит на каждом
- Какие атаки возможны на каждом уровне
- Что видит SOC-аналитик на каждом уровне

**Ключевые протоколы:**
- **TCP/IP** — трёхстороннее рукопожатие, флаги (SYN, ACK, RST, FIN)
- **DNS** — как работает, типы записей (A, AAAA, MX, TXT, CNAME)
- **HTTP/HTTPS** — методы (GET, POST), коды ответов, заголовки
- **DHCP** — процесс получения адреса (DORA)
- **ARP** — резолвинг MAC-адресов

**IP-адресация:**
- IPv4: подсети, CIDR, частные диапазоны (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
- NAT, порты (0–65535), well-known ports (22, 53, 80, 443, 445, 3389)
- IPv6 basics

#### Практика: Wireshark

```bash
# Установка
sudo apt install wireshark tshark

# Захват трафика (100 пакетов)
sudo tshark -i eth0 -c 100 -w ~/capture.pcap

# Просмотр TCP-рукопожатия
tshark -r ~/capture.pcap -Y "tcp.flags.syn==1" -T fields -e ip.src -e ip.dst -e tcp.dstport

# Просмотр DNS-запросов
tshark -r ~/capture.pcap -Y "dns.qry.name" -T fields -e ip.src -e dns.qry.name -e dns.a

# Просмотр HTTP-запросов
tshark -r ~/capture.pcap -Y "http.request" -T fields -e ip.src -e http.host -e http.request.uri

# Фильтр подозрительных портов
tshark -r ~/capture.pcap -Y "tcp.dstport==4444 || tcp.dstport==1234 || tcp.dstport==31337"
```

#### Задание недели 1
1. Установить Wireshark на Linux
2. Захватить 5 минут собственного трафика
3. Найти все DNS-запросы и записать топ-10 доменов
4. Найти все HTTP-запросы (если есть)
5. Определить свой IP, шлюз и DNS-сервер командами: `ip addr`, `ip route`, `cat /etc/resolv.conf`

#### Задание недели 2
1. Скачать учебный PCAP (Malware Traffic Analysis — malware-traffic-analysis.net)
2. Найти в нём:
   - IP-адреса источника и назначения
   - DNS-запросы к подозрительным доменам
   - HTTP POST запросы (возможная эксфильтрация)
3. Составить таймлайн событий
4. Написать краткий отчёт (1 страница)

#### Ключевые порты для запоминания

| Порт | Сервис | Зачем SOC-аналитику |
|------|--------|---------------------|
| 22 | SSH | Brute-force, lateral movement |
| 25 | SMTP | Фишинг, спам |
| 53 | DNS | DNS tunneling, C2 |
| 80 | HTTP | Web-атаки, C2 |
| 88 | Kerberos | Kerberoasting, Golden Ticket |
| 135 | RPC | Lateral movement |
| 139/445 | SMB | EternalBlue, PsExec, lateral movement |
| 443 | HTTPS | C2, exfiltration |
| 636 | LDAPS | AD-атаки |
| 1433 | MSSQL | SQLi, lateral movement |
| 3306 | MySQL | SQLi |
| 3389 | RDP | Brute-force, lateral movement |
| 5985/5986 | WinRM | Lateral movement |
| 8080 | HTTP Alt | Web-атаки, прокси |

---

### Неделя 3–4: Windows и Linux для защитника

#### Linux: основные команды

```bash
# Файловая система
ls -la /var/log/           # Логи системы
cat /etc/passwd            # Пользователи
cat /etc/shadow            # Хеши паролей (нужен root)
cat /etc/group             # Группы
cat /etc/crontab           # Запланированные задачи
crontab -l                 # Задачи текущего пользователя

# Процессы
ps aux                     # Все процессы
ps aux | grep -i suspicious # Поиск подозрительного
top / htop                 # Мониторинг в реальном времени
lsof -i                    # Открытые сетевые соединения
lsof -p <PID>              # Файлы процесса

# Сеть
ss -tlnp                   # Открытые порты (TCP)
ss -ulnp                   # Открытые порты (UDP)
netstat -antp              # Соединения (если нет ss)
ip addr show               # IP-адреса
ip route show              # Маршруты
iptables -L -n -v          # Правила файрвола

# Логи
journalctl -xe             # Системный журнал
journalctl -u ssh          # Логи SSH
cat /var/log/auth.log      # Аутентификация (Debian/Ubuntu)
cat /var/log/secure         # Аутентификация (RHEL/CentOS)
last                       # Последние входы
lastb                      # Неудачные входы
who                        # Текущие сессии

# Поиск
find / -name "*.sh" -perm -o+x 2>/dev/null  # Исполняемые скрипты
find / -newer /tmp/marker -ls 2>/dev/null     # Файлы новее маркера
find / -mtime -1 -ls 2>/dev/null              # Изменены за последний день
grep -r "password" /etc/ 2>/dev/null           # Пароли в конфигах
```

#### Windows: ключевые Event ID

| Event ID | Источник | Что означает | Важность |
|----------|----------|-------------|----------|
| 4624 | Security | Успешный вход | Высокая — тип входа важен |
| 4625 | Security | Неудачный вход | Высокая — brute-force |
| 4648 | Security | Вход с явными учётными данными | Высокая — runas, lateral |
| 4672 | Security | Специальные привилегии | Высокая — admin logon |
| 4688 | Security | Создание процесса | Высокая — выполнение |
| 4689 | Security | Завершение процесса | Средняя |
| 4697 | Security | Установка службы | Высокая — persistence |
| 4698 | Security | Создание запланированной задачи | Высокая — persistence |
| 4720 | Security | Создание учётной записи | Высокая — persistence |
| 4732 | Security | Добавление в группу Administrators | Критическая |
| 4768 | Security | Запрос TGT (Kerberos) | Средняя — Kerberoasting |
| 4769 | Security | Запрос TGS (Kerberos) | Средняя — Kerberoasting |
| 1102 | Security | Очистка журнала | Критическая — anti-forensics |
| 7045 | System | Установка службы | Высокая — persistence |

#### Sysmon Event ID

| Event ID | Что отслеживает | Зачем |
|----------|----------------|-------|
| 1 | Создание процесса | Полная командная строка, родитель, хеши |
| 3 | Сетевое подключение | C2, lateral movement |
| 6 | Загрузка драйвера | Руткиты |
| 7 | Загрузка DLL | DLL hijacking |
| 8 | CreateRemoteThread | Process injection |
| 10 | ProcessAccess | LSASS dumping |
| 11 | Создание файла | Dropped files |
| 12/13 | Изменение реестра | Persistence |
| 15 | FileCreateStreamHash | ADS |
| 22 | DNS-запрос | C2 domains |
| 23 | Удаление файла | Anti-forensics |
| 25 | Process tampering | Hollowing, herpaderping |

#### Установка Sysmon

```powershell
# Скачать Sysmon
# https://learn.microsoft.com/en-us/sysinternals/downloads/sysmon

# Установить с конфигурацией SwiftOnSecurity
Sysmon64.exe -accepteula -i sysmonconfig-export.xml

# Проверить установку
Get-Service Sysmon64

# Просмотр логов Sysmon
Get-WinEvent -LogName "Microsoft-Windows-Sysmon/Operational" -MaxEvents 10

# Фильтр по Event ID (создание процесса)
Get-WinEvent -LogName "Microsoft-Windows-Sysmon/Operational" |
    Where-Object { $_.Id -eq 1 } |
    Select-Object -First 10 |
    Format-List TimeCreated, Message
```

#### Практика: развернуть лабораторию

**Архитектура лаборатории:**

```
┌─────────────────────────────────────────┐
│             Хост (ваш ПК)               │
│                                         │
│  ┌───────────────┐  ┌───────────────┐   │
│  │ Ubuntu Server  │  │  Windows 10   │   │
│  │ (Wazuh Server) │  │  (Wazuh Agent │   │
│  │ + Suricata     │  │   + Sysmon)   │   │
│  │ 192.168.56.10  │  │ 192.168.56.20 │   │
│  └───────────────┘  └───────────────┘   │
│         Host-Only Network               │
│         192.168.56.0/24                 │
└─────────────────────────────────────────┘
```

**Шаг 1: Создать VM с Ubuntu Server 22.04**
```bash
# После установки Ubuntu Server:
sudo apt update && sudo apt upgrade -y

# Установить Wazuh (all-in-one)
curl -sO https://packages.wazuh.com/4.7/wazuh-install.sh
sudo bash ./wazuh-install.sh -a

# Запомнить пароль admin из вывода!
# Доступ к дашборду: https://192.168.56.10
```

**Шаг 2: Установить Suricata на Ubuntu**
```bash
sudo apt install suricata -y

# Настроить интерфейс
sudo nano /etc/suricata/suricata.yaml
# Найти HOME_NET и задать: "[192.168.56.0/24]"

# Обновить правила
sudo suricata-update

# Запустить
sudo systemctl enable suricata
sudo systemctl start suricata

# Проверить
sudo suricata --build-info | head -20
tail -f /var/log/suricata/fast.log
```

**Шаг 3: Установить Wazuh Agent на Windows**
```powershell
# Скачать агент: https://packages.wazuh.com/4.x/windows/wazuh-agent-4.7.0-1.msi
# Установить, указав IP сервера: 192.168.56.10

# Проверить
Get-Service WazuhSvc
```

**Шаг 4: Установить Sysmon на Windows**
```powershell
# Скачать Sysmon + конфигурацию SwiftOnSecurity
Sysmon64.exe -accepteula -i sysmonconfig-export.xml

# Проверить в Event Viewer:
# Applications and Services Logs → Microsoft → Windows → Sysmon → Operational
```

#### Чеклист навыков месяца 1

- [ ] Объяснить модель OSI и TCP/IP
- [ ] Назвать 10 ключевых портов и связанные с ними атаки
- [ ] Работать с Wireshark: захват, фильтрация, анализ
- [ ] Работать в Linux: файлы, процессы, сеть, логи
- [ ] Знать ключевые Windows Event ID (4624, 4625, 4688, 1102)
- [ ] Знать ключевые Sysmon Event ID (1, 3, 8, 10, 11, 12/13)
- [ ] Развернуть лабораторию: Wazuh + Sysmon + Suricata
- [ ] Анализировать учебный PCAP-файл
- [ ] Написать краткий отчёт по найденным аномалиям

---

## МЕСЯЦ 2: ИНСТРУМЕНТЫ SOC

> **Цель:** освоить SIEM (Wazuh), научиться писать Sigma-правила.
> **Время:** 10–15 часов/неделю

---

### Неделя 5–6: SIEM (Wazuh)

#### Что такое SIEM

SIEM (Security Information and Event Management) — система, которая:
1. **Собирает** логи со всех источников (серверы, рабочие станции, сетевое оборудование)
2. **Нормализует** — приводит к единому формату
3. **Коррелирует** — связывает события из разных источников
4. **Алертит** — сообщает о подозрительной активности
5. **Хранит** — для расследований и compliance

#### Навигация по Wazuh

```
Wazuh Dashboard (https://192.168.56.10)

├── Security Events     — все события безопасности
├── Integrity Monitoring — контроль целостности файлов
├── Vulnerabilities      — обнаруженные уязвимости
├── MITRE ATT&CK        — маппинг на фреймворк
├── Agents               — список агентов
│   └── Agent details    — детали по агенту
├── Management
│   ├── Rules            — правила детекции
│   ├── Decoders         — декодеры логов
│   └── Configuration    — настройка
└── Dashboards           — визуализация
```

#### Работа с правилами Wazuh

```xml
<!-- Пример: обнаружение brute-force SSH -->
<group name="ssh_brute_force">
  <rule id="100001" level="10" frequency="5" timeframe="60">
    <if_matched_sid>5710</if_matched_sid>
    <description>SSH brute-force: 5+ неудачных входов за 60 секунд</description>
    <mitre>
      <id>T1110</id>
    </mitre>
  </rule>
</group>

<!-- Пример: обнаружение очистки логов Windows -->
<group name="windows_log_clearing">
  <rule id="100002" level="12">
    <if_sid>60103</if_sid>
    <field name="win.system.eventID">1102</field>
    <description>Очистка журнала безопасности Windows</description>
    <mitre>
      <id>T1070.001</id>
    </mitre>
  </rule>
</group>

<!-- Пример: обнаружение Mimikatz по командной строке -->
<group name="mimikatz_detection">
  <rule id="100003" level="14">
    <if_sid>61603</if_sid>
    <field name="win.eventdata.commandLine" type="pcre2">(?i)(mimikatz|sekurlsa|kerberos::list|lsadump::)</field>
    <description>Возможное использование Mimikatz</description>
    <mitre>
      <id>T1003</id>
    </mitre>
  </rule>
</group>
```

#### Создание дашбордов

**Дашборд 1: Обзор безопасности**
- Виджет: Количество алертов по уровням (pie chart)
- Виджет: Топ-10 source IP по количеству алертов (bar chart)
- Виджет: Алерты по времени (line chart)
- Виджет: Топ-10 правил, сработавших чаще всего

**Дашборд 2: Аутентификация**
- Виджет: Успешные vs неудачные входы
- Виджет: Входы по времени суток (heatmap)
- Виджет: Топ пользователей с неудачными входами
- Виджет: Входы с необычных IP

**Дашборд 3: Endpoint**
- Виджет: Новые процессы (Sysmon EID 1)
- Виджет: Сетевые подключения (Sysmon EID 3)
- Виджет: Изменения реестра (Sysmon EID 12/13)
- Виджет: Агенты — статус подключения

**Дашборд 4: Сеть**
- Виджет: Suricata алерты по категориям
- Виджет: Топ source/destination IP
- Виджет: Протоколы по объёму трафика
- Виджет: Подозрительные DNS-запросы

**Дашборд 5: MITRE ATT&CK**
- Виджет: Тактики по количеству алертов
- Виджет: Маппинг на матрицу ATT&CK
- Виджет: Топ-10 техник, обнаруженных за неделю

#### Задание недели 5–6

1. Подключить Windows-агент к Wazuh серверу
2. Убедиться, что логи Sysmon поступают в SIEM
3. Добавить 3 кастомных правила (brute-force, очистка логов, подозрительный процесс)
4. Создать 5 дашбордов (по описанию выше)
5. Сгенерировать тестовые алерты:
   - Неудачные SSH-входы: `for i in $(seq 1 10); do ssh baduser@192.168.56.10; done`
   - Подозрительный процесс на Windows: запустить PowerShell с `-EncodedCommand`
6. Убедиться, что алерты видны на дашбордах

---

### Неделя 7–8: Sigma-правила

#### Что такое Sigma

Sigma — универсальный формат описания правил детекции для SIEM-систем. Написали один раз — конвертировали в Wazuh, Splunk, ELK, QRadar, ArcSight.

**Аналогии:**
- YARA — для файлов
- Snort/Suricata — для трафика
- **Sigma — для логов**

#### Структура Sigma-правила

```yaml
title: Обнаружение Encoded PowerShell команды
id: f7b4e5a1-2c3d-4e5f-6a7b-8c9d0e1f2a3b
status: stable
description: >
    Обнаруживает выполнение PowerShell с параметром -EncodedCommand,
    что часто используется для обфускации вредоносных команд.
references:
    - https://attack.mitre.org/techniques/T1059/001/
author: Курс «Кибер-Щит»
date: 2024/01/15
modified: 2024/06/01
tags:
    - attack.execution
    - attack.t1059.001
logsource:
    category: process_creation
    product: windows
detection:
    selection:
        CommandLine|contains:
            - '-EncodedCommand'
            - '-enc'
            - '-ec'
        Image|endswith:
            - '\powershell.exe'
            - '\pwsh.exe'
    condition: selection
falsepositives:
    - Легитимные скрипты администрирования
    - SCCM/Intune развёртывания
level: high
```

#### 10 Sigma-правил для практики

**Правило 1: Brute-force RDP**
```yaml
title: Обнаружение brute-force RDP
id: a1b2c3d4-e5f6-7890-abcd-ef1234567890
status: test
description: Множественные неудачные входы по RDP за короткий период.
logsource:
    product: windows
    service: security
detection:
    selection:
        EventID: 4625
        LogonType: 10
    condition: selection | count(TargetUserName) by SourceNetworkAddress > 5
    timeframe: 5m
level: high
tags:
    - attack.credential_access
    - attack.t1110.001
```

**Правило 2: Создание учётной записи**
```yaml
title: Создание новой учётной записи
id: b2c3d4e5-f6a7-8901-bcde-f12345678901
status: stable
description: Обнаружение создания локальной учётной записи.
logsource:
    product: windows
    service: security
detection:
    selection:
        EventID: 4720
    condition: selection
level: medium
tags:
    - attack.persistence
    - attack.t1136.001
```

**Правило 3: Запуск PsExec**
```yaml
title: Обнаружение PsExec
id: c3d4e5f6-a7b8-9012-cdef-123456789012
status: stable
description: Обнаружение инструмента PsExec по имени пайпа или сервиса.
logsource:
    category: process_creation
    product: windows
detection:
    selection_image:
        Image|endswith: '\PsExec.exe'
    selection_service:
        EventID: 7045
        ServiceName: 'PSEXESVC'
    condition: selection_image or selection_service
level: high
tags:
    - attack.lateral_movement
    - attack.t1021.002
```

**Правило 4: Доступ к LSASS**
```yaml
title: Подозрительный доступ к LSASS
id: d4e5f6a7-b8c9-0123-defa-234567890123
status: stable
description: Обнаружение доступа к процессу lsass.exe (credential dumping).
logsource:
    category: process_access
    product: windows
detection:
    selection:
        TargetImage|endswith: '\lsass.exe'
        GrantedAccess|contains:
            - '0x1010'
            - '0x1038'
            - '0x1FFFFF'
    filter_legitimate:
        SourceImage|endswith:
            - '\MsMpEng.exe'
            - '\csrss.exe'
            - '\wininit.exe'
            - '\wmiprvse.exe'
    condition: selection and not filter_legitimate
level: critical
tags:
    - attack.credential_access
    - attack.t1003.001
```

**Правило 5: Использование certutil для загрузки**
```yaml
title: Certutil используется для загрузки файла
id: e5f6a7b8-c9d0-1234-efab-345678901234
status: stable
description: Обнаружение использования certutil для скачивания файлов (LOLBin).
logsource:
    category: process_creation
    product: windows
detection:
    selection:
        Image|endswith: '\certutil.exe'
        CommandLine|contains:
            - 'urlcache'
            - '-urlcache'
            - '/urlcache'
    condition: selection
level: high
tags:
    - attack.command_and_control
    - attack.t1105
```

**Правило 6: Подозрительный PowerShell download**
```yaml
title: PowerShell Web Download
id: f6a7b8c9-d0e1-2345-fabc-456789012345
status: stable
description: Обнаружение загрузки через PowerShell (IWR, WebClient).
logsource:
    category: process_creation
    product: windows
detection:
    selection:
        Image|endswith:
            - '\powershell.exe'
            - '\pwsh.exe'
        CommandLine|contains:
            - 'Invoke-WebRequest'
            - 'IWR'
            - 'wget'
            - 'curl'
            - 'Net.WebClient'
            - 'DownloadString'
            - 'DownloadFile'
    condition: selection
level: medium
tags:
    - attack.execution
    - attack.t1059.001
```

**Правило 7: Регистрация нового сервиса**
```yaml
title: Создание нового Windows-сервиса
id: a7b8c9d0-e1f2-3456-abcd-567890123456
status: stable
description: Обнаружение создания нового сервиса (возможная persistence).
logsource:
    product: windows
    service: system
detection:
    selection:
        EventID: 7045
    filter_known:
        ServiceName:
            - 'Wazuh'
            - 'WinDefend'
            - 'MsMpSvc'
    condition: selection and not filter_known
level: medium
tags:
    - attack.persistence
    - attack.t1543.003
```

**Правило 8: Запуск задачи через schtasks**
```yaml
title: Создание запланированной задачи через schtasks
id: b8c9d0e1-f2a3-4567-bcde-678901234567
status: stable
description: Обнаружение создания scheduled task через командную строку.
logsource:
    category: process_creation
    product: windows
detection:
    selection:
        Image|endswith: '\schtasks.exe'
        CommandLine|contains: '/create'
    condition: selection
level: medium
tags:
    - attack.persistence
    - attack.t1053.005
```

**Правило 9: Подозрительный DNS (длинный домен)**
```yaml
title: Подозрительно длинный DNS-запрос
id: c9d0e1f2-a3b4-5678-cdef-789012345678
status: test
description: DNS-запросы с длиной домена > 50 символов могут указывать на DNS tunneling.
logsource:
    category: dns
    product: windows
detection:
    selection:
        EventID: 22
        QueryName|re: '.{50,}'
    condition: selection
level: medium
tags:
    - attack.command_and_control
    - attack.t1071.004
```

**Правило 10: SSH brute-force (Linux)**
```yaml
title: SSH Brute-Force на Linux
id: d0e1f2a3-b4c5-6789-defa-890123456789
status: stable
description: Множественные неудачные SSH-входы.
logsource:
    product: linux
    service: sshd
detection:
    selection:
        - 'Failed password'
        - 'authentication failure'
    condition: selection | count() by src_ip > 10
    timeframe: 5m
level: high
tags:
    - attack.credential_access
    - attack.t1110.001
```

#### Конвертация Sigma-правил

```bash
# Установка sigma-cli
pip install sigma-cli pySigma-backend-elasticsearch pySigma-backend-splunk

# Конвертация в Elasticsearch (ELK/Wazuh)
sigma convert -t elasticsearch -p ecs_windows rule.yml

# Конвертация в Splunk
sigma convert -t splunk rule.yml

# Пакетная конвертация всех правил в папке
sigma convert -t elasticsearch -p ecs_windows rules/*.yml
```

#### Задание недели 7–8

1. Написать 10 Sigma-правил (по шаблонам выше)
2. Конвертировать их в формат Wazuh/ELK
3. Загрузить в свой SIEM
4. Сгенерировать тестовые события для каждого правила
5. Убедиться, что алерты срабатывают
6. Задокументировать: правило → тестовый кейс → скриншот алерта

#### Чеклист навыков месяца 2

- [ ] Навигация по Wazuh Dashboard
- [ ] Создание и настройка агентов
- [ ] Написание кастомных правил Wazuh (XML)
- [ ] Создание дашбордов и визуализаций
- [ ] Формат Sigma-правил — структура, поля, модификаторы
- [ ] Написание 10 Sigma-правил с нуля
- [ ] Конвертация Sigma → ELK / Splunk
- [ ] Тестирование правил на реальных логах
- [ ] Триаж алертов: определение true/false positive

---

## МЕСЯЦ 3: ДЕТЕКЦИЯ АТАК

> **Цель:** знать 20 ключевых техник ATT&CK и уметь их обнаруживать.
> **Время:** 10–15 часов/неделю

---

### Неделя 9–10: Топ-20 техник MITRE ATT&CK для SOC L1

#### Что должен обнаруживать Junior SOC-аналитик

| # | Техника ATT&CK | Тактика | Что искать |
|---|----------------|---------|------------|
| 1 | T1059.001 PowerShell | Execution | Event 4104, Sysmon 1 (powershell.exe), -enc, IEX |
| 2 | T1059.003 cmd.exe | Execution | Sysmon 1, подозрительный родитель (Excel→cmd) |
| 3 | T1003.001 LSASS Dump | Credential Access | Sysmon 10 (target=lsass.exe), Event 4656 |
| 4 | T1110.001 Brute Force | Credential Access | Event 4625 (массовые), fail2ban |
| 5 | T1078 Valid Accounts | Defense Evasion | Event 4624 тип 10 с необычных IP |
| 6 | T1021.002 SMB/Admin Shares | Lateral Movement | Event 5140/5145, Sysmon 3 (port 445) |
| 7 | T1053.005 Scheduled Task | Persistence | Event 4698, Sysmon 1 (schtasks.exe) |
| 8 | T1543.003 Windows Service | Persistence | Event 7045, Sysmon 1 (sc.exe create) |
| 9 | T1547.001 Registry Run Keys | Persistence | Sysmon 12/13 (Run, RunOnce) |
| 10 | T1105 Ingress Tool Transfer | C2 | Sysmon 11 + 3, certutil, bitsadmin |
| 11 | T1071.001 Web Protocols | C2 | Suricata, подозрительный user-agent |
| 12 | T1071.004 DNS | C2 | Sysmon 22 (длинные домены), DNS entropy |
| 13 | T1027 Obfuscated Files | Defense Evasion | Base64 в командной строке, -enc |
| 14 | T1070.001 Clear Logs | Defense Evasion | Event 1102 (Security log cleared) |
| 15 | T1055 Process Injection | Defense Evasion | Sysmon 8 (CreateRemoteThread), 10 |
| 16 | T1218.011 Rundll32 | Defense Evasion | Sysmon 1 (rundll32 с подозрительной DLL) |
| 17 | T1486 Data Encrypted | Impact | Массовое переименование файлов, VSS deletion |
| 18 | T1048 Exfiltration via Alt Protocol | Exfiltration | DNS, ICMP, нестандартные порты |
| 19 | T1136.001 Create Account | Persistence | Event 4720, net user add |
| 20 | T1569.002 Service Execution | Execution | Event 7045 + 4688, PsExec |

#### Детальный разбор: T1003.001 — LSASS Credential Dumping

**Что происходит:**
Атакующий получает доступ к процессу lsass.exe для извлечения паролей, хешей и Kerberos-тикетов из памяти.

**Инструменты атакующих:**
- Mimikatz (sekurlsa::logonpasswords)
- procdump.exe -ma lsass.exe
- comsvcs.dll + MiniDump
- Task Manager → Create dump file

**Как обнаружить:**

```
Sysmon Event ID 10:
  TargetImage: C:\Windows\System32\lsass.exe
  GrantedAccess: 0x1010 или 0x1FFFFF
  SourceImage: НЕ стандартные системные процессы

Windows Event ID 4656:
  ObjectName: \NTDS\lsass
  AccessMask: 0x143A (необычный доступ)
```

**Sigma-правило (уже написали выше — правило 4)**

**PowerShell-скрипт мониторинга:**

```powershell
# Поиск подозрительного доступа к LSASS за последние 24 часа
$start = (Get-Date).AddDays(-1)
Get-WinEvent -LogName "Microsoft-Windows-Sysmon/Operational" -FilterXPath "*[System[EventID=10] and EventData[Data[@Name='TargetImage']='C:\Windows\System32\lsass.exe']]" |
    Where-Object { $_.TimeCreated -ge $start } |
    ForEach-Object {
        $xml = [xml]$_.ToXml()
        [PSCustomObject]@{
            Time       = $_.TimeCreated
            SourcePID  = ($xml.Event.EventData.Data | Where-Object Name -eq 'SourceProcessId').'#text'
            SourceImg  = ($xml.Event.EventData.Data | Where-Object Name -eq 'SourceImage').'#text'
            Access     = ($xml.Event.EventData.Data | Where-Object Name -eq 'GrantedAccess').'#text'
        }
    } | Format-Table -AutoSize
```

**Hardening:**
```powershell
# Включить Credential Guard
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Lsa" /v RunAsPPL /t REG_DWORD /d 1 /f

# Включить Credential Guard через GPO
# Computer Configuration → Administrative Templates → System → Device Guard
# → Turn On Virtualization Based Security → Enabled
# → Credential Guard Configuration → Enabled with UEFI lock
```

#### Детальный разбор: T1110 — Brute Force

**Python-скрипт обнаружения brute-force:**

```python
#!/usr/bin/env python3
"""Обнаружение brute-force атак в логах auth.log (Linux)."""

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta


def parse_auth_log(logfile="/var/log/auth.log"):
    """Парсит auth.log и возвращает неудачные входы."""
    failed_pattern = re.compile(
        r"(\w+\s+\d+\s+[\d:]+)\s+\S+\s+sshd\[\d+\]:\s+"
        r"Failed password for (?:invalid user )?(\S+) from (\S+)"
    )
    
    failed_logins = []
    with open(logfile) as f:
        for line in f:
            match = failed_pattern.search(line)
            if match:
                timestamp_str, user, ip = match.groups()
                failed_logins.append({
                    "timestamp": timestamp_str,
                    "user": user,
                    "ip": ip
                })
    return failed_logins


def detect_brute_force(failed_logins, threshold=5, window_minutes=5):
    """Обнаруживает IP с threshold+ неудачных входов за window минут."""
    ip_counter = Counter(entry["ip"] for entry in failed_logins)
    
    alerts = []
    for ip, count in ip_counter.most_common():
        if count >= threshold:
            users_targeted = set(
                e["user"] for e in failed_logins if e["ip"] == ip
            )
            alerts.append({
                "ip": ip,
                "total_failures": count,
                "users_targeted": list(users_targeted),
                "severity": "CRITICAL" if count > 50 else "HIGH"
            })
    return alerts


if __name__ == "__main__":
    logins = parse_auth_log()
    print(f"[*] Всего неудачных входов: {len(logins)}")
    
    alerts = detect_brute_force(logins, threshold=5)
    if alerts:
        print(f"\n[!] ОБНАРУЖЕНО {len(alerts)} brute-force атак:\n")
        for alert in alerts:
            print(f"  IP: {alert['ip']}")
            print(f"  Попыток: {alert['total_failures']}")
            print(f"  Пользователи: {', '.join(alert['users_targeted'])}")
            print(f"  Критичность: {alert['severity']}")
            print()
    else:
        print("\n[+] Brute-force атак не обнаружено.")
```

---

### Неделя 11–12: Suricata IDS + Network Analysis

#### Правила Suricata для SOC

```bash
# Обнаружение Nmap SYN-скана
alert tcp any any -> $HOME_NET any (msg:"SCAN Nmap SYN scan detected"; \
    flags:S; threshold:type threshold, track by_src, count 100, seconds 10; \
    classtype:attempted-recon; sid:1000001; rev:1;)

# Обнаружение DNS tunneling (длинные запросы)
alert dns any any -> any any (msg:"POSSIBLE DNS Tunneling - Long query"; \
    dns.query; content:"."; pcre:"/^.{50,}/"; \
    classtype:bad-unknown; sid:1000002; rev:1;)

# Обнаружение подозрительного User-Agent
alert http any any -> $EXTERNAL_NET any (msg:"SUSPICIOUS User-Agent - curl"; \
    http.user_agent; content:"curl/"; \
    classtype:bad-unknown; sid:1000003; rev:1;)

# Обнаружение передачи больших объёмов данных (>50MB)
alert tcp $HOME_NET any -> $EXTERNAL_NET any (msg:"POSSIBLE Data Exfiltration - Large transfer"; \
    flow:established,to_server; \
    threshold:type threshold, track by_src, count 50000000, seconds 3600; \
    classtype:bad-unknown; sid:1000004; rev:1;)

# Обнаружение reverse shell (bash)
alert tcp $HOME_NET any -> $EXTERNAL_NET any (msg:"POSSIBLE Reverse Shell - bash"; \
    flow:established; content:"/bin/bash"; \
    classtype:trojan-activity; sid:1000005; rev:1;)
```

#### Wireshark: ключевые фильтры для SOC

```
# Поиск подозрительных DNS-запросов
dns.qry.name contains ".xyz" || dns.qry.name contains ".top" || dns.qry.name contains ".tk"

# HTTP POST (возможная эксфильтрация)
http.request.method == "POST"

# Нестандартные порты
tcp.dstport > 1024 && tcp.dstport != 3389 && tcp.dstport != 8080 && tcp.dstport != 8443

# SMB-трафик (lateral movement)
smb || smb2

# Kerberos (Kerberoasting)
kerberos.msg_type == 13 && kerberos.cipher == 23

# TLS handshake (для анализа JA3)
tls.handshake.type == 1
```

#### Задание недели 11–12

1. Добавить 5 кастомных правил Suricata (по шаблонам выше)
2. Протестировать правила:
   - Запустить Nmap-скан на свою сеть
   - Сделать длинный DNS-запрос
   - Передать файл через HTTP POST
3. Скачать PCAP с malware-traffic-analysis.net
4. Проанализировать в Wireshark:
   - Найти C2-трафик
   - Определить IoC (IP, домены, хеши)
   - Составить таймлайн
5. Написать отчёт об инциденте (по PCAP)

#### Чеклист навыков месяца 3

- [ ] Перечислить и объяснить топ-20 техник ATT&CK
- [ ] Для каждой техники: назвать Event ID, Sysmon ID, инструмент детекции
- [ ] Написать Sigma-правило для любой из 20 техник
- [ ] Работать с Suricata: правила, логи, тюнинг
- [ ] Анализировать PCAP в Wireshark: DNS, HTTP, SMB, Kerberos
- [ ] Находить C2-трафик в дампе
- [ ] Определять IoC из сетевого трафика
- [ ] Различать true positive и false positive

---

## МЕСЯЦ 4: INCIDENT RESPONSE

> **Цель:** уметь реагировать на инцидент по стандарту NIST.
> **Время:** 10–15 часов/неделю

---

### Неделя 13–14: IR Process (NIST SP 800-61)

#### 4 фазы Incident Response

```
┌─────────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────────┐
│ Preparation │───→│  Detection &  │───→│ Containment, │───→│  Post-       │
│ Подготовка  │    │  Analysis     │    │ Eradication, │    │  Incident    │
│             │    │  Обнаружение  │    │  Recovery    │    │  Activity    │
│             │    │  и анализ     │    │  Сдерживание │    │  Разбор      │
└─────────────┘    └───────────────┘    └──────────────┘    └──────────────┘
```

**Фаза 1: Preparation (Подготовка)**
- IR Policy (политика реагирования)
- IR Team (команда, контакты, роли)
- Инструменты (SIEM, EDR, форензика)
- Playbooks (плейбуки на типовые инциденты)
- Тренировки (табличные учения, purple team)

**Фаза 2: Detection & Analysis (Обнаружение и анализ)**
- Источники: SIEM алерты, EDR, IDS, обращения пользователей, TI
- Триаж: severity (P1–P4), true/false positive
- Анализ: scope (масштаб), IoC, таймлайн
- Документирование: всё в тикет

**Фаза 3: Containment, Eradication, Recovery**
- **Containment (сдерживание):**
  - Изоляция хоста (отключить от сети, но не выключать!)
  - Блокировка IP/домена на файрволе
  - Отключение скомпрометированных учёток
  - Сохранение улик (memory dump, disk image)

- **Eradication (устранение):**
  - Удаление малвари
  - Сброс паролей
  - Закрытие уязвимости / вектора атаки
  - Удаление persistence-механизмов

- **Recovery (восстановление):**
  - Переустановка ОС или восстановление из бэкапа
  - Мониторинг 72 часа после восстановления
  - Проверка работоспособности сервисов
  - Постепенное снятие ограничений

**Фаза 4: Post-Incident Activity (Разбор)**
- Lessons learned meeting (что пошло не так, что улучшить)
- Итоговый отчёт (таймлайн, IoC, причина, рекомендации)
- Обновление плейбуков и правил детекции
- Дообучение персонала

#### Шаблон тикета инцидента

```
ИНЦИДЕНТ #[номер]

Дата обнаружения: YYYY-MM-DD HH:MM UTC
Обнаружил: [кто/что — SIEM алерт / пользователь / EDR]
Приоритет: P1 / P2 / P3 / P4
Статус: Open / Investigating / Contained / Resolved / Closed

ОПИСАНИЕ:
[Краткое описание — что произошло]

ЗАТРОНУТЫЕ АКТИВЫ:
- Хост: [hostname / IP]
- Пользователь: [username]
- Сервис: [если применимо]

IoC:
- IP: [список]
- Домены: [список]
- Хеши: [MD5/SHA256]
- Файлы: [пути]

ТАЙМЛАЙН:
| Время (UTC) | Событие |
|-------------|---------|
| HH:MM       | [событие] |

ДЕЙСТВИЯ:
| Время | Действие | Кто |
|-------|----------|-----|
| HH:MM | [действие] | [аналитик] |

РЕШЕНИЕ:
[Как инцидент был закрыт]

РЕКОМЕНДАЦИИ:
[Что нужно сделать, чтобы не повторилось]
```

#### Практика: разбор инцидента

**Сценарий: Фишинг → Malware → Lateral Movement**

```
08:15 — Пользователь ivan.petrov открыл фишинговое письмо
08:16 — Скачан файл invoice_2024.docm
08:17 — Word запустил макрос → PowerShell с -EncodedCommand
08:18 — PowerShell скачал payload.exe с attacker.com
08:19 — payload.exe → Sysmon EID 1, EID 3 (conn to 185.x.x.x:443)
08:25 — Mimikatz → Sysmon EID 10 (lsass.exe access)
08:30 — PsExec lateral movement → Event 5140, 4624(type 3)
08:35 — Алерт SIEM: "Подозрительный доступ к LSASS"
```

**Задание:** проведите расследование:
1. Определите первоначальный вектор
2. Составьте полный таймлайн
3. Перечислите IoC (IP, домены, хеши, файлы)
4. Опишите действия по containment
5. Определите, какие системы скомпрометированы
6. Напишите рекомендации по предотвращению

---

### Неделя 15–16: Форензика (основы)

#### Volatility: анализ дампа памяти

```bash
# Установка Volatility 3
pip install volatility3

# Определить профиль ОС
vol -f memory.dmp windows.info

# Список процессов
vol -f memory.dmp windows.pslist
vol -f memory.dmp windows.psscan    # Включая скрытые

# Сетевые подключения
vol -f memory.dmp windows.netscan

# Поиск инъекций в процессы
vol -f memory.dmp windows.malfind

# Командная строка процессов
vol -f memory.dmp windows.cmdline

# DLL-библиотеки процесса
vol -f memory.dmp windows.dlllist --pid <PID>

# Дерево процессов
vol -f memory.dmp windows.pstree

# Хеши процессов
vol -f memory.dmp windows.hashdump

# Дамп процесса
vol -f memory.dmp windows.pslist --pid <PID> --dump
```

#### Timeline Analysis

```bash
# Создание timeline с помощью plaso (log2timeline)
log2timeline.py timeline.plaso /path/to/evidence/

# Конвертация в CSV
psort.py -o l2tcsv timeline.plaso -w timeline.csv

# Фильтрация по дате
psort.py -o l2tcsv timeline.plaso "date > '2024-01-15' AND date < '2024-01-16'" -w filtered.csv
```

#### Windows артефакты для форензики

| Артефакт | Путь | Что даёт |
|----------|------|----------|
| Prefetch | C:\Windows\Prefetch\ | Запускавшиеся программы + время |
| Amcache | C:\Windows\AppCompat\Programs\Amcache.hve | Установка программ, хеши |
| ShimCache | SYSTEM hive | Выполнявшиеся программы |
| Recent Files | %USERPROFILE%\AppData\Roaming\Microsoft\Windows\Recent | Открытые файлы |
| PowerShell History | %USERPROFILE%\AppData\Roaming\Microsoft\Windows\PowerShell\PSReadLine | Команды PowerShell |
| Event Logs | C:\Windows\System32\winevt\Logs\ | Все события |
| Registry Hives | C:\Windows\System32\config\ | SAM, SYSTEM, SOFTWARE, SECURITY |
| Browser History | %USERPROFILE%\AppData\Local\... | Посещённые сайты |
| $MFT | Корень NTFS | Все файлы, включая удалённые |

#### Задание недели 15–16

1. Скачать учебный дамп памяти (MemLabs, Volatility samples)
2. Анализ в Volatility:
   - Найти подозрительные процессы (pstree, malfind)
   - Извлечь сетевые подключения (netscan)
   - Найти инъекции (malfind)
   - Извлечь командные строки (cmdline)
3. Составить отчёт с таймлайном и IoC
4. Изучить Windows-артефакты на своей лаборатории

#### Чеклист навыков месяца 4

- [ ] Описать 4 фазы NIST IR
- [ ] Заполнить тикет инцидента по шаблону
- [ ] Провести containment: изоляция хоста, блокировка IP, отключение учётки
- [ ] Работать с Volatility: pslist, netscan, malfind, cmdline
- [ ] Анализировать Windows-артефакты: Prefetch, Amcache, Event Logs
- [ ] Составить таймлайн инцидента
- [ ] Написать полный IR-отчёт
- [ ] Расследовать 3 учебных инцидента от начала до конца

---

## МЕСЯЦ 5: ПОРТФОЛИО И ТРУДОУСТРОЙСТВО

> **Цель:** собрать портфолио и подготовиться к собеседованию.
> **Время:** 10–15 часов/неделю

---

### Неделя 17–18: Сборка портфолио

#### Что должно быть в портфолио Junior SOC-аналитика

**1. 5 разобранных инцидентов** (можно на учебных данных):
- Фишинг → компрометация учётной записи
- Brute-force атака на SSH / RDP
- Malware: заражение через macro в документе
- Lateral movement в корпоративной сети
- Ransomware / cryptominer

Каждый инцидент — отдельный документ:
```
Инцидент: [название]
Дата расследования: [дата]

1. Обнаружение: как был обнаружен (какой алерт)
2. Анализ: что произошло (с таймлайном)
3. IoC: IP, домены, хеши, файлы
4. Containment: что было сделано
5. Recommendations: как предотвратить
6. Приложения: скриншоты SIEM, Wireshark, Volatility
```

**2. Набор Sigma-правил (10–20 штук):**
- С описанием, тестовыми кейсами
- Выложить на GitHub
- README с описанием каждого правила

**3. Скрипты автоматизации (3–5 штук):**
- Python: анализ логов
- Python: парсинг PCAP
- Bash: мониторинг системы
- PowerShell: сбор артефактов

**4. Лаборатория SOC:**
- Описание архитектуры
- Конфигурация Wazuh, Sysmon, Suricata
- Скриншоты дашбордов

**5. Блог / заметки (опционально):**
- Writeups CTF-задач
- Разборы CVE
- Заметки по инструментам

#### Структура GitHub-репозитория

```
soc-portfolio/
├── README.md                    # Обзор портфолио
├── incidents/
│   ├── 01-phishing-compromise/
│   │   ├── report.md
│   │   └── screenshots/
│   ├── 02-ssh-bruteforce/
│   ├── 03-macro-malware/
│   ├── 04-lateral-movement/
│   └── 05-ransomware/
├── sigma-rules/
│   ├── README.md
│   ├── brute-force-rdp.yml
│   ├── lsass-access.yml
│   ├── encoded-powershell.yml
│   └── ...
├── scripts/
│   ├── log-analyzer.py
│   ├── pcap-parser.py
│   ├── sysmon-hunter.ps1
│   └── linux-triage.sh
├── lab-setup/
│   ├── architecture.md
│   ├── wazuh-config/
│   ├── sysmon-config/
│   └── suricata-rules/
└── certifications/
    └── study-notes/
```

---

### Неделя 19–20: Резюме и собеседование

#### Площадки для поиска работы SOC-аналитика

| Площадка | URL | Особенности |
|----------|-----|-------------|
| hh.ru | hh.ru | Основная в РФ, фильтр «информационная безопасность» |
| Хабр Карьера | career.habr.com | IT-специфика, зарплаты |
| LinkedIn | linkedin.com | Международные компании, SOC в РФ и удалёнка |
| Telegram | @cybersec_jobs, @infosec_work | Быстрые вакансии, стартапы |
| Карьерные страницы | BI.ZONE, Positive Technologies, Solar, Kaspersky | Прямые вакансии |

#### Ключевые слова для поиска

```
"SOC analyst" OR "SOC-аналитик" OR "аналитик SOC"
"специалист мониторинга" OR "инженер мониторинга ИБ"
"L1 analyst" OR "junior security analyst"
"специалист по информационной безопасности" (фильтр: мониторинг)
```

#### Mock-собеседование

**Типичная структура интервью SOC L1:**
1. Расскажите о себе (2 мин)
2. Технические вопросы (20 мин) — см. [Топ-50 вопросов](#топ-50-вопросов-собеседования-soc-l1)
3. Практическое задание (15 мин) — анализ алерта / лога
4. Ситуационные вопросы (10 мин) — «что вы будете делать, если...»
5. Ваши вопросы (5 мин)

**Практическое задание на собеседовании (пример):**

```
Вам показывают алерт из SIEM:

Правило: "Multiple failed login attempts from single IP"
Severity: High
Source IP: 203.0.113.42
Target: DC01 (Domain Controller)
User: Administrator
Event ID: 4625
Count: 150 за 10 минут
Time: 03:17 UTC (ночь)

Вопросы:
1. Это true positive или false positive? Почему?
2. Какую дополнительную информацию вы запросите?
3. Какие действия предпримете?
4. Какой приоритет присвоите?
```

**Ожидаемый ответ:**
1. Скорее всего true positive: 150 неудачных входов в DC за 10 минут ночью — нехарактерно
2. Запрошу: геолокацию IP, историю этого IP в SIEM, успешные входы после серии неудачных (4624), есть ли этот IP в TI
3. Действия: проверить TI по IP, если подозрителен — заблокировать на файрволе, проверить, не было ли успешного входа, эскалировать на L2
4. Приоритет: P2 (High) — атака на DC, но пока нет признаков компрометации

---

## МЕСЯЦ 6: СЕРТИФИКАЦИЯ (опционально)

> **Цель:** получить признанную сертификацию для усиления резюме.
> **Время:** 10–15 часов/неделю

---

### CompTIA Security+ (SY0-701)

**Что даёт:** базовая международная сертификация по ИБ, признаётся во всём мире.

**Домены экзамена:**
1. General Security Concepts (12%)
2. Threats, Vulnerabilities, and Mitigations (22%)
3. Security Architecture (18%)
4. Security Operations (28%)
5. Security Program Management & Oversight (20%)

**Ресурсы для подготовки:**
- Professor Messer (YouTube, бесплатно)
- CompTIA CertMaster (платно)
- Jason Dion Practice Tests (Udemy)
- Книга: CompTIA Security+ Get Certified Get Ahead (Darril Gibson)

**Стоимость экзамена:** ~$400 (можно найти ваучеры со скидкой)

### CompTIA CySA+ (CS0-003)

**Что даёт:** сертификация для SOC-аналитиков, следующий уровень после Security+.

**Домены:**
1. Security Operations (33%)
2. Vulnerability Management (30%)
3. Incident Response & Management (20%)
4. Reporting & Communication (17%)

### Бесплатные платформы для практики

| Платформа | Фокус | Цена |
|-----------|-------|------|
| **TryHackMe** | SOC Level 1 path, Blue Team | Бесплатно + Premium $10/мес |
| **LetsDefend** | SOC Analyst path, алерты, IR | Бесплатно + Premium |
| **CyberDefenders** | Forensics challenges | Бесплатно |
| **Blue Team Labs Online** | IR, forensics, SIEM | Бесплатно + Premium |
| **HackTheBox** | Defensive tracks (Sherlocks) | Бесплатно + VIP |
| **RangeForce** | SOC, IR | По подписке |

### План подготовки к Security+ (8 недель)

| Неделя | Тема | Часов |
|--------|------|-------|
| 1 | General Security Concepts | 10 |
| 2 | Threats & Vulnerabilities (часть 1) | 12 |
| 3 | Threats & Vulnerabilities (часть 2) | 12 |
| 4 | Security Architecture | 10 |
| 5 | Security Operations (часть 1) | 12 |
| 6 | Security Operations (часть 2) | 12 |
| 7 | Program Management & Oversight | 10 |
| 8 | Пробные экзамены + повторение | 15 |

---

## ТОП-50 ВОПРОСОВ СОБЕСЕДОВАНИЯ SOC L1

> С развёрнутыми ответами. Используйте для подготовки к интервью.

---

### Основы сетей и протоколов

**1. Что такое модель OSI? Назовите 7 уровней.**

Модель OSI — эталонная модель взаимодействия открытых систем. 7 уровней (снизу вверх):
1. Physical (физический) — кабели, сигналы
2. Data Link (канальный) — MAC-адреса, Ethernet, ARP
3. Network (сетевой) — IP-адресация, маршрутизация
4. Transport (транспортный) — TCP/UDP, порты
5. Session (сеансовый) — управление сессиями
6. Presentation (представления) — шифрование, сжатие
7. Application (прикладной) — HTTP, DNS, SMTP

**2. В чём разница между TCP и UDP?**

TCP — надёжный протокол с установлением соединения (трёхстороннее рукопожатие), гарантирует доставку и порядок пакетов. Используется для HTTP, SSH, SMTP.
UDP — ненадёжный, без установления соединения, быстрее, но без гарантий доставки. Используется для DNS, VoIP, стриминга.

**3. Что такое трёхстороннее рукопожатие TCP?**

SYN → SYN-ACK → ACK. Клиент посылает SYN, сервер отвечает SYN-ACK, клиент подтверждает ACK. После этого соединение установлено.

**4. Какие порты должен знать SOC-аналитик?**

22 (SSH), 25 (SMTP), 53 (DNS), 80 (HTTP), 88 (Kerberos), 135 (RPC), 139/445 (SMB), 443 (HTTPS), 636 (LDAPS), 1433 (MSSQL), 3306 (MySQL), 3389 (RDP), 5985/5986 (WinRM).

**5. Что такое DNS и как его могут использовать злоумышленники?**

DNS — система преобразования доменных имён в IP-адреса. Злоумышленники используют DNS для: tunneling (C2-канал через DNS-запросы), DGA (генерация доменов для C2), DNS spoofing, exfiltration через TXT-записи.

### Windows Security

**6. Что такое Event ID 4624 и какие типы входа вы знаете?**

Event ID 4624 — успешный вход. Типы: 2 (Interactive — локальный), 3 (Network — сетевой, SMB), 4 (Batch), 5 (Service), 7 (Unlock), 10 (RemoteInteractive — RDP). Для SOC важны: тип 3 (lateral movement), тип 10 (RDP).

**7. Что такое Event ID 4625?**

Неудачная попытка входа. Массовые события 4625 от одного IP — признак brute-force. Важно смотреть: SubStatus code (причина отказа), TargetUserName, источник.

**8. Как обнаружить очистку логов?**

Event ID 1102 (Security log cleared) + Sysmon EID 23 (файл удалён, если логи удалены физически). Централизованный сбор логов в SIEM предотвращает потерю при очистке на хосте.

**9. Что такое Sysmon и зачем он нужен?**

Sysmon — бесплатный системный монитор от Sysinternals (Microsoft). Расширяет стандартное логирование Windows: записывает создание процессов с полной командной строкой и хешами (EID 1), сетевые подключения (EID 3), загрузку драйверов (EID 6), доступ к процессам (EID 10), изменения реестра (EID 12/13), DNS-запросы (EID 22).

**10. Как обнаружить Pass-the-Hash?**

Event ID 4624 с LogonType 3 (Network) и NTLM-аутентификацией (не Kerberos). Подозрительно, если NTLM используется для входа на сервер, где обычно используется Kerberos. Sigma-правило проверяет LogonType=3 + AuthenticationPackageName=NTLM + привилегированный аккаунт.

### SIEM и детекция

**11. Что такое SIEM?**

Security Information and Event Management — система сбора, нормализации, корреляции и анализа событий безопасности из множества источников. Основные функции: централизованный сбор логов, правила корреляции, алерты, дашборды, хранение для расследований.

**12. Что такое Sigma-правило?**

Sigma — универсальный формат описания правил детекции для SIEM. Аналог Snort/Suricata для сетевого трафика, но для логов. Можно написать одно правило и конвертировать в формат любого SIEM (Splunk, ELK, QRadar).

**13. Как отличить true positive от false positive?**

Проверить контекст: кто (легитимный пользователь или подозрительный?), когда (рабочее время?), откуда (обычный IP?), что именно (штатная операция?). Проверить TI по IoC. Сопоставить с другими алертами. Если сомневаетесь — эскалируйте.

**14. Что такое IoC?**

Indicators of Compromise — артефакты, указывающие на компрометацию: IP-адреса C2-серверов, хеши вредоносных файлов, домены, URL, паттерны в реестре, подозрительные email-адреса. Источники: TI feeds (OTX, MISP, AbuseIPDB), отчёты вендоров, собственные расследования.

**15. Назовите 5 источников Threat Intelligence.**

AlienVault OTX (бесплатный), MISP (open source), AbuseIPDB (IP reputation), VirusTotal (хеши, URL), CIRCL (CERT Luxembourg). Российские: НКЦКИ, Kaspersky TI, Positive Technologies.

### Incident Response

**16. Назовите 4 фазы NIST Incident Response.**

1. Preparation (подготовка)
2. Detection & Analysis (обнаружение и анализ)
3. Containment, Eradication & Recovery (сдерживание, устранение, восстановление)
4. Post-Incident Activity (разбор после инцидента)

**17. Что вы сделаете первым делом при обнаружении инцидента?**

Не паниковать. Задокументировать (время, что обнаружено, как). Определить масштаб (сколько хостов, какие данные). НЕ выключать заражённую машину (потеряете данные из RAM). Изолировать хост от сети. Уведомить руководителя / L2. Начать сбор улик.

**18. Почему нельзя сразу выключать заражённый компьютер?**

В оперативной памяти находятся: ключи шифрования (могут понадобиться для восстановления), C2-соединения (адреса серверов), запущенные процессы и их командные строки, расшифрованный malware payload. При выключении всё это теряется. Вместо этого — снять дамп памяти, затем изолировать от сети.

**19. Что такое containment и какие типы бывают?**

Containment — сдерживание инцидента, чтобы он не распространился. Типы: Short-term (быстрое — отключить от сети, заблокировать IP, отключить учётку) и Long-term (подготовка к eradication — создать forensic image, настроить мониторинг, подготовить чистую среду).

**20. Как составить таймлайн инцидента?**

Собрать события из всех источников: SIEM (алерты), Event Logs (Windows), auth.log (Linux), Sysmon, Suricata, proxy logs, DNS logs, email logs. Привести к UTC. Отсортировать хронологически. Пометить ключевые точки: initial access, execution, persistence, lateral movement, exfiltration.

### Форензика

**21. Что такое Volatility?**

Open source фреймворк для анализа дампов оперативной памяти. Позволяет: извлекать список процессов (pslist/psscan), сетевые подключения (netscan), инъекции в процессы (malfind), командные строки (cmdline), DLL и хендлы.

**22. Какие артефакты Windows вы знаете?**

Prefetch (запускавшиеся программы), Amcache (установка ПО), ShimCache (AppCompatCache), Event Logs, PowerShell History, $MFT (файловая система), Registry Hives (SAM, SYSTEM, SOFTWARE), Browser History, Recent Files, UserAssist.

**23. Что такое Prefetch?**

Windows Prefetch — механизм ускорения запуска программ. Файлы .pf в C:\Windows\Prefetch\ содержат: имя программы, время последних 8 запусков, количество запусков, список файлов/DLL, которые программа загружала. Для форензики — доказательство выполнения программы.

**24. Как извлечь данные из дампа памяти?**

С помощью Volatility: `vol -f dump.raw windows.pslist` (процессы), `windows.netscan` (сеть), `windows.malfind` (инъекции), `windows.cmdline` (командные строки), `windows.hashdump` (хеши паролей), `windows.filescan` + `windows.dumpfiles` (извлечение файлов).

**25. Что такое Chain of Custody?**

Цепочка хранения улик — документирование каждого человека, который имел доступ к цифровым уликам: кто собрал, когда, как передал, кому, где хранилось. Нужна для юридической значимости улик. Каждое действие с уликой — запись в журнал.

### Инструменты

**26. Какие фильтры Wireshark вы используете чаще всего?**

`tcp.port == 445` (SMB), `dns.qry.name contains "suspicious"`, `http.request.method == "POST"`, `ip.addr == 10.0.0.1`, `tcp.flags.syn == 1 && tcp.flags.ack == 0` (SYN scan), `tls.handshake.type == 1` (TLS Client Hello), `frame contains "password"`.

**27. Что такое Suricata?**

Open source IDS/IPS (Intrusion Detection/Prevention System). Анализирует сетевой трафик в реальном времени и сравнивает с правилами (signatures). Может работать в режиме IDS (обнаружение) или IPS (блокировка). Поддерживает Emerging Threats и собственные правила.

**28. Что такое YARA?**

Инструмент для идентификации и классификации malware по паттернам (строки, hex, regex). YARA-правило описывает характеристики файла: строки внутри, размер, заголовки. Используется: при сканировании файловых систем, в sandbox, в SIEM.

**29. Какие инструменты для Threat Hunting вы знаете?**

SIEM (Wazuh, Splunk, ELK) для корреляции, Sigma для правил, Sysmon для endpoint, Wireshark/Zeek для сети, Volatility для memory, MITRE ATT&CK Navigator для маппинга, Jupyter Notebooks для data analysis, OSQuery для endpoint queries.

**30. Что такое ELK Stack?**

Elasticsearch (поиск и хранение), Logstash (сбор и обработка логов), Kibana (визуализация). Используется как SIEM-платформа. Wazuh работает поверх ELK, добавляя правила детекции и агенты.

### MITRE ATT&CK

**31. Что такое MITRE ATT&CK?**

База знаний о тактиках и техниках кибератак, основанная на реальных наблюдениях. Структура: Tactics (что атакующий хочет достичь) → Techniques (как это делает) → Sub-techniques (конкретные методы). 14 тактик: от Reconnaissance до Impact.

**32. Назовите 5 тактик ATT&CK.**

Initial Access (первоначальный доступ), Execution (выполнение), Persistence (закрепление), Privilege Escalation (повышение привилегий), Defense Evasion (обход защиты), Credential Access (доступ к учётным данным), Lateral Movement (горизонтальное перемещение).

**33. Как SOC использует ATT&CK?**

Для: маппинга алертов на техники (понимание контекста), оценки покрытия детекции (какие техники мы видим, какие нет), threat hunting (формулировка гипотез), gap analysis (какие правила нужно написать), отчётов (единый язык с командой).

### Ситуационные вопросы

**34. Вы видите 1000 алертов «failed login» с одного IP за 5 минут. Ваши действия?**

Проверить IP в TI (VirusTotal, AbuseIPDB). Проверить геолокацию. Проверить, есть ли успешные входы (4624) после серии неудачных. Если IP внешний и подозрительный — заблокировать на файрволе и эскалировать. Если внутренний — проверить, не сломался ли скрипт или сервис.

**35. Пользователь сообщил, что получил подозрительное письмо. Что делать?**

Попросить НЕ открывать вложения и не переходить по ссылкам. Получить заголовки письма (full headers). Проверить: отправителя (SPF/DKIM/DMARC), ссылки (hover, не кликать), вложения (хеш → VirusTotal). Если фишинг — заблокировать отправителя, проверить, кто ещё получил это письмо, удалить из всех ящиков.

**36. На дашборде появился алерт «Suspicious PowerShell execution» в 3 часа ночи. Ваши действия?**

PowerShell ночью — подозрительно. Проверить: какой пользователь, какой хост, полная командная строка (Sysmon EID 1 или Event 4104). Если -EncodedCommand — декодировать Base64. Проверить родительский процесс (Word → PowerShell — плохо). Проверить сетевые подключения от этого хоста. Если подозрительно — эскалировать, изолировать хост.

**37. Как вы будете расследовать инцидент с ransomware?**

Немедленно: изолировать заражённые хосты от сети (НЕ выключать). Определить масштаб (какие хосты, какие данные). Определить вектор заражения (фишинг? RDP? уязвимость?). Собрать IoC: расширение зашифрованных файлов, ransom note, C2-адреса. Проверить наличие бэкапов. НЕ платить выкуп. Привлечь IR-команду. Проверить, не утекли ли данные (double extortion).

**38. Вы обнаружили C2-трафик с внутреннего хоста. Что делать?**

Изолировать хост от сети. Снять дамп памяти. Определить C2-адрес (IP/домен). Заблокировать на файрволе и прокси. Проверить, какие ещё хосты общались с этим C2. Определить, как malware попал на хост (initial access). Проверить, было ли lateral movement. Собрать IoC и отчёт.

**39. Как отличить легитимный PowerShell от вредоносного?**

Признаки вредоносного: -EncodedCommand / -enc, IEX (Invoke-Expression), DownloadString / DownloadFile, bypass ExecutionPolicy, запуск из необычного родителя (Word, Excel, wscript), обфускация (case mixing, concat), ночное время, несистемные пользователи.

**40. Вы нашли алерт, который оказался false positive. Что делать?**

Задокументировать (почему false positive). Оценить, можно ли тюнить правило, чтобы исключить этот сценарий (добавить filter). Если нельзя без потери покрытия — добавить в исключения с обоснованием. Обновить runbook для этого типа алерта.

### Общие и карьерные

**41. Почему вы хотите работать в SOC?**

Мне нравится расследовать инциденты и защищать организации. SOC — передний край обороны, где каждый алерт — потенциальный инцидент. Я люблю работать с данными, видеть паттерны и принимать решения в условиях неопределённости. Это область с высоким спросом и постоянным развитием.

**42. Как вы поддерживаете свои знания в актуальном состоянии?**

Telegram-каналы по ИБ, Twitter/X (security researchers), MITRE ATT&CK updates, бюллетени CERT, TryHackMe/HackTheBox для практики, подкасты (Darknet Diaries), блоги (Krebs on Security), конференции (PHDays, OFFZONE).

**43. Что такое kill chain?**

Cyber Kill Chain (Lockheed Martin) — 7 этапов атаки: Reconnaissance → Weaponization → Delivery → Exploitation → Installation → C2 → Actions on Objectives. Если прервать на любом этапе — атака провалена. SOC работает на этапах 3–7.

**44. В чём разница между IDS и IPS?**

IDS (Intrusion Detection System) — обнаруживает атаки и генерирует алерты, но не блокирует. IPS (Intrusion Prevention System) — обнаруживает И блокирует трафик. Suricata может работать в обоих режимах. IDS безопаснее (не ломает легитимный трафик), IPS эффективнее (блокирует сразу).

**45. Что такое Zero Trust?**

Модель безопасности «никому не доверяй, всё проверяй». Принципы: проверяй каждый запрос, минимум привилегий, считай, что сеть уже скомпрометирована. Практика: MFA, микросегментация, continuous verification, least privilege, assume breach.

**46. Что такое SOAR?**

Security Orchestration, Automation and Response — платформа автоматизации IR. Позволяет: автоматически обогащать алерты (TI lookup), запускать playbooks, изолировать хосты, блокировать IP. Примеры: Cortex XSOAR, Splunk SOAR, Wazuh active response.

**47. Расскажите о работе SOC L1 / L2 / L3.**

L1 (Triage): мониторинг алертов 24/7, первичный анализ, true/false positive, эскалация. L2 (Investigation): глубокий анализ эскалированных инцидентов, форензика, containment. L3 (Hunt/Engineering): threat hunting, разработка правил детекции, тюнинг SIEM, malware analysis.

**48. Что вы знаете о российском законодательстве в сфере ИБ?**

152-ФЗ (о персональных данных), 187-ФЗ (о КИИ), приказы ФСТЭК (21, 17, 31, 239), ГОСТы (ГОСТ Р 57580 — банки). Регуляторы: ФСТЭК, ФСБ, Роскомнадзор (ПДн). НКЦКИ — координация реагирования на инциденты.

**49. Какие метрики SOC вы знаете?**

MTTD (Mean Time To Detect) — среднее время обнаружения. MTTR (Mean Time To Respond) — среднее время реагирования. False positive rate. Alert volume. Escalation rate. Incidents closed per analyst. Coverage (% техник ATT&CK, которые мы детектируем).

**50. Где вы видите себя через 2–3 года?**

Рост в SOC: L1 → L2 → Threat Hunter / IR Specialist / Detection Engineer. Или переход в purple team / DevSecOps. Сертификации: CySA+, GCIH. Цель — стать экспертом, который строит детекцию и расследует сложные инциденты, а не только мониторит алерты.

---

## ШАБЛОН РЕЗЮМЕ JUNIOR SOC-АНАЛИТИКА

```
ИВАНОВ ИВАН ИВАНОВИЧ
SOC-аналитик (L1) | Blue Team

📧 email@example.com | 📱 +7 (9XX) XXX-XX-XX
🔗 LinkedIn: linkedin.com/in/ivanov | GitHub: github.com/ivanov-soc

──────────────────────────────────────────────

ОБЗОР

Junior SOC-аналитик с практическим опытом работы с SIEM (Wazuh/ELK),
IDS (Suricata), Sysmon и Sigma-правилами. Развернул собственную SOC-лабораторию,
провёл расследования 5+ инцидентов (учебных), написал 20+ Sigma-правил детекции.
Готов к работе в режиме 24/7 мониторинга.

──────────────────────────────────────────────

НАВЫКИ

SIEM:          Wazuh, ELK (Elasticsearch, Kibana), базовый Splunk
IDS/IPS:       Suricata (правила, тюнинг)
Endpoint:      Sysmon (настройка, анализ логов), Windows Event Logs
Detection:     Sigma-правила (написание, конвертация)
Network:       Wireshark, tcpdump, базовый Zeek
Forensics:     Volatility (memory analysis), базовый KAPE
Scripting:     Python (анализ логов, парсинг), Bash, PowerShell
Frameworks:    MITRE ATT&CK, NIST SP 800-61, Cyber Kill Chain
ОС:            Linux (Ubuntu, Kali), Windows Server 2019/2022

──────────────────────────────────────────────

ПРОЕКТЫ / ПОРТФОЛИО

SOC-лаборатория (home lab)
• Развернул: Wazuh Server + 2 агента (Windows + Linux) + Suricata IDS
• Настроил сбор логов Sysmon + Windows Event Logs + auth.log
• Создал 5 дашбордов мониторинга
• github.com/ivanov-soc/lab-setup

Набор Sigma-правил (20 правил)
• Написал правила для: brute-force, LSASS access, PsExec, encoded PowerShell,
  DNS tunneling, lateral movement (SMB), log clearing, persistence
• Каждое правило с описанием, тестовым кейсом и маппингом на ATT&CK
• github.com/ivanov-soc/sigma-rules

Расследования инцидентов (5 кейсов)
• Фишинг → компрометация учётки → lateral movement
• SSH brute-force → успешный вход → криптомайнер
• Malware через макрос Word → C2 → credential dumping
• Ransomware: WannaCry-подобный (учебный PCAP)
• Insider: подозрительное копирование на USB
• github.com/ivanov-soc/incident-reports

──────────────────────────────────────────────

ОБРАЗОВАНИЕ

[Ваш вуз], [специальность]                           20XX – 20XX

──────────────────────────────────────────────

СЕРТИФИКАЦИИ (если есть)

• CompTIA Security+ (SY0-701)                         20XX
• TryHackMe: SOC Level 1 (completion badge)           20XX
• LetsDefend: SOC Analyst path                        20XX

──────────────────────────────────────────────

ДОПОЛНИТЕЛЬНО

Языки:     Русский (родной), Английский (B1-B2, чтение документации)
Готовность: Сменный график, удалёнка, переезд
```

**Советы по резюме:**
- Фокус на практику, не теорию: «написал 20 Sigma-правил» > «изучил Sigma»
- GitHub-портфолио = ваш главный аргумент
- Для каждого проекта: что сделали, какие инструменты, результат
- Не врите об опыте — на техническом интервью спросят детали
- Даже учебные проекты — это проекты, если они качественные

---

## ЧЕКЛИСТ ГОТОВНОСТИ К РАБОТЕ

### Hard Skills

| Навык | Уровень | Статус |
|-------|---------|--------|
| TCP/IP, DNS, HTTP — объяснить основы | Базовый | ☐ |
| Wireshark — захват и анализ трафика | Базовый | ☐ |
| Linux — командная строка, логи, процессы | Средний | ☐ |
| Windows Event Logs — ключевые Event ID | Средний | ☐ |
| Sysmon — установка, конфигурация, анализ | Средний | ☐ |
| SIEM (Wazuh/ELK) — навигация, алерты, правила | Средний | ☐ |
| Sigma-правила — формат, написание, конвертация | Средний | ☐ |
| Suricata — установка, правила | Базовый | ☐ |
| MITRE ATT&CK — 20 ключевых техник | Средний | ☐ |
| IR по NIST — 4 фазы, практика | Базовый | ☐ |
| Volatility — анализ дампа памяти | Базовый | ☐ |
| Python — скрипты анализа логов | Базовый | ☐ |
| Триаж алертов — true/false positive | Средний | ☐ |

### Soft Skills

| Навык | Почему важно | Статус |
|-------|-------------|--------|
| Документирование | Всё записывать — основа IR | ☐ |
| Коммуникация | Объяснить инцидент нетехническому человеку | ☐ |
| Работа под давлением | P1-инцидент ночью — нужна холодная голова | ☐ |
| Командная работа | SOC — это команда, не одиночка | ☐ |
| Любопытство | Копать глубже, а не закрывать тикет | ☐ |
| Английский (чтение) | Документация, TI, правила — всё на английском | ☐ |
| Сменный график | Многие SOC работают 24/7 | ☐ |

### Портфолио

| Элемент | Количество | Статус |
|---------|-----------|--------|
| Расследования инцидентов | 5 | ☐ |
| Sigma-правила | 10–20 | ☐ |
| Python/Bash скрипты | 3–5 | ☐ |
| SOC-лаборатория (описание) | 1 | ☐ |
| GitHub-репозиторий | 1 | ☐ |

---

## РЕСУРСЫ ДЛЯ САМООБУЧЕНИЯ

### Бесплатные курсы

| Ресурс | Тема | URL |
|--------|------|-----|
| TryHackMe: SOC Level 1 | Полный путь SOC L1 | tryhackme.com |
| LetsDefend: SOC Analyst | Алерты, расследования | letsdefend.io |
| CyberDefenders | Blue Team challenges | cyberdefenders.org |
| Wazuh Documentation | SIEM | documentation.wazuh.com |
| Sigma Rule Repository | Примеры правил | github.com/SigmaHQ/sigma |
| MITRE ATT&CK | Фреймворк | attack.mitre.org |

### Книги

| Книга | Автор | Тема |
|-------|-------|------|
| Blue Team Handbook | Don Murdoch | SOC operations |
| The Practice of Network Security Monitoring | Richard Bejtlich | NSM |
| Applied Incident Response | Steve Anson | IR |
| Practical Malware Analysis | Sikorski, Honig | Malware basics |
| BTFM (Blue Team Field Manual) | Alan White | Шпаргалка |

### Telegram-каналы (РФ)

| Канал | Тематика |
|-------|----------|
| @cybersachok | Новости ИБ |
| @Hacker_freeworld | Практика, CTF |
| @codeby_sec | Разборы атак |
| @infosec_globe | Общее по ИБ |
| @RuCTF | CTF-соревнования |

### YouTube

| Канал | Тематика |
|-------|----------|
| John Hammond | Blue Team, CTF |
| 13Cubed | DFIR, Forensics |
| Professor Messer | Security+, CySA+ |
| NetworkChuck | Сети, Linux |
| SANS Institute | IR, Forensics |

### Лаборатории

| Платформа | Что практиковать |
|-----------|-----------------|
| TryHackMe | SOC Level 1 path |
| HackTheBox Sherlocks | Blue team challenges |
| LetsDefend | Алерты, триаж, IR |
| Blue Team Labs Online | IR scenarios |
| CyberDefenders | Forensics PCAP |
| Malware Traffic Analysis | PCAP analysis |
| SANS Holiday Hack | Ежегодный CTF |

---

## ДОРОЖНАЯ КАРТА: ОТ L1 ДО SENIOR

```
Год 0–1: SOC Analyst L1
├── Триаж алертов 24/7
├── Первичный анализ инцидентов
├── Документирование
├── Сертификация: Security+
└── Навыки: SIEM, Sigma, Sysmon, Wireshark

Год 1–2: SOC Analyst L2
├── Глубокий анализ инцидентов
├── Forensics (memory, disk, network)
├── Разработка правил детекции
├── Сертификация: CySA+ / GCIH
└── Навыки: Volatility, KAPE, threat hunting basics

Год 2–3: Специализация (выбрать одно)
├── → Threat Hunter
│   ├── Проактивный поиск угроз
│   ├── Data analysis (Jupyter, Python)
│   └── Сертификация: GCIH, GCIA
│
├── → Incident Responder
│   ├── Ведение сложных инцидентов
│   ├── Forensics (advanced)
│   └── Сертификация: GCFE, GCFA
│
├── → Detection Engineer
│   ├── Разработка и тюнинг SIEM
│   ├── SOAR, автоматизация
│   └── Навыки: Python, API, CI/CD для правил
│
└── → Purple Team
    ├── Совместные учения (red + blue)
    ├── Atomic Red Team, Caldera
    └── Сертификация: OSCP + GCIH

Год 3–5: Senior / Lead
├── SOC Team Lead
├── CISO / Head of Security
├── Principal IR Consultant
└── Сертификация: CISSP, CISM
```

---

## ИТОГО: ПЛАН ДЕЙСТВИЙ

| Месяц | Фокус | Результат |
|-------|-------|-----------|
| 1 | Сети + Linux + Windows + Лаборатория | Развёрнут SOC-стенд |
| 2 | SIEM (Wazuh) + Sigma | 10 Sigma-правил, 5 дашбордов |
| 3 | 20 техник ATT&CK + Suricata + Wireshark | Детекция 20 техник |
| 4 | Incident Response + Форензика | 3 расследованных инцидента |
| 5 | Портфолио + Резюме + Собеседование | GitHub-портфолио, отклик на вакансии |
| 6 | Сертификация (Security+) | Сертификат |

**Общее время:** 4–6 месяцев × 10–15 часов/неделю = 160–360 часов

**Вы готовы к собеседованию на Junior SOC Analyst после месяца 4.**
**Месяцы 5–6 — усиление позиции (портфолио + сертификация).**

---

*Этот трек — часть курса «Кибер-Щит». Полный курс (12 недель, 100 модулей защиты, 300 защитных действий) доступен отдельно.*
