# Стартовый комплект к уроку 1.4.4

Набор скриптов для управления ML-сервером. Один скрипт готов и служит образцом, два — заготовки,
которые нужно дописать. Постановка задачи, пошаговый план и критерии приёмки — в `../lesson_1_4_4.md`.

## Состав

| Файл | Состояние | Назначение |
|---|---|---|
| `scripts/sysinfo.sh` | готов, образец стиля | сводка по CPU, памяти, диску, GPU и процессам |
| `scripts/cleanup_checkpoints.sh` | заготовка | оставить N последних чекпоинтов, остальные удалить |
| `scripts/run_training.sh` | заготовка | обёртка запуска обучения: лог, блокировка, сигналы, код возврата |
| `systemd/nightly-retrain.service`, `systemd/nightly-retrain.timer` | образец | ночной запуск обучения по расписанию |
| `tests/test_cleanup_checkpoints.sh` | готов | тесты и одновременно спецификация для `cleanup_checkpoints.sh` |

Требования к системе: bash 4+, GNU coreutils и findutils, `flock` (пакет `util-linux`). Всё это есть
в Ubuntu и Debian, в том числе в WSL2. GPU не обязательна.

## Быстрый старт

```bash
scripts/sysinfo.sh                      # работает сразу
scripts/sysinfo.sh --help
scripts/sysinfo.sh --path /tmp --top 10 --disk-warn 80 --strict

tests/test_cleanup_checkpoints.sh       # на заготовке: 11 тестов из 15 не пройдены, код возврата 1
```

Тесты падают, пока не написана логика в функциях с пометкой `TODO`: заготовка завершается с кодом 99
и сообщением «функция ... не реализована». Четыре теста на разбор аргументов проходят сразу — эта часть
уже готова.

## Что нужно доделать

### 1. `scripts/cleanup_checkpoints.sh`

Разбор и проверка аргументов готовы. Напишите тела функций `list_checkpoints_oldest_first`,
`select_for_deletion`, `remove_checkpoint` и итоговую строку в `main`. Контракт каждой функции описан
в комментарии над ней.

Поведение, которое проверяют тесты:

- остаются `N` самых свежих файлов (по умолчанию 3), подходящих под шаблон (по умолчанию `epoch_*.pt`);
- «свежесть» определяется временем изменения файла, а не номером в имени;
- файлы, не подходящие под шаблон (`best.pt`, логи, конфиги), и вложенные папки не затрагиваются;
- имена файлов и папок могут содержать пробелы;
- `--dry-run` ничего не удаляет;
- в stdout печатаются пути только тех файлов, которые удалены (или были бы удалены при `--dry-run`),
  оставленные файлы поимённо не перечисляются;
- пустая папка и папка, где файлов не больше `N`, — не ошибка, код возврата 0;
- ошибка в аргументах — код возврата 2, при этом ничего не удаляется.

Готово, когда `tests/test_cleanup_checkpoints.sh` сообщает «не пройдено: 0».

### 2. `scripts/run_training.sh`

Разбор аргументов, имена файла лога и файла блокировки, каркас `main` готовы. Напишите тела функций
`acquire_lock`, `on_signal`, `run_and_log`. Требования:

- stdout и stderr команды попадают и в лог `LOG_DIR/NAME_ГГГГММДД_ЧЧММСС.log`, и на экран;
- в начале лога — время старта, хост и команда; в конце — код возврата и длительность;
- скрипт возвращает код возврата обёрнутой команды;
- второй запуск с тем же `--name`, пока идёт первый, сразу завершается с кодом 75 и ничего не запускает;
- по SIGINT и SIGTERM сигнал передаётся команде, в лог пишется отметка о прерывании, код возврата —
  130 или 143 соответственно; после завершения обёртки команда не остаётся работать в фоне.

Готовых тестов для этого скрипта нет: написать `tests/test_run_training.sh` по образцу имеющегося —
часть задания.

### 3. Расписание

Адаптируйте юниты из `systemd/` под свои пути и пользователя и проверьте их:

```bash
systemd-analyze verify systemd/nightly-retrain.service systemd/nightly-retrain.timer
systemd-analyze calendar '*-*-* 02:30:00'
```

`systemd-analyze verify` сообщит, что команда из `ExecStart` не найдена, пока пути в образце не заменены
на существующие. Установка на машине с systemd:

```bash
sudo cp systemd/nightly-retrain.service systemd/nightly-retrain.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nightly-retrain.timer
systemctl list-timers nightly-retrain.timer
sudo systemctl start nightly-retrain.service      # пробный запуск, не дожидаясь расписания
journalctl -u nightly-retrain.service -n 50
```

## Проверка перед сдачей

```bash
for f in scripts/*.sh tests/*.sh; do bash -n "$f"; done
shellcheck scripts/*.sh tests/*.sh
tests/test_cleanup_checkpoints.sh
tests/test_run_training.sh
```

Все четыре команды должны завершаться без замечаний и с кодом возврата 0.
