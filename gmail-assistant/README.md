# Gmail Assistant (Google Apps Script)

Ассистент принимает команды по почте и обрабатывает документы. Он работает внутри вашего аккаунта Google: не нужны ни проект в Google Cloud, ни `credentials.json`, ни включённый компьютер.

| Тема письма | Что делает |
|---|---|
| `EXTRACT` | Извлекает текст из вложений; ответ — текст в письме и файлы `.txt` |
| `SUMMARY` | Краткое резюме каждого вложения |
| `ORGANIZE [папка]` | Сохраняет вложения в Google Drive: `AI Assistant/<папка или дата>/<тип>` |
| `CONVERT <формат>` | Конвертирует: `pdf`, `docx`, `xlsx`, `pptx`, `txt`, `html`, `csv`, `odt` |
| `HELP` | Справка |

Форматы: PDF (с распознаванием текста), DOC/DOCX/ODT/RTF, XLS/XLSX/ODS, PPT/PPTX/ODP, TXT, CSV, HTML, JSON, MD.

## Установка

1. Включите **Google Apps Script API**: https://script.google.com/home/usersettings
2. Загрузите скрипт в свой аккаунт:
   ```bash
   cd gmail-assistant && npm install
   npx clasp login
   cd apps-script
   npx clasp create --type standalone --title "Gmail Assistant" --rootDir .
   git checkout appsscript.json   # clasp create перезаписывает манифест
   npx clasp push --force
   ```
3. Откройте редактор (`npx clasp open-script`), выберите функцию **setup** → «Выполнить» → «Разрешить».
   `setup` ставит проверку почты раз в минуту и отправляет вам тестовое письмо `HELP`.
4. Проверка с файлами: запустите функцию **selfTest**. Она отправит вам `SUMMARY` с PDF и `CONVERT pdf` с DOCX.

Обновить код после правок: `npx clasp push --force` в папке `apps-script`.

## Настройки

«Настройки проекта» → «Свойства скрипта»:

| Свойство | Назначение |
|---|---|
| `ALLOWED_SENDERS` | Дополнительные адреса через запятую (ваш адрес разрешён всегда) |
| `ANTHROPIC_API_KEY` | Резюме через Claude; без ключа резюме простое (объём и первые предложения) |
| `CLAUDE_MODEL` | По умолчанию `claude-opus-5-5` |
| `DRIVE_ROOT_FOLDER` | По умолчанию `AI Assistant` |

Остановить ассистента: запустите функцию `uninstall`.

## Безопасность

- Команды принимаются только от разрешённых адресов. Кроме того, письмо должно быть отправлено с вашего же аккаунта или пройти проверку DMARC: подделанное поле From не сработает.
- Темы вида `Re: …` командами не считаются, поэтому ассистент не отвечает сам себе.
- С `ANTHROPIC_API_KEY` текст документов для `SUMMARY` отправляется в Anthropic API.
- Для конвертации создаются временные файлы в Google Drive, после обработки они отправляются в корзину.
