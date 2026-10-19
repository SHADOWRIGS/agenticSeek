/**
 * Gmail Assistant on Google Apps Script.
 *
 * Runs inside your own Google account: no Cloud project, no credentials.json.
 * Run setup() once from the editor and click "Allow"; after that a trigger checks
 * the inbox every minute and answers commands in the email subject:
 *   EXTRACT | SUMMARY | ORGANIZE [folder] | CONVERT <format> | HELP
 *
 * Optional Script Properties (Project Settings -> Script Properties):
 *   ALLOWED_SENDERS     extra addresses allowed to send commands, comma-separated
 *                       (your own address is always allowed)
 *   ANTHROPIC_API_KEY   enables Claude summaries for SUMMARY; without it the summary is local
 *   CLAUDE_MODEL        defaults to claude-opus-5-5
 *   DRIVE_ROOT_FOLDER   defaults to "AI Assistant"
 */

const COMMANDS = ['EXTRACT', 'SUMMARY', 'ORGANIZE', 'CONVERT', 'HELP'];
const PROCESSED_LABEL = 'AI-Assistant/Processed';
const PROCESSED_KEY = 'processedIds';
const MAX_PROCESSED_IDS = 500;
const MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024;
const MAX_REPLY_ATTACHMENT_BYTES = 20 * 1024 * 1024; // Gmail rejects messages over ~25 MB
const MAX_REPLY_TEXT_CHARS = 20000;

const MIME_BY_EXT = {
  pdf: 'application/pdf',
  doc: 'application/msword',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  odt: 'application/vnd.oasis.opendocument.text',
  rtf: 'application/rtf',
  xls: 'application/vnd.ms-excel',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  ods: 'application/vnd.oasis.opendocument.spreadsheet',
  ppt: 'application/vnd.ms-powerpoint',
  pptx: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
  odp: 'application/vnd.oasis.opendocument.presentation',
  txt: 'text/plain',
  html: 'text/html',
  csv: 'text/csv',
};

const GOOGLE_DOC = 'application/vnd.google-apps.document';
const GOOGLE_SHEET = 'application/vnd.google-apps.spreadsheet';
const GOOGLE_SLIDES = 'application/vnd.google-apps.presentation';

// Which native Google type each file type is imported as (PDF is OCR'd into a Doc).
const IMPORT_AS = {
  pdf: GOOGLE_DOC, doc: GOOGLE_DOC, docx: GOOGLE_DOC, odt: GOOGLE_DOC, rtf: GOOGLE_DOC,
  xls: GOOGLE_SHEET, xlsx: GOOGLE_SHEET, ods: GOOGLE_SHEET,
  ppt: GOOGLE_SLIDES, pptx: GOOGLE_SLIDES, odp: GOOGLE_SLIDES,
};

const EXPORTS = {
  [GOOGLE_DOC]: ['pdf', 'docx', 'txt', 'html', 'odt'],
  [GOOGLE_SHEET]: ['pdf', 'xlsx', 'csv'],
  [GOOGLE_SLIDES]: ['pdf', 'pptx', 'txt'],
};

const TEXT_EXT = ['txt', 'md', 'csv', 'json', 'log', 'xml', 'yaml', 'yml'];

function prop(name, fallback) {
  return PropertiesService.getScriptProperties().getProperty(name) || fallback;
}

function myEmail() {
  return Session.getEffectiveUser().getEmail().toLowerCase();
}

function allowedSenders() {
  const extra = prop('ALLOWED_SENDERS', '')
    .split(',')
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);
  return [myEmail()].concat(extra);
}

function helpText() {
  const root = prop('DRIVE_ROOT_FOLDER', 'AI Assistant');
  return [
    'Команды (первое слово темы письма), документы — во вложении:',
    '',
    'EXTRACT          — извлечь текст (PDF, DOC/DOCX, XLS/XLSX, PPT/PPTX, TXT, CSV, HTML…)',
    'SUMMARY          — краткое резюме каждого вложения',
    `ORGANIZE [папка] — сохранить вложения в Google Drive: «${root}/<папка или дата>/<тип>»`,
    'CONVERT <формат> — конвертировать: pdf, docx, xlsx, pptx, txt, html, csv, odt',
    'HELP             — эта справка',
    '',
    'Пример: тема «CONVERT pdf», вложение report.docx.',
  ].join('\n');
}

// ---------------------------------------------------------------- setup

/** Run once from the editor: installs the 1-minute trigger and sends a test HELP email. */
function setup() {
  uninstall();
  ScriptApp.newTrigger('processInbox').timeBased().everyMinutes(1).create();
  getOrCreateLabel_();
  GmailApp.sendEmail(myEmail(), 'HELP', 'Тестовая команда для Gmail Assistant. Ответ придёт в эту ветку в течение пары минут.');
  Logger.log('Готово: триггер установлен, тестовое письмо HELP отправлено на %s', myEmail());
}

/** Stops the assistant (removes its trigger). */
function uninstall() {
  ScriptApp.getProjectTriggers()
    .filter((t) => t.getHandlerFunction() === 'processInbox')
    .forEach((t) => ScriptApp.deleteTrigger(t));
}

function getOrCreateLabel_() {
  return GmailApp.getUserLabelByName(PROCESSED_LABEL) || GmailApp.createLabel(PROCESSED_LABEL);
}

// ---------------------------------------------------------------- main loop

function processInbox() {
  const lock = LockService.getScriptLock();
  if (!lock.tryLock(1000)) return; // previous run still going
  try {
    const props = PropertiesService.getUserProperties();
    const processed = JSON.parse(props.getProperty(PROCESSED_KEY) || '[]');
    const seen = new Set(processed);
    const label = getOrCreateLabel_();

    const threads = GmailApp.search('in:inbox newer_than:2d', 0, 50);
    const messages = [];
    threads.forEach((t) => t.getMessages().forEach((m) => messages.push(m)));
    messages.sort((a, b) => a.getDate() - b.getDate());

    for (const msg of messages) {
      if (seen.has(msg.getId())) continue;
      const command = parseCommand(msg.getSubject());
      if (!command) continue; // ordinary email — leave untouched

      // Record first so a crash can't make the assistant answer the same email forever.
      seen.add(msg.getId());
      processed.push(msg.getId());
      props.setProperty(PROCESSED_KEY, JSON.stringify(processed.slice(-MAX_PROCESSED_IDS)));
      msg.getThread().addLabel(label);

      if (!isTrustedSender_(msg)) {
        Logger.log('Пропуск: «%s» от %s не прошёл проверку отправителя', msg.getSubject(), msg.getFrom());
        continue;
      }
      handleMessage_(msg, command);
    }
  } finally {
    lock.releaseLock();
  }
}

function parseCommand(subject) {
  const words = String(subject || '').trim().split(/\s+/);
  const name = (words[0] || '').toUpperCase().replace(/[:.!]$/, '');
  if (COMMANDS.indexOf(name) === -1) return null;
  return { name: name, args: words.slice(1) };
}

function extractAddress_(from) {
  const m = String(from).match(/<([^>]+)>/);
  return (m ? m[1] : String(from)).trim().toLowerCase();
}

/**
 * The From header is easy to forge. Trust a message only if it is from an allowed
 * address AND was sent from this account (SENT label) or passed DMARC per Gmail.
 */
function isTrustedSender_(msg) {
  if (allowedSenders().indexOf(extractAddress_(msg.getFrom())) === -1) return false;
  const meta = Gmail.Users.Messages.get('me', msg.getId(), { format: 'minimal' });
  if ((meta.labelIds || []).indexOf('SENT') !== -1) return true;
  return /\bdmarc=pass\b/i.test(msg.getHeader('Authentication-Results') || '');
}

function handleMessage_(msg, command) {
  let reply;
  try {
    const files = [];
    const skipped = [];
    msg.getAttachments({ includeInlineImages: false }).forEach((a) => {
      if (a.getSize() > MAX_ATTACHMENT_BYTES) skipped.push(a.getName() + ' (больше лимита)');
      else files.push(a.copyBlob().setName(a.getName()));
    });
    reply = HANDLERS[command.name](files, command.args, msg);
    if (skipped.length) reply.text += '\n\nПропущены: ' + skipped.join(', ');
  } catch (err) {
    reply = { text: `Не удалось выполнить ${command.name}: ${err.message}` };
  }
  sendReply_(msg, reply);
}

function sendReply_(msg, reply) {
  const attachments = [];
  const dropped = [];
  let total = 0;
  (reply.attachments || []).forEach((b) => {
    const size = b.getBytes().length;
    if (total + size > MAX_REPLY_ATTACHMENT_BYTES) dropped.push(b.getName());
    else {
      total += size;
      attachments.push(b);
    }
  });
  let text = reply.text;
  if (dropped.length) text += '\n\nНе поместились в письмо (лимит Gmail): ' + dropped.join(', ');
  msg.reply(text, { attachments: attachments });
}

// ---------------------------------------------------------------- commands

const HANDLERS = {
  HELP: () => ({ text: helpText() }),
  EXTRACT: cmdExtract_,
  SUMMARY: cmdSummary_,
  ORGANIZE: cmdOrganize_,
  CONVERT: cmdConvert_,
};

function noFiles_() {
  return { text: 'В письме нет вложений. Прикрепите документ и отправьте команду ещё раз.\n\n' + helpText() };
}

function clip_(text, max) {
  return text.length > max ? text.slice(0, max) + '\n\n… [обрезано, полный текст во вложении]' : text;
}

function cmdExtract_(files) {
  if (!files.length) return noFiles_();
  const parts = [];
  const attachments = [];
  const perFile = Math.floor(MAX_REPLY_TEXT_CHARS / files.length);
  files.forEach((f) => {
    try {
      const text = extractText_(f);
      parts.push(`■ ${f.getName()} (${text.length} символов)\n\n${clip_(text, perFile) || '[текст не найден]'}`);
      attachments.push(Utilities.newBlob(text, 'text/plain', baseName_(f.getName()) + '.txt'));
    } catch (err) {
      parts.push(`■ ${f.getName()}: ${err.message}`);
    }
  });
  return { text: parts.join('\n\n' + '─'.repeat(40) + '\n\n'), attachments: attachments };
}

function cmdSummary_(files) {
  if (!files.length) return noFiles_();
  const parts = files.map((f) => {
    try {
      const text = extractText_(f);
      if (!text) return `■ ${f.getName()}: текст не найден`;
      const s = summarize_(f.getName(), text);
      return `■ ${f.getName()}  [${s.provider}]\n\n${s.text}`;
    } catch (err) {
      return `■ ${f.getName()}: ${err.message}`;
    }
  });
  return { text: parts.join('\n\n') };
}

function cmdOrganize_(files, args, msg) {
  if (!files.length) return noFiles_();
  const root = prop('DRIVE_ROOT_FOLDER', 'AI Assistant');
  const date = Utilities.formatDate(msg.getDate(), Session.getScriptTimeZone(), 'yyyy-MM-dd');
  const sub = args.join(' ').trim() || date;
  const lines = files.map((f) => {
    const type = (extOf_(f.getName()) || 'other').toUpperCase();
    const folder = ensureFolderPath_([root, sub, type]);
    const file = folder.createFile(f);
    return `• ${f.getName()} → ${root}/${sub}/${type}\n  ${file.getUrl()}`;
  });
  return { text: `Сохранено в Google Drive (${files.length}):\n\n` + lines.join('\n') };
}

function cmdConvert_(files, args) {
  if (!files.length) return noFiles_();
  const target = String(args[0] || '').toLowerCase().replace(/^\./, '');
  if (!target) return { text: 'Укажите формат, например тема «CONVERT pdf».\n\n' + helpText() };

  const lines = [];
  const attachments = [];
  files.forEach((f) => {
    const ext = extOf_(f.getName());
    const outName = baseName_(f.getName()) + '.' + target;
    try {
      let blob;
      const native = IMPORT_AS[ext];
      if (native && ext !== 'pdf' && EXPORTS[native].indexOf(target) !== -1) {
        blob = withNativeCopy_(f, native, (id) => exportFile_(id, target));
      } else if (target === 'txt') {
        blob = Utilities.newBlob(extractText_(f), 'text/plain');
      } else {
        lines.push(`• ${f.getName()}: ${ext || '?'} → ${target} не поддерживается`);
        return;
      }
      attachments.push(blob.setName(outName));
      lines.push(`• ${f.getName()} → ${outName}`);
    } catch (err) {
      lines.push(`• ${f.getName()}: ошибка — ${err.message}`);
    }
  });
  return { text: 'Результат конвертации:\n\n' + lines.join('\n'), attachments: attachments };
}

// ---------------------------------------------------------------- documents & Drive

function extOf_(name) {
  const m = String(name).toLowerCase().match(/\.([a-z0-9]+)$/);
  return m ? m[1] : '';
}

function baseName_(name) {
  return String(name).replace(/\.[^.]+$/, '');
}

function htmlToText_(html) {
  return html
    .replace(/<(script|style)[\s\S]*?<\/\1>/gi, '')
    .replace(/<br\s*\/?>|<\/(p|div|h\d|li|tr)>/gi, '\n')
    .replace(/<[^>]+>/g, '')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/** Imports a file into Google Docs/Sheets/Slides, runs fn(fileId), then trashes the temp copy. */
function withNativeCopy_(blob, nativeMime, fn) {
  const ext = extOf_(blob.getName());
  const source = blob.copyBlob().setContentType(MIME_BY_EXT[ext] || blob.getContentType());
  const temp = Drive.Files.create({ name: 'tmp-ai-assistant-' + blob.getName(), mimeType: nativeMime }, source);
  try {
    return fn(temp.id);
  } finally {
    try {
      DriveApp.getFileById(temp.id).setTrashed(true);
    } catch (e) {
      // ignore: a leftover temp file is harmless
    }
  }
}

function exportFile_(fileId, target) {
  const mime = target === 'txt' ? 'text/plain' : MIME_BY_EXT[target];
  const res = UrlFetchApp.fetch(
    `https://www.googleapis.com/drive/v3/files/${fileId}/export?mimeType=${encodeURIComponent(mime)}`,
    { headers: { Authorization: 'Bearer ' + ScriptApp.getOAuthToken() }, muteHttpExceptions: true },
  );
  if (res.getResponseCode() !== 200) throw new Error('экспорт из Drive: HTTP ' + res.getResponseCode());
  return res.getBlob().setContentType(mime);
}

function sheetToText_(fileId) {
  return SpreadsheetApp.openById(fileId)
    .getSheets()
    .map((sh) => {
      const rows = sh.getDataRange().getDisplayValues().map((r) => r.join('\t'));
      return '## ' + sh.getName() + '\n' + rows.join('\n');
    })
    .join('\n\n');
}

/** Extracts plain text; Office files and PDFs (via OCR) go through a temporary Google copy. */
function extractText_(blob) {
  const ext = extOf_(blob.getName());
  if (ext === 'html' || ext === 'htm') return htmlToText_(blob.getDataAsString('UTF-8'));
  if (TEXT_EXT.indexOf(ext) !== -1) return blob.getDataAsString('UTF-8').trim();
  const native = IMPORT_AS[ext];
  if (!native) throw new Error(`формат .${ext || '?'} не поддерживается`);
  if (native === GOOGLE_SHEET) return withNativeCopy_(blob, native, sheetToText_).trim();
  return withNativeCopy_(blob, native, (id) => exportFile_(id, 'txt').getDataAsString('UTF-8')).trim();
}

function ensureFolderPath_(parts) {
  let folder = DriveApp.getRootFolder();
  parts.forEach((name) => {
    const it = folder.getFoldersByName(name);
    folder = it.hasNext() ? it.next() : folder.createFolder(name);
  });
  return folder;
}

// ---------------------------------------------------------------- summaries

function localSummary_(text) {
  const clean = text.replace(/\s+/g, ' ').trim();
  const sentences = clean.match(/[^.!?…]+[.!?…]+/g) || [clean];
  const words = clean ? clean.split(' ').length : 0;
  return [`Объём: ~${words} слов, ${clean.length} символов.`, '', 'Начало документа:']
    .concat(sentences.slice(0, 5).map((s) => '• ' + s.trim()))
    .join('\n');
}

/** Claude summary when ANTHROPIC_API_KEY is set (raw HTTP: Apps Script has no Anthropic SDK). */
function summarize_(filename, text) {
  const apiKey = prop('ANTHROPIC_API_KEY', '');
  if (!apiKey) return { text: localSummary_(text), provider: 'local' };
  if (text.length > 3000000) return { text: localSummary_(text), provider: 'local (документ слишком большой)' };

  const res = UrlFetchApp.fetch('https://api.anthropic.com/v1/messages', {
    method: 'post',
    contentType: 'application/json',
    muteHttpExceptions: true,
    headers: {
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'anthropic-beta': 'server-side-fallback-2026-07-01',
    },
    payload: JSON.stringify({
      model: prop('CLAUDE_MODEL', 'claude-opus-5-5'),
      max_tokens: 16000,
      output_config: { effort: 'low' },
      fallbacks: 'default',
      system:
        'Ты делаешь краткие резюме документов. Пиши на языке документа. Формат: 1–2 предложения о сути, ' +
        'затем 3–7 пунктов с ключевыми фактами, цифрами, датами и действиями. Не добавляй того, чего нет в документе.',
      messages: [
        { role: 'user', content: `<document name="${filename}">\n${text}\n</document>\n\nСделай резюме этого документа.` },
      ],
    }),
  });

  if (res.getResponseCode() !== 200) {
    Logger.log('Claude API HTTP %s: %s', res.getResponseCode(), res.getContentText().slice(0, 500));
    return { text: localSummary_(text), provider: 'local (Claude недоступен)' };
  }
  const message = JSON.parse(res.getContentText());
  if (message.stop_reason === 'refusal') return { text: localSummary_(text), provider: 'local (Claude отказался)' };
  const summary = (message.content || [])
    .filter((b) => b.type === 'text')
    .map((b) => b.text)
    .join('\n')
    .trim();
  return { text: summary || localSummary_(text), provider: message.model };
}
