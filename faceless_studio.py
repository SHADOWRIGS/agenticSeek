#!/usr/bin/env python3
"""
FACELESS STUDIO — контент-конвейер для каналов без лица (YouTube / Instagram).

По нише генерирует через локальный Ollama:
  идеи роликов -> сценарий с озвучкой и видеорядом -> SEO -> нарезки Shorts/Reels -> план недели.

Запуск:
  venv/bin/python faceless_studio.py "личные финансы для новичков"
  venv/bin/python faceless_studio.py "история загадок" --ideas 10 --scripts 2 --model dolphin-llama3:latest
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "mannix/llama3.1-8b-abliterated:latest"
OUTPUT_DIR = "./faceless_output"

SYSTEM = (
    "Ты продюсер YouTube и Instagram каналов без лица. Пишешь на русском, конкретно, "
    "без воды. Контент должен быть оригинальным и полезным зрителю, а не шаблонным: "
    "так он проходит правила монетизации YouTube."
)


def ask(model: str, prompt: str, as_json: bool = False, temperature: float = 0.7) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "stream": True,
        "options": {"temperature": temperature},
    }
    if as_json:
        payload["format"] = "json"
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/chat",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    # Потоковый режим: таймаут считается между токенами, а не на весь ответ,
    # поэтому длинные сценарии на медленной машине не обрываются.
    chunks = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            if not line.strip():
                continue
            part = json.loads(line)
            if "error" in part:
                raise RuntimeError(part["error"])
            chunks.append(part.get("message", {}).get("content", ""))
            if len(chunks) % 50 == 0:
                print(".", end="", flush=True)
            if part.get("done"):
                break
    if len(chunks) >= 50:
        print()
    return "".join(chunks).strip()


def ask_json(model: str, prompt: str, retries: int = 2):
    for attempt in range(retries + 1):
        raw = ask(model, prompt, as_json=True, temperature=0.6)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.S)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        print(f"   ⚠️  невалидный JSON, повтор {attempt + 1}/{retries}")
    raise RuntimeError("Модель не вернула JSON")


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.U)
    return re.sub(r"[\s_]+", "-", text).strip("-")[:60] or "niche"


# ==================== ЭТАПЫ ====================
def gen_ideas(model: str, niche: str, n: int) -> list:
    data = ask_json(model, f"""Ниша канала: "{niche}".
Придумай {n} идей роликов для канала без лица. Для каждой укажи:
- title: рабочее название
- hook: первая фраза ролика (до 15 слов), цепляющая за 3 секунды
- angle: чем ролик отличается от того, что уже есть на YouTube
- format: один из ["озвучка+стоки", "запись экрана", "текст на экране", "только руки"]
- potential: оценка интереса аудитории от 1 до 10
Верни JSON: {{"ideas": [{{"title": "...", "hook": "...", "angle": "...", "format": "...", "potential": 8}}]}}""")
    ideas = data.get("ideas", [])
    return sorted(ideas, key=lambda i: -float(i.get("potential", 0) or 0))


def gen_script(model: str, niche: str, idea: dict) -> str:
    return ask(model, f"""Ниша: "{niche}". Ролик: "{idea['title']}".
Хук: "{idea.get('hook', '')}". Формат: {idea.get('format', 'озвучка+стоки')}.

Напиши сценарий длинного ролика на 7–9 минут в формате Markdown:
## Структура
Таблица: | Время | Блок | Текст озвучки | Видеоряд (что показать) | Ключевые слова для стоков (англ.) |
Блоки: Хук (0:00–0:15), Интро, 3–5 основных пунктов, Удержание (тизер на середине), Вывод, Призыв к действию.

## Полный текст озвучки
Сплошной текст для диктора/TTS, разговорный стиль, короткие предложения.

## Где нужны факты
Список утверждений, которые надо проверить перед публикацией.""")


def gen_seo(model: str, niche: str, idea: dict) -> dict:
    return ask_json(model, f"""Ниша: "{niche}". Ролик: "{idea['title']}".
Сделай SEO для YouTube. Верни JSON:
{{"titles": ["3 варианта заголовка до 60 символов"],
  "description": "описание 3–5 предложений с ключевыми словами и таймкодами-заглушками",
  "tags": ["15 тегов"],
  "thumbnail_text": "2–4 слова крупным текстом на превью",
  "thumbnail_idea": "что изображено на превью без лица"}}""")


def gen_shorts(model: str, niche: str, idea: dict) -> dict:
    return ask_json(model, f"""Ниша: "{niche}". Длинный ролик: "{idea['title']}".
Сделай 3 нарезки для Shorts / Reels (до 45 секунд каждая). Верни JSON:
{{"shorts": [{{"hook": "первые 2 секунды", "voiceover": "текст озвучки",
   "on_screen_text": "текст поверх видео", "visual": "что в кадре",
   "caption": "подпись для Instagram с призывом", "hashtags": ["5 хэштегов"]}}]}}""")


def gen_week_plan(model: str, niche: str, ideas: list) -> str:
    titles = "\n".join(f"- {i['title']}" for i in ideas)
    return ask(model, f"""Ниша: "{niche}". Идеи:
{titles}

Составь план публикаций на 1 неделю в Markdown-таблице:
| День | Платформа | Формат | Что публикуем | Время публикации |
Ритм: 2–3 длинных ролика YouTube, 5–7 Shorts/Reels (нарезки из длинных), 2 сторис-опроса в Instagram.
Под таблицей — 3 совета по монетизации именно этой ниши (партнёрки, свои продукты, спонсоры).""")


# ==================== СБОРКА ====================
def render_video_md(idea: dict, script: str, seo: dict, shorts: dict) -> str:
    out = [f"# {idea['title']}\n",
           f"**Хук:** {idea.get('hook', '')}  ",
           f"**Угол:** {idea.get('angle', '')}  ",
           f"**Формат:** {idea.get('format', '')}\n",
           "## SEO\n",
           "**Заголовки:**"]
    out += [f"- {t}" for t in seo.get("titles", [])]
    out += ["", f"**Описание:**\n\n{seo.get('description', '')}\n",
            f"**Теги:** {', '.join(seo.get('tags', []))}\n",
            f"**Превью:** «{seo.get('thumbnail_text', '')}» — {seo.get('thumbnail_idea', '')}\n",
            "---\n", script, "\n---\n", "## Нарезки Shorts / Reels\n"]
    for n, s in enumerate(shorts.get("shorts", []), 1):
        out += [f"### Short {n}",
                f"- **Хук:** {s.get('hook', '')}",
                f"- **Озвучка:** {s.get('voiceover', '')}",
                f"- **Текст на экране:** {s.get('on_screen_text', '')}",
                f"- **Видеоряд:** {s.get('visual', '')}",
                f"- **Подпись:** {s.get('caption', '')} {' '.join(s.get('hashtags', []))}", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Контент-конвейер для каналов без лица")
    ap.add_argument("niche", help="ниша канала")
    ap.add_argument("--ideas", type=int, default=8, help="сколько идей сгенерировать")
    ap.add_argument("--scripts", type=int, default=2, help="для скольких лучших идей писать сценарии")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    started = time.time()
    folder = os.path.join(OUTPUT_DIR, f"{slugify(args.niche)}-{datetime.now():%Y%m%d-%H%M}")
    os.makedirs(folder, exist_ok=True)
    print(f"🎬 Ниша: {args.niche}\n🤖 Модель: {args.model}\n📁 Папка: {folder}\n")

    try:
        print("💡 Генерирую идеи...")
        ideas = gen_ideas(args.model, args.niche, args.ideas)
    except Exception as e:
        sys.exit(f"❌ Ollama недоступна или ошибка модели: {e}")
    if not ideas:
        sys.exit("❌ Модель не вернула идей")

    lines = [f"# Идеи: {args.niche}\n", "| # | Потенциал | Название | Хук | Формат |", "|---|---|---|---|---|"]
    for n, i in enumerate(ideas, 1):
        lines.append(f"| {n} | {i.get('potential', '')} | {i['title']} | {i.get('hook', '')} | {i.get('format', '')} |")
    with open(os.path.join(folder, "00_ideas.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"   ✅ {len(ideas)} идей")

    for n, idea in enumerate(ideas[:args.scripts], 1):
        print(f"\n📝 Ролик {n}: {idea['title']}")
        print("   сценарий..."); script = gen_script(args.model, args.niche, idea)
        print("   SEO...");      seo = gen_seo(args.model, args.niche, idea)
        print("   Shorts...");   shorts = gen_shorts(args.model, args.niche, idea)
        path = os.path.join(folder, f"{n:02d}_{slugify(idea['title'])}.md")
        with open(path, "w") as f:
            f.write(render_video_md(idea, script, seo, shorts))
        print(f"   ✅ {path}")

    print("\n📅 План недели...")
    with open(os.path.join(folder, "99_week_plan.md"), "w") as f:
        f.write(f"# План недели: {args.niche}\n\n" + gen_week_plan(args.model, args.niche, ideas))

    print(f"\n🏁 Готово за {time.time() - started:.0f} сек. Результат: {folder}")


if __name__ == "__main__":
    main()
