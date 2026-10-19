#!/usr/bin/env python3
"""
АВТОНОМНЫЙ СУПЕРАГЕНТ "МОЛОТ OMEGA"
С Selenium, Playwright, графом задач, долгосрочной памятью и самоулучшением
"""

import asyncio
import subprocess
import sys
import os
import json
import re
import aiohttp
import aiofiles
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import pickle
import zlib
from collections import defaultdict, deque
import networkx as nx
import warnings
from concurrent.futures import ThreadPoolExecutor
import threading
import gc
warnings.filterwarnings('ignore')

# ==================== КОНФИГУРАЦИЯ ====================
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "dolphin-llama3:8b"  # 8B для скорости, без цензуры
VECTOR_DB_PATH = "./vector_db_omega"
MEMORY_COMPRESSION_THRESHOLD = 100  # Сжимать после 100 воспоминаний
MAX_CONTEXT_LENGTH = 32000  # Для long-context моделей

# ==================== ГРАФ ЗАДАЧ ====================
class TaskGraph:
    """Граф зависимостей задач в стиле LangGraph"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.execution_history = {}
        
    def add_task(self, task: Dict, dependencies: List[int] = None) -> int:
        """Добавляет задачу в граф"""
        task_id = self.node_counter
        self.node_counter += 1
        
        self.graph.add_node(task_id, 
                           task=task,
                           status="pending",
                           start_time=None,
                           end_time=None,
                           result=None)
        
        if dependencies:
            for dep_id in dependencies:
                self.graph.add_edge(dep_id, task_id)
        
        return task_id
    
    def get_ready_tasks(self) -> List[int]:
        """Возвращает задачи готовые к выполнению"""
        ready = []
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            
            if node_data['status'] == "pending":
                # Проверяем зависимости
                predecessors = list(self.graph.predecessors(node_id))
                
                if all(self.graph.nodes[p]['status'] == "completed" 
                      for p in predecessors):
                    ready.append(node_id)
        
        return ready
    
    def mark_completed(self, task_id: int, result: Any):
        """Отмечает задачу как выполненную"""
        self.graph.nodes[task_id]['status'] = "completed"
        self.graph.nodes[task_id]['end_time'] = datetime.now()
        self.graph.nodes[task_id]['result'] = result
        self.execution_history[task_id] = result
    
    def mark_failed(self, task_id: int, error: str):
        """Отмечает задачу как неудачную"""
        self.graph.nodes[task_id]['status'] = "failed"
        self.graph.nodes[task_id]['end_time'] = datetime.now()
        self.graph.nodes[task_id]['error'] = error
    
    def visualize(self, filename: str = "task_graph.png"):
        """Визуализирует граф задач"""
        try:
            import matplotlib.pyplot as plt
            
            pos = nx.spring_layout(self.graph)
            
            # Цвета по статусу
            colors = []
            for node_id in self.graph.nodes():
                status = self.graph.nodes[node_id]['status']
                if status == "completed":
                    colors.append("green")
                elif status == "failed":
                    colors.append("red")
                elif status == "executing":
                    colors.append("yellow")
                else:
                    colors.append("gray")
            
            plt.figure(figsize=(12, 8))
            nx.draw(self.graph, pos, node_color=colors, with_labels=True,
                   node_size=2000, font_size=10, font_weight='bold')
            
            # Подписи задач
            labels = {}
            for node_id in self.graph.nodes():
                task = self.graph.nodes[node_id]['task']
                action = task.get('action', '')[:20]
                labels[node_id] = f"{node_id}: {action}"
            
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
            
            plt.title("Граф выполнения задач", fontsize=16)
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            return f"Граф сохранен: {filename}"
        except Exception as e:
            return f"Ошибка визуализации: {e}"
    
    def get_execution_path(self) -> List[List[int]]:
        """Возвращает путь выполнения по слоям"""
        # Топологическая сортировка
        try:
            layers = []
            remaining = set(self.graph.nodes())
            
            while remaining:
                # Находим узлы без незавершенных зависимостей
                current_layer = []
                for node_id in remaining:
                    preds = set(self.graph.predecessors(node_id))
                    if preds.issubset(set(self.graph.nodes()) - remaining):
                        current_layer.append(node_id)
                
                if not current_layer:
                    break
                
                layers.append(current_layer)
                remaining -= set(current_layer)
            
            return layers
        except:
            # Простой BFS если граф циклический
            return [list(self.graph.nodes())]
    
    def get_task_summary(self) -> Dict:
        """Статистика выполнения"""
        total = len(self.graph.nodes())
        completed = sum(1 for n in self.graph.nodes() 
                       if self.graph.nodes[n]['status'] == "completed")
        failed = sum(1 for n in self.graph.nodes() 
                    if self.graph.nodes[n]['status'] == "failed")
        
        return {
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "pending": total - completed - failed,
            "success_rate": completed / total if total > 0 else 0
        }

# ==================== ДОЛГОСРОЧНАЯ ПАМЯТЬ ====================
class LongTermMemory:
    """Память со сжатием и RAG"""
    
    def __init__(self, storage_path="./memory_db"):
        self.storage_path = storage_path
        self.memories = []
        self.compressed_memories = []
        self.embeddings_cache = {}
        
        os.makedirs(storage_path, exist_ok=True)
        self.load_memories()
    
    def add_memory(self, content: str, metadata: Dict = None, 
                   importance: float = 0.5):
        """Добавляет воспоминание"""
        memory = {
            "id": hashlib.md5(content.encode()).hexdigest(),
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "importance": importance,
            "access_count": 0,
            "last_accessed": datetime.now().isoformat()
        }
        
        self.memories.append(memory)
        
        # Авто-сжатие если много воспоминаний
        if len(self.memories) > MEMORY_COMPRESSION_THRESHOLD:
            self.compress_memories()
        
        self.save_memories()
        return memory["id"]
    
    def compress_memories(self):
        """Сжимает старые/неважные воспоминания"""
        if len(self.memories) < 10:
            return
        
        # Сортируем по важности и давности
        sorted_memories = sorted(self.memories, 
                                key=lambda x: (
                                    x['importance'], 
                                    -x['access_count'],
                                    x['timestamp']
                                ))
        
        # Оставляем 70% самых важных
        keep_count = int(len(self.memories) * 0.7)
        memories_to_keep = sorted_memories[-keep_count:]
        memories_to_compress = sorted_memories[:-keep_count]
        
        # Сжимаем старые воспоминания
        for memory in memories_to_compress:
            compressed = {
                "id": memory["id"],
                "summary": self._summarize_memory(memory["content"]),
                "original_length": len(memory["content"]),
                "compressed_length": len(self._summarize_memory(memory["content"])),
                "metadata": memory["metadata"],
                "timestamp": memory["timestamp"],
                "importance": memory["importance"] * 0.5  # Понижаем важность
            }
            self.compressed_memories.append(compressed)
        
        # Обновляем список
        self.memories = memories_to_keep
        self.save_memories()
        
        print(f"🧠 Сжато {len(memories_to_compress)} воспоминаний. "
              f"Осталось {len(self.memories)} активных.")
    
    def _summarize_memory(self, content: str) -> str:
        """Суммаризирует воспоминание для сжатия"""
        if len(content) <= 500:
            return content
        
        # Простая суммаризация (можно заменить на LLM)
        sentences = re.split(r'[.!?]+', content)
        
        if len(sentences) <= 3:
            return content
        
        # Берем первое, последнее и самые длинные предложения
        important = []
        if sentences[0]:
            important.append(sentences[0])
        
        if len(sentences) > 1 and sentences[-1]:
            important.append(sentences[-1])
        
        # Ищем ключевые предложения
        words = content.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        for sentence in sentences[1:-1]:
            if len(sentence) > 50:
                score = sum(word_freq.get(word.lower(), 0) 
                           for word in sentence.split() 
                           if len(word) > 3)
                if score > 10:
                    important.append(sentence)
        
        return ". ".join(important[:5]) + "."
    
    def retrieve_context(self, query: str, n_results: int = 5) -> str:
        """Извлекает релевантный контекст"""
        # Объединяем активные и сжатые воспоминания
        all_memories = self.memories + [
            {"content": m["summary"], "importance": m["importance"], 
             "last_accessed": m.get("last_accessed", datetime.now().isoformat())}
            for m in self.compressed_memories
        ]
        
        # Простой семантический поиск по ключевым словам
        query_words = set(re.findall(r'\w+', query.lower()))
        
        scored_memories = []
        for memory in all_memories:
            content = memory["content"].lower()
            memory_words = set(re.findall(r'\w+', content))
            
            # Счет совпадения слов
            word_overlap = len(query_words.intersection(memory_words))
            
            # Учитываем важность и давность доступа
            importance = memory.get("importance", 0.5)
            last_accessed = memory.get("last_accessed", datetime.now().isoformat())
            recency = 1.0  # Можно добавить расчет давности
            
            score = (word_overlap * 0.5 + 
                    importance * 0.3 + 
                    recency * 0.2)
            
            scored_memories.append((score, memory))
        
        # Сортируем по релевантности
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        
        # Обновляем счетчики доступа
        for score, memory in scored_memories[:n_results]:
            if 'id' in memory:
                for m in self.memories:
                    if m['id'] == memory['id']:
                        m['access_count'] += 1
                        m['last_accessed'] = datetime.now().isoformat()
        
        # Формируем контекст
        context_parts = []
        for i, (score, memory) in enumerate(scored_memories[:n_results]):
            source = "🟢" if memory in self.memories else "🟡"
            context_parts.append(f"{source} Воспоминание {i+1} (релевантность: {score:.2f}):")
            context_parts.append(memory["content"][:500])
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def save_memories(self):
        """Сохраняет память на диск"""
        try:
            data = {
                "memories": self.memories,
                "compressed_memories": self.compressed_memories,
                "last_updated": datetime.now().isoformat()
            }
            
            # Сохраняем в несколько форматов
            with open(f"{self.storage_path}/memories.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            with open(f"{self.storage_path}/memories.pkl", "wb") as f:
                pickle.dump(data, f)
            
            # Сжимаем для долгосрочного хранения
            compressed = zlib.compress(pickle.dumps(data))
            with open(f"{self.storage_path}/memories_compressed.z", "wb") as f:
                f.write(compressed)
                
        except Exception as e:
            print(f"⚠️  Ошибка сохранения памяти: {e}")
    
    def load_memories(self):
        """Загружает память с диска"""
        try:
            if os.path.exists(f"{self.storage_path}/memories.json"):
                with open(f"{self.storage_path}/memories.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                self.memories = data.get("memories", [])
                self.compressed_memories = data.get("compressed_memories", [])
                print(f"🧠 Загружено {len(self.memories)} активных и "
                      f"{len(self.compressed_memories)} сжатых воспоминаний")
            else:
                print("🧠 Память не найдена, создаю новую")
        except Exception as e:
            print(f"⚠️  Ошибка загрузки памяти: {e}")
    
    def get_stats(self) -> Dict:
        """Статистика памяти"""
        total_size = sum(len(m["content"]) for m in self.memories)
        compressed_size = sum(len(m["summary"]) for m in self.compressed_memories)
        
        return {
            "active_memories": len(self.memories),
            "compressed_memories": len(self.compressed_memories),
            "total_size_kb": (total_size + compressed_size) / 1024,
            "compression_ratio": compressed_size / total_size if total_size > 0 else 0,
            "most_accessed": sorted(self.memories, 
                                  key=lambda x: x.get("access_count", 0), 
                                  reverse=True)[:3]
        }

# ==================== БРАУЗЕР АВТОМАТИЗАЦИЯ ====================
class BrowserAutomation:
    """Автоматизация браузера через Selenium и Playwright"""
    
    def __init__(self):
        self.selenium_driver = None
        self.playwright_browser = None
        self.playwright_context = None
        self.browser_type = "chrome"  # chrome, firefox, webkit
        self.headless = True
        self.screenshots_dir = "./browser_screenshots"
        
        os.makedirs(self.screenshots_dir, exist_ok=True)
    
    async def init_selenium(self):
        """Инициализирует Selenium"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            from webdriver_manager.chrome import ChromeDriverManager
            
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            
            # Скрываем WebDriver
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            service = Service(ChromeDriverManager().install())
            self.selenium_driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Изменяем свойства navigator
            self.selenium_driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            
            print("✅ Selenium инициализирован")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка Selenium: {e}")
            return False
    
    async def init_playwright(self):
        """Инициализирует Playwright"""
        try:
            from playwright.async_api import async_playwright
            
            playwright = await async_playwright().start()
            
            if self.browser_type == "chrome":
                self.playwright_browser = await playwright.chromium.launch(
                    headless=self.headless,
                    args=['--disable-blink-features=AutomationControlled']
                )
            elif self.browser_type == "firefox":
                self.playwright_browser = await playwright.firefox.launch(
                    headless=self.headless
                )
            else:
                self.playwright_browser = await playwright.webkit.launch(
                    headless=self.headless
                )
            
            # Создаем контекст с user agent
            self.playwright_context = await self.playwright_browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            print("✅ Playwright инициализирован")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка Playwright: {e}")
            return False
    
    async def browse_with_selenium(self, url: str, actions: List[Dict] = None) -> Dict:
        """Навигация с Selenium"""
        if not self.selenium_driver:
            if not await self.init_selenium():
                return {"error": "Selenium не доступен"}
        
        try:
            results = {
                "url": url,
                "title": "",
                "actions_performed": [],
                "screenshots": [],
                "page_source_size": 0,
                "cookies": []
            }
            
            # Открываем страницу
            self.selenium_driver.get(url)
            await asyncio.sleep(2)  # Ждем загрузки
            
            results["title"] = self.selenium_driver.title
            results["page_source_size"] = len(self.selenium_driver.page_source)
            results["cookies"] = self.selenium_driver.get_cookies()
            
            # Делаем скриншот
            screenshot_path = f"{self.screenshots_dir}/selenium_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.selenium_driver.save_screenshot(screenshot_path)
            results["screenshots"].append(screenshot_path)
            
            # Выполняем действия
            if actions:
                for action in actions:
                    action_result = await self._perform_selenium_action(action)
                    results["actions_performed"].append(action_result)
            
            # Получаем все ссылки
            links = self.selenium_driver.find_elements("tag name", "a")
            results["links_found"] = len(links)
            
            return results
            
        except Exception as e:
            return {"error": f"Selenium ошибка: {str(e)}"}
    
    async def browse_with_playwright(self, url: str, actions: List[Dict] = None) -> Dict:
        """Навигация с Playwright"""
        if not self.playwright_context:
            if not await self.init_playwright():
                return {"error": "Playwright не доступен"}
        
        try:
            results = {
                "url": url,
                "title": "",
                "actions_performed": [],
                "screenshots": [],
                "page_content": "",
                "performance_metrics": {}
            }
            
            # Создаем страницу
            page = await self.playwright_context.new_page()
            
            # Открываем URL
            response = await page.goto(url, wait_until="networkidle", timeout=30000)
            
            results["title"] = await page.title()
            results["status"] = response.status if response else "no response"
            
            # Делаем скриншот
            screenshot_path = f"{self.screenshots_dir}/playwright_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            results["screenshots"].append(screenshot_path)
            
            # Получаем метрики производительности
            try:
                metrics = await page.evaluate("""() => {
                    return {
                        timing: performance.timing,
                        navigation: performance.getEntriesByType('navigation')[0]
                    };
                }""")
                results["performance_metrics"] = metrics
            except:
                pass
            
            # Выполняем действия
            if actions:
                for action in actions:
                    action_result = await self._perform_playwright_action(page, action)
                    results["actions_performed"].append(action_result)
            
            # Получаем контент
            results["page_content"] = await page.content()
            
            # Закрываем страницу
            await page.close()
            
            return results
            
        except Exception as e:
            return {"error": f"Playwright ошибка: {str(e)}"}
    
    async def _perform_selenium_action(self, action: Dict) -> Dict:
        """Выполняет действие в Selenium"""
        action_type = action.get("type", "click")
        selector = action.get("selector", "")
        value = action.get("value", "")
        
        try:
            element = self.selenium_driver.find_element("css selector", selector)
            
            if action_type == "click":
                element.click()
                return {"type": "click", "selector": selector, "success": True}
                
            elif action_type == "type":
                element.clear()
                element.send_keys(value)
                return {"type": "type", "selector": selector, "value": value, "success": True}
                
            elif action_type == "scroll":
                if value == "up":
                    self.selenium_driver.execute_script("window.scrollTo(0, 0);")
                elif value == "down":
                    self.selenium_driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                else:
                    self.selenium_driver.execute_script(f"window.scrollBy(0, {value});")
                return {"type": "scroll", "direction": value, "success": True}
                
            else:
                return {"type": action_type, "error": "Неизвестный тип действия"}
                
        except Exception as e:
            return {"type": action_type, "selector": selector, "error": str(e)}
    
    async def _perform_playwright_action(self, page, action: Dict) -> Dict:
        """Выполняет действие в Playwright"""
        action_type = action.get("type", "click")
        selector = action.get("selector", "")
        value = action.get("value", "")
        
        try:
            if action_type == "click":
                await page.click(selector)
                return {"type": "click", "selector": selector, "success": True}
                
            elif action_type == "type":
                await page.fill(selector, value)
                return {"type": "type", "selector": selector, "value": value, "success": True}
                
            elif action_type == "scroll":
                if value == "up":
                    await page.evaluate("window.scrollTo(0, 0)")
                elif value == "down":
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                else:
                    await page.evaluate(f"window.scrollBy(0, {value})")
                return {"type": "scroll", "direction": value, "success": True}
                
            elif action_type == "hover":
                await page.hover(selector)
                return {"type": "hover", "selector": selector, "success": True}
                
            elif action_type == "wait_for":
                await page.wait_for_selector(selector, timeout=10000)
                return {"type": "wait_for", "selector": selector, "success": True}
                
            else:
                return {"type": action_type, "error": "Неизвестный тип действия"}
                
        except Exception as e:
            return {"type": action_type, "selector": selector, "error": str(e)}
    
    async def close(self):
        """Закрывает браузеры"""
        try:
            if self.selenium_driver:
                self.selenium_driver.quit()
                self.selenium_driver = None
            
            if self.playwright_browser:
                await self.playwright_browser.close()
                self.playwright_browser = None
                self.playwright_context = None
                
            print("✅ Браузеры закрыты")
        except Exception as e:
            print(f"⚠️  Ошибка закрытия браузеров: {e}")

# ==================== СИСТЕМА САМОУЛУЧШЕНИЯ ====================
class SelfImprovementSystem:
    """Система самоулучшения агента"""
    
    def __init__(self, memory: LongTermMemory):
        self.memory = memory
        self.improvement_log = []
        self.failure_patterns = {}
        self.success_patterns = {}
        
    async def analyze_execution(self, task_graph: TaskGraph, 
                               execution_results: List[Dict]) -> Dict:
        """Анализирует выполнение и предлагает улучшения"""
        
        # Анализ статистики выполнения
        summary = task_graph.get_task_summary()
        
        # Находим проблемные задачи
        problematic_tasks = []
        for node_id in task_graph.graph.nodes():
            node_data = task_graph.graph.nodes[node_id]
            if node_data['status'] == "failed":
                problematic_tasks.append({
                    "id": node_id,
                    "task": node_data['task'],
                    "error": node_data.get('error', 'Unknown')
                })
        
        # Анализ паттернов
        patterns = self._analyze_patterns(task_graph, execution_results)
        
        # Генерация рекомендаций
        recommendations = await self._generate_recommendations(
            summary, problematic_tasks, patterns
        )
        
        # Проверяем нужно ли перепланирование
        need_replan = (
            summary["success_rate"] < 0.5 or 
            len(problematic_tasks) > len(task_graph.graph.nodes()) * 0.3
        )
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "problematic_tasks": problematic_tasks,
            "patterns": patterns,
            "recommendations": recommendations,
            "need_replan": need_replan,
            "improvement_score": self._calculate_improvement_score(summary)
        }
        
        # Сохраняем анализ в память
        self.memory.add_memory(
            content=json.dumps(analysis, ensure_ascii=False, indent=2),
            metadata={"type": "self_improvement", "score": analysis["improvement_score"]},
            importance=0.7
        )
        
        self.improvement_log.append(analysis)
        
        return analysis
    
    def _analyze_patterns(self, task_graph: TaskGraph, 
                         execution_results: List[Dict]) -> Dict:
        """Анализирует паттерны успеха/провала"""
        patterns = {
            "common_success_factors": [],
            "common_failure_causes": [],
            "dependency_issues": [],
            "timing_issues": [],
            "tool_preferences": {}
        }
        
        # Анализ по инструментам
        tool_success = defaultdict(list)
        for result in execution_results:
            if isinstance(result, dict):
                tool = result.get("tool_used", "unknown")
                success = result.get("success", False)
                tool_success[tool].append(success)
        
        for tool, successes in tool_success.items():
            success_rate = sum(successes) / len(successes) if successes else 0
            patterns["tool_preferences"][tool] = {
                "usage_count": len(successes),
                "success_rate": success_rate,
                "recommended": success_rate > 0.7
            }
        
        # Анализ зависимостей
        for node_id in task_graph.graph.nodes():
            preds = list(task_graph.graph.predecessors(node_id))
            if preds:
                all_preds_success = all(
                    task_graph.graph.nodes[p]['status'] == "completed"
                    for p in preds
                )
                
                if not all_preds_success:
                    patterns["dependency_issues"].append({
                        "task_id": node_id,
                        "failed_dependencies": [
                            p for p in preds 
                            if task_graph.graph.nodes[p]['status'] != "completed"
                        ]
                    })
        
        return patterns
    
    async def _generate_recommendations(self, summary: Dict,
                                       problematic_tasks: List[Dict],
                                       patterns: Dict) -> List[str]:
        """Генерирует рекомендации по улучшению"""
        recommendations = []
        
        # Если низкий процент успеха
        if summary["success_rate"] < 0.6:
            recommendations.append(
                "⚠️  Низкий процент успеха. Рекомендую: "
                "1) Упростить задачи\n"
                "2) Добавить проверки перед выполнением\n"
                "3) Использовать более надежные инструменты"
            )
        
        # Если много проблем с зависимостями
        if patterns["dependency_issues"]:
            recommendations.append(
                "🔗 Проблемы с зависимостями. Рекомендую: "
                "1) Проверять готовность зависимостей перед выполнением\n"
                "2) Добавить автоматическое восстановление\n"
                "3) Создать альтернативные пути выполнения"
            )
        
        # Рекомендации по инструментам
        for tool, stats in patterns["tool_preferences"].items():
            if stats["usage_count"] > 3 and stats["success_rate"] < 0.4:
                recommendations.append(
                    f"🛠️  Инструмент '{tool}' имеет низкую успешность "
                    f"({stats['success_rate']*100:.0f}%). "
                    f"Рассмотри альтернативы или улучши обработку ошибок."
                )
        
        # Если есть специфичные проблемы
        for problem in problematic_tasks[:3]:
            error = problem.get("error", "")
            if "timeout" in error.lower():
                recommendations.append(
                    f"⏰ Таймаут в задаче {problem['id']}. "
                    f"Увеличить таймауты или добавить retry логику."
                )
            elif "permission" in error.lower():
                recommendations.append(
                    f"🔒 Проблема прав в задаче {problem['id']}. "
                    f"Проверить права доступа или использовать sudo."
                )
        
        # Добавляем рекомендации из памяти
        memory_context = self.memory.retrieve_context(
            "рекомендации по улучшению производительности", 3
        )
        if memory_context and len(memory_context) > 100:
            recommendations.append(
                f"🧠 Из памяти: {memory_context[:300]}..."
            )
        
        return recommendations[:5]  # Ограничиваем 5 рекомендациями
    
    def _calculate_improvement_score(self, summary: Dict) -> float:
        """Рассчитывает score улучшений"""
        base_score = summary["success_rate"] * 100
        
        # Бонус за отсутствие провалов
        if summary["failed"] == 0:
            base_score += 20
        
        # Штраф за много ожидающих задач
        if summary["pending"] > summary["completed"] * 0.5:
            base_score -= 15
        
        return max(0, min(100, base_score))
    
    async def propose_new_plan(self, original_plan: Dict, 
                              analysis: Dict) -> Dict:
        """Предлагает улучшенный план на основе анализа"""
        
        prompt = f"""Оригинальный план имел проблемы. Проанализируй и предложи улучшенную версию.

Оригинальный план:
{json.dumps(original_plan, ensure_ascii=False, indent=2)}

Анализ проблем:
{json.dumps(analysis, ensure_ascii=False, indent=2)}

Предложи улучшенный план в JSON формате:
{{
    "improvements": ["список улучшений"],
    "new_steps": [
        {{"id": 1, "action": "действие", "tool": "инструмент", 
          "dependencies": [], "timeout_seconds": 60, "retry_count": 3}}
    ],
    "fallback_strategies": ["стратегии на случай ошибок"],
    "monitoring_checks": ["проверки во время выполнения"],
    "expected_improvement": "процент"
}}"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3}
                    },
                    timeout=60
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return json.loads(result.get("response", "{}"))
        except Exception as e:
            print(f"⚠️  Ошибка генерации улучшенного плана: {e}")
        
        # Fallback: простое улучшение
        return {
            "improvements": ["Увеличить таймауты", "Добавить retry логику"],
            "new_steps": original_plan.get("steps", []),
            "fallback_strategies": ["Повторить с альтернативным инструментом"],
            "expected_improvement": "10%"
        }

# ==================== ГЛАВНЫЙ АГЕНТ OMEGA ====================
class AutonomousOmegaAgent:
    """Конечный агент со всеми функциями"""
    
    def __init__(self):
        self.name = "⚡ МОЛОТ OMEGA v6.0"
        self.task_graph = TaskGraph()
        self.long_term_memory = LongTermMemory()
        self.browser = BrowserAutomation()
        self.self_improvement = SelfImprovementSystem(self.long_term_memory)
        
        self.execution_history = []
        self.iteration_count = 0
        self.max_iterations = 3  # Максимум итераций самоулучшения
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                           {self.name}                                   ║
║  С графом задач, долгосрочной памятью, браузером и самоулучшением       ║
╚══════════════════════════════════════════════════════════════════════════╝
        """)
    
    async def process_complex_task(self, task: str) -> Dict:
        """Обрабатывает сложную задачу с самоулучшением"""
        print(f"\n🎯 НАЧИНАЮ СЛОЖНУЮ ЗАДАЧУ: {task}")
        
        # Сохраняем в память
        self.long_term_memory.add_memory(
            content=f"Новая задача: {task}",
            metadata={"type": "task_start", "iteration": 0},
            importance=0.8
        )
        
        # Получаем контекст из памяти
        context = self.long_term_memory.retrieve_context(task, 5)
        print(f"🧠 Контекст из памяти ({len(context)} символов)")
        
        # Основной цикл с самоулучшением
        best_result = None
        best_score = 0
        
        for iteration in range(self.max_iterations):
            print(f"\n🔄 ИТЕРАЦИЯ {iteration + 1}/{self.max_iterations}")
            
            # Генерируем/улучшаем план
            if iteration == 0:
                plan = await self._generate_initial_plan(task, context)
            else:
                # Анализируем предыдущее выполнение и улучшаем план
                analysis = await self.self_improvement.analyze_execution(
                    self.task_graph, self.execution_history[-1]
                )
                
                if analysis.get("need_replan"):
                    print("🔄 Перепланирование на основе анализа...")
                    new_plan = await self.self_improvement.propose_new_plan(
                        plan, analysis
                    )
                    plan = self._merge_plans(plan, new_plan)
                else:
                    print("✅ План хорош, продолжаем")
            
            # Выполняем план
            result = await self._execute_with_graph(plan, task)
            
            # Сохраняем результат
            self.execution_history.append(result)
            
            # Оцениваем результат
            score = self._evaluate_result(result)
            print(f"📊 Оценка итерации: {score}/100")
            
            if score > best_score:
                best_score = score
                best_result = result
            
            # Если результат отличный - завершаем
            if score >= 85:
                print(f"🎉 Отличный результат! Завершаю итерации.")
                break
            
            # Если мало улучшений - возможно стоит остановиться
            if iteration > 0 and score < best_score * 1.1:
                print(f"⚠️  Незначительное улучшение. Завершаю итерации.")
                break
        
        # Финальный анализ
        final_analysis = await self.self_improvement.analyze_execution(
            self.task_graph, 
            self.execution_history[-1] if self.execution_history else []
        )
        
        # Формируем финальный отчет
        final_result = {
            "task": task,
            "best_iteration_score": best_score,
            "total_iterations": min(self.max_iterations, len(self.execution_history)),
            "task_graph_summary": self.task_graph.get_task_summary(),
            "final_analysis": final_analysis,
            "memory_stats": self.long_term_memory.get_stats(),
            "execution_path": self.task_graph.get_execution_path(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Сохраняем в долгосрочную память
        self.long_term_memory.add_memory(
            content=json.dumps(final_result, ensure_ascii=False, indent=2),
            metadata={"type": "task_completion", "score": best_score},
            importance=0.9
        )
        
        # Визуализируем граф
        graph_viz = self.task_graph.visualize(
            f"task_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        print(f"📊 {graph_viz}")
        
        return final_result
    
    async def _generate_initial_plan(self, task: str, context: str) -> Dict:
        """Генерирует начальный план через LLM"""
        print("🧠 Генерирую начальный план...")
        
        prompt = f"""Задача: {task}

Контекст из памяти агента:
{context}

Создай детальный план выполнения с зависимостями. 
Используй разные инструменты: shell, python, selenium, playwright, анализ данных.

Верни JSON:
{{
    "reasoning": "рассуждения о задаче",
    "steps": [
        {{
            "id": 1,
            "action": "детальное описание действия",
            "tool": "shell/python/selenium/playwright/browser/search/analyze",
            "dependencies": [номера шагов],
            "timeout_seconds": 30,
            "retry_count": 2,
            "priority": "high/medium/low",
            "expected_output": "что ожидается"
        }}
    ],
    "parallel_groups": [[номера шагов которые можно выполнять параллельно]],
    "browser_actions_needed": true/false,
    "data_analysis_needed": true/false,
    "risk_assessment": "оценка рисков"
}}"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.4,
                            "num_ctx": MAX_CONTEXT_LENGTH
                        }
                    },
                    timeout=90
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        plan = json.loads(result.get("response", "{}"))
                        
                        # Валидация плана
                        if "steps" not in plan:
                            raise ValueError("Неверный формат плана")
                        
                        print(f"✅ План создан: {len(plan['steps'])} шагов")
                        return plan
                        
        except Exception as e:
            print(f"⚠️  Ошибка генерации плана: {e}")
        
        # Fallback план
        return self._create_fallback_plan(task)
    
    def _create_fallback_plan(self, task: str) -> Dict:
        """Создает резервный план"""
        return {
            "reasoning": "Автономный план (LLM недоступен)",
            "steps": [
                {
                    "id": 1,
                    "action": f"Проанализировать задачу: {task}",
                    "tool": "analyze",
                    "dependencies": [],
                    "timeout_seconds": 30,
                    "priority": "high"
                },
                {
                    "id": 2,
                    "action": "Выполнить необходимые действия",
                    "tool": "shell",
                    "dependencies": [1],
                    "timeout_seconds": 60,
                    "priority": "medium"
                },
                {
                    "id": 3,
                    "action": "Проверить и проанализировать результат",
                    "tool": "analyze",
                    "dependencies": [2],
                    "timeout_seconds": 30,
                    "priority": "medium"
                }
            ],
            "parallel_groups": [],
            "browser_actions_needed": False,
            "data_analysis_needed": True,
            "risk_assessment": "Средний"
        }
    
    async def _execute_with_graph(self, plan: Dict, original_task: str) -> Dict:
        """Выполняет план используя граф задач"""
        print("🔄 Строю граф выполнения...")
        
        # Очищаем предыдущий граф
        self.task_graph = TaskGraph()
        
        # Добавляем задачи в граф
        task_ids = {}
        for step in plan["steps"]:
            dependencies = step.get("dependencies", [])
            dep_ids = [task_ids[dep] for dep in dependencies if dep in task_ids]
            
            task_id = self.task_graph.add_task(step, dep_ids)
            task_ids[step["id"]] = task_id
        
        # Выполняем по слоям
        execution_layers = self.task_graph.get_execution_path()
        print(f"📊 Граф построен: {len(execution_layers)} слоев выполнения")
        
        results = []
        
        for layer_num, layer in enumerate(execution_layers, 1):
            print(f"\n📦 Слой {layer_num}: {len(layer)} задач")
            
            # Выполняем задачи слоя параллельно
            layer_tasks = []
            for task_id in layer:
                task_data = self.task_graph.graph.nodes[task_id]['task']
                layer_tasks.append(self._execute_single_task(task_id, task_data))
            
            layer_results = await asyncio.gather(*layer_tasks, return_exceptions=True)
            
            for task_id, result in zip(layer, layer_results):
                if isinstance(result, Exception):
                    self.task_graph.mark_failed(task_id, str(result))
                    results.append({"task_id": task_id, "error": str(result)})
                else:
                    self.task_graph.mark_completed(task_id, result)
                    results.append({"task_id": task_id, "result": result})
        
        return {
            "original_task": original_task,
            "plan": plan,
            "execution_results": results,
            "graph_summary": self.task_graph.get_task_summary(),
            "execution_time": datetime.now().isoformat()
        }
    
    async def _execute_single_task(self, task_id: int, task: Dict) -> Any:
        """Выполняет одну задачу"""
        action = task.get("action", "")
        tool = task.get("tool", "auto")
        timeout = task.get("timeout_seconds", 30)
        
        print(f"  ▶️  Задача {task_id}: {action[:50]}...")
        
        try:
            # Определяем инструмент если auto
            if tool == "auto":
                tool = self._detect_tool(action)
            
            # Выполняем в зависимости от инструмента
            if tool == "shell":
                result = await self._execute_shell(action, timeout)
            elif tool == "python":
                result = await self._execute_python(action, timeout)
            elif tool in ["selenium", "browser"]:
                result = await self._execute_browser_action(action, tool)
            elif tool == "playwright":
                result = await self._execute_playwright_action(action, timeout)
            elif tool == "search":
                result = await self._execute_web_search(action)
            elif tool == "analyze":
                result = await self._analyze_data(action)
            else:
                result = f"⚠️  Неизвестный инструмент: {tool}"
            
            return {
                "action": action,
                "tool_used": tool,
                "result": result,
                "success": "❌" not in str(result) and "⚠️" not in str(result),
                "task_id": task_id
            }
            
        except asyncio.TimeoutError:
            return f"⏰ Таймаут задачи {task_id} ({timeout} сек)"
        except Exception as e:
            return f"❌ Ошибка в задаче {task_id}: {str(e)}"
    
    def _detect_tool(self, action: str) -> str:
        """Определяет инструмент по действию"""
        action_lower = action.lower()
        
        if any(word in action_lower for word in ["nmap", "ping", "curl", "wget", "ls", "cat"]):
            return "shell"
        elif any(word in action_lower for word in ["python", "import", "def ", "class "]):
            return "python"
        elif any(word in action_lower for word in ["браузер", "сайт", "веб", "клик", "скролл"]):
            return "selenium"
        elif any(word in action_lower for word in ["поиск", "найди", "google", "duckduckgo"]):
            return "search"
        elif any(word in action_lower for word in ["анализ", "данные", "график", "отчет"]):
            return "analyze"
        else:
            return "shell"
    
    async def _execute_shell(self, command: str, timeout: int) -> str:
        """Выполняет shell команду"""
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), 
                    timeout=timeout
                )
                
                output = stdout.decode() if stdout else stderr.decode()
                return f"✅ Код {proc.returncode}:\n{output[:1000]}"
                
            except asyncio.TimeoutError:
                proc.kill()
                return f"⏰ Таймаут команды: {command}"
                
        except Exception as e:
            return f"❌ Ошибка shell: {e}"
    
    async def _execute_python(self, code: str, timeout: int) -> str:
        """Выполняет Python код"""
        try:
            # Создаем временный файл
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                tmp_file = f.name
            
            # Выполняем через subprocess для изоляции
            proc = await asyncio.create_subprocess_exec(
                sys.executable, tmp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout
                )
                
                output = stdout.decode() if stdout else stderr.decode()
                result = f"✅ Выполнено:\n{output[:1000]}"
                
            except asyncio.TimeoutError:
                proc.kill()
                result = "⏰ Таймаут выполнения Python"
                
            # Удаляем временный файл
            os.unlink(tmp_file)
            
            return result
            
        except Exception as e:
            return f"❌ Ошибка Python: {e}"
    
    async def _execute_browser_action(self, action: str, tool: str) -> str:
        """Выполняет действие в браузере"""
        try:
            # Парсим URL из действия
            url_match = re.search(r'https?://[^\s]+', action)
            if not url_match:
                return "❌ Не найден URL в действии"
            
            url = url_match.group(0)
            
            # Парсим действия если есть
            actions = []
            if "клик" in action.lower():
                # Ищем селектор
                selector_match = re.search(r'селектор[:\s]+([^\s]+)', action)
                if selector_match:
                    actions.append({
                        "type": "click",
                        "selector": selector_match.group(1)
                    })
            
            if "введи" in action.lower() or "ввод" in action.lower():
                # Ищем текст для ввода
                text_match = re.search(r'текст[:\s]+([^\n]+)', action)
                if text_match:
                    selector_match = re.search(r'в поле[:\s]+([^\s]+)', action)
                    selector = selector_match.group(1) if selector_match else "input"
                    
                    actions.append({
                        "type": "type",
                        "selector": selector,
                        "value": text_match.group(1)
                    })
            
            # Выполняем в выбранном браузере
            if tool == "selenium":
                result = await self.browser.browse_with_selenium(url, actions)
            else:
                result = await self.browser.browse_with_playwright(url, actions)
            
            return f"🌐 Результат браузера:\n{json.dumps(result, ensure_ascii=False, indent=2)[:1000]}"
            
        except Exception as e:
            return f"❌ Ошибка браузера: {e}"
    
    async def _execute_playwright_action(self, action: str, timeout: int) -> str:
        """Выполняет действие через Playwright"""
        # Для начала используем общий метод браузера
        return await self._execute_browser_action(action, "playwright")
    
    async def _execute_web_search(self, query: str) -> str:
        """Выполняет веб-поиск"""
        try:
            # Очищаем запрос
            clean_query = re.sub(r'поиск|найди|ищи', '', query, flags=re.IGNORECASE).strip()
            
            async with aiohttp.ClientSession() as session:
                # Используем DuckDuckGo
                url = f"https://api.duckduckgo.com/?q={aiohttp.helpers.quote(clean_query)}&format=json"
                
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        result = f"🔍 Результаты по '{clean_query}':\n"
                        
                        if data.get("AbstractText"):
                            result += f"📖 {data['AbstractText'][:300]}\n\n"
                        
                        if data.get("RelatedTopics"):
                            for i, topic in enumerate(data["RelatedTopics"][:5], 1):
                                if isinstance(topic, dict) and "Text" in topic:
                                    result += f"{i}. {topic['Text'][:150]}\n"
                        
                        return result
                    else:
                        return f"❌ Ошибка поиска: HTTP {response.status}"
                        
        except Exception as e:
            return f"❌ Ошибка поиска: {e}"
    
    async def _analyze_data(self, action: str) -> str:
        """Анализирует данные"""
        try:
            # Ищем файлы данных
            files = []
            for ext in ['.csv', '.json', '.xlsx', '.xls']:
                for file in Path('.').glob(f'*{ext}'):
                    files.append(str(file))
            
            if not files:
                # Создаем тестовые данные
                import pandas as pd
                import numpy as np
                
                data = pd.DataFrame({
                    'Дата': pd.date_range('2024-01-01', periods=100),
                    'Значение': np.random.randn(100).cumsum(),
                    'Категория': np.random.choice(['A', 'B', 'C'], 100)
                })
                
                data.to_csv('тестовые_данные.csv', index=False)
                files = ['тестовые_данные.csv']
            
            # Анализируем первый найденный файл
            import pandas as pd
            
            df = pd.read_csv(files[0])
            
            analysis = f"📊 Анализ файла {files[0]}:\n"
            analysis += f"Размер: {df.shape[0]} строк, {df.shape[1]} столбцов\n"
            analysis += f"Колонки: {', '.join(df.columns)}\n\n"
            
            analysis += "📈 Статистика:\n"
            analysis += df.describe().to_string() + "\n\n"
            
            # Проверяем на пропуски
            missing = df.isnull().sum().sum()
            if missing > 0:
                analysis += f"⚠️  Пропущенных значений: {missing}\n"
            
            return analysis
            
        except Exception as e:
            return f"❌ Ошибка анализа: {e}"
    
    def _evaluate_result(self, result: Dict) -> float:
        """Оценивает результат выполнения"""
        if not result or "graph_summary" not in result:
            return 0
        
        summary = result["graph_summary"]
        
        # Базовый score
        score = summary["success_rate"] * 100
        
        # Бонусы
        if summary["total_tasks"] > 5:
            score += 5  # За сложность
        
        if summary["failed"] == 0:
            score += 10  # За безошибочность
        
        # Штрафы
        if "execution_results" in result:
            errors = sum(1 for r in result["execution_results"] 
                        if isinstance(r, dict) and r.get("error"))
            score -= errors * 5
        
        return max(0, min(100, score))
    
    def _merge_plans(self, old_plan: Dict, improvements: Dict) -> Dict:
        """Объединяет старый план с улучшениями"""
        if "new_steps" in improvements:
            # Используем полностью новые шаги
            old_plan["steps"] = improvements["new_steps"]
        
        if "improvements" in improvements:
            old_plan["improvements_applied"] = improvements["improvements"]
        
        if "fallback_strategies" in improvements:
            old_plan["fallback_strategies"] = improvements["fallback_strategies"]
        
        return old_plan
    
    async def run_interactive(self):
        """Интерактивный режим"""
        print("\n🎮 РЕЖИМЫ:")
        print("  1. Сложная задача с самоулучшением")
        print("  2. Быстрая задача (без итераций)")
        print("  3. Управление памятью")
        print("  4. Тест браузера")
        print("  5. Выход")
        
        while True:
            try:
                choice = input("\nВыбери режим (1-5): ").strip()
                
                if choice == "1":
                    task = input("Введи сложную задачу: ").strip()
                    if task.lower() in ["выход", "exit"]:
                        break
                    
                    result = await self.process_complex_task(task)
                    self._print_complex_result(result)
                
                elif choice == "2":
                    task = input("Введи быструю задачу: ").strip()
                    if not task:
                        continue
                    
                    # Простое выполнение
                    if "браузер" in task.lower() or "сайт" in task.lower():
                        result = await self.browser.browse_with_selenium(
                            "https://www.google.com" if "http" not in task else task,
                            []
                        )
                        print(f"🌐 Результат: {json.dumps(result, ensure_ascii=False)[:500]}")
                    else:
                        # Простая shell команда
                        result = await self._execute_shell(task, 30)
                        print(f"🖥️  Результат: {result}")
                
                elif choice == "3":
                    self._manage_memory()
                
                elif choice == "4":
                    await self._test_browser()
                
                elif choice == "5":
                    break
                
                else:
                    print("❌ Неверный выбор")
                    
            except KeyboardInterrupt:
                print("\n\n🛑 Остановлено пользователем")
                break
            except Exception as e:
                print(f"\n💥 Ошибка: {e}")
                import traceback
                traceback.print_exc()
    
    def _print_complex_result(self, result: Dict):
        """Выводит результат сложной задачи"""
        print(f"\n{'='*80}")
        print("🎉 РЕЗУЛЬТАТ ВЫПОЛНЕНИЯ СЛОЖНОЙ ЗАДАЧИ")
        print('='*80)
        
        print(f"📋 Задача: {result['task'][:100]}...")
        print(f"⭐ Лучшая оценка: {result['best_iteration_score']:.1f}/100")
        print(f"🔄 Итераций: {result['total_iterations']}")
        
        if "graph_summary" in result:
            summary = result["graph_summary"]
            print(f"📊 Граф задач: {summary['total_tasks']} всего, "
                  f"{summary['completed']} выполнено, "
                  f"{summary['failed']} провалено")
            print(f"📈 Успешность: {summary['success_rate']*100:.1f}%")
        
        if "final_analysis" in result:
            analysis = result["final_analysis"]
            if "recommendations" in analysis:
                print(f"\n💡 РЕКОМЕНДАЦИИ:")
                for i, rec in enumerate(analysis["recommendations"][:3], 1):
                    print(f"  {i}. {rec[:80]}...")
        
        if "memory_stats" in result:
            stats = result["memory_stats"]
            print(f"\n🧠 ПАМЯТЬ: {stats['active_memories']} активных, "
                  f"{stats['compressed_memories']} сжатых, "
                  f"{stats['total_size_kb']:.1f} KB")
        
        print('='*80)
    
    def _manage_memory(self):
        """Управление памятью"""
        print("\n🧠 УПРАВЛЕНИЕ ПАМЯТЬЮ:")
        print("  1. Показать статистику")
        print("  2. Поиск в памяти")
        print("  3. Очистить старые воспоминания")
        print("  4. Назад")
        
        choice = input("Выбери действие: ").strip()
        
        if choice == "1":
            stats = self.long_term_memory.get_stats()
            print(f"\n📊 Статистика памяти:")
            print(f"   Активных: {stats['active_memories']}")
            print(f"   Сжатых: {stats['compressed_memories']}")
            print(f"   Размер: {stats['total_size_kb']:.1f} KB")
            print(f"   Коэффициент сжатия: {stats['compression_ratio']:.2f}")
            
            if stats['most_accessed']:
                print(f"\n🔥 Самые часто используемые:")
                for i, mem in enumerate(stats['most_accessed'], 1):
                    print(f"   {i}. Использовано {mem.get('access_count', 0)} раз")
                    print(f"      {mem.get('content', '')[:60]}...")
        
        elif choice == "2":
            query = input("Поисковый запрос: ").strip()
            if query:
                results = self.long_term_memory.retrieve_context(query, 3)
                print(f"\n🔍 Результаты поиска:")
                print(results[:1000])
        
        elif choice == "3":
            confirm = input("Очистить старые воспоминания? (y/n): ").strip().lower()
            if confirm == 'y':
                old_count = len(self.long_term_memory.memories)
                self.long_term_memory.compress_memories()
                new_count = len(self.long_term_memory.memories)
                print(f"✅ Очищено {old_count - new_count} воспоминаний")
    
    async def _test_browser(self):
        """Тестирует браузер"""
        print("\n🌐 ТЕСТ БРАУЗЕРА:")
        print("  1. Тест Selenium")
        print("  2. Тест Playwright")
        print("  3. Оба")
        print("  4. Назад")
        
        choice = input("Выбери тест: ").strip()
        
        if choice in ["1", "3"]:
            print("\n🧪 Тестирую Selenium...")
            result = await self.browser.browse_with_selenium(
                "https://www.google.com",
                []
            )
            print(f"✅ Selenium: {result.get('title', 'Ошибка')}")
        
        if choice in ["2", "3"]:
            print("\n🎭 Тестирую Playwright...")
            result = await self.browser.browse_with_playwright(
                "https://www.github.com",
                []
            )
            print(f"✅ Playwright: {result.get('title', 'Ошибка')}")
    
    async def cleanup(self):
        """Очистка ресурсов"""
        await self.browser.close()
        self.long_term_memory.save_memories()
        print("🧹 Ресурсы освобождены, память сохранена")

# ==================== УСТАНОВКА ====================
async def setup_omega():
    """Настраивает Omega агента"""
    print("🔧 Настройка MOLOT OMEGA...")
    
    # Проверяем Ollama
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{OLLAMA_HOST}/api/tags", timeout=10) as resp:
                if resp.status == 200:
                    models = await resp.json()
                    model_names = [m['name'] for m in models.get('models', [])]
                    
                    if OLLAMA_MODEL not in model_names:
                        print(f"⚠️  Модель {OLLAMA_MODEL} не найдена")
                        install = input(f"Скачать {OLLAMA_MODEL}? (y/n): ")
                        if install.lower() == 'y':
                            print(f"📦 Скачиваю {OLLAMA_MODEL}...")
                            subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
                            print("✅ Модель скачана")
                    else:
                        print(f"✅ Модель {OLLAMA_MODEL} доступна")
                else:
                    print("❌ Ollama не доступен")
    except:
        print("⚠️  Ollama не запущен. Запусти: ollama serve")
    
    # Проверяем браузерные зависимости
    browser_deps = ["selenium", "playwright", "webdriver-manager"]
    missing_browser = []
    
    for dep in browser_deps:
        try:
            __import__(dep.replace('-', '_'))
        except ImportError:
            missing_browser.append(dep)
    
    if missing_browser:
        print(f"🌐 Отсутствуют браузерные зависимости: {', '.join(missing_browser)}")
        install = input("Установить? (y/n): ")
        if install.lower() == 'y':
            import pip
            for dep in missing_browser:
                if dep == "playwright":
                    # Playwright требует отдельной установки
                    subprocess.run([sys.executable, "-m", "pip", "install", "playwright"])
                    subprocess.run(["playwright", "install", "chromium"])
                else:
                    pip.main(['install', dep])
    
    # Проверяем остальные зависимости
    other_deps = ["aiohttp", "aiofiles", "pandas", "numpy", "networkx", "matplotlib"]
    missing_other = []
    
    for dep in other_deps:
        try:
            __import__(dep.replace('-', '_'))
        except ImportError:
            missing_other.append(dep)
    
    if missing_other:
        print(f"📦 Отсутствуют другие зависимости: {', '.join(missing_other)}")
        install = input("Установить? (y/n): ")
        if install.lower() == 'y':
            import pip
            for dep in missing_other:
                pip.main(['install', dep])
    
    return AutonomousOmegaAgent()

async def main():
    """Главная функция"""
    try:
        agent = await setup_omega()
        
        # Если есть аргументы - выполнить задачу
        if len(sys.argv) > 1:
            task = " ".join(sys.argv[1:])
            result = await agent.process_complex_task(task)
            agent._print_complex_result(result)
        else:
            # Интерактивный режим
            await agent.run_interactive()
        
        await agent.cleanup()
        
    except Exception as e:
        print(f"💥 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Запускаем асинхронно
    asyncio.run(main())
