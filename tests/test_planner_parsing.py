"""Tests for PlannerAgent parsing logic using lightweight stubs (no torch dependency)."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sources.logger import Logger
from sources.tools.tools import Tools


class PlannerStub:
    """Lightweight stand-in with just the parsing methods from PlannerAgent."""

    def __init__(self):
        self.logger = Logger("test_planner.log")
        tool = Tools()
        tool.tag = "json"
        self.tools = {"json": tool}
        self.agents = {"coder": None, "file": None, "web": None, "casual": None}

    def get_task_names(self, text: str) -> list[str]:
        tasks_names = []
        lines = text.strip().split('\n')
        for line in lines:
            if line is None:
                continue
            line = line.strip()
            if len(line) == 0:
                continue
            if '##' in line or line[0].isdigit():
                tasks_names.append(line)
                continue
        self.logger.info(f"Found {len(tasks_names)} tasks names.")
        return tasks_names

    def parse_agent_tasks(self, text: str) -> list:
        tasks = []
        tasks_names = self.get_task_names(text)
        blocks, _ = self.tools["json"].load_exec_block(text)
        if blocks is None:
            return []
        for block in blocks:
            line_json = json.loads(block)
            if 'plan' in line_json:
                for task in line_json['plan']:
                    if task['agent'].lower() not in [ag_name.lower() for ag_name in self.agents]:
                        self.logger.warning(f"Agent {task['agent']} does not exist.")
                        return []
                    try:
                        agent = {
                            'agent': task['agent'],
                            'id': task['id'],
                            'task': task['task']
                        }
                    except (KeyError, TypeError):
                        self.logger.warning("Missing field in json plan.")
                        return []
                    if 'need' in task:
                        agent['need'] = task['need']
                    tasks.append(agent)
        if len(tasks_names) != len(tasks):
            names = [task['task'] for task in tasks]
            return list(map(list, zip(names, tasks)))
        return list(map(list, zip(tasks_names, tasks)))


class TestGetTaskNames(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault('WORK_DIR', os.path.join(os.path.dirname(__file__), '..'))
        self.planner = PlannerStub()

    def test_extracts_numbered_tasks(self):
        text = "1. Search the web\n2. Write code\n3. Summarize"
        result = self.planner.get_task_names(text)
        self.assertEqual(len(result), 3)
        self.assertIn("1. Search the web", result)

    def test_extracts_header_tasks(self):
        text = "## Task A\nsome text\n## Task B\nmore text"
        result = self.planner.get_task_names(text)
        self.assertEqual(len(result), 2)
        self.assertIn("## Task A", result)

    def test_empty_input(self):
        result = self.planner.get_task_names("")
        self.assertEqual(result, [])

    def test_no_tasks(self):
        text = "Just a regular paragraph with no numbers or headers."
        result = self.planner.get_task_names(text)
        self.assertEqual(result, [])

    def test_mixed_content(self):
        text = "intro\n1. First task\ndetails\n## Second heading\nmore"
        result = self.planner.get_task_names(text)
        self.assertEqual(len(result), 2)


class TestParseAgentTasks(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault('WORK_DIR', os.path.join(os.path.dirname(__file__), '..'))
        self.planner = PlannerStub()

    def test_valid_plan_json(self):
        plan = {
            "plan": [
                {"agent": "coder", "id": "1", "task": "Write tests"},
                {"agent": "web", "id": "2", "task": "Search docs", "need": "1"}
            ]
        }
        text = "1. Write tests\n2. Search docs\n```json\n" + json.dumps(plan) + "\n```"
        result = self.planner.parse_agent_tasks(text)
        self.assertEqual(len(result), 2)

    def test_unknown_agent_returns_empty(self):
        plan = {"plan": [{"agent": "unknown_agent", "id": "1", "task": "Do stuff"}]}
        text = "1. Do stuff\n```json\n" + json.dumps(plan) + "\n```"
        result = self.planner.parse_agent_tasks(text)
        self.assertEqual(result, [])

    def test_no_json_block_returns_empty(self):
        text = "Just some text without any json blocks"
        result = self.planner.parse_agent_tasks(text)
        self.assertEqual(result, [])

    def test_missing_field_returns_empty(self):
        plan = {"plan": [{"agent": "coder"}]}
        text = "1. task\n```json\n" + json.dumps(plan) + "\n```"
        result = self.planner.parse_agent_tasks(text)
        self.assertEqual(result, [])

    def test_task_with_need_field(self):
        plan = {
            "plan": [
                {"agent": "coder", "id": "1", "task": "Code it"},
                {"agent": "file", "id": "2", "task": "Save it", "need": "1"}
            ]
        }
        text = "1. Code it\n2. Save it\n```json\n" + json.dumps(plan) + "\n```"
        result = self.planner.parse_agent_tasks(text)
        self.assertEqual(len(result), 2)
        self.assertIn("need", result[1][1])


if __name__ == '__main__':
    unittest.main()
