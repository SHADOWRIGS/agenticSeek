import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sources.tools.tools import Tools


class TestToolsSaveBlock(unittest.TestCase):
    def setUp(self):
        os.environ['WORK_DIR'] = tempfile.mkdtemp()
        self.tool = Tools()
        self.tool.tag = "python"

    def test_save_block_creates_file(self):
        blocks = ["print('hello')"]
        save_path = "test_output.py"
        self.tool.save_block(blocks, save_path)
        full_path = os.path.join(self.tool.work_dir, save_path)
        self.assertTrue(os.path.exists(full_path))
        with open(full_path) as f:
            self.assertEqual(f.read(), "print('hello')")

    def test_save_block_none_path_does_nothing(self):
        self.tool.save_block(["code"], None)

    def test_save_block_creates_subdirectory(self):
        blocks = ["code here"]
        save_path = "subdir/deep/file.py"
        self.tool.save_block(blocks, save_path)
        full_path = os.path.join(self.tool.work_dir, "subdir/deep", "file.py")
        self.assertTrue(os.path.exists(full_path))


class TestToolsGetParameterValue(unittest.TestCase):
    def setUp(self):
        os.environ['WORK_DIR'] = tempfile.mkdtemp()
        self.tool = Tools()
        self.tool.tag = "python"

    def test_finds_parameter(self):
        block = "name = hello\nage = 25"
        self.assertEqual(self.tool.get_parameter_value(block, "name"), "hello")
        self.assertEqual(self.tool.get_parameter_value(block, "age"), "25")

    def test_missing_parameter_returns_none(self):
        block = "name = hello"
        self.assertIsNone(self.tool.get_parameter_value(block, "missing"))


class TestToolsLoadExecBlockEdgeCases(unittest.TestCase):
    def setUp(self):
        os.environ['WORK_DIR'] = tempfile.mkdtemp()
        self.tool = Tools()
        self.tool.tag = "python"

    def test_nested_backticks_handled(self):
        text = "```python\nprint('```')\n```"
        blocks, path = self.tool.load_exec_block(text)
        self.assertIsNotNone(blocks)
        self.assertEqual(len(blocks), 1)

    def test_multiple_tags_only_matches_correct(self):
        text = "```javascript\nconsole.log('hi')\n```\n```python\nprint('hi')\n```"
        blocks, path = self.tool.load_exec_block(text)
        self.assertIsNotNone(blocks)
        self.assertEqual(len(blocks), 1)
        self.assertIn("print('hi')", blocks[0])

    def test_unclosed_block_returns_empty(self):
        text = "```python\nprint('hi')"
        blocks, path = self.tool.load_exec_block(text)
        self.assertEqual(blocks, [])

    def test_empty_block(self):
        text = "```python\n```"
        blocks, path = self.tool.load_exec_block(text)
        self.assertIsNotNone(blocks)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].strip(), "")

    def test_block_with_save_path(self):
        text = "```python:src/main.py\nprint('saved')\n```"
        blocks, path = self.tool.load_exec_block(text)
        self.assertIsNotNone(blocks)
        self.assertEqual(path, "src/main.py")
        self.assertIn("print('saved')", blocks[0])


if __name__ == '__main__':
    unittest.main()
