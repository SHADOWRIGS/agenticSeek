import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sources.tools.safety import is_any_unsafe, is_unsafe


class TestIsUnsafe(unittest.TestCase):
    def test_unsafe_unix_commands(self):
        dangerous = ["rm -rf /", "dd if=/dev/zero", "chmod 777 file",
                      "chown root file", "shutdown -h now", "kill -9 1234"]
        for cmd in dangerous:
            with self.subTest(cmd=cmd):
                self.assertTrue(is_unsafe(cmd), f"'{cmd}' should be unsafe")

    def test_safe_unix_commands(self):
        safe = ["ls -la", "pwd", "cat file.txt", "echo hello",
                "python script.py", "pip install flask", "cd /tmp"]
        for cmd in safe:
            with self.subTest(cmd=cmd):
                self.assertFalse(is_unsafe(cmd), f"'{cmd}' should be safe")

    def test_git_detected_as_unsafe_unix(self):
        self.assertTrue(is_unsafe("git push origin main"))

    def test_empty_string_is_safe(self):
        self.assertFalse(is_unsafe(""))


class TestIsAnyUnsafe(unittest.TestCase):
    def test_all_safe(self):
        self.assertFalse(is_any_unsafe(["ls", "pwd", "echo hi"]))

    def test_mixed_has_unsafe(self):
        self.assertTrue(is_any_unsafe(["ls", "rm -rf /", "pwd"]))

    def test_empty_list(self):
        self.assertFalse(is_any_unsafe([]))

    def test_all_unsafe(self):
        self.assertTrue(is_any_unsafe(["rm file", "kill 123"]))


if __name__ == '__main__':
    unittest.main()
