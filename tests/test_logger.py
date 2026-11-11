import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sources.logger import Logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.log_folder = '.logs'
        self.log_file = 'test_unit.log'
        self.logger = Logger(self.log_file)

    def tearDown(self):
        log_path = os.path.join(self.log_folder, self.log_file)
        if os.path.exists(log_path):
            os.remove(log_path)

    def test_creates_log_folder(self):
        self.assertTrue(os.path.isdir(self.log_folder))

    def test_creates_log_file_on_write(self):
        self.logger.info("init message")
        self.assertTrue(os.path.exists(os.path.join(self.log_folder, self.log_file)))

    def test_info_writes_to_file(self):
        self.logger.info("info test message")
        with open(os.path.join(self.log_folder, self.log_file)) as f:
            content = f.read()
        self.assertIn("info test message", content)

    def test_error_writes_to_file(self):
        self.logger.error("error test message")
        with open(os.path.join(self.log_folder, self.log_file)) as f:
            content = f.read()
        self.assertIn("error test message", content)

    def test_warning_writes_to_file(self):
        self.logger.warning("warning test message")
        with open(os.path.join(self.log_folder, self.log_file)) as f:
            content = f.read()
        self.assertIn("warning test message", content)

    def test_duplicate_message_suppression(self):
        self.logger.info("duplicate msg")
        self.logger.info("duplicate msg")
        with open(os.path.join(self.log_folder, self.log_file)) as f:
            content = f.read()
        self.assertEqual(content.count("duplicate msg"), 1)

    def test_different_messages_not_suppressed(self):
        self.logger.info("msg one")
        self.logger.info("msg two")
        with open(os.path.join(self.log_folder, self.log_file)) as f:
            content = f.read()
        self.assertIn("msg one", content)
        self.assertIn("msg two", content)


if __name__ == '__main__':
    unittest.main()
