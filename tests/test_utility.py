import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sources.utility import pretty_print, get_color_map, timer_decorator


class TestGetColorMap(unittest.TestCase):
    def test_returns_expected_keys(self):
        cmap = get_color_map()
        expected = {"success", "failure", "status", "code", "warning", "output", "info"}
        self.assertEqual(set(cmap.keys()), expected)

    def test_values_are_strings(self):
        for v in get_color_map().values():
            self.assertIsInstance(v, str)


class TestPrettyPrint(unittest.TestCase):
    def test_valid_colors_no_crash(self):
        for color in ("success", "failure", "status", "code", "warning", "output", "info"):
            try:
                pretty_print("test", color=color)
            except Exception as exc:
                self.fail(f"pretty_print crashed with color={color}: {exc}")

    def test_unknown_color_falls_back(self):
        try:
            pretty_print("test", color="nonexistent_color")
        except Exception as exc:
            self.fail(f"pretty_print crashed with unknown color: {exc}")


class TestTimerDecorator(unittest.TestCase):
    def test_decorated_function_returns_value(self):
        @timer_decorator
        def add(a, b):
            return a + b

        self.assertEqual(add(2, 3), 5)

    def test_decorated_function_preserves_none(self):
        @timer_decorator
        def noop():
            pass

        self.assertIsNone(noop())


if __name__ == '__main__':
    unittest.main()
