import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sources.schemas import executorResult, QueryRequest, QueryResponse


class TestExecutorResult(unittest.TestCase):
    def test_creation(self):
        r = executorResult(block="print('hi')", feedback="ok", success=True, tool_type="python")
        self.assertEqual(r.block, "print('hi')")
        self.assertEqual(r.feedback, "ok")
        self.assertTrue(r.success)
        self.assertEqual(r.tool_type, "python")

    def test_jsonify(self):
        r = executorResult(block="code", feedback="done", success=False, tool_type="bash")
        j = r.jsonify()
        self.assertEqual(j["block"], "code")
        self.assertEqual(j["feedback"], "done")
        self.assertFalse(j["success"])
        self.assertEqual(j["tool_type"], "bash")

    def test_str(self):
        r = executorResult(block="b", feedback="f", success=True, tool_type="t")
        s = str(r)
        self.assertIn("Tool: t", s)
        self.assertIn("Block: b", s)


class TestQueryRequest(unittest.TestCase):
    def test_creation_defaults(self):
        q = QueryRequest(query="hello")
        self.assertEqual(q.query, "hello")
        self.assertTrue(q.tts_enabled)

    def test_jsonify(self):
        q = QueryRequest(query="test", tts_enabled=False)
        j = q.jsonify()
        self.assertEqual(j["query"], "test")
        self.assertFalse(j["tts_enabled"])


class TestQueryResponse(unittest.TestCase):
    def test_creation(self):
        r = QueryResponse(
            done="true", answer="ans", reasoning="reason",
            agent_name="agent", success="true",
            blocks={}, status="ok", uid="123"
        )
        self.assertEqual(r.done, "true")
        self.assertEqual(r.answer, "ans")
        self.assertEqual(r.uid, "123")

    def test_jsonify(self):
        r = QueryResponse(
            done="false", answer="a", reasoning="r",
            agent_name="n", success="false",
            blocks={"k": "v"}, status="s", uid="u"
        )
        j = r.jsonify()
        self.assertEqual(j["done"], "false")
        self.assertEqual(j["blocks"], {"k": "v"})
        self.assertEqual(j["uid"], "u")


if __name__ == '__main__':
    unittest.main()
