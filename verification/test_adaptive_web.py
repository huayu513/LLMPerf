import json
import tempfile
import unittest
from pathlib import Path

from web.server import adaptive_decisions, summarize_run


class AdaptiveWebTests(unittest.TestCase):
    def test_summary_and_decision_view_expose_safe_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "decisions").mkdir()
            (root / "plan.json").write_text(json.dumps({"candidates": [], "metadata": {
                "workflow_mode": "tune", "search": {"strategy": "adaptive"}}}))
            (root / "tune-state.json").write_text(json.dumps({"adaptive": {"records": [
                {"id": 1, "manifest": "decisions/000001.json"},
                {"id": 2, "manifest": "../experiment.json"}]}}))
            (root / "decisions/000001.json").write_text(json.dumps({"id": 1, "source": "rule",
                "status": "FALLBACK", "reason": "runtime_advisor_unavailable", "selected": ["c0"],
                "input": {"environment": "not part of the compact UI response"}}))
            summary = summarize_run(root)
            rows = adaptive_decisions(root)
            self.assertEqual(summary["strategy"], "adaptive")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["reason"], "runtime_advisor_unavailable")
            self.assertNotIn("input", rows[0])


if __name__ == "__main__":
    unittest.main()
