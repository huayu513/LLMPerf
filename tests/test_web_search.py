import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from s1slow.Automation.web import server


class WebSearchTests(unittest.TestCase):
    def test_debug_search_increases_concurrency_by_sixteen(self):
        calls = []
        metadata = {
            "expected_request_count": 64,
            "search": {"concurrency_max": 64, "start_concurrency": 16},
        }
        candidate = {"id": "c0", "gpu_indexes": (0,), "static_config": {}}

        class Control:
            def check(self):
                return None

        def fake_context(run_dir, candidate_id):
            return {}, metadata, {"c0": candidate}, candidate

        def fake_attempt(
            emit,
            control,
            run_dir,
            metadata,
            candidate,
            candidate_id,
            concurrency,
            run_class,
            mode,
            scale,
            task_prefix,
        ):
            calls.append(concurrency)
            score = {16: 10, 32: 18, 48: 25, 64: 23}[concurrency]
            return {
                "status": "VALID",
                "output_tokens_per_second": score,
                "concurrency": concurrency,
            }

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(server, "load_debug_context", side_effect=fake_context), \
                patch.object(server, "run_debug_attempt", side_effect=fake_attempt):
            summary = server.debug_search_worker(
                lambda event: None,
                Control(),
                Path(directory),
                "c0",
            )

        self.assertEqual(calls, [16, 32, 48, 64])
        self.assertEqual(summary["best_concurrency"], 48)


if __name__ == "__main__":
    unittest.main()
