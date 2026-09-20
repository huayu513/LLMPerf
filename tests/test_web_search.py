import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from s1slow.Automation.web import server


class WebSearchTests(unittest.TestCase):
    def test_plan_candidates_exposes_failed_and_skipped_state(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "plan.json").write_text(
                '{"candidates":[{"id":"c0","static_config":{}},'
                '{"id":"c1","static_config":{}}]}'
            )
            (run_dir / "results-index.json").write_text('{"rows":[]}')
            (run_dir / "search-state.json").write_text(
                '{"candidate_states":{'
                '"c0":{"status":"FAILED","reasons":["server_info_missing"]},'
                '"c1":{"status":"SKIPPED","reasons":["backend_blocked"]}'
                '}}'
            )
            rows = server.plan_candidates(run_dir)
        self.assertEqual(
            [(row["latest_status"], row["latest_reasons"]) for row in rows],
            [
                ("FAILED", ["server_info_missing"]),
                ("SKIPPED", ["backend_blocked"]),
            ],
        )

    def test_preview_uses_high_thinking_when_plan_metadata_is_older(self):
        preview = server.preview_launch(
            {"served_model_name": "model", "expected_request_count": 4},
            {"id": "c0", "gpu_indexes": [0], "static_config": {}},
            1,
        )
        self.assertEqual(preview["requested_parameters"]["chat_template_kwargs"], {
            "enable_thinking": True,
            "reasoning_effort": "high",
            "thinking": True,
        })

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
