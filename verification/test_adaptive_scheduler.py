import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from adaptive_fixtures import SCENE
from llmperf.adaptive import AdaptiveScheduler
from llmperf.artifacts import sha256_file, write_json_atomic
from llmperf.errors import ConfigError
from llmperf.types import CandidateConfig


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.candidates = {str(index): asdict(CandidateConfig(str(index), static_config={
            "tp": 1, "mem_fraction_static": 0.75 + index * 0.01})) for index in range(12)}
        self.metadata = {"adaptive_policy": {"version": 1, "batch_size": 2, "max_advisor_calls": 2},
                         "baseline_candidate_id": "0", "scenario": SCENE,
                         "expected_source_sha256": "workload", "expected_request_count": 36,
                         "workload_source": "synthetic"}
        self.state = {"plan_hash": "plan", "observations": [], "decisions": []}

    def scheduler(self, advisor=None):
        return AdaptiveScheduler(self.metadata, self.candidates, self.root, self.state,
                                 lambda: None, advisor=advisor)

    def observe(self, scheduler, ident):
        result = {"candidate_id": ident, "task_id": f"trial-{ident}", "score": -1.0, "status": "VALID"}
        path = self.root / "trials" / f"{ident}.json"
        write_json_atomic(path, result)
        result["evidence"] = {"manifest": str(path.relative_to(self.root)), "sha256": sha256_file(path)}
        result["evidence_summary"] = scheduler.summary(result, ident)
        self.state["observations"].append(result)
        scheduler.consume(ident, result)

    def bootstrap(self, advisor=None):
        scheduler = self.scheduler(advisor)
        for ident in list(scheduler.next_batch(8, 5)):
            self.observe(scheduler, ident)
        return scheduler

    def test_feedback_changes_order_and_is_persisted(self):
        class Advisor:
            def ask(self, task, payload):
                self.payload = payload
                return {"next_candidates": [{"candidate_id": payload["shortlist"][-1]["candidate_id"],
                                              "reason": "test"}]}
        advisor = Advisor()
        scheduler = self.bootstrap(advisor)
        batch = scheduler.next_batch(6, 5)
        self.assertEqual(batch[0], advisor.payload["shortlist"][-1]["candidate_id"])
        self.assertTrue(advisor.payload["recent"])
        document = json.loads((self.root / "decisions/000002.json").read_text())
        self.assertEqual(document["source"], "advisor")
        self.assertEqual(document["selected"], batch)

    def test_runtime_timeout_is_capped_at_300_and_remaining_budget(self):
        self.metadata["runtime_advisor"] = {
            "base_url": "http://localhost/v1", "model": "local", "timeout_seconds": 1000,
        }
        scheduler = self.bootstrap()
        self.assertEqual(scheduler.advisor.settings["timeout_seconds"], 300)
        captured = []

        def reply(client, task, payload):
            captured.append(client.settings["timeout_seconds"])
            return {"next_candidates": [{"candidate_id": payload["shortlist"][-1]["candidate_id"]}]}

        with patch("llmperf.adaptive.LocalAdvisor.ask", autospec=True, side_effect=reply):
            for remaining_seconds in (1000, 7):
                with self.subTest(remaining_seconds=remaining_seconds):
                    batch = scheduler.next_batch(6, remaining_seconds)
                    self.assertEqual(captured[-1], min(300, remaining_seconds))
                    for ident in list(batch):
                        self.observe(scheduler, ident)

    def test_unknown_duplicate_and_control_fields_fall_back(self):
        class Advisor:
            def ask(self, task, payload):
                return {"next_candidates": [{"candidate_id": "outside"}], "winner": "outside"}
        scheduler = self.bootstrap(Advisor())
        batch = scheduler.next_batch(6, 5)
        self.assertTrue(set(batch) <= set(self.candidates))
        self.assertEqual(json.loads((self.root / "decisions/000002.json").read_text())["status"], "FALLBACK")

    def test_interrupted_call_consumes_budget_without_reasking(self):
        class Advisor:
            def ask(self, *_):
                raise KeyboardInterrupt()
        scheduler = self.bootstrap(Advisor())
        with self.assertRaises(KeyboardInterrupt):
            scheduler.next_batch(6, 5)
        allocated = copy.deepcopy(self.state)
        self.state = allocated
        resumed = self.scheduler(Advisor())
        self.assertTrue(resumed.next_batch(6, 5))
        self.assertEqual(self.state["adaptive"]["advisor_calls"], 1)
        document = json.loads((self.root / "decisions/000002.json").read_text())
        self.assertEqual(document["reason"], "interrupted_advisor_call")

    def test_modified_decision_or_evidence_is_rejected(self):
        scheduler = self.bootstrap()
        scheduler.next_batch(6, 5)
        path = self.root / "trials/0.json"
        path.write_text("{}")
        with self.assertRaises(ConfigError):
            self.scheduler()

    def test_call_limit_and_pending_reuse(self):
        class Advisor:
            calls = 0
            def ask(self, task, payload):
                self.calls += 1
                return {"next_candidates": []}
        advisor = Advisor()
        scheduler = self.bootstrap(advisor)
        for _ in range(4):
            batch = list(scheduler.next_batch(6, 5))
            self.assertEqual(batch, self.scheduler(advisor).next_batch(6, 5))
            for ident in batch:
                self.observe(scheduler, ident)
        self.assertEqual(advisor.calls, 2)


if __name__ == "__main__":
    unittest.main()
