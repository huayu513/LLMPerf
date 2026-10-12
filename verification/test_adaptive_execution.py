import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adaptive_fixtures import EmptyPlanningAdvisor, environment, write_config
from llmperf.artifacts import sha256_file, write_json_atomic
from llmperf.errors import ConfigError
from llmperf.tuning import plan_tuning
from llmperf.workflow import run_saved_plan


class FeedbackAdvisor:
    def __init__(self):
        self.calls = []

    def ask(self, task, payload):
        self.calls.append(copy.deepcopy(payload))
        return {"diagnosis": {"kind": "hypothesis", "evidence_ids": [payload["tested"][-1]["task_id"]],
                              "hypothesis": "验证邻域配置"},
                "next_candidates": [{"candidate_id": payload["shortlist"][-1]["candidate_id"],
                                      "reason": "选择尚未测量的配置"}]}


class AdaptiveExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = write_config(self.root)
        self.executed = []

    def plan(self, **adaptive):
        document = json.loads(self.config.read_text())
        document["search"]["adaptive"].update(adaptive)
        self.config.write_text(json.dumps(document))
        result = plan_tuning(self.config, probe=lambda _: environment(),
                             advisor=EmptyPlanningAdvisor(), calibrate=False)
        self.plan_path = Path(result["plan"])
        self.saved_plan = self.plan_path.read_bytes()
        self.metadata = json.loads(self.saved_plan)["metadata"]
        self.candidates = {row["id"]: row for row in json.loads(self.saved_plan)["candidates"]}
        return result

    def execute(self, task, attempt):
        self.executed.append((task.id, attempt, task.run_class))
        latency = 1.0 if task.candidate_id == self.metadata["baseline_candidate_id"] else 0.5
        return {"status": "VALID", "output_tokens_per_second": 100,
                "latency_seconds": {"p95": latency}, "ttft_seconds": {"p95": 0.1},
                "tpot_seconds": {"p95": 0.01}, "reasons": [],
                "effective_static_config": self.candidates[task.candidate_id]["static_config"]}

    def run_plan(self, advisor=None, executor=None, resume=False):
        return run_saved_plan(self.plan_path, probe=lambda _: environment(),
                              executor=executor or self.execute, advisor=advisor or FeedbackAdvisor(), resume=resume)

    def test_measured_feedback_and_final_verification_preserve_plan(self):
        self.plan()
        advisor = FeedbackAdvisor()
        report = self.run_plan(advisor)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(self.plan_path.read_bytes(), self.saved_plan)
        self.assertLessEqual(report["trials"], 14)
        self.assertEqual(sum(row[2] == "final_repeat" for row in self.executed), 3)
        self.assertEqual(len(report["adaptive"]["finalist_ids"]), 2)
        self.assertTrue(report["adaptive"]["baseline_verified"])
        self.assertAlmostEqual(report["comparison"]["relative_improvement"], 0.5)
        self.assertTrue(advisor.calls[0]["recent"][0]["ttft_seconds"])
        self.assertEqual(advisor.calls[0]["scenario"]["target_concurrency"], 4)
        state = json.loads((self.plan_path.parent / "tune-state.json").read_text())
        self.assertLessEqual(state["adaptive"]["advisor_calls"], 3)
        self.assertEqual(state["adaptive"]["pending"], [])
        self.assertNotIn("test-token", (self.plan_path.parent / "best.json").read_text())
        self.assertEqual(self.run_plan(resume=True)["best"], report["best"])

    def test_failure_triggers_diagnosis_without_claiming_baseline_gain(self):
        self.plan()
        advisor = FeedbackAdvisor()
        def execute(task, attempt):
            if task.candidate_id == self.metadata["baseline_candidate_id"]:
                self.executed.append((task.id, attempt, task.run_class))
                return {"status": "FAILED", "failure_kind": "oom", "failure_phase": "startup",
                        "reasons": ["out_of_memory"]}
            return self.execute(task, attempt)
        report = self.run_plan(advisor, execute)
        self.assertEqual(advisor.calls[0]["recent"][0]["failure_kind"], "oom")
        self.assertEqual(len(advisor.calls[0]["tested"]), 1)
        self.assertFalse(report["adaptive"]["baseline_verified"])
        self.assertIsNone(report["comparison"]["relative_improvement"])

    def test_missing_credentials_fall_back_without_loading_search_config(self):
        self.plan()
        self.config.unlink()
        with patch("llmperf.advisor.LocalAdvisor.ask", side_effect=AssertionError("unexpected network")), \
             patch.dict("os.environ", {"LLMPERF_ADVISOR_KEY": ""}):
            report = run_saved_plan(self.plan_path, probe=lambda _: environment(), executor=self.execute)
        self.assertEqual(report["status"], "PASS")
        records = report["adaptive"]["decision_records"]
        documents = [json.loads((self.plan_path.parent / row["manifest"]).read_text()) for row in records]
        self.assertTrue(any(row["reason"] == "runtime_advisor_unavailable" for row in documents))

    def test_interrupted_trial_reuses_decisions_and_completed_trials(self):
        self.plan(max_advisor_calls=1)
        advisor = FeedbackAdvisor()
        def interrupt(task, attempt):
            if len(self.executed) == 3:
                raise KeyboardInterrupt()
            return self.execute(task, attempt)
        with self.assertRaises(KeyboardInterrupt):
            self.run_plan(advisor, interrupt)
        completed = list(self.executed)
        decision = self.plan_path.parent / "decisions/000002.json"
        digest = sha256_file(decision)
        report = self.run_plan(advisor, resume=True)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(sha256_file(decision), digest)
        self.assertEqual(len(advisor.calls), 1)
        for task_id, attempt, _ in completed:
            self.assertEqual(sum(row[0] == task_id and row[1] == attempt for row in self.executed), 1)
        self.assertLessEqual(report["trials"], 14)

    def test_interrupted_advisor_allocation_is_not_reissued(self):
        self.plan(max_advisor_calls=1)
        class InterruptedAdvisor:
            def ask(self, *_):
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_plan(InterruptedAdvisor())
        advisor = FeedbackAdvisor()
        report = self.run_plan(advisor, resume=True)
        self.assertFalse(advisor.calls)
        self.assertEqual(report["adaptive"]["advisor_calls"], 1)
        document = json.loads((self.plan_path.parent / "decisions/000002.json").read_text())
        self.assertEqual(document["reason"], "interrupted_advisor_call")

    def test_decision_file_survives_interrupted_state_update(self):
        self.plan(max_advisor_calls=1)
        advisor = FeedbackAdvisor()
        def interrupt(path, value):
            write_json_atomic(path, value)
            if Path(path).name == "000002.json":
                raise KeyboardInterrupt()
        with patch("llmperf.adaptive.write_json_atomic", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.run_plan(advisor)
        report = self.run_plan(advisor, resume=True)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(advisor.calls), 1)

    def test_last_exploration_result_survives_interrupted_state_update(self):
        self.plan(max_advisor_calls=0)
        exploration_limit = self.metadata["planning"]["exploration_capacity"]
        def interrupt(path, value):
            write_json_atomic(path, value)
            if Path(path).name == "trial.json" and len(self.executed) == exploration_limit:
                raise KeyboardInterrupt()
        with patch("llmperf.tuning.write_json_atomic", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.run_plan()
        report = self.run_plan(resume=True)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(self.executed), report["trials"])
        state = json.loads((self.plan_path.parent / "tune-state.json").read_text())
        self.assertEqual(len(state["observations"]), exploration_limit)

    def test_completed_resume_rejects_changed_trial_decision_and_report(self):
        self.plan()
        self.run_plan()
        state = json.loads((self.plan_path.parent / "tune-state.json").read_text())
        trial = next(entry for entries in state["trials"].values() for entry in entries if entry["completed"])
        paths = [self.plan_path.parent / trial["manifest"],
                 self.plan_path.parent / "decisions/000001.json", self.plan_path.parent / "best.json"]
        for path in paths:
            original = path.read_bytes()
            path.write_text("{}")
            with self.subTest(path=path), self.assertRaises(ConfigError):
                self.run_plan(resume=True)
            path.write_bytes(original)

    def test_final_repeats_override_exploration_predictions(self):
        self.plan()
        def execute(task, attempt):
            result = self.execute(task, attempt)
            if task.run_class == "final_repeat" and task.candidate_id != self.metadata["baseline_candidate_id"]:
                result["latency_seconds"]["p95"] = 2.0
            return result
        report = self.run_plan(executor=execute)
        self.assertEqual(report["best"]["candidate_id"], self.metadata["baseline_candidate_id"])
        self.assertEqual(report["comparison"]["relative_improvement"], 0)

    def test_time_budget_reserves_verification(self):
        document = json.loads(self.config.read_text())
        document["search"]["max_seconds"] = 5
        self.config.write_text(json.dumps(document))
        self.plan()
        clock = [0.0]
        def execute(task, attempt):
            clock[0] += 1
            return self.execute(task, attempt)
        with patch("llmperf.tuning.time.monotonic", side_effect=lambda: clock[0]):
            report = self.run_plan(executor=execute)
        self.assertEqual(report["status"], "PASS")
        self.assertLessEqual(report["adaptive"]["elapsed_seconds"], 5)
        self.assertTrue(any(row["reason"] == "time_reserved_for_final_verification"
                            for row in report["stop_reasons"]))

    def test_advisor_wait_counts_against_time_and_cannot_launch_after_deadline(self):
        document = json.loads(self.config.read_text())
        document["search"]["max_seconds"] = 5
        self.config.write_text(json.dumps(document))
        self.plan(batch_size=1, max_advisor_calls=1)
        clock = [0.0]
        class SlowAdvisor(FeedbackAdvisor):
            def ask(self, task, payload):
                clock[0] += 6
                return super().ask(task, payload)
        def execute(task, attempt):
            clock[0] += 0.1
            return self.execute(task, attempt)
        with patch("llmperf.tuning.time.monotonic", side_effect=lambda: clock[0]):
            report = self.run_plan(SlowAdvisor(), execute)
        self.assertEqual(report["status"], "INCONCLUSIVE")
        self.assertEqual(len(self.executed), 1)
        self.assertGreater(report["adaptive"]["elapsed_seconds"], 5)

    def test_final_result_survives_interrupted_state_update_at_budget_limit(self):
        self.plan(max_advisor_calls=0)
        def interrupt(path, value):
            write_json_atomic(path, value)
            if Path(path).name == "trial.json" and len(self.executed) == 14:
                raise KeyboardInterrupt()
        with patch("llmperf.tuning.write_json_atomic", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.run_plan()
        report = self.run_plan(resume=True)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["trials"], 14)
        self.assertEqual(len(self.executed), 14)

    def test_resume_checks_native_usage_artifacts_and_observation_scores(self):
        self.plan(max_advisor_calls=0)
        def execute(task, attempt):
            result = self.execute(task, attempt)
            directory = self.plan_path.parent / "trials" / task.id / f"attempt-{attempt:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "native.requests.jsonl").write_text(json.dumps({
                "ok": True, "prompt_tokens": 100, "completion_tokens": 100}) + "\n")
            result["result_path"] = str(directory)
            return result
        self.run_plan(executor=execute)
        state_path = self.plan_path.parent / "tune-state.json"
        original_state = state_path.read_bytes()
        state = json.loads(original_state)
        state["observations"][0]["score"] = 9999
        state_path.write_text(json.dumps(state))
        with self.assertRaises(ConfigError):
            self.run_plan(resume=True)
        state_path.write_bytes(original_state)
        usage = next((self.plan_path.parent / "trials").rglob("native.requests.jsonl"))
        usage.write_text("{}\n")
        with self.assertRaises(ConfigError):
            self.run_plan(resume=True)


if __name__ == "__main__":
    unittest.main()
