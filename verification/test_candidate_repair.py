import argparse
import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.server import image_capabilities
from llmperf.adaptive import AdaptiveScheduler
from llmperf.artifacts import sha256_file, write_json_atomic
from llmperf.candidate_repair import proposal_candidate, runtime_candidate_records
from llmperf.errors import ConfigError
from llmperf.feature_catalog import engine_catalog, validate_value
from llmperf.native_planning import observed_knobs
from llmperf.planner import write_plan
from llmperf.tuning import execute_tuning_plan
from llmperf.types import CandidateConfig, Plan


def catalog():
    return engine_catalog("vllm", {"schema": {"parameters": {
        "speculative_config": {"flags": ["--speculative-config"], "type": "object"},
        "gpu_memory_utilization": {"flags": ["--gpu-memory-utilization"], "type": "number"},
        "moe_backend": {"flags": ["--moe-backend"], "type": "string", "choices": ["auto", "marlin"]}},
        "nested": {"speculative_config": {
            "method": {"type": "string", "choices": ["dspark", "mtp"]},
            "num_speculative_tokens": {"type": "integer", "default": None},
            "model": {"type": "string"}}}}})


class NativeDependencyTests(unittest.TestCase):
    def test_fallback_does_not_generate_method_only_speculation(self):
        evidence = catalog()
        self.assertNotIn("speculative_config", observed_knobs(evidence))
        evidence["nested"]["speculative_config"]["num_speculative_tokens"]["default"] = 3
        self.assertEqual(observed_knobs(evidence)["speculative_config"], [
            {"method": "dspark", "num_speculative_tokens": 3},
            {"method": "mtp", "num_speculative_tokens": 3}])

    def test_dspark_needs_a_length_without_imposing_the_rule_on_mtp(self):
        evidence = catalog()
        with self.assertRaisesRegex(ValueError, "num_speculative_tokens"):
            validate_value("speculative_config", {"method": "dspark"}, evidence)
        validate_value("speculative_config", {"method": "mtp"}, evidence)
        validate_value("speculative_config", {"method": "dspark", "num_speculative_tokens": 7}, evidence)
        with self.assertRaises(ValueError):
            validate_value("speculative_config", {"method": "dspark", "num_speculative_tokens": True}, evidence)

    def test_declared_required_fields_are_checked(self):
        evidence = catalog()
        evidence["nested"]["speculative_config"]["num_speculative_tokens"]["required"] = True
        with self.assertRaisesRegex(ValueError, "num_speculative_tokens"):
            validate_value("speculative_config", {"method": "mtp"}, evidence)

    def test_preflight_constructs_native_config_after_json_parsing(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--speculative-config", type=json.loads)
        constructed = []

        def config(**spec):
            constructed.append(spec)
            if not spec.get("num_speculative_tokens"):
                raise ValueError("num_speculative_tokens must be set")

        with patch.object(image_capabilities, "build_parser", return_value=(parser, None)), \
             patch.object(image_capabilities.importlib, "import_module", return_value=SimpleNamespace(SpeculativeConfig=config)):
            results = image_capabilities.preflight("vllm", [
                {"id": "bad", "argv": ["--speculative-config", '{"method":"dspark"}']},
                {"id": "good", "argv": ["--speculative-config", '{"method":"dspark","num_speculative_tokens":3}']}])
        self.assertEqual(len(constructed), 2)
        self.assertEqual(results[0]["status"], "parser_rejected")
        self.assertEqual(results[0]["scope"], "native_configuration")
        self.assertEqual(results[0]["failure_kind"], "configuration")
        self.assertEqual(results[1]["status"], "preflight_passed")
        self.assertFalse(results[1]["runtime_verified"])

    def test_preflight_with_model_context_uses_engine_configuration(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--speculative-config", type=json.loads)
        calls = []
        args = SimpleNamespace(create_engine_config=lambda: calls.append("engine_config"))
        engine_args = SimpleNamespace(from_cli_args=lambda namespace: args)
        with patch.object(image_capabilities, "build_parser", return_value=(parser, None)), \
             patch.object(image_capabilities.importlib, "import_module", return_value=SimpleNamespace(EngineArgs=engine_args)):
            results = image_capabilities.preflight("vllm", [{"id": "context", "model_context": True,
                "argv": ["--speculative-config", '{"method":"dspark","num_speculative_tokens":3}']}])
        self.assertEqual(calls, ["engine_config"])
        self.assertEqual(results[0]["status"], "preflight_passed")

    def test_native_error_input_context_does_not_hide_missing_count(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--speculative-config", type=json.loads)
        class NativeError(ValueError):
            def errors(self):
                return [{"loc": (), "type": "value_error", "msg": "num_speculative_tokens must be set"}]
        def reject(**spec):
            raise NativeError("num_speculative_tokens must be set [type=value_error, input_value={'target_model_config':None}]")
        with patch.object(image_capabilities, "build_parser", return_value=(parser, None)), \
             patch.object(image_capabilities.importlib, "import_module", return_value=SimpleNamespace(SpeculativeConfig=reject)):
            results = image_capabilities.preflight("vllm", [{"id": "missing",
                "argv": ["--speculative-config", '{"method":"dspark"}']}])
        self.assertEqual(results[0]["status"], "parser_rejected")
        self.assertEqual(results[0]["configuration_errors"][0]["type"], "value_error")


class RuntimeRepairTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.candidates = {"baseline": asdict(CandidateConfig("baseline", static_config={
            "engine": "vllm", "tp": 1, "dp": 1, "pp": 1, "gpu_memory_utilization": 0.9})),
            "broken": asdict(CandidateConfig("broken", static_config={
                "engine": "vllm", "tp": 1, "dp": 1, "pp": 1, "gpu_memory_utilization": 0.9,
                "speculative_config": {"method": "dspark"}})),
            "same-defect": asdict(CandidateConfig("same-defect", tp=2, static_config={
                "engine": "vllm", "tp": 2, "dp": 1, "pp": 1, "speculative_config": {"method": "dspark"}}))}
        self.frozen_candidates = copy.deepcopy(self.candidates)
        self.metadata = {"engine": "vllm", "image": "fixture", "engine_catalog": catalog(),
            "configuration": {}, "search": {}, "adaptive_policy": {"version": 2, "batch_size": 2,
                "max_advisor_calls": 3, "max_generated_candidates": 2}, "baseline_candidate_id": "baseline",
            "scenario": {"target_concurrency": 4}, "expected_source_sha256": "workload",
            "expected_request_count": 4, "workload_source": "captured"}
        self.state = {"plan_hash": "plan", "observations": [], "decisions": []}

    def save(self):
        write_json_atomic(self.root / "tune-state.json", self.state)

    def scheduler(self, advisor=None):
        return AdaptiveScheduler(self.metadata, self.candidates, self.root, self.state, self.save, advisor)

    def proposal(self, **changes):
        return {"base_id": "broken", "patch": {"speculative_config": {"num_speculative_tokens": 3}},
                "reason": "补齐缺失字段", "evidence_ids": ["nested:speculative_config"], **changes}

    def preflight(self, draft, config, root, **kwargs):
        self.assertGreater(kwargs["timeout_seconds"], 0)
        report = {"results": [{"id": draft.candidates[0].id, "status": "preflight_passed"}],
                  "runtime_verified": False}
        write_json_atomic(root / "native-preflight.json", report)
        return report

    def bootstrap(self, advisor):
        scheduler = self.scheduler(advisor)
        self.assertEqual(scheduler.next_batch(8, 5), ["baseline"])
        result = {"candidate_id": "baseline", "task_id": "trial-baseline", "score": 1.0, "status": "VALID"}
        path = self.root / "trials/baseline.json"
        write_json_atomic(path, result)
        result["evidence"] = {"manifest": "trials/baseline.json", "sha256": sha256_file(path)}
        result["evidence_summary"] = scheduler.summary(result, "baseline")
        self.state["observations"].append(result)
        scheduler.consume("baseline", result)
        self.save()
        return scheduler

    def test_patch_merges_nested_fields_and_allows_new_numeric_values(self):
        candidate = proposal_candidate(self.proposal(), self.candidates, self.metadata,
                                       {"broken"}, {"nested:speculative_config"})
        self.assertEqual(candidate["static_config"]["speculative_config"], {
            "method": "dspark", "num_speculative_tokens": 3})
        self.assertEqual(self.candidates, self.frozen_candidates)
        self.assertTrue(candidate["id"].startswith("repair-"))

    def test_protected_conditions_and_explicit_restrictions_cannot_be_patched(self):
        patches = [{"tp": 4}, {"max_model_len": 10}, {"gpu_indexes": [1]}, {"model": "other"},
                   {"speculative_config": {"model": "other"}}, {"gpu_memory_utilization": 1},
                   {"speculative_config": {"num_speculative_tokens": False}}]
        for value in patches:
            with self.subTest(patch=value), self.assertRaises(ValueError):
                proposal_candidate(self.proposal(patch=value), self.candidates, self.metadata,
                                   {"broken"}, {"nested:speculative_config"})
        self.metadata["configuration"] = {"engine_options": {"gpu_memory_utilization": 0.9}}
        with self.assertRaisesRegex(ValueError, "user restriction"):
            proposal_candidate(self.proposal(patch={"gpu_memory_utilization": 0.83,
                "speculative_config": {"num_speculative_tokens": 3}}), self.candidates, self.metadata,
                {"broken"}, {"nested:speculative_config"})

    def test_native_preflight_then_registration_resume_and_ui_records(self):
        request = self.proposal()
        advisor = SimpleNamespace(ask=lambda *args: {"candidate_proposals": [request]})
        scheduler = self.bootstrap(advisor)
        self.assertEqual(scheduler.remaining(), [])
        with patch("llmperf.adaptive.preflight_candidates", side_effect=self.preflight):
            batch = scheduler.next_batch(7, 5)
        self.assertEqual(len(batch), 1)
        self.assertTrue(batch[0].startswith("repair-"))
        self.assertNotIn("broken", scheduler.remaining())
        self.assertNotIn("same-defect", scheduler.remaining())
        records = runtime_candidate_records(self.root)
        self.assertEqual(records[0]["request"], request)
        self.assertEqual(records[0]["candidate"]["id"], batch[0])
        from web.server import adaptive_decisions, plan_candidates
        write_json_atomic(self.root / "plan.json", {"metadata": self.metadata,
                                                    "candidates": list(self.frozen_candidates.values())})
        rows = plan_candidates(self.root)
        generated = next(row for row in rows if row["id"] == batch[0])
        self.assertEqual(generated["runtime_proposal"]["base_id"], "broken")
        self.assertEqual(generated["native_preflight"]["status"], "preflight_passed")
        self.assertEqual(adaptive_decisions(self.root)[-1]["proposals"][0]["candidate_id"], batch[0])
        self.candidates = copy.deepcopy(self.frozen_candidates)
        resumed = self.scheduler(advisor)
        self.assertIn(batch[0], self.candidates)
        self.assertEqual(resumed.next_batch(7, 5), batch)
        artifact = self.root / records[0]["preflight_manifest"]
        artifact.write_text("{}")
        with self.assertRaisesRegex(ConfigError, "preflight evidence changed"):
            self.scheduler(advisor)

    def test_unknown_native_validation_is_not_admitted(self):
        advisor = SimpleNamespace(ask=lambda *args: {"candidate_proposals": [self.proposal()]})
        scheduler = self.bootstrap(advisor)
        report = {"results": [{"id": "unknown", "status": "unknown", "reason": "device unavailable"}]}
        with patch("llmperf.adaptive.preflight_candidates", return_value=report):
            self.assertEqual(scheduler.next_batch(7, 5), [])
        self.assertEqual(self.candidates, self.frozen_candidates)
        self.assertTrue(scheduler.can_propose())
        payload = scheduler._payload([], 1, 7, 5)
        self.assertTrue(payload["previous_proposal_rejections"])

    def test_interruption_after_decision_write_restores_candidate_without_reasking(self):
        advisor = SimpleNamespace(ask=lambda *args: {"candidate_proposals": [self.proposal()]})
        scheduler = self.bootstrap(advisor)
        def interrupt(path, value):
            write_json_atomic(path, value)
            if Path(path).name == "000002.json":
                raise KeyboardInterrupt()
        with patch("llmperf.adaptive.preflight_candidates", side_effect=self.preflight), \
             patch("llmperf.adaptive.write_json_atomic", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            scheduler.next_batch(7, 5)
        self.candidates = copy.deepcopy(self.frozen_candidates)
        advisor.ask = lambda *args: self.fail("recorded proposal must not be requested twice")
        resumed = self.scheduler(advisor)
        batch = resumed.next_batch(7, 5)
        self.assertTrue(batch[0].startswith("repair-"))
        self.assertIn(batch[0], self.candidates)
        self.assertEqual(self.state["adaptive"]["advisor_calls"], 1)

    def test_old_policy_rejects_runtime_proposals(self):
        self.metadata["adaptive_policy"]["version"] = 1
        scheduler = self.scheduler(SimpleNamespace(ask=lambda *_: {}))
        with self.assertRaisesRegex(ValueError, "unsupported feedback fields"):
            scheduler._validate_reply({"candidate_proposals": [self.proposal()]}, {"shortlist": [], "batch_size": 1})

    def test_proposal_and_batch_allowances_are_enforced(self):
        scheduler = self.scheduler(SimpleNamespace(ask=lambda *_: {}))
        payload = {"shortlist": [{"candidate_id": "baseline"}], "batch_size": 1,
                   "tested": [], "remaining_proposals": 0}
        with self.assertRaisesRegex(ValueError, "too many"):
            scheduler._validate_proposals({"candidate_proposals": [self.proposal()]}, payload)
        with self.assertRaisesRegex(ValueError, "exceed batch_size"):
            scheduler._validate_reply({"next_candidates": [{"candidate_id": "baseline"}],
                                       "candidate_proposals": [self.proposal()]}, payload)

    def test_repaired_candidate_is_measured_repeated_and_resumed_within_budget(self):
        self.metadata.update(workflow_mode="tune", scene_calibration=None,
            scenario={"target_concurrency": 4, "priority": "latency"},
            search={"strategy": "adaptive", "max_trials": 6, "repetitions": 1,
                    "max_seconds": 30, "promotion_tolerance": 0.05},
            planning={"reserved_final_trials": 2, "advisor_calls": []})
        self.metadata["adaptive_policy"]["final_top_k"] = 1
        plan = Plan("integration", candidates=tuple(CandidateConfig(**row) for row in self.candidates.values()),
                    metadata=self.metadata)
        plan_path = self.root / "plan.json"
        write_plan(plan_path, plan)
        original = plan_path.read_bytes()
        calls = []
        advisor_calls = []
        def advise(task, payload):
            advisor_calls.append(payload)
            return {"candidate_proposals": [self.proposal()]}
        advisor = SimpleNamespace(ask=advise)

        def execute(task, attempt):
            if task.candidate_id.startswith("repair-") and not calls[-1][0].startswith("repair-"):
                raise KeyboardInterrupt()
            calls.append((task.candidate_id, task.run_class))
            return measured(task)

        def measured(task):
            candidates = {**self.frozen_candidates, **{record["candidate"]["id"]: record["candidate"]
                for record in runtime_candidate_records(self.root)}}
            return {"status": "VALID", "output_tokens_per_second": 100,
                    "latency_seconds": {"p95": 0.5 if task.candidate_id.startswith("repair-") else 1.0},
                    "effective_static_config": candidates[task.candidate_id]["static_config"], "reasons": []}

        with patch("llmperf.adaptive.preflight_candidates", side_effect=self.preflight), self.assertRaises(KeyboardInterrupt):
            execute_tuning_plan(plan, self.root, executor=execute, advisor=advisor)
        self.assertEqual(len(advisor_calls), 1)
        recorded = self.root / "decisions/000002.json"
        decision_hash = sha256_file(recorded)
        def resume(task, attempt):
            calls.append((task.candidate_id, task.run_class))
            return measured(task)
        # Repeated duplicate proposals consume only the remaining advisor quota.
        report = execute_tuning_plan(plan, self.root, executor=resume, advisor=advisor, resume=True)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["best"]["candidate_id"].startswith("repair-"))
        self.assertEqual(report["best"]["repetitions"], 1)
        self.assertLessEqual(report["trials"], 6)
        self.assertEqual(report["adaptive"]["generated_candidates"], [report["best"]["candidate_id"]])
        self.assertEqual(sha256_file(recorded), decision_hash)
        self.assertEqual(plan_path.read_bytes(), original)
        completed_calls = len(calls)
        resumed = execute_tuning_plan(plan, self.root, executor=resume, advisor=advisor, resume=True)
        self.assertEqual(resumed["best"], report["best"])
        self.assertEqual(len(calls), completed_calls)

    def test_learned_native_missing_field_blocks_other_topologies_only_until_repaired(self):
        self.metadata["engine_catalog"]["nested"] = {}
        scheduler = self.scheduler(SimpleNamespace(ask=lambda *_: {}))
        self.state["observations"] = [{"candidate_id": "broken", "status": "FAILED", "score": None,
            "evidence_summary": {"log_excerpts": [{"tail":
                "ValidationError for SpeculativeConfig: num_speculative_tokens must be set"}]}}]
        self.assertTrue(scheduler.blocked(self.candidates["same-defect"]))
        fixed = copy.deepcopy(self.candidates["same-defect"])
        fixed["static_config"]["speculative_config"]["num_speculative_tokens"] = 3
        self.assertFalse(scheduler.blocked(fixed))
        self.assertFalse(scheduler.blocked(self.candidates["baseline"]))

    def test_failure_preserves_independent_pending_candidates_for_both_policies(self):
        self.metadata["engine_catalog"]["nested"] = {}
        self.metadata["adaptive_policy"]["batch_size"] = 3
        for version in (1, 2):
            for kind, phase in (("oom", "startup"), ("configuration", "startup"), ("timeout", "replay")):
                with self.subTest(version=version, kind=kind, phase=phase):
                    self.metadata["adaptive_policy"]["version"] = version
                    self.state = {"plan_hash": "plan", "observations": [], "decisions": []}
                    advisor = SimpleNamespace(ask=lambda *_: self.fail("pending batch must finish before feedback"))
                    scheduler = self.scheduler(advisor)
                    batch = list(scheduler.next_batch(8, 5))
                    self.assertEqual(len(batch), 3)
                    result = {"candidate_id": batch[0], "status": "FAILED", "score": None,
                              "failure_kind": kind, "failure_phase": phase}
                    self.state["observations"].append(result)
                    scheduler.consume(batch[0], result)
                    self.assertEqual(scheduler.next_batch(7, 5), batch[1:])
                    self.assertEqual(self.state["decisions"], [])

    def test_failed_batch_only_skips_peers_with_the_confirmed_missing_field(self):
        self.metadata["engine_catalog"]["nested"] = {}
        self.metadata["adaptive_policy"].update(batch_size=4)
        self.metadata["baseline_candidate_id"] = "broken"
        fixed = copy.deepcopy(self.candidates["same-defect"])
        fixed["id"] = "fixed"
        fixed["static_config"]["speculative_config"]["num_speculative_tokens"] = 3
        self.candidates["fixed"] = fixed
        advisor = SimpleNamespace(ask=lambda *_: self.fail("independent pending trials must still run"))
        scheduler = self.scheduler(advisor)
        batch = list(scheduler.next_batch(8, 5))
        self.assertEqual(batch[0], "broken")
        self.assertEqual(set(batch), set(self.candidates))
        result = {"candidate_id": "broken", "status": "FAILED", "score": None,
            "failure_kind": "configuration", "failure_phase": "startup", "evidence_summary": {
                "log_excerpts": [{"tail": "ValidationError for SpeculativeConfig: num_speculative_tokens must be set"}]}}
        self.state["observations"].append(result)
        scheduler.consume("broken", result)
        expected = [ident for ident in batch[1:] if ident != "same-defect"]
        self.assertEqual(scheduler.next_batch(7, 5), expected)
        self.assertEqual(self.state["decisions"], [{"reason": "known_configuration_defect",
            "observed_candidate_id": "broken", "cancelled_candidates": ["same-defect"]}])
        self.assertEqual(self.scheduler(advisor).next_batch(7, 5), expected)


if __name__ == "__main__":
    unittest.main()
