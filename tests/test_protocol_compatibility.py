import copy
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from benchmarks.server.protocol_check import deployment_report, exchange, inspect_response, report_is_valid, validate_protocol
from llmperf.adapters import ReplayAdapter, read_attempt
from llmperf.discovery import discover_model
from llmperf.evidence import environment_summary, summarize_trial
from llmperf.planner import create_search_plan
from llmperf.types import PlanTask, WorkloadManifest
import test_auto_planner as planner_fixtures


FIXTURES = Path(__file__).parent / "fixtures" / "models"


def response(stream=False, tools=False, reasoning=True):
    message = {"content": None if tools else "7"}
    if reasoning:
        message["reasoning_content"] = "Three plus four is seven."
    if tools:
        message["tool_calls"] = [{"index": 0, "type": "function", "function": {
            "name": "llmperf_probe", "arguments": '{"value":7}'}}]
    return {"items": [{"choices": [{"delta" if stream else "message": message,
                                   "finish_reason": "tool_calls" if tools else "stop"}],
                       "usage": {"completion_tokens": 20}}], "completed": True}


class ProtocolCheckTests(unittest.TestCase):
    def test_reasoning_and_tool_protocols_are_observed_without_execution(self):
        sent = []
        def sender(base_url, payload, timeout):
            sent.append(copy.deepcopy(payload))
            return response(payload["stream"], "tools" in payload)
        specification = {"streaming": True, "tools": True, "reasoning": True}
        report = validate_protocol("http://127.0.0.1:25080", "test", specification, sender)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["tools_executed"])
        self.assertFalse(report["scored"])
        self.assertEqual(len(sent), 4)
        self.assertEqual([row["case"] for row in report["cases"]], ["chat", "streaming", "tools", "tools_streaming"])
        self.assertNotIn("3 + 4", json.dumps(report))

    def test_fragmented_streaming_tool_calls_are_reassembled(self):
        data = {"items": [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "llmperf_", "arguments": '{"val'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "probe", "arguments": 'ue":7}'}}]},
                          "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"completion_tokens": 10}},
        ], "completed": True}
        report = inspect_response(data, stream=True, tools=True)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["tool_call_count"], 1)

    def test_raw_v41_or_thinking_tags_in_content_are_not_silent_success(self):
        for marker in ("<｜DSML｜ calls>", "<tool_call>", "<think>"):
            data = response()
            data["items"][0]["choices"][0]["message"]["content"] = marker
            self.assertEqual(inspect_response(data)["status"], "FAIL")

    def test_missing_reasoning_or_usage_is_unknown_not_verified(self):
        data = response(reasoning=False)
        self.assertEqual(inspect_response(data, reasoning=True)["status"], "UNKNOWN")
        data["items"][0].pop("usage")
        self.assertEqual(inspect_response(data)["status"], "UNKNOWN")

    def test_new_vllm_reasoning_alias_is_recognized(self):
        data = response()
        message = data["items"][0]["choices"][0]["message"]
        message["reasoning"] = message.pop("reasoning_content")
        message["reasoning_content"] = None
        self.assertEqual(inspect_response(data, reasoning=True)["status"], "PASS")

    def test_missing_generation_finish_reason_is_not_verified(self):
        data = response()
        data["items"][0]["choices"][0].pop("finish_reason")
        self.assertEqual(inspect_response(data)["status"], "UNKNOWN")

    def test_truncated_tool_arguments_are_unknown_not_misclassified_as_parser_failure(self):
        data = response(tools=True)
        choice = data["items"][0]["choices"][0]
        choice["finish_reason"] = "length"
        choice["message"]["tool_calls"][0]["function"]["arguments"] = '{"value":'
        self.assertEqual(inspect_response(data, tools=True)["status"], "UNKNOWN")

    def test_transport_failure_and_http_rejection_are_distinguished(self):
        with patch("benchmarks.server.protocol_check.time.monotonic", return_value=1):
            report = validate_protocol("http://localhost", "test", {}, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        self.assertEqual(report["status"], "UNKNOWN")
        def rejected(*args):
            raise urllib.error.HTTPError("http://localhost", 400, "bad request", {}, None)
        self.assertEqual(validate_protocol("http://localhost", "test", {}, rejected)["status"], "FAIL")

    def test_multimodal_correctness_is_not_claimed_by_text_probes(self):
        report = validate_protocol("http://localhost", "test", {"multimodal": True}, lambda *_: response())
        self.assertEqual(report["status"], "UNKNOWN")
        self.assertFalse(report["multimodal_verified"])
        self.assertFalse(report["runtime_verified"])

    def test_sse_exchange_parses_usage_and_done_without_returning_raw_output(self):
        import io
        body = io.BytesIO(b'data: {"choices":[{"delta":{"content":"7"},"finish_reason":"stop"}]}\n\n'
                          b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
                          b'data: [DONE]\n\n')
        with patch("benchmarks.server.protocol_check.urllib.request.urlopen", return_value=body):
            result = exchange("http://localhost", {"stream": True}, 5)
        self.assertEqual(inspect_response(result, stream=True)["status"], "PASS")

    def test_forged_pass_or_missing_case_or_instance_is_rejected(self):
        instance = {"id": "i0", **validate_protocol("http://localhost", "test", {}, lambda *_: response())}
        report = deployment_report([instance])
        self.assertTrue(report_is_valid(report, {}, ["i0"]))
        report["instances"][0]["cases"][0]["status"] = "FAIL"
        self.assertFalse(report_is_valid(report, {}, ["i0"]))
        self.assertFalse(report_is_valid(deployment_report([instance]), {"streaming": True}, ["i0"]))
        self.assertFalse(report_is_valid(deployment_report([instance]), {}, ["i0", "i1"]))

    def test_malformed_status_or_version_does_not_crash_evidence_validation(self):
        for value in ([], {}, None, True):
            instance = {"id": "i0", **validate_protocol("http://localhost", "test", {}, lambda *_: response())}
            report = deployment_report([instance])
            report["instances"][0]["cases"][0]["status"] = value
            self.assertFalse(report_is_valid(report, {}, ["i0"]))
        report["version"] = True
        self.assertFalse(report_is_valid(report, {}, ["i0"]))


class ModelPlanningTests(unittest.TestCase):
    def test_unknown_fp4_layout_uses_native_auto_instead_of_guessing_mxfp4(self):
        model = discover_model(FIXTURES / "DeepSeek-V4.1-Flash-1017")
        config = planner_fixtures.configuration(Path("/tmp/compatibility-planner"))
        environment = planner_fixtures.environment()
        environment.container["capabilities"].update(tool_parsers=["deepseekv41"], reasoning_parsers=["deepseek-v41"],
            runner_backends=["auto", "triton", "deep_gemm", "flashinfer_mxfp4", "flashinfer_trtllm"])
        workload = WorkloadManifest("test", raw={"models": {"captured": 1}, "count": 1,
                                                  "requirements": {"tools": True, "streaming": True}})
        plan = create_search_plan(config, model, workload, environment, config.model_path / "results")
        self.assertTrue(plan.candidates)
        self.assertTrue(all(candidate.backend is None for candidate in plan.candidates))
        self.assertTrue(plan.metadata["protocol_validation"]["tools"])
        self.assertTrue(plan.metadata["protocol_validation"]["reasoning"])
        self.assertFalse(plan.metadata["model_constraints"]["runtime_verified"])
        config.search.backends = ("triton",)
        explicit = create_search_plan(config, model, workload, environment, config.model_path / "results")
        self.assertTrue(all(candidate.backend == "triton" for candidate in explicit.candidates))


class ProtocolEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = PlanTask("task", 1, "candidate", run_class="formal", concurrency=8)
        self.metadata = {"expected_source_sha256": "a" * 64, "expected_request_count": 1,
                         "protocol_validation": {"version": 1}}
        documents = {
            "formal.summary.json": {"jsonl_sha256": "a" * 64, "request_count": 1, "successes": 1,
                "failures": 0, "server_usage_available": 1, "server_usage_missing": 0, "completion_tokens": 20,
                "measured_seconds": 1.0, "output_tokens_per_second": 20.0, "base_url": "http://127.0.0.1:25080"},
            "server.requested.json": {"model_path": "/model", "max_running_requests": 8},
            "server.info.json": {"server_args": {"model_path": "/model", "max_running_requests": 8}},
            "server.evidence.json": {"readiness": True, "server_info_captured": True, "resolved": True,
                "requested_server_parameters": {"model_path": "/model", "max_running_requests": 8},
                "unsupported_parameters": [], "mismatched_parameters": []},
            "run_manifest.json": {"state": "complete", "exit_code": 0},
        }
        for name, value in documents.items():
            (self.root / name).write_text(json.dumps(value))

    def write_protocol(self, sender):
        report = deployment_report([{"id": "i0", **validate_protocol("http://localhost", "test", {}, sender)}])
        (self.root / "server.protocol.json").write_text(json.dumps(report))
        return report

    def test_required_protocol_artifact_cannot_be_skipped(self):
        result = read_attempt(self.root, self.task, self.metadata)
        self.assertIn("protocol_validation_missing", result["reasons"])
        self.assertNotEqual(result["status"], "VALID")

    def test_verified_protocol_is_bound_to_trial_evidence(self):
        self.write_protocol(lambda *_: response())
        result = read_attempt(self.root, self.task, self.metadata)
        self.assertEqual(result["status"], "VALID", result["reasons"])
        self.assertTrue(result["model_runtime_verified"])

    def test_unknown_protocol_does_not_turn_into_a_verified_model(self):
        self.write_protocol(lambda *_: {"items": []})
        result = read_attempt(self.root, self.task, self.metadata)
        self.assertEqual(result["status"], "VALID", result["reasons"])
        self.assertFalse(result["model_runtime_verified"])
        self.assertIn("protocol_validation_incomplete", result["warnings"])

    def test_legacy_attempts_without_protocol_requirements_keep_working(self):
        self.metadata.pop("protocol_validation")
        self.assertEqual(read_attempt(self.root, self.task, self.metadata)["status"], "VALID")

    def test_advisor_evidence_contains_protocol_and_normalized_constraints(self):
        model = discover_model(FIXTURES / "DeepSeek-V4.1-Flash-1017")
        context = environment_summary({"model_snapshot": {"raw": model.raw}})
        self.assertEqual(context["model"]["protocol"], "deepseek-v41")
        self.assertEqual(context["model"]["quantization_details"]["expert_dtype"], "fp4")
        report = self.write_protocol(lambda *_: response())
        candidate = {"id": "candidate", "config_hash": "hash", "static_config": {}}
        summary = summarize_trial({"protocol_validation": report, "model_runtime_verified": True}, candidate, self.root)
        self.assertTrue(summary["model_runtime_verified"])
        self.assertEqual(summary["protocol_validation"]["instances"][0]["cases"], [{"case": "chat", "status": "PASS"}])
        for invalid in (None, {}, [{"cases": None}], [{"cases": 1}]):
            summarize_trial({"protocol_validation": {"instances": invalid}}, candidate, self.root)
