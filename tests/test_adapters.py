import json
import tempfile
import unittest
from pathlib import Path
from s1slow.Automation.automation.adapters import ReplayAdapter, read_attempt
from s1slow.Automation.automation.types import PlanTask
from s1slow.Automation.automation.docker_runtime import DockerRunResult

class FakeRuntime:
    def __init__(self): self.specs=[]
    def run(self, spec, log):
        self.specs.append(spec)
        Path(log).write_text("ok")
        return DockerRunResult(0, "", "", {}, spec.name, spec.command)

class AdapterTests(unittest.TestCase):
    def test_build_spec_and_execute(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); model=root/"model"; model.mkdir(); src=root/"input.jsonl"; src.write_text("{}\n"); idx=root/"index.json"; idx.write_text("{}")
            runtime=FakeRuntime(); metadata={"image":"x/y@sha256:"+"a"*64,"model_host":str(model),"jsonl_host":str(src),"index_host":str(idx),"benchmark_dir":str(root),"candidates":{"c":{"gpu_indexes":[0]}}}
            adapter=ReplayAdapter(metadata, root/"run", runtime); task=PlanTask("t",0,"c",run_class="smoke")
            result=adapter(task,1); self.assertTrue(Path(result).is_dir()); self.assertEqual(len(runtime.specs),1); self.assertIn("Docker", runtime.specs[0].__class__.__name__)
            self.assertEqual(runtime.specs[0].gpu_indexes,(0,)); self.assertIn("/opt/s1slow/benchmarks/run_point.sh", runtime.specs[0].command)

    def test_default_profile_is_portable_auto_profile(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); model = root / "model"; model.mkdir()
            source = root / "input.jsonl"; source.write_text("{}\n")
            index = root / "index.json"; index.write_text("{}")
            adapter = ReplayAdapter({
                "image": "x/y@sha256:" + "a" * 64,
                "model_host": str(model), "jsonl_host": str(source),
                "index_host": str(index), "benchmark_dir": str(root),
            }, root / "run")
            spec = adapter.build_spec(PlanTask("t", 0, "c", run_class="smoke"), 1)
            self.assertEqual(spec.command[4], "AUTO")

    def test_non_controller_stage_uses_formal_run_class(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"; model.mkdir()
            source = root / "input.jsonl"; source.write_text("{}\n")
            index = root / "index.json"; index.write_text("{}")
            adapter = ReplayAdapter({
                "image": "x/y@sha256:" + "a" * 64,
                "model_host": str(model), "jsonl_host": str(source),
                "index_host": str(index), "benchmark_dir": str(root),
            }, root / "run", FakeRuntime())
            spec = adapter.build_spec(PlanTask("t", 3, "c", run_class="concurrency", concurrency=8), 1)
            self.assertEqual(spec.command[2], "formal")

    def test_exploration_limit_uses_diagnostic_controller(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"; model.mkdir()
            source = root / "input.jsonl"; source.write_text("{}\n")
            index = root / "index.json"; index.write_text("{}")
            adapter = ReplayAdapter({
                "image": "x/y@sha256:" + "a" * 64,
                "model_host": str(model), "jsonl_host": str(source),
                "index_host": str(index), "benchmark_dir": str(root),
                "expected_request_count": 3000,
                "search": {"explore_request_limit": 128},
            }, root / "run", FakeRuntime())
            spec = adapter.build_spec(PlanTask("t", 3, "c", run_class="concurrency", concurrency=16), 1)
            self.assertEqual(spec.command[2], "diagnostic")
            command = list(spec.command)
            self.assertEqual(command[command.index("--limit") + 1], "128")

    def test_multi_instance_candidate_uses_deployment_runner_and_logical_splits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"; model.mkdir()
            source = root / "input.jsonl"; source.write_text("{}\n")
            index = root / "index.json"; index.write_text("{}")
            metadata = {
                "image": "x/y@sha256:" + "a" * 64,
                "model_host": str(model), "jsonl_host": str(source),
                "index_host": str(index), "benchmark_dir": str(root),
                "expected_request_count": 3000,
                "search": {"explore_request_limit": 128},
                "service_port": 54322,
                "candidates": {"c": {"gpu_indexes": [6, 7], "static_config": {
                    "tp": 1, "dp": 1, "pp": 1, "max_running_requests": 64,
                    "deployment": {
                        "total_gpu_count": 2,
                        "instance_count": 2,
                        "gpus_per_instance": 1,
                        "label": "2卡2实例",
                        "ascii_label": "2g2i",
                        "gpu_indexes": [6, 7],
                        "instances": [
                            {"id": "i0", "ordinal": 0, "gpu_indexes": [6]},
                            {"id": "i1", "ordinal": 1, "gpu_indexes": [7]},
                        ],
                    },
                }}},
            }
            spec = ReplayAdapter(metadata, root / "run").build_spec(
                PlanTask("t", 3, "c", run_class="concurrency", concurrency=16), 1
            )
            self.assertEqual(spec.gpu_indexes, (6, 7))
            self.assertEqual(spec.command[0], "python3")
            self.assertIn("/opt/s1slow/benchmarks/run_deployment_point.py", spec.command)
            self.assertEqual(spec.command[2], "diagnostic")
            self.assertEqual(spec.env["MAX_RUNNING_REQUESTS"], "8")
            deployment = json.loads(spec.env["S1_DEPLOYMENT"])
            self.assertEqual(deployment["base_urls"], [
                "http://127.0.0.1:54322", "http://127.0.0.1:54323",
            ])
            self.assertEqual(deployment["instances"][0]["logical_gpu_indexes"], [0])
            self.assertEqual(deployment["instances"][1]["logical_gpu_indexes"], [1])

    def test_build_spec_forwards_runtime_contract_and_uses_logical_gpu_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"; model.mkdir()
            source = root / "input.jsonl"; source.write_text("{}\n")
            index = root / "index.json"; index.write_text("{}")
            metadata = {
                "image": "x/y@sha256:" + "a" * 64,
                "model_host": str(model), "jsonl_host": str(source),
                "index_host": str(index), "benchmark_dir": str(root),
                "served_model_name": "served", "tool_call_parser": "tools",
                "reasoning_parser": "reasoning",
                "chat_template_kwargs": {"thinking": False},
                "quantization": "fp8", "profile_env": {"CUSTOM_SETTING": "yes"},
                "warmup": 3, "request_timeout": 45.5, "ready_timeout": 87, "name_prefix": "custom",
                "expected_source_sha256": "f" * 64, "expected_request_count": 5,
                "candidates": {"c": {"gpu_indexes": [3, 7], "static_config": {
                    "tp": 2, "dp": 2, "pp": 1, "dp_attention": True,
                    "dp_lm_head": True, "backend": "flashinfer",
                    "moe_a2a_backend": "deepep", "dspark": False,
                    "mem_fraction_static": 0.73, "max_running_requests": 99,
                    "chunked_prefill_size": 4096,
                }}},
            }
            spec = ReplayAdapter(metadata, root / "run").build_spec(
                PlanTask("t", 0, "c", run_class="smoke", mode="closed_loop", concurrency=4), 1
            )
            self.assertTrue(spec.name.startswith("custom-"))
            self.assertEqual(spec.gpu_indexes, (3, 7))
            self.assertEqual(spec.env["CUDA_VISIBLE_DEVICES"], "0,1")
            expected_env = {
                "TP_SIZE": "2", "DP_SIZE": "2", "PP_SIZE": "1",
                "ENABLE_DP_ATTENTION": "1", "ENABLE_DP_LM_HEAD": "1",
                "MOE_RUNNER_BACKEND": "flashinfer", "MOE_A2A_BACKEND": "deepep",
                "ENABLE_DSPARK": "0", "MEM_FRACTION_STATIC": "0.73",
                "MAX_RUNNING_REQUESTS": "4", "CHUNKED_PREFILL_SIZE": "4096",
                "TOOL_CALL_PARSER": "tools", "REASONING_PARSER": "reasoning",
                "DEFAULT_CHAT_TEMPLATE_KWARGS": '{"thinking":false}',
                "QUANTIZATION": "fp8", "CUSTOM_SETTING": "yes",
            }
            for key, value in expected_env.items():
                self.assertEqual(spec.env[key], value, key)
            command = list(spec.command)
            self.assertEqual(command[command.index("--mode") + 1], "closed-loop")
            self.assertEqual(command[command.index("--warmup") + 1], "3")
            self.assertEqual(command[command.index("--request-timeout") + 1], "45.5")
            self.assertEqual(command[command.index("--ready-timeout") + 1], "87")
            self.assertEqual(command[command.index("--limit") + 1], "5")

    def test_read_attempt_accepts_complete_formal_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            attempt = Path(td)
            output = attempt / "formal" / "AUTO" / "run_001"
            output.mkdir(parents=True)
            summary = {
                "jsonl_sha256": "a" * 64, "request_count": 2,
                "successes": 2, "failures": 0, "server_usage_available": 2,
                "server_usage_missing": 0, "completion_tokens": 40,
                "measured_seconds": 2.0, "output_tokens_per_second": 20.0,
                "base_url": "http://127.0.0.1:25080",
            }
            (output / "formal.summary.json").write_text(json.dumps(summary))
            evidence = {
                "readiness": True, "server_info_captured": True, "resolved": True,
                "requested_server_parameters": {"model_path": "/model", "tp": 2, "backend": "auto", "max_running_requests": 8},
                "resolved_server_parameters": {"tp": 2, "backend": "flashinfer", "max_running_requests": 8},
                "parameter_checks": {
                    "tp": {"supported": True, "matches": True},
                    "backend": {"supported": True, "matches": True, "auto_resolved": True},
                    "max_running_requests": {"supported": True, "matches": True},
                },
                "unsupported_parameters": [], "mismatched_parameters": [],
            }
            (output / "server.evidence.json").write_text(json.dumps(evidence))
            (output / "server.requested.json").write_text(json.dumps(evidence["requested_server_parameters"]))
            (output / "server.info.json").write_text(json.dumps({
                "server_args": {"model_path": "/model", "tp_size": 2, "moe_runner_backend": "flashinfer", "max_running_requests": 8}
            }))
            (output / "run_manifest.json").write_text(json.dumps({"state": "complete", "exit_code": 0}))
            task = PlanTask("task", 1, "candidate", run_class="formal", concurrency=8)
            result = read_attempt(attempt, task, {
                "expected_source_sha256": "a" * 64, "expected_request_count": 2,
            })
            self.assertEqual(result["status"], "VALID")
            self.assertEqual(result["reasons"], [])
            self.assertEqual(result["output_tokens_per_second"], 20.0)
            self.assertEqual(result["candidate_id"], "candidate")
            self.assertEqual(result["concurrency"], 8)

    def test_read_attempt_accepts_limited_exploration_and_multi_endpoint_summary(self):
        with tempfile.TemporaryDirectory() as td:
            attempt = Path(td)
            output = attempt / "diagnostic"
            output.mkdir()
            summary = {
                "jsonl_sha256": "a" * 64, "request_count": 128,
                "successes": 128, "failures": 0, "server_usage_available": 128,
                "server_usage_missing": 0, "completion_tokens": 2560,
                "measured_seconds": 10.0, "output_tokens_per_second": 256.0,
                "base_url": "http://127.0.0.1:54322",
                "base_urls": ["http://127.0.0.1:54322", "http://127.0.0.1:54323"],
                "endpoint_summaries": [
                    {"endpoint_id": "i0", "request_count": 64},
                    {"endpoint_id": "i1", "request_count": 64},
                ],
            }
            (output / "limited.summary.json").write_text(json.dumps(summary))
            requested = {"model_path": "/model", "max_running_requests": 8}
            (output / "server.requested.json").write_text(json.dumps(requested))
            (output / "server.info.json").write_text(json.dumps({
                "server_args": {"model_path": "/model", "max_running_requests": 8}
            }))
            (output / "server.evidence.json").write_text(json.dumps({
                "readiness": True, "server_info_captured": True, "resolved": True,
                "requested_server_parameters": requested,
                "unsupported_parameters": [], "mismatched_parameters": [],
            }))
            (output / "run_manifest.json").write_text(json.dumps({"state": "complete", "exit_code": 0}))
            result = read_attempt(attempt, PlanTask("task", 1, "candidate", run_class="concurrency", concurrency=16), {
                "expected_source_sha256": "a" * 64, "expected_request_count": 3000,
                "service_port": 54322,
                "search": {"explore_request_limit": 128},
                "candidates": {"candidate": {"static_config": {
                    "deployment": {"instance_count": 2},
                }}},
            })
            self.assertEqual(result["status"], "VALID")
            self.assertEqual(result["request_count"], 128)
            self.assertEqual(result["endpoint_summaries"][1]["endpoint_id"], "i1")

    def test_read_attempt_rejects_unsupported_or_incomplete_evidence(self):
        base_summary = {
            "jsonl_sha256": "b" * 64, "request_count": 3,
            "successes": 3, "failures": 0, "server_usage_available": 3,
            "server_usage_missing": 0, "completion_tokens": 30,
            "measured_seconds": 1.0, "output_tokens_per_second": 30.0,
            "base_url": "http://localhost:54322",
        }
        cases = (
            ({"readiness": False, "resolved": True}, "readiness_missing", "INCONCLUSIVE"),
            ({"readiness": True, "resolved": False, "unsupported_parameters": ["pp"]}, "unsupported_server_parameters", "UNSUPPORTED"),
            ({"readiness": True, "resolved": False, "mismatched_parameters": ["tp"]}, "server_parameter_mismatch", "INCONCLUSIVE"),
        )
        for evidence, reason, status in cases:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as td:
                attempt = Path(td)
                (attempt / "x.summary.json").write_text(json.dumps(base_summary))
                (attempt / "server.evidence.json").write_text(json.dumps(evidence))
                result = read_attempt(attempt, PlanTask("t", 0, "c", run_class="formal"), {
                    "expected_source_sha256": "b" * 64, "expected_request_count": 3,
                })
                self.assertEqual(result["status"], status)
                self.assertIn(reason, result["reasons"])

    def test_read_attempt_requires_positive_finite_measurement(self):
        with tempfile.TemporaryDirectory() as td:
            attempt = Path(td)
            summary = {
                "jsonl_sha256": "c" * 64, "request_count": 1,
                "successes": 1, "failures": 0, "server_usage_available": 1,
                "server_usage_missing": 0, "completion_tokens": 1,
                "measured_seconds": float("nan"), "output_tokens_per_second": 0,
                "base_url": "http://127.0.0.1:54322",
            }
            (attempt / "x.summary.json").write_text(json.dumps(summary))
            (attempt / "server.evidence.json").write_text(json.dumps({"readiness": True, "resolved": True}))
            result = read_attempt(attempt, PlanTask("t", 0, "c", run_class="formal"), {
                "expected_source_sha256": "c" * 64, "expected_request_count": 1,
            })
            self.assertEqual(result["status"], "INCONCLUSIVE")
            self.assertIn("measured_seconds_invalid", result["reasons"])
            self.assertIn("output_tokens_per_second_invalid", result["reasons"])

    def test_read_attempt_failed_exit_is_failed_even_without_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            result = read_attempt(
                Path(td), PlanTask("t", 0, "c", run_class="formal"), {}, exit_code=17
            )
            self.assertEqual(result["status"], "FAILED")
            self.assertIn("process_exit_17", result["reasons"])

    def test_read_attempt_recomputes_server_parameter_and_score_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            attempt = Path(td)
            summary = {
                "jsonl_sha256": "d" * 64, "request_count": 1, "successes": 1,
                "failures": 0, "server_usage_available": 1, "server_usage_missing": 0,
                "completion_tokens": 10, "measured_seconds": 2.0,
                "output_tokens_per_second": 99.0,
                "base_url": "http://127.0.0.1:54322",
            }
            (attempt / "x.summary.json").write_text(json.dumps(summary))
            requested = {"max_running_requests": 1}
            (attempt / "server.requested.json").write_text(json.dumps(requested))
            (attempt / "server.info.json").write_text(json.dumps({"max_running_requests": 9}))
            (attempt / "server.evidence.json").write_text(json.dumps({
                "readiness": True, "server_info_captured": True, "resolved": True,
                "requested_server_parameters": requested, "unsupported_parameters": [],
                "mismatched_parameters": [],
            }))
            (attempt / "run_manifest.json").write_text(json.dumps({"state": "complete", "exit_code": 0}))
            result = read_attempt(attempt, PlanTask("t", 0, "c", run_class="formal", concurrency=1), {
                "expected_source_sha256": "d" * 64, "expected_request_count": 1,
            })
            self.assertEqual(result["status"], "INCONCLUSIVE")
            self.assertIn("server_parameter_mismatch", result["reasons"])
            self.assertIn("throughput_inconsistent", result["reasons"])

    def test_read_attempt_requires_controller_completion(self):
        with tempfile.TemporaryDirectory() as td:
            attempt = Path(td)
            (attempt / "x.summary.json").write_text(json.dumps({
                "jsonl_sha256": "e" * 64, "request_count": 1, "successes": 1,
                "failures": 0, "server_usage_available": 1, "server_usage_missing": 0,
                "completion_tokens": 1, "measured_seconds": 1.0,
                "output_tokens_per_second": 1.0, "base_url": "http://localhost:54322",
            }))
            (attempt / "server.evidence.json").write_text(json.dumps({
                "readiness": True, "server_info_captured": True, "resolved": True,
            }))
            result = read_attempt(attempt, PlanTask("t", 0, "c", run_class="formal"), {
                "expected_source_sha256": "e" * 64, "expected_request_count": 1,
            })
            self.assertIn("controller_completion_missing", result["reasons"])

if __name__ == "__main__": unittest.main()
