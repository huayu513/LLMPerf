import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from llmperf.adapters import ReplayAdapter, read_attempt
from llmperf.deployment_ports import deployment_service_ports
from llmperf.types import PlanTask
from llmperf.docker_runtime import DockerRunResult

class FakeRuntime:
    def __init__(self, server_command=False, server_command_subdir="", server_command_text="echo server\n"):
        self.specs=[]
        self.server_command=server_command
        self.server_command_subdir=server_command_subdir
        self.server_command_text=server_command_text
    def run(self, spec, log):
        self.specs.append(spec)
        Path(log).write_text("ok")
        if self.server_command:
            command_dir = Path(log).parent / self.server_command_subdir
            command_dir.mkdir(parents=True, exist_ok=True)
            (command_dir / "server.command.sh").write_text(self.server_command_text, encoding="utf-8")
        return DockerRunResult(0, "", "", {}, spec.name, spec.command)

class AdapterTests(unittest.TestCase):
    def test_dp_attention_deployment_ports_use_separate_blocks(self):
        self.assertEqual(
            deployment_service_ports(25080, 4),
            [25080, 25336, 25592, 25848],
        )

    def test_deployment_port_plan_rejects_overflow(self):
        with self.assertRaises(ValueError):
            deployment_service_ports(65500, 2)

    def test_crlf_benchmark_bundle_is_staged_as_lf_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"; model.mkdir()
            source = root / "input.jsonl"; source.write_text("{}\n")
            index = root / "index.json"; index.write_text("{}")
            benchmark = root / "benchmarks"
            (benchmark / "server").mkdir(parents=True)
            script = benchmark / "server" / "launch_server.sh"
            script.write_bytes(b"#!/usr/bin/env bash\r\nset -euo pipefail\r\necho ok\r\n")
            adapter = ReplayAdapter(
                {
                    "image": "x/y@sha256:" + "a" * 64,
                    "model_host": str(model), "jsonl_host": str(source),
                    "index_host": str(index), "benchmark_dir": str(benchmark),
                    "candidates": {"c": {"gpu_indexes": [0]}},
                },
                root / "run",
            )
            spec = adapter.build_spec(PlanTask("t", 0, "c", run_class="smoke"), 1)
            mount = next(mount for mount in spec.mounts if mount.dst == "/opt/s1slow/benchmarks")
            self.assertNotEqual(mount.src, benchmark.resolve())
            self.assertIn(b"\r\n", script.read_bytes())
            self.assertEqual(
                (mount.src / "server" / "launch_server.sh").read_bytes(),
                b"#!/usr/bin/env bash\nset -euo pipefail\necho ok\n",
            )

    def test_build_spec_and_execute(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); model=root/"model"; model.mkdir(); src=root/"input.jsonl"; src.write_text("{}\n"); idx=root/"index.json"; idx.write_text("{}")
            runtime=FakeRuntime(); metadata={"image":"x/y@sha256:"+"a"*64,"model_host":str(model),"jsonl_host":str(src),"index_host":str(idx),"benchmark_dir":str(root),"candidates":{"c":{"gpu_indexes":[0]}}}
            adapter=ReplayAdapter(metadata, root/"run", runtime); task=PlanTask("t",0,"c",run_class="smoke")
            result=adapter(task,1); self.assertTrue(Path(result).is_dir()); self.assertEqual(len(runtime.specs),1); self.assertIn("Docker", runtime.specs[0].__class__.__name__)
            self.assertEqual(runtime.specs[0].gpu_indexes,(0,)); self.assertIn("/opt/s1slow/benchmarks/run_point.sh", runtime.specs[0].command)

    def test_execute_writes_host_server_reproduce_script(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); model=root/"model"; model.mkdir(); src=root/"input.jsonl"; src.write_text("{}\n"); idx=root/"index.json"; idx.write_text("{}")
            runtime=FakeRuntime(server_command=True)
            metadata={"image":"x/y@sha256:"+"a"*64,"model_host":str(model),"jsonl_host":str(src),"index_host":str(idx),"benchmark_dir":str(root),"candidates":{"c":{"gpu_indexes":[6]}}}
            adapter=ReplayAdapter(metadata, root/"run", runtime); task=PlanTask("t",0,"c",run_class="smoke")
            result=Path(adapter(task,1))
            script=result/"server.reproduce.sh"
            self.assertTrue(script.is_file())
            self.assertTrue(script.stat().st_mode & 0o111)
            content=script.read_text(encoding="utf-8")
            self.assertIn("docker run --rm --name", content)
            self.assertIn("--gpus 1", content)
            self.assertIn("--publish 127.0.0.1:25080:25080", content)
            self.assertIn("bash -lc", content)
            self.assertNotIn("/run/results", content)
            self.assertNotIn(str(src.resolve()), content)
            self.assertNotIn(str(idx.resolve()), content)
            self.assertNotIn("server.reproduce.container.sh", content)
            self.assertEqual(subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode, 0)
            fake_docker = root / "docker"
            fake_docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_PATH"\n')
            fake_docker.chmod(0o755)
            (result / "server.command.sh").unlink()
            src.unlink(); idx.unlink()
            captured = root / "docker.argv"
            env = {**os.environ, "PATH": f"{root}:{os.environ['PATH']}",
                   "CAPTURE_PATH": str(captured)}
            launched = subprocess.run(["bash", str(script)], env=env, capture_output=True)
            self.assertEqual(launched.returncode, 0, launched.stderr)
            argv = captured.read_text().splitlines()
            self.assertEqual(argv.count("--mount"), 1)
            self.assertIn(f"type=bind,src={model.resolve()},dst=/model,readonly", argv)
            self.assertNotIn("--env", argv)

    def test_nested_server_command_gets_reproduce_script_next_to_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); model=root/"model"; model.mkdir(); src=root/"input.jsonl"; src.write_text("{}\n"); idx=root/"index.json"; idx.write_text("{}")
            runtime=FakeRuntime(server_command=True, server_command_subdir="formal/AUTO/run_001")
            metadata={"image":"x/y@sha256:"+"a"*64,"model_host":str(model),"jsonl_host":str(src),"index_host":str(idx),"benchmark_dir":str(root),"candidates":{"c":{"gpu_indexes":[0]}}}
            adapter=ReplayAdapter(metadata, root/"run", runtime); task=PlanTask("t",0,"c",run_class="formal")
            result=Path(adapter(task,1))
            artifact=result/"formal/AUTO/run_001"
            script=artifact/"server.reproduce.sh"
            self.assertTrue(script.is_file())
            content=script.read_text(encoding="utf-8")
            self.assertIn("bash -lc", content)
            self.assertNotIn("/run/results", content)
            self.assertNotIn(str(src.resolve()), content)
            self.assertNotIn(str(idx.resolve()), content)
            self.assertFalse((result/"server.reproduce.sh").exists())

    def test_multi_instance_reproduction_embeds_commands_and_publishes_each_port(self):
        command = (
            "# instance i0 http://127.0.0.1:25080\n"
            "setsid env CUDA_VISIBLE_DEVICES=0 sglang serve --model-path /model --port 25080\n"
            "# instance i1 http://127.0.0.1:25081\n"
            "setsid env CUDA_VISIBLE_DEVICES=1 sglang serve --model-path /model --port 25081\n"
        )
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); model=root/"model"; model.mkdir(); src=root/"input.jsonl"; src.write_text("{}\n"); idx=root/"index.json"; idx.write_text("{}")
            runtime=FakeRuntime(server_command=True, server_command_text=command)
            metadata={"image":"x/y@sha256:"+"a"*64,"model_host":str(model),"jsonl_host":str(src),"index_host":str(idx),"benchmark_dir":str(root),"candidates":{"c":{"gpu_indexes":[0,1]}}}
            adapter=ReplayAdapter(metadata, root/"run", runtime); task=PlanTask("t",0,"c",run_class="formal")
            result=Path(adapter(task,1))
            script=result/"server.reproduce.sh"
            content=script.read_text(encoding="utf-8")
            self.assertIn("--publish 127.0.0.1:25080:25080", content)
            self.assertIn("--publish 127.0.0.1:25081:25081", content)
            self.assertIn("CUDA_VISIBLE_DEVICES=0", content)
            self.assertIn("CUDA_VISIBLE_DEVICES=1", content)
            self.assertNotIn("/run/workload", content)
            self.assertNotIn("/opt/s1slow/benchmarks", content)
            self.assertEqual(subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode, 0)

    def test_aggregate_command_wins_over_instance_command(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); model=root/"model"; model.mkdir(); src=root/"input.jsonl"; src.write_text("{}\n"); idx=root/"index.json"; idx.write_text("{}")
            runtime=FakeRuntime(server_command=True, server_command_subdir="formal/AUTO/run_001")
            metadata={"image":"x/y@sha256:"+"a"*64,"model_host":str(model),"jsonl_host":str(src),"index_host":str(idx),"benchmark_dir":str(root),"candidates":{"c":{"gpu_indexes":[0]}}}
            adapter=ReplayAdapter(metadata, root/"run", runtime); task=PlanTask("t",0,"c",run_class="formal")
            result=Path(adapter(task,1))
            aggregate=result/"formal/AUTO/run_001"
            instances=aggregate/"instances/0"
            instances.mkdir(parents=True)
            (instances/"server.command.sh").write_text("echo instance\n", encoding="utf-8")
            (aggregate/"server.evidence.json").write_text("{}\n", encoding="utf-8")
            adapter._write_server_reproduce_script(runtime.specs[0], result)
            self.assertTrue((aggregate/"server.reproduce.sh").is_file())
            self.assertFalse((instances/"server.reproduce.sh").exists())

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
            self.assertEqual(json.loads(spec.env["DEFAULT_CHAT_TEMPLATE_KWARGS"]), {
                "enable_thinking": True,
                "reasoning_effort": "max",
                "thinking": True,
            })
            legacy_metadata = {
                "image": "x/y@sha256:" + "a" * 64,
                "model_host": str(model), "jsonl_host": str(source),
                "index_host": str(index), "benchmark_dir": str(root),
                "chat_template_kwargs": {},
                "model_snapshot": {"raw": {"provenance": {
                    "chat_template_kwargs": "safe empty default",
                }}},
            }
            legacy_spec = ReplayAdapter(
                legacy_metadata, root / "legacy-run"
            ).build_spec(PlanTask("t", 0, "c", run_class="smoke"), 1)
            self.assertEqual(
                json.loads(legacy_spec.env["DEFAULT_CHAT_TEMPLATE_KWARGS"]),
                json.loads(spec.env["DEFAULT_CHAT_TEMPLATE_KWARGS"]),
            )
            explicit_empty = dict(legacy_metadata)
            explicit_empty["model_snapshot"] = {
                "raw": {"provenance": {
                    "chat_template_kwargs": "model_overrides.chat_template_kwargs",
                }}
            }
            explicit_spec = ReplayAdapter(
                explicit_empty, root / "explicit-run"
            ).build_spec(PlanTask("t", 0, "c", run_class="smoke"), 1)
            self.assertEqual(explicit_spec.env["DEFAULT_CHAT_TEMPLATE_KWARGS"], "{}")

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
                "http://127.0.0.1:54322", "http://127.0.0.1:54578",
            ])
            self.assertEqual(deployment["service_port_stride"], 256)
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
                self.assertEqual(result["failure_phase"], "startup")
                self.assertFalse(result["backend_unsupported"])

    def test_read_attempt_marks_explicit_backend_unsupported_only_with_ready_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            attempt = Path(td)
            summary = {
                "jsonl_sha256": "u" * 64, "request_count": 1,
                "successes": 1, "failures": 0, "server_usage_available": 1,
                "server_usage_missing": 0, "completion_tokens": 1,
                "measured_seconds": 1.0, "output_tokens_per_second": 1.0,
                "base_url": "http://127.0.0.1:25080",
            }
            requested = {"model_path": "/model", "backend": "triton"}
            (attempt / "x.summary.json").write_text(json.dumps(summary))
            (attempt / "server.requested.json").write_text(json.dumps(requested))
            (attempt / "server.info.json").write_text(json.dumps({
                "server_args": {"model_path": "/model"},
            }))
            (attempt / "server.evidence.json").write_text(json.dumps({
                "readiness": True, "server_info_captured": True, "resolved": False,
                "requested_server_parameters": requested,
                "unsupported_parameters": ["backend"], "mismatched_parameters": [],
                "backend_unsupported": True,
            }))
            result = read_attempt(
                attempt,
                PlanTask("t", 0, "c", run_class="formal"),
                {
                    "expected_source_sha256": "u" * 64,
                    "expected_request_count": 1,
                    "candidates": {"c": {"static_config": {"backend": "triton"}}},
                },
            )
            self.assertEqual(result["status"], "UNSUPPORTED")
            self.assertTrue(result["backend_unsupported"])

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
            self.assertEqual(result["failure_phase"], "startup")

    def test_read_attempt_marks_request_failure_as_replay(self):
        with tempfile.TemporaryDirectory() as td:
            attempt = Path(td)
            summary = {
                "jsonl_sha256": "r" * 64, "request_count": 1,
                "successes": 0, "failures": 1, "server_usage_available": 0,
                "server_usage_missing": 0, "completion_tokens": 1,
                "measured_seconds": 1.0, "output_tokens_per_second": 1.0,
                "base_url": "http://127.0.0.1:25080",
            }
            (attempt / "x.summary.json").write_text(json.dumps(summary))
            requested = {"model_path": "/model", "max_running_requests": 1}
            (attempt / "server.requested.json").write_text(json.dumps(requested))
            (attempt / "server.info.json").write_text(json.dumps(requested))
            (attempt / "server.evidence.json").write_text(json.dumps({
                "readiness": True, "server_info_captured": True, "resolved": True,
                "requested_server_parameters": requested,
                "unsupported_parameters": [], "mismatched_parameters": [],
            }))
            (attempt / "run_manifest.json").write_text(json.dumps({
                "state": "complete", "exit_code": 0,
            }))
            result = read_attempt(attempt, PlanTask("t", 0, "c", run_class="formal"), {
                "expected_source_sha256": "r" * 64, "expected_request_count": 1,
            })
            self.assertEqual(result["status"], "INCONCLUSIVE")
            self.assertIn("request_failures", result["reasons"])
            self.assertEqual(result["failure_phase"], "replay")

    def test_read_attempt_failed_exit_is_failed_even_without_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            result = read_attempt(
                Path(td), PlanTask("t", 0, "c", run_class="formal"), {}, exit_code=17
            )
            self.assertEqual(result["status"], "FAILED")
            self.assertIn("process_exit_17", result["reasons"])

    def test_read_attempt_classifies_startup_oom_without_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            attempt = Path(td)
            (attempt / "server.log").write_text(
                "torch.OutOfMemoryError: CUDA out of memory\n",
                encoding="utf-8",
            )
            result = read_attempt(
                attempt,
                PlanTask("t", 0, "c", run_class="concurrency", concurrency=64),
                {"expected_source_sha256": "x", "expected_request_count": 1},
                exit_code=1,
            )
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["failure_kind"], "oom")
            self.assertIn("out_of_memory", result["reasons"])
            self.assertEqual(result["failure_phase"], "startup")
            self.assertEqual(result["oom_log_paths"], [str(attempt / "server.log")])

    def test_read_attempt_detects_oomkilled_in_nested_instance_log(self):
        with tempfile.TemporaryDirectory() as td:
            attempt = Path(td)
            log = attempt / "instances" / "i0" / "server.log"
            log.parent.mkdir(parents=True)
            log.write_text("container terminated: OOMKilled\n", encoding="utf-8")
            result = read_attempt(
                attempt,
                PlanTask("t", 0, "c", run_class="concurrency", concurrency=64),
                {},
                exit_code=137,
            )
            self.assertEqual(result["failure_kind"], "oom")
            self.assertEqual(result["oom_log_paths"], [str(log)])

    def test_read_attempt_detects_container_oom_marker(self):
        with tempfile.TemporaryDirectory() as td:
            attempt = Path(td)
            (attempt / "container-oom-killed").write_text("1\n", encoding="utf-8")
            result = read_attempt(
                attempt,
                PlanTask("t", 0, "c", run_class="formal"),
                {},
                exit_code=137,
            )
            self.assertEqual(result["failure_kind"], "oom")
            self.assertTrue(result["container_oom_killed"])

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
