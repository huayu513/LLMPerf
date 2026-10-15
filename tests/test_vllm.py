import copy
import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmarks.run_deployment_point import compare_parameters
from benchmarks.server.vllm_config import command_args, launch_config, requested_parameters, server_parameters
from llmperf.adapters import ReplayAdapter, _compare_server_parameters, read_attempt
from llmperf.capabilities import EnvironmentSnapshot, GPUInfo
from llmperf.errors import ConfigError
from llmperf.planner import create_search_plan
from llmperf.types import DockerConfig, ModelManifest, PlanTask, WorkloadManifest
from llmperf.workflow import run_workflow
from llmperf.adaptive import build_candidate_pool
from llmperf.tuning import _catalog, _from_suggestion
from web.server import preview_launch
from benchmarks import run_deployment_point as controller
from benchmarks.server import vllm_server


ROOT = Path(__file__).resolve().parents[1]
VLLM_OPTIONS = [
    "--trust-remote-code", "--served-model-name", "--host", "--port",
    "--tensor-parallel-size", "--data-parallel-size", "--pipeline-parallel-size",
    "--gpu-memory-utilization", "--max-num-seqs", "--max-num-batched-tokens",
    "--max-model-len", "--enforce-eager", "--enable-expert-parallel",
    "--all2all-backend", "--moe-backend", "--speculative-config",
    "--tool-call-parser", "--reasoning-parser", "--enable-auto-tool-choice",
    "--default-chat-template-kwargs",
]


def environment(count=2):
    return EnvironmentSnapshot("PASS", gpus=tuple(
        GPUInfo(index, "A800", 81920, "8.0", f"GPU-{index}") for index in range(count)),
        image={"status": "PASS", "image_id": "sha256:fixture"},
        container={"versions": {"vllm": "0.19.1+backport"}, "capabilities": {
            "options": list(VLLM_OPTIONS), "runner_backends": ["auto", "marlin", "triton"],
            "tool_parsers": ["deepseek_v4"], "reasoning_parsers": ["deepseek_v4"],
        }})


def native_info(config):
    return {"vllm_config": {
        "model_config": {"model": "/model", "max_model_len": config.get("max_model_len", 32768),
                         "enforce_eager": config.get("enforce_eager", False)},
        "parallel_config": {"tensor_parallel_size": config["tp"], "data_parallel_size": config["dp"],
                            "pipeline_parallel_size": config["pp"], "enable_expert_parallel": False},
        "cache_config": {"gpu_memory_utilization": config["gpu_memory_utilization"]},
        "scheduler_config": {"max_num_seqs": config["max_num_seqs"],
                             "max_num_batched_tokens": config["max_num_batched_tokens"]},
        "kernel_config": {"moe_backend": "marlin"}, "speculative_config": None,
    }, "models": {"data": [{"id": config["served_model_name"]}]}}


class VLLMTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = ModelManifest("native-0731", "/model", tool_call_parser="deepseekv4",
            reasoning_parser="deepseek-v4", raw={"model_type": "deepseek_v4", "is_moe": True,
                                               "num_attention_heads": 64, "num_hidden_layers": 80})
        self.config = SimpleNamespace(engine="vllm", engine_options={}, model_path=self.root / "model",
            input_path=self.root / "input.jsonl", image="repo/vllm@sha256:" + "a" * 64,
            gpu_indexes=None, search=SimpleNamespace(backends=None, concurrency_max=16),
            docker=DockerConfig(), warmup=0, ready_timeout=30, request_timeout=6666)
        self.workload = WorkloadManifest("input", sha256="a" * 64,
            raw={"count": 4, "models": {"captured-alias": 4}})

    def plan(self, count=2):
        return create_search_plan(self.config, self.model, self.workload, environment(count), self.root / "results")

    def fixture(self, *, instances=1):
        plan = self.plan()
        candidate = next(candidate for candidate in plan.candidates
                         if candidate.static_config["deployment"]["instance_count"] == instances)
        metadata = plan.metadata
        for name in ("model",):
            (self.root / name).mkdir(exist_ok=True)
        (self.root / "input.jsonl").write_text("{}\n")
        index = self.root / "index.json"
        index.write_text("{}")
        metadata["index_host"] = str(index)
        task = PlanTask("vllm-test", 0, candidate.id, run_class="formal", concurrency=4)
        return metadata, candidate, task

    def test_native_plan_does_not_require_or_generate_sglang_features(self):
        plan = self.plan(8)
        self.assertEqual(plan.metadata["engine"], "vllm")
        self.assertEqual(plan.metadata["tool_call_parser"], "deepseek_v4")
        self.assertEqual(plan.metadata["reasoning_parser"], "deepseek_v4")
        self.assertTrue(any(candidate.tp == 8 for candidate in plan.candidates))
        for candidate in plan.candidates:
            self.assertFalse(candidate.dp_attention or candidate.dspark)
            self.assertEqual(candidate.tp * candidate.dp * candidate.pp,
                             candidate.static_config["deployment"]["gpus_per_instance"])
            self.assertNotIn("mem_fraction_static", candidate.static_config)
            self.assertNotIn("chunked_prefill_size", candidate.static_config)
            self.assertEqual(candidate.static_config["max_model_len"], 32768)

    def test_reasoning_stream_fields_both_count_toward_first_token_activity(self):
        source = (ROOT / "benchmarks/replay/replay_jsonl_sglang.py").read_text()
        node = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "observe_response_item")
        scope = {"Any": object, "json": json}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "replay-parser", "exec"), scope)
        for delta in ({"reasoning": "thinking"}, {"reasoning_content": "thinking"},
                      {"reasoning": "thinking", "reasoning_content": "thinking"}):
            state = {"usage": None, "finish_reason": None, "first_activity": None,
                     "last_activity": None, "content_chars": 0, "reasoning_chars": 0,
                     "tool_call_chunks": 0, "tool_call_chars": 0}
            scope["observe_response_item"]({"choices": [{"delta": delta}]}, 0.25, state)
            self.assertEqual(state["reasoning_chars"], 8)
            self.assertEqual(state["first_activity"], 0.25)
            scope["observe_response_item"]({"choices": [{"delta": {"content": "answer"}}]}, 0.75, state)
            self.assertEqual(state["first_activity"], 0.25)
            self.assertEqual(state["last_activity"], 0.75)

    def test_explicit_native_backend_and_speculation_are_supported(self):
        self.config.engine_options = {"moe_backend": "marlin", "speculative_config": {
            "method": "deepseek_mtp", "num_speculative_tokens": 5}}
        plan = self.plan()
        self.assertTrue(all(candidate.backend == "marlin" for candidate in plan.candidates))
        native = launch_config(plan.metadata, plan.candidates[0].static_config, 4)
        command = command_args(native, 25080)
        self.assertIn("--moe-backend", command)
        self.assertIn("--speculative-config", command)
        self.assertNotIn("--speculative-algorithm", command)
        info = native_info(native)
        info["vllm_config"]["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 5}
        expected = requested_parameters(native)
        self.assertEqual(compare_parameters(expected, info)[:2], ([], []))
        self.assertEqual(_compare_server_parameters(expected, info), ([], []))

    def test_fixed_parallelism_filters_out_smaller_native_weight_instances(self):
        self.config.engine_options = {"tensor_parallel_size": 8, "data_parallel_size": 1,
                                      "pipeline_parallel_size": 1}
        plan = self.plan(8)
        self.assertEqual(len(plan.candidates), 1)
        self.assertEqual(plan.candidates[0].tp, 8)
        self.assertEqual(plan.candidates[0].static_config["deployment"]["instance_count"], 1)
        self.assertNotIn("tensor_parallel_size", plan.candidates[0].static_config)

    def test_native_tuning_catalog_pool_and_web_preview(self):
        plan = self.plan()
        catalog = _catalog(plan, {"target_concurrency": 4})
        self.assertEqual(set(catalog["knobs"]), {
            "gpu_memory_utilization", "max_num_seqs", "max_num_batched_tokens"})
        changed = _from_suggestion({"base_id": plan.candidates[0].id,
            "params": {"max_num_batched_tokens": 2048, "gpu_memory_utilization": 0.8}}, catalog)
        self.assertEqual(changed.static_config["max_num_batched_tokens"], 2048)
        with self.assertRaises(ValueError):
            _from_suggestion({"base_id": plan.candidates[0].id, "params": {"max_num_seqs": True}}, catalog)
        pool = build_candidate_pool(list(plan.candidates), catalog["knobs"], 4, 24)
        self.assertTrue(pool)
        self.assertTrue(all("max_running_requests" not in candidate.static_config for candidate in pool))
        self.assertTrue(all(candidate.static_config["max_num_seqs"] <= 4 for candidate in pool))
        candidate = plan.metadata["candidates"][plan.candidates[0].id]
        preview = preview_launch(plan.metadata, candidate, 4)
        self.assertEqual(preview["engine"], "vllm")
        self.assertIn("--max-num-batched-tokens", preview["server_command_preview"])
        self.assertNotIn("sglang_command_preview", preview)

    def test_controller_captures_each_native_instance_before_replay(self):
        metadata, candidate, task = self.fixture(instances=2)
        spec = ReplayAdapter(metadata, self.root / "run").build_spec(task, 1)
        output = self.root / "controller"
        env = {**os.environ, **spec.env, "S1_RESULTS_ROOT": str(output)}
        config = json.loads(spec.env["S1_VLLM_CONFIG"])
        info = native_info(config)
        calls = []
        def execute(command, **kwargs):
            calls.append(command)
            if command[:2] == ["bash", str(controller.LAUNCHER)] and command[2] == "start":
                state = Path(kwargs["env"]["S1_SERVER_STATE_DIR"])
                state.mkdir(parents=True)
                (state / "server.requested.json").write_text(json.dumps(requested_parameters(config)))
                (state / "server.command.sh").write_text("vllm serve /model --port " + kwargs["env"]["SERVER_PORT"] + "\n")
            return SimpleNamespace(returncode=0)
        def fetch(url, **kwargs):
            if url.endswith("/server_info?config_format=json"):
                return True, {"vllm_config": info["vllm_config"]}
            self.assertTrue(url.endswith("/v1/models"), url)
            return True, info["models"]
        urls = {}
        def health(url, **kwargs):
            urls[url] = urls.get(url, 0) + 1
            return urls[url] > 1
        with patch.dict(os.environ, env, clear=True), patch.object(sys, "argv", [
            "controller", "formal", "--profile", "AUTO", "--concurrency", "4"]), \
                patch.object(controller, "http_ok", side_effect=health), \
                patch.object(controller, "http_json", side_effect=fetch), \
                patch.object(controller.subprocess, "run", side_effect=execute):
            self.assertEqual(controller.main(), 0)
        evidence = json.loads((output / "server.evidence.json").read_text())
        self.assertTrue(evidence["resolved"])
        self.assertEqual(len(evidence["instances"]), 2)
        self.assertEqual(sum(command[1] == str(controller.REPLAY) for command in calls), 1)
        for instance in candidate.static_config["deployment"]["instances"]:
            self.assertTrue((output / "instances" / instance["id"] / "server.info.json").is_file())
        (output / "formal.summary.json").write_text(json.dumps({"jsonl_sha256": "a" * 64,
            "request_count": 4, "successes": 4, "failures": 0,
            "server_usage_available": 4, "server_usage_missing": 0,
            "completion_tokens": 40, "measured_seconds": 2, "output_tokens_per_second": 20,
            "base_urls": ["http://127.0.0.1:25080", "http://127.0.0.1:25336"]}))
        (output / "container-exit-code").write_text("0\n")
        result = read_attempt(output, task, metadata)
        self.assertEqual(result["status"], "VALID", result["reasons"])
        damaged = output / "instances" / "i1" / "server.info.json"
        payload = json.loads(damaged.read_text())
        payload["vllm_config"]["parallel_config"]["tensor_parallel_size"] = 99
        damaged.write_text(json.dumps(payload))
        result = read_attempt(output, task, metadata)
        self.assertIn("instance_server_parameter_mismatch", result["reasons"])
        self.assertNotEqual(result["status"], "VALID")

    def test_missing_required_options_and_conflicting_backends_fail_closed(self):
        self.config.search.backends = ("marlin",)
        self.config.engine_options = {"moe_backend": "marlin"}
        with self.assertRaisesRegex(ConfigError, "either"):
            self.plan()
        self.config.engine_options = {}
        snapshot = environment()
        snapshot.container["capabilities"]["options"].remove("--moe-backend")
        with self.assertRaisesRegex(ConfigError, "--moe-backend"):
            create_search_plan(self.config, self.model, self.workload, snapshot, self.root)

    def test_single_and_multi_instance_specs_use_native_controller(self):
        for instances in (1, 2):
            metadata, candidate, task = self.fixture(instances=instances)
            spec = ReplayAdapter(metadata, self.root / "run").build_spec(task, 1)
            self.assertEqual(spec.entrypoint, "")
            self.assertEqual(spec.command[:2], ("python3", "/opt/s1slow/benchmarks/run_deployment_point.py"))
            native = json.loads(spec.env["S1_VLLM_CONFIG"])
            self.assertEqual(native["max_num_seqs"], 4 // instances)
            self.assertEqual(native["chat_template_kwargs"], self.model.chat_template_kwargs)
            command = command_args(native, 25080)
            self.assertEqual(command[:3], ["vllm", "serve", "/model"])
            self.assertIn("--enable-auto-tool-choice", command)
            self.assertNotIn("--model-path", command)
            self.assertNotIn("--moe-runner-backend", command)
            self.assertEqual(json.loads(spec.env["S1_DEPLOYMENT"])["instance_count"], instances)

    def test_native_evidence_is_checked_without_trusting_requested_values(self):
        metadata, candidate, task = self.fixture()
        config = launch_config(metadata, candidate.static_config, 4)
        requested = requested_parameters(config)
        info = native_info(config)
        self.assertEqual(compare_parameters(requested, info)[:2], ([], []))
        self.assertNotIn("tool_call_parser", requested)
        tampered = copy.deepcopy(info)
        tampered["vllm_config"]["parallel_config"]["tensor_parallel_size"] = 99
        self.assertIn("tp", compare_parameters(requested, tampered)[1])
        self.assertEqual(server_parameters({"vllm_config": "ModelConfig(...)"}), {})
        attempt = self.root / "attempt"
        attempt.mkdir()
        def write(name, payload):
            (attempt / name).write_text(json.dumps(payload))
        write("server.requested.json", requested)
        write("server.info.json", info)
        write("server.evidence.json", {"readiness": True, "server_info_captured": True, "resolved": True,
            "requested_server_parameters": requested, "unsupported_parameters": [], "mismatched_parameters": []})
        write("run_manifest.json", {"state": "complete", "exit_code": 0})
        write("formal.summary.json", {"jsonl_sha256": "a" * 64, "request_count": 4,
            "successes": 4, "failures": 0, "server_usage_available": 4, "server_usage_missing": 0,
            "completion_tokens": 40, "measured_seconds": 2, "output_tokens_per_second": 20,
            "base_url": "http://127.0.0.1:25080"})
        (attempt / "container-exit-code").write_text("0\n")
        result = read_attempt(attempt, task, metadata)
        self.assertEqual(result["status"], "VALID", result["reasons"])
        self.assertEqual(result["effective_static_config"]["max_num_seqs"], 4)
        write("server.info.json", tampered)
        result = read_attempt(attempt, task, metadata)
        self.assertNotEqual(result["status"], "VALID")
        self.assertIn("server_parameter_mismatch", result["reasons"])

    def test_launcher_argv_and_reproduction_keep_dev_endpoints_private(self):
        metadata, candidate, task = self.fixture()
        metadata["profile_env"] = {"VLLM_DSV4_SM80_FUSE_PRE_GATE": "1"}
        spec = ReplayAdapter(metadata, self.root / "run").build_spec(task, 1)
        process = subprocess.run(["bash", str(ROOT / "benchmarks/server/launch_server.sh"), "argv-json", "AUTO"],
            env={**os.environ, **spec.env, "S1_PYTHON_BIN": sys.executable}, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        argv = json.loads(process.stdout)
        self.assertIn("VLLM_DSV4_SM80_FUSE_PRE_GATE=1", argv["env"])
        self.assertEqual(argv["argv"][argv["argv"].index("--host") + 1], "127.0.0.1")
        attempt = self.root / "reproduce"
        attempt.mkdir()
        (attempt / "server.command.sh").write_text("env VLLM_SERVER_DEV_MODE=1 vllm serve /model --host 127.0.0.1 --port 25080\n")
        script = ReplayAdapter._write_server_reproduce_script(spec, attempt)
        text = script.read_text()
        self.assertIn("VLLM_SERVER_DEV_MODE=0", text)
        self.assertNotIn("VLLM_SERVER_DEV_MODE=1", text)
        self.assertIn("--entrypoint ''", text)
        self.assertIn("--host 0.0.0.0", text)

    def test_lifecycle_refuses_reused_pids_and_unmanaged_process_groups(self):
        record = json.dumps({"pid": 4321, "start_time": "100"})
        fields = ["S"] + ["0"] * 18 + ["100"]
        stat = "4321 (vllm serve) " + " ".join(fields)
        with patch.object(Path, "read_text", side_effect=[record, stat]), \
                patch.object(vllm_server.os, "getpgid", return_value=4321):
            self.assertEqual(vllm_server.managed_pid(self.root), 4321)
        fields[-1] = "101"
        stale = "4321 (vllm serve) " + " ".join(fields)
        with patch.object(Path, "read_text", side_effect=[record, stale]), \
                patch.object(vllm_server.os, "killpg") as kill:
            vllm_server.stop(self.root)
            kill.assert_not_called()
        with patch.object(Path, "read_text", side_effect=[record, stat]), \
                patch.object(vllm_server.os, "getpgid", return_value=999):
            self.assertIsNone(vllm_server.managed_pid(self.root))

    def test_workflow_preserves_engine_and_skips_sglang_backend_resolver(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": "deepseek_v4",
            "architectures": ["DeepseekV4ForCausalLM"], "num_attention_heads": 64,
            "num_hidden_layers": 80, "n_routed_experts": 256}))
        source = self.root / "input.jsonl"
        source.write_text(json.dumps({"api": "openai_chat_completions", "request_id": "r0",
            "source_message_id": "s0", "captured_at": "2026-10-15T00:00:00Z",
            "request": {"model": "captured-alias",
            "messages": [{"role": "user", "content": "hello"}], "max_tokens": 4}}) + "\n")
        path = self.root / "experiment.json"
        path.write_text(json.dumps({"engine": "vllm", "model_path": "model", "input_path": "input.jsonl",
                                   "image": self.config.image}))
        probe = Mock(return_value=environment())
        resolver = Mock(side_effect=AssertionError("must not invoke SGLang resolver"))
        report = run_workflow(path, stop_after="plan", probe=probe, backend_resolver=resolver)
        resolver.assert_not_called()
        self.assertEqual(probe.call_args.args[0].engine, "vllm")
        saved = json.loads((Path(report["result_dir"]) / "plan.json").read_text())
        self.assertEqual(saved["metadata"]["configuration"]["engine"], "vllm")


if __name__ == "__main__":
    unittest.main()
