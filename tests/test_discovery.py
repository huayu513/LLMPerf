import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import mock_open, patch

from llmperf.model_weights import inspect_model_weights, screen_weight_capacity
from llmperf.types import CandidateConfig, ModelManifest

from llmperf.discovery import (
    discover_model,
    inspect_workload,
    prepare_workload,
)
from llmperf.errors import ConfigError
from llmperf.capabilities import EnvironmentSnapshot, GPUInfo
from llmperf.tuning import plan_tuning
from llmperf.workflow import run_saved_plan, run_workflow
from llmperf.planner import fingerprint


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def model_dir(self, name, chat_template="<tool_call></tool_call>", **metadata):
        path = self.root / name
        path.mkdir()
        base = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "num_attention_heads": 32,
            "num_hidden_layers": 48,
        }
        base.update(metadata)
        (path / "config.json").write_text(json.dumps(base), encoding="utf-8")
        if chat_template is not None:
            (path / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": chat_template}), encoding="utf-8"
            )
        return path

    @staticmethod
    def record(index, model="captured-alias", request_updates=None):
        request = {
            "model": model,
            "messages": [{"role": "user", "content": f"hello {index}"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.2,
        }
        request.update(request_updates or {})
        return {
            "api": "openai_chat_completions",
            "request_id": f"request-{index}",
            "source_message_id": index,
            "captured_at": f"2026-09-09T00:00:0{index}Z",
            "request": request,
        }

    def write_jsonl(self, records):
        path = self.root / "captured.jsonl"
        source = b"".join(
            (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            for row in records
        )
        path.write_bytes(source)
        return path, source

    def test_renamed_qwen_directories_are_classified_only_from_metadata(self):
        first = discover_model(self.model_dir("unrelated-one"))
        second = discover_model(self.model_dir("also-unrelated"))

        self.assertEqual(first.tool_call_parser, "qwen")
        self.assertEqual(first.reasoning_parser, "qwen3")
        self.assertEqual(second.tool_call_parser, first.tool_call_parser)
        self.assertEqual(second.reasoning_parser, first.reasoning_parser)
        self.assertEqual(first.chat_template_kwargs, {
            "enable_thinking": True,
            "reasoning_effort": "max",
            "thinking": True,
        })
        self.assertEqual(first.raw["model_type"], "qwen3")
        self.assertEqual(first.raw["num_attention_heads"], 32)
        self.assertEqual(first.raw["num_hidden_layers"], 48)
        self.assertFalse(first.raw["is_moe"])
        self.assertIn("metadata", first.raw)
        self.assertIn("provenance", first.raw)

    def test_glm_and_deepseek_architectures_map_to_supported_runtime_metadata(self):
        glm = discover_model(self.model_dir(
            "x", model_type="glm4_moe", architectures=["Glm4MoeForCausalLM"],
            n_routed_experts=160,
        ))
        deepseek = discover_model(self.model_dir(
            "y", model_type="deepseek_v4", architectures=["DeepseekV4ForCausalLM"],
            n_routed_experts=256,
        ))

        self.assertEqual((glm.tool_call_parser, glm.reasoning_parser), ("glm", "glm45"))
        self.assertEqual((deepseek.tool_call_parser, deepseek.reasoning_parser),
                         ("deepseekv4", "deepseek-v4"))
        self.assertTrue(glm.raw["is_moe"])
        self.assertTrue(deepseek.raw["is_moe"])

    def test_deepseek_v3_tool_parser_uses_static_template_markers(self):
        metadata = {
            "model_type": "deepseek_v3",
            "architectures": ["DeepseekV3ForCausalLM"],
            "n_routed_experts": 256,
        }
        v3 = discover_model(self.model_dir(
            "v3", chat_template=(
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
                "function<｜tool▁sep｜>name"
            ), **metadata,
        ))
        v32 = discover_model(self.model_dir(
            "v32", chat_template=(
                '<｜DSML｜function_calls><｜DSML｜invoke name="tool">'
            ), **metadata,
        ))

        self.assertEqual(v3.tool_call_parser, "deepseekv3")
        self.assertEqual(v32.tool_call_parser, "deepseekv32")
        self.assertIn(
            "tokenizer_config.json",
            v3.raw["provenance"]["tool_call_parser"],
        )

    def test_actual_deepseek_v3_concatenated_jinja_selects_v3_parser(self):
        fence = chr(96) * 3
        template = (
            "{{ '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + "
            "tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + "
            "'\n' + '" + fence + "json' + '\n' + formatted_args + '\n' + "
            "'" + fence + "' + '<｜tool▁call▁end｜>' }}"
        )
        model = discover_model(self.model_dir(
            "actual-v3", chat_template=template,
            model_type="deepseek_v3",
            architectures=["DeepseekV3ForCausalLM"],
            n_routed_experts=256,
        ))

        self.assertEqual(model.tool_call_parser, "deepseekv3")

    def test_qwen3_coder_marker_selects_coder_parser(self):
        model = discover_model(self.model_dir(
            "coder", model_type="qwen3_moe",
            architectures=["Qwen3MoeForCausalLM"],
            chat_template="<function=tool><parameter=arg>value</parameter></function>",
            num_experts=128,
        ))

        self.assertEqual(model.tool_call_parser, "qwen3_coder")

    def test_ambiguous_tool_parser_requires_explicit_override(self):
        deepseek = self.model_dir(
            "ambiguous-deepseek", chat_template=None,
            model_type="deepseek_v3", architectures=["DeepseekV3ForCausalLM"],
            n_routed_experts=256,
        )
        qwen = self.model_dir("ambiguous-qwen", chat_template=None)

        for path in (deepseek, qwen):
            with self.subTest(path=path), self.assertRaisesRegex(
                ConfigError, "model_overrides.tool_call_parser"
            ):
                discover_model(path)
        explicit = discover_model(deepseek, {"tool_call_parser": "deepseekv3"})
        self.assertEqual(explicit.tool_call_parser, "deepseekv3")
        self.assertEqual(
            explicit.raw["provenance"]["tool_call_parser"],
            "model_overrides.tool_call_parser",
        )

    def test_quantization_is_reported_without_loading_model_code(self):
        model = discover_model(self.model_dir(
            "quantized",
            quantization_config={"quant_method": "fp8", "activation_scheme": "dynamic"},
        ))

        self.assertEqual(model.quantization, "fp8")
        self.assertEqual(model.raw["metadata"]["quantization_config"]["activation_scheme"],
                         "dynamic")
        self.assertEqual(model.raw["provenance"]["quantization"],
                         "config.json:quantization_config.quant_method")

    def test_model_inventory_hashes_metadata_and_stats_weight_shards(self):
        path = self.model_dir("inventory")
        tokenizer_bytes = json.dumps({"version": "1.0"}).encode() + bytes([10])
        (path / "tokenizer.json").write_bytes(tokenizer_bytes)
        shard = path / "model-00001-of-00002.safetensors"
        shard.write_bytes(b"weight-placeholder")

        inventory = discover_model(path).raw["inventory"]

        self.assertEqual(
            inventory["metadata_files"]["tokenizer.json"]["sha256"],
            hashlib.sha256(tokenizer_bytes).hexdigest(),
        )
        self.assertEqual(inventory["weight_files"][0]["path"], shard.name)
        self.assertEqual(inventory["weight_files"][0]["size"], len(b"weight-placeholder"))
        self.assertIsInstance(inventory["weight_files"][0]["mtime_ns"], int)
        self.assertEqual(inventory["weight_identity"], "filename-size-mtime_ns")

    def test_explicit_overrides_are_validated_and_recorded(self):
        model = discover_model(self.model_dir("override"), {
            "tool_call_parser": None,
            "chat_template_kwargs": {"thinking": False},
            "profile_env": {"SAFE_FLAG": "1"},
        })

        self.assertEqual(model.served_model_name, "override")
        self.assertIsNone(model.tool_call_parser)
        self.assertEqual(model.chat_template_kwargs, {"thinking": False})
        self.assertEqual(model.profile_env, {"SAFE_FLAG": "1"})
        self.assertEqual(model.raw["provenance"]["tool_call_parser"],
                         "model_overrides.tool_call_parser")
        with self.assertRaisesRegex(ConfigError, "unknown model override"):
            discover_model(self.root / "override", {"backend": "ignored"})

    def test_unknown_or_incomplete_model_metadata_is_actionable(self):
        unknown = self.model_dir(
            "mystery", model_type="mystery", architectures=["MysteryForCausalLM"])
        model = discover_model(unknown)
        self.assertEqual(model.raw["family"], "generic")
        self.assertFalse(model.raw["protocol"]["runtime_verified"])
        self.assertEqual(model.chat_template_kwargs, {})
        incomplete = self.model_dir("incomplete", num_attention_heads=None)
        with self.assertRaisesRegex(ConfigError, "num_attention_heads"):
            discover_model(incomplete)

    def test_workload_validation_hashes_counts_and_preserves_source_bytes(self):
        path, source = self.write_jsonl([self.record(1), self.record(2)])
        workload = inspect_workload(path)

        self.assertEqual(path.read_bytes(), source)
        self.assertEqual(workload.sha256, hashlib.sha256(source).hexdigest())
        self.assertEqual(workload.raw["count"], 2)
        self.assertEqual(workload.raw["models"], {"captured-alias": 2})
        self.assertEqual(workload.raw["model"], "captured-alias")
        self.assertEqual(workload.jsonl_resolved, str(path.resolve()))

    def test_workload_rejects_malformed_duplicate_and_mixed_alias_records(self):
        cases = [
            [self.record(1), self.record(2, model="other")],
            [self.record(1), self.record(1)],
            [{"api": "wrong"}],
            [],
        ]
        for records in cases:
            with self.subTest(records=records):
                path, _ = self.write_jsonl(records)
                with self.assertRaises(ConfigError):
                    inspect_workload(path)

    def test_prepare_workload_uses_canonical_token_free_indexer(self):
        source_path, source = self.write_jsonl([self.record(1), self.record(2)])
        index_path = self.root / "data" / "replay_index.json"

        result = prepare_workload(source_path, index_path)

        self.assertEqual(source_path.read_bytes(), source)
        self.assertEqual(result["format"], "s1_jsonl_chat_requests")
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["models"], {"captured-alias": 2})
        self.assertIsNone(result["model_path"])
        self.assertIsNone(result["tokenizer_class"])
        self.assertIsNone(result["input_tokens"])
        self.assertEqual(json.loads(index_path.read_text()), result)


def write_weights(path, tensors):
    cursor = 0
    header = {}
    for name, dtype, shape, size in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + size]}
        cursor += size
    raw = json.dumps(header).encode()
    with path.open("wb") as output:
        output.write(len(raw).to_bytes(8, "little"))
        output.write(raw)
        output.truncate(8 + len(raw) + cursor)
    return raw


class ModelWeightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_auto_reads_shapes_dtype_and_payload_size_without_reading_tensor_data(self):
        path = self.root / "model.safetensors"
        header = write_weights(path, [
            ("model.layers.0.weight", "BF16", [4, 5], 40),
            ("lm_head.weight", "BF16", [2, 5], 20),
            ("scalar", "F32", [], 4),
        ])
        opener = mock_open(read_data=len(header).to_bytes(8, "little") + header)
        with patch("llmperf.model_weights.Path.open", opener):
            report = inspect_model_weights(self.root, {"num_hidden_layers": 1})
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["parameter_count"], 31)
        self.assertEqual(report["stored_weight_bytes"], 64)
        self.assertEqual(report["minimum_resident_weight_bytes"], 40)
        self.assertEqual(report["dtype_summary"]["BF16"]["elements"], 30)
        self.assertIsNone(report["logical_parameter_count"])
        self.assertEqual([call.args[0] for call in opener().read.call_args_list], [8, len(header)])

    def test_index_selects_only_the_actual_checkpoint_and_ignores_extra_weights(self):
        write_weights(self.root / "part-1.safetensors", [("model.a.weight", "F16", [4], 8)])
        write_weights(self.root / "part-2.safetensors", [("model.b.weight", "F16", [6], 12)])
        write_weights(self.root / "adapter_model.safetensors", [("adapter.weight", "F16", [99], 198)])
        write_weights(self.root / "unrelated.safetensors", [("other.weight", "F16", [99], 198)])
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 20}, "weight_map": {
                "model.a.weight": "part-1.safetensors", "model.b.weight": "part-2.safetensors"}}))
        report = inspect_model_weights(self.root, {})
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["file_count"], 2)
        self.assertEqual(report["parameter_count"], 10)
        self.assertEqual(report["stored_weight_bytes"], 20)

    def test_missing_indexed_shard_or_index_path_escape_never_proves_capacity(self):
        for name in ("missing.safetensors", "../outside.safetensors", "/outside.safetensors"):
            with self.subTest(name=name):
                (self.root / "model.safetensors.index.json").write_text(json.dumps({
                    "weight_map": {"model.weight": name}}))
                report = inspect_model_weights(self.root, {})
                self.assertNotEqual(report["status"], "complete")
                self.assertIsNone(report["minimum_resident_weight_bytes"])
                self.assertTrue(report["errors"])

    def test_quantized_storage_is_not_claimed_to_be_logical_parameter_count(self):
        write_weights(self.root / "model.safetensors", [("model.weight", "I32", [4], 16)])
        report = inspect_model_weights(self.root, {"quantization_config": {"quant_method": "gptq", "bits": 4}}, "gptq")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["stored_tensor_elements"], 4)
        self.assertEqual(report["stored_weight_bytes"], 16)
        self.assertIsNone(report["parameter_count"])
        self.assertIsNone(report["minimum_resident_weight_bytes"])

    def test_fp32_lower_bound_allows_loading_at_half_precision(self):
        write_weights(self.root / "model.safetensors", [("model.weight", "F32", [10], 40)])
        report = inspect_model_weights(self.root, {})
        self.assertEqual(report["minimum_resident_weight_bytes"], 20)

    def test_tied_output_heads_and_optional_draft_layers_are_excluded_from_lower_bound(self):
        write_weights(self.root / "model.safetensors", [
            ("model.layers.0.weight", "F16", [10], 20),
            ("model.layers.1.weight", "F16", [20], 40),
            ("model.mtp.weight", "F16", [10], 20),
            ("lm_head.weight", "F16", [10], 20),
        ])
        report = inspect_model_weights(self.root, {"num_hidden_layers": 1})
        self.assertEqual(report["minimum_resident_weight_bytes"], 20)

    def test_binary_index_size_is_recorded_but_never_unpickled_or_used_as_GPU_bound(self):
        (self.root / "pytorch_model.bin").write_bytes(b"not a pickle")
        (self.root / "pytorch_model.bin.index.json").write_text(json.dumps({
            "metadata": {"total_size": 300000000000},
            "weight_map": {"model.weight": "pytorch_model.bin"}}))
        report = inspect_model_weights(self.root, {})
        self.assertEqual(report["index_declared_weight_bytes"], 300000000000)
        self.assertIsNone(report["parameter_count"])
        self.assertIsNone(report["minimum_resident_weight_bytes"])

    def test_invalid_huge_header_and_truncated_payload_are_reported_not_loaded(self):
        path = self.root / "model.safetensors"
        path.write_bytes((2 ** 63).to_bytes(8, "little"))
        report = inspect_model_weights(self.root, {})
        self.assertTrue(report["errors"])
        self.assertIsNone(report["minimum_resident_weight_bytes"])
        write_weights(path, [("model.weight", "F16", [10], 20)])
        with path.open("r+b") as output:
            output.truncate(path.stat().st_size - 1)
        report = inspect_model_weights(self.root, {})
        self.assertTrue(report["errors"])
        self.assertIsNone(report["minimum_resident_weight_bytes"])

    def test_ambiguous_checkpoint_does_not_become_a_hard_bound(self):
        write_weights(self.root / "weights.safetensors", [("model.weight", "F16", [10], 20)])
        report = inspect_model_weights(self.root, {})
        self.assertEqual(report["status"], "partial")
        self.assertIsNone(report["minimum_resident_weight_bytes"])

    def test_discovery_automatically_reads_model_path_without_manual_size_configuration(self):
        (self.root / "config.json").write_text(json.dumps({"model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 2, "num_attention_heads": 2}))
        (self.root / "tokenizer_config.json").write_text('{"chat_template":"<tool_call></tool_call>"}')
        write_weights(self.root / "model.safetensors", [("model.layers.0.weight", "BF16", [10], 20)])
        model = discover_model(self.root)
        self.assertEqual(model.raw["weights"]["parameter_count"], 10)
        self.assertEqual(model.raw["weights"]["minimum_resident_weight_bytes"], 20)

    def test_nonstandard_runtime_dtype_cannot_prove_a_sixteen_bit_lower_bound(self):
        write_weights(self.root / "model.safetensors", [("model.weight", "F32", [10], 40)])
        report = inspect_model_weights(self.root, {"dtype": "float8_e4m3fn"})
        self.assertEqual(report["parameter_count"], 10)
        self.assertIsNone(report["minimum_resident_weight_bytes"])

    def test_unspecified_quantization_or_compression_method_does_not_enable_hard_exclusion(self):
        write_weights(self.root / "model.safetensors", [("model.weight", "F16", [10], 20)])
        for metadata in ({"quantization_config": {}}, {"compression_config": {"format": "custom"}}):
            report = inspect_model_weights(self.root, metadata)
            self.assertTrue(report["quantization_declared"])
            self.assertIsNone(report["minimum_resident_weight_bytes"])
            self.assertIsNone(report["parameter_count"])


class WeightCapacityTests(unittest.TestCase):
    def model(self, **kwargs):
        return ModelManifest("test", "/model", raw={"weights": {"minimum_resident_weight_bytes": 300 * 1024 ** 3},
                                                    "is_moe": False}, **kwargs)

    def environment(self):
        return SimpleNamespace(gpus=[SimpleNamespace(index=index, memory_total_mb=80 * 1024) for index in range(8)])

    def candidate(self, count, dp=1):
        instances = [{"gpu_indexes": list(range(offset, offset + count))} for offset in range(0, 8, count)]
        return CandidateConfig(str(count) + "-dp" + str(dp), tp=count // dp, dp=dp, gpu_indexes=tuple(range(8)),
            static_config={"deployment": {"instance_count": len(instances), "gpus_per_instance": count,
                "instances": instances, "label": "8 GPUs / " + str(len(instances)) + " instances"}})

    def test_independent_instances_do_not_pool_all_eight_gpu_memories(self):
        for engine in ("sglang", "vllm"):
            with self.subTest(engine=engine):
                candidates = [replace(self.candidate(count), static_config={**self.candidate(count).static_config,
                              "engine": engine}) for count in (1, 2, 4, 8)]
                kept, report = screen_weight_capacity(candidates, self.model(), self.environment())
                self.assertEqual([candidate.tp for candidate in kept], [4, 8])
                self.assertEqual(report["excluded_count"], 2)
                self.assertEqual(report["candidates"][0]["optimistic_replica_capacity_bytes"], 80 * 1024 ** 3)
                self.assertFalse(report["runtime_verified"])

    def test_dense_data_parallelism_does_not_split_the_model_weights(self):
        kept, report = screen_weight_capacity([self.candidate(8, dp=8)], self.model(), self.environment())
        self.assertFalse(kept)
        self.assertEqual(report["excluded_count"], 1)

    def test_moe_uses_optimistic_instance_capacity_and_never_active_parameter_count(self):
        model = replace(self.model(), raw={"is_moe": True, "active_parameters": 1,
                                         "weights": {"minimum_resident_weight_bytes": 300 * 1024 ** 3}})
        kept, report = screen_weight_capacity([self.candidate(1), self.candidate(8, dp=8)], model, self.environment())
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].dp, 8)
        self.assertEqual(report["excluded_count"], 1)

    def test_unknown_quantization_custom_loading_or_missing_gpu_memory_does_not_hard_exclude(self):
        for model in (self.model(quantization="fp8"), self.model(profile_env={"CUSTOM_LOADING": "1"}),
                      replace(self.model(), raw={})):
            kept, report = screen_weight_capacity([self.candidate(1)], model, self.environment())
            self.assertEqual(len(kept), 1)
            self.assertEqual(report["excluded_count"], 0)
        environment = self.environment()
        environment.gpus[0].memory_total_mb = None
        kept, report = screen_weight_capacity([self.candidate(8)], self.model(), environment)
        self.assertEqual(len(kept), 1)
        self.assertEqual(report["unknown_count"], 1)


class WeightPlanningTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def configuration(self, engine="sglang", size_mb=300):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 8, "num_attention_heads": 8}))
        (model / "tokenizer_config.json").write_text('{"chat_template":"<tool_call></tool_call>"}')
        write_weights(model / "model.safetensors", [
            ("model.layers.0.weight", "BF16", [size_mb * 1024 ** 2 // 2], size_mb * 1024 ** 2)])
        (self.root / "input.jsonl").write_text(json.dumps(DiscoveryTests.record(1)) + "\n")
        config = self.root / "experiment.json"
        config.write_text(json.dumps({"engine": engine, "model_path": "model",
            "image": "repo/engine@sha256:" + "a" * 64, "output_dir": "results", "input_path": "input.jsonl",
            "scenario": {"target_concurrency": 4, "priority": "latency", "max_input_tokens": 8192,
                         "max_output_tokens": 2048},
            "advisor": {"base_url": "http://127.0.0.1:9000/v1", "model": "local", "authorization": "Bearer test"},
            "search": {"max_trials": 14, "repetitions": 1,
                       "adaptive": {"candidate_pool_limit": 32, "max_advisor_calls": 3}}}))
        return config

    def environment(self, engine="sglang"):
        options = ["--trust-remote-code", "--served-model-name", "--host", "--port", "--tool-call-parser",
                   "--reasoning-parser", "--default-chat-template-kwargs"]
        if engine == "sglang":
            options += ["--tp-size", "--dp", "--pp-size", "--model-path", "--enable-metrics",
                        "--enable-cache-report", "--mem-fraction-static", "--max-running-requests",
                        "--chunked-prefill-size"]
        else:
            options += ["--tensor-parallel-size", "--data-parallel-size", "--pipeline-parallel-size",
                        "--gpu-memory-utilization", "--max-num-seqs", "--max-num-batched-tokens",
                        "--max-model-len", "--enforce-eager", "--enable-auto-tool-choice"]
        return EnvironmentSnapshot("PASS", gpus=tuple(GPUInfo(index, "A800", 80, "8.0", f"GPU-{index}")
            for index in range(8)), container={"capabilities": {"options": options,
                "tool_parsers": ["qwen"], "reasoning_parsers": ["qwen3"]}})

    def test_both_engines_send_automatic_weight_facts_to_advisor_and_freeze_reports(self):
        class Advisor:
            def __init__(self):
                self.contexts = []

            def ask(self, task, payload):
                if task == "capabilities":
                    self.contexts.append(payload["context"])
                    return {"features": []}
                return {"candidates": []}

        for engine in ("sglang", "vllm"):
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as directory:
                self.root = Path(directory)
                config = self.configuration(engine)
                original = config.read_bytes(), (self.root / "input.jsonl").read_bytes()
                advisor = Advisor()
                with patch("llmperf.tuning.preflight_candidates", return_value={"status": "unknown", "results": []}):
                    result = plan_tuning(config, probe=lambda _: self.environment(engine), advisor=advisor, calibrate=False)
                plan = json.loads(Path(result["plan"]).read_text())
                metadata = plan["metadata"]
                context, = advisor.contexts
                self.assertEqual(context["model_weights"]["parameter_count"], 300 * 1024 ** 2 // 2)
                self.assertEqual(context["model_weights"]["stored_weight_bytes"], 300 * 1024 ** 2)
                self.assertNotIn("files", context["model_weights"])
                self.assertGreater(context["weight_capacity"]["excluded_count"], 0)
                self.assertIn("deployment_summary", context["weight_capacity"])
                self.assertTrue(all(candidate["tp"] * candidate["pp"] >= 4 for candidate in plan["candidates"]))
                self.assertEqual(metadata["adaptive_policy"]["max_advisor_calls"], 3)
                for name in ("model-weights.json", "weight-capacity.json"):
                    report = Path(result["result_dir"]) / name
                    self.assertEqual(hashlib.sha256(report.read_bytes()).hexdigest(), metadata["capability_artifacts"][name])
                self.assertEqual(original, (config.read_bytes(), (self.root / "input.jsonl").read_bytes()))
                report = Path(result["result_dir"]) / "model-weights.json"
                report.write_text("{}")
                with self.assertRaisesRegex(ConfigError, "capability evidence changed"):
                    run_saved_plan(result["plan"], probe=lambda _: self.environment(engine))

    def test_auto_screens_weights_without_calling_any_advisor(self):
        config = self.configuration()
        with patch("llmperf.advisor.LocalAdvisor.ask", side_effect=AssertionError("AUTO must not call LLM")), \
                patch("llmperf.workflow.preflight_candidates", return_value={"status": "unknown", "results": []}):
            result = run_workflow(config, stop_after="plan", probe=lambda _: self.environment())
        plan = json.loads(Path(result["plan"]).read_text())
        self.assertGreater(plan["metadata"]["weight_capacity"]["excluded_count"], 0)
        self.assertTrue(all(candidate["tp"] * candidate["pp"] >= 4 for candidate in plan["candidates"]))
        self.assertNotIn("capability_analysis", plan["metadata"])
        self.assertTrue((Path(result["result_dir"]) / "model-weights.json").is_file())

    def test_all_impossible_candidates_stop_before_llm_with_diagnostic_reports(self):
        config = self.configuration(size_mb=1000)
        with patch("llmperf.advisor.LocalAdvisor.ask", side_effect=AssertionError("LLM not needed for proven exclusions")):
            with self.assertRaisesRegex(ConfigError, "all candidates exceed proven weight capacity"):
                plan_tuning(config, probe=lambda _: self.environment(), calibrate=False)
        report, = (self.root / "results").glob("*/weight-capacity.json")
        weights = report.with_name("model-weights.json")
        self.assertTrue(weights.is_file())
        capacity = json.loads(report.read_text())
        self.assertEqual(capacity["excluded_count"], capacity["input_candidates"])
        self.assertEqual(capacity["retained_count"], 0)

    def test_old_plan_without_weight_snapshot_preserves_original_model_identity(self):
        config = self.configuration()
        with patch("llmperf.workflow.preflight_candidates", return_value={"status": "unknown", "results": []}):
            result = run_workflow(config, stop_after="plan", probe=lambda _: self.environment())
        path = Path(result["plan"])
        plan = json.loads(path.read_text())
        metadata = plan["metadata"]
        metadata["model_snapshot"]["raw"].pop("weights")
        metadata["model_fingerprint"] = fingerprint(metadata["model_snapshot"])
        metadata.pop("weight_capacity")
        for name in ("model-weights.json", "weight-capacity.json"):
            metadata["capability_artifacts"].pop(name)
        path.write_text(json.dumps(plan))
        with patch("llmperf.workflow.execute_search", return_value={"status": "PASS"}) as executor:
            report = run_saved_plan(path, probe=lambda _: self.environment())
        self.assertEqual(report["status"], "PASS")
        executor.assert_called_once()


if __name__ == "__main__":
    unittest.main()
