import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from s1slow.Automation.automation.discovery import (
    discover_model,
    inspect_workload,
    prepare_workload,
)
from s1slow.Automation.automation.errors import ConfigError


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
            "reasoning_effort": "high",
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
        with self.assertRaisesRegex(ConfigError, "model_overrides"):
            discover_model(unknown)
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


if __name__ == "__main__":
    unittest.main()
