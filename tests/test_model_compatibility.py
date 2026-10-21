import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from llmperf.discovery import discover_model
from llmperf.errors import ConfigError
from llmperf.model_protocols import parser_candidates


FIXTURES = Path(__file__).parent / "fixtures" / "models"


class ModelMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def discover(self, metadata, overrides=None):
        (self.root / "config.json").write_text(json.dumps(metadata))
        return discover_model(self.root, overrides)

    def test_unknown_architecture_is_not_rejected_or_given_thinking_defaults(self):
        model = self.discover({"model_type": "future_llm", "architectures": ["FutureForCausalLM"]})
        self.assertEqual(model.raw["family"], "generic")
        self.assertIsNone(model.raw["num_attention_heads"])
        self.assertIsNone(model.raw["is_moe"])
        self.assertIsNone(model.tool_call_parser)
        self.assertEqual(model.chat_template_kwargs, {})
        self.assertFalse(model.raw["protocol"]["runtime_verified"])
        self.assertTrue(model.raw["normalized"]["unknowns"])

    def test_nested_facts_have_correct_provenance_and_overrides(self):
        metadata = {"model_type": "future", "num_hidden_layers": 2,
                    "text_config": {"num_hidden_layers": 40, "num_attention_heads": 64,
                                    "num_experts": 8},
                    "vision_config": {"num_hidden_layers": 32, "num_attention_heads": 16}}
        model = self.discover(metadata, {"num_attention_heads": 32})
        self.assertEqual(model.raw["num_hidden_layers"], 40)
        self.assertEqual(model.raw["num_attention_heads"], 32)
        self.assertTrue(model.raw["is_moe"])
        self.assertTrue(model.raw["normalized"]["has_vision"])
        self.assertEqual(model.raw["provenance"]["num_hidden_layers"],
                         "config.json:text_config.num_hidden_layers")
        self.assertEqual(model.raw["provenance"]["num_attention_heads"],
                         "model_overrides.num_attention_heads")
        self.assertEqual(model.raw["metadata"], metadata)

    def test_nested_quantization_and_dtype_are_preserved(self):
        model = self.discover({"model_type": "future", "text_config": {
            "dtype": "bfloat16", "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4"}}})
        self.assertEqual(model.quantization, "fp8")
        self.assertEqual(model.raw["provenance"]["quantization"],
                         "config.json:text_config.quantization_config.quant_method")
        normalized = model.raw["normalized"]
        self.assertEqual(normalized["text"]["dtype"], "bfloat16")
        self.assertEqual(normalized["quantization"]["expert_dtype"], "fp4")
        self.assertIsNone(normalized["quantization"]["expert_layout"])

    def test_invalid_declared_structure_is_not_treated_as_unknown(self):
        for value in (None, True, 0, -2, "64"):
            with self.subTest(value=value), self.assertRaisesRegex(ConfigError, "num_attention_heads"):
                self.discover({"model_type": "future", "num_attention_heads": value})
        with self.assertRaisesRegex(ConfigError, "vision_config"):
            self.discover({"model_type": "future", "vision_config": []})

    def test_checkpoint_python_is_never_imported(self):
        (self.root / "configuration_future.py").write_text("raise RuntimeError('must not execute')")
        model = self.discover({"model_type": "future", "auto_map": {
            "AutoConfig": "configuration_future.FutureConfig"}})
        self.assertEqual(model.raw["family"], "generic")

    def test_parser_names_are_engine_specific(self):
        self.assertEqual(parser_candidates("deepseekv41", "sglang", "tool_call_parser"), ("deepseekv41",))
        self.assertEqual(parser_candidates("deepseekv41", "vllm", "tool_call_parser"), ("deepseek_v41",))
        self.assertEqual(parser_candidates("deepseek-v41", "vllm", "reasoning_parser"), ("deepseek_v41",))
        self.assertNotIn("deepseek_v4", parser_candidates("deepseekv41", "vllm", "tool_call_parser"))


class OfficialCheckpointTests(unittest.TestCase):
    def test_official_snapshots_are_pinned_and_unchanged(self):
        directories = sorted(path for path in FIXTURES.iterdir() if path.is_dir())
        self.assertEqual(len(directories), 5)
        for directory in directories:
            with self.subTest(model=directory.name):
                source = json.loads((directory / "source.json").read_text())
                self.assertEqual(len(source["sha"]), 40)
                self.assertEqual(hashlib.sha256((directory / "config.json").read_bytes()).hexdigest(),
                                 source["config_sha256"])

    def test_flash_models_are_discovered_without_tokenizer_or_gpu(self):
        expected = {
            "DeepSeek-V4.1-Flash-1017": ("deepseek-v41", 40, 64),
            "Qwen3.8-Flash-Next": ("qwen4-exp", 48, 24),
            "Qwen3.8-Flash-Next-FP8": ("qwen4-exp", 48, 24),
            "GLM-5.3-Flash": ("glm5-next", 45, 64),
            "GLM-5.3-Flash-BF16": ("glm5-next", 45, 64),
        }
        for name, (family, layers, heads) in expected.items():
            with self.subTest(model=name):
                model = discover_model(FIXTURES / name)
                self.assertEqual(model.raw["family"], family)
                self.assertEqual(model.raw["num_hidden_layers"], layers)
                self.assertEqual(model.raw["num_attention_heads"], heads)
                self.assertTrue(model.raw["is_moe"])
                self.assertFalse(model.raw["protocol"]["runtime_verified"])

    def test_v41_has_its_own_protocol_and_mixed_quantization(self):
        model = discover_model(FIXTURES / "DeepSeek-V4.1-Flash-1017")
        self.assertEqual(model.tool_call_parser, "deepseekv41")
        self.assertEqual(model.reasoning_parser, "deepseek-v41")
        self.assertFalse(model.raw["protocol"]["static_template_present"])
        self.assertEqual(model.raw["protocol"]["encoding"], "engine_native")
        quantization = model.raw["normalized"]["quantization"]
        self.assertEqual(quantization["expert_dtype"], "fp4")
        self.assertEqual(quantization["components"]["model"]["weight_block_size"], [32, 32])
        self.assertIsNone(quantization["expert_layout"])

    def test_bf16_and_fp8_variants_keep_their_real_quantization(self):
        for name in ("Qwen3.8-Flash-Next", "GLM-5.3-Flash-BF16"):
            self.assertIsNone(discover_model(FIXTURES / name).quantization)
        for name in ("Qwen3.8-Flash-Next-FP8", "GLM-5.3-Flash"):
            self.assertEqual(discover_model(FIXTURES / name).quantization, "fp8")
