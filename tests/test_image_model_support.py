import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmarks.server.model_capabilities import collect_model_support
from llmperf.capabilities import EnvironmentSnapshot, _container_probe, probe_environment
from llmperf.discovery import discover_model
from llmperf.errors import ConfigError
from llmperf.model_compatibility import assess_model_compatibility, require_model_compatibility
from llmperf.workflow import run_workflow
import test_capabilities as capability_fixtures


FIXTURES = Path(__file__).parent / "fixtures" / "models"


def environment(engine="sglang", architectures=None, parser_choices=True):
    tool_parser = "deepseekv41" if engine == "sglang" else "deepseek_v41"
    reasoning_parser = "deepseek-v41" if engine == "sglang" else "deepseek_v41"
    return EnvironmentSnapshot("PASS", image={"image_id": "sha256:test"}, container={
        "model_support": {"registry": {"status": "discovered", "architectures":
                          architectures if architectures is not None else ["DeepseekV41ForCausalLM"]},
                          "config": {"status": "accepted", "weights_loaded": False}},
        "capabilities": {"tool_parsers": [tool_parser] if parser_choices else ["deepseekv4"],
                         "reasoning_parsers": [reasoning_parser] if parser_choices else ["deepseek-v4"]}})


class ImageModelSupportTests(unittest.TestCase):
    def test_environment_mounts_only_config_and_merges_native_parser_choices(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = capability_fixtures.CapabilityTests().config(temporary)
            (config.paths.model_host / "config.json").write_text("{}")
            output = capability_fixtures.probe_output()
            lines = output.splitlines()
            payload = json.loads(lines[0].split("=", 1)[1])
            payload["model_support"] = {"parsers": {"tool_parsers": {
                "status": "discovered", "choices": ["deepseekv41"]}}}
            lines[0] = "S1SLOW_PROBE_JSON=" + json.dumps(payload)
            runner = capability_fixtures.FakeRunner(config, probe="\n".join(lines) + "\n")
            snapshot = probe_environment(config, runner)
            command = next(call for call in runner.calls if call[:3] == ["docker", "run", "--rm"])
            self.assertEqual(command[command.index("-v") + 1],
                             str(config.paths.model_host / "config.json") + ":/tmp/llmperf-model/config.json:ro")
            self.assertIn("HF_HUB_OFFLINE=1", command)
            self.assertIn("deepseekv41", snapshot.container["capabilities"]["tool_parsers"])

    def test_config_probe_only_uses_installed_engine_code(self):
        getter = Mock(return_value=SimpleNamespace(model_type="deepseek_v41", architectures=["DeepseekV41ForCausalLM"]))
        modules = {
            "sglang.srt.plugins": SimpleNamespace(load_plugins=Mock()),
            "sglang.srt.models.registry": SimpleNamespace(ModelRegistry=SimpleNamespace(
                get_supported_archs=lambda: {"DeepseekV41ForCausalLM"})),
            "sglang.srt.hf_transformers_utils": SimpleNamespace(get_config=getter),
            "sglang.srt.function_call.function_call_parser": SimpleNamespace(FunctionCallParser=SimpleNamespace(
                ToolCallParserEnum={"deepseekv41": object})),
            "sglang.srt.parser.reasoning_parser": SimpleNamespace(ReasoningParser=SimpleNamespace(
                DetectorMap={"deepseek-v41": object})),
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
                "benchmarks.server.model_capabilities.importlib.import_module", side_effect=modules.__getitem__):
            (Path(temporary) / "config.json").write_text("{}")
            report = collect_model_support("sglang", temporary)
            getter.assert_called_once_with(temporary, trust_remote_code=False)
        self.assertEqual(report["registry"]["status"], "discovered")
        self.assertEqual(report["config"]["status"], "accepted")
        self.assertEqual(report["parsers"]["reasoning_parsers"]["choices"], ["deepseek-v41"])
        self.assertFalse(report["runtime_verified"])
        self.assertFalse(report["config"]["weights_loaded"])

    def test_unavailable_registry_or_config_stays_unknown(self):
        with patch("benchmarks.server.model_capabilities.importlib.import_module", side_effect=ImportError("missing")):
            report = collect_model_support("sglang")
        self.assertEqual(report["registry"]["status"], "unknown")
        self.assertEqual(report["config"]["status"], "unknown")
        self.assertFalse(report["runtime_verified"])

    def test_vllm_lazy_parser_names_are_discovered(self):
        modules = {
            "vllm.plugins": SimpleNamespace(load_general_plugins=lambda: None),
            "vllm.model_executor.models.registry": SimpleNamespace(ModelRegistry=SimpleNamespace(
                get_supported_archs=lambda: {"DeepseekV41ForCausalLM"})),
            "vllm.tool_parsers": SimpleNamespace(ToolParserManager=SimpleNamespace(
                tool_parsers={}, lazy_parsers={"deepseek_v41": ("module", "Parser")})),
            "vllm.reasoning": SimpleNamespace(ReasoningParserManager=SimpleNamespace(
                reasoning_parsers={}, lazy_parsers={"deepseek_v41": ("module", "Parser")})),
        }
        with patch("benchmarks.server.model_capabilities.importlib.import_module", side_effect=modules.__getitem__):
            report = collect_model_support("vllm")
        self.assertEqual(report["parsers"]["tool_parsers"]["choices"], ["deepseek_v41"])
        self.assertEqual(report["parsers"]["reasoning_parsers"]["choices"], ["deepseek_v41"])

    def test_embedded_probe_is_valid_python_for_both_engines(self):
        for engine in ("sglang", "vllm"):
            source = _container_probe(engine, model_config=True)
            compile(source, "model-probe", "exec")
            self.assertIn("collect_model_support", source)
            self.assertIn("/tmp/llmperf-model", source)

    def test_new_parser_names_are_bound_without_v4_fallback(self):
        model = discover_model(FIXTURES / "DeepSeek-V4.1-Flash-1017")
        for engine in ("sglang", "vllm"):
            report = assess_model_compatibility(model, environment(engine), engine)
            require_model_compatibility(report)
            self.assertEqual(report["checks"]["architecture"]["status"], "SUPPORTED")
            self.assertEqual(report["checks"]["config"]["status"], "SUPPORTED")
            self.assertEqual(report["checks"]["runtime"]["status"], "NOT_RUN")
            self.assertFalse(report["runtime_verified"])
            with self.assertRaisesRegex(ConfigError, "parser"):
                require_model_compatibility(assess_model_compatibility(model, environment(engine, parser_choices=False), engine))

    def test_absent_native_architecture_is_not_hidden_by_metadata_overrides(self):
        model = discover_model(FIXTURES / "DeepSeek-V4.1-Flash-1017", {
            "model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"]})
        report = assess_model_compatibility(model, environment(architectures=["DeepseekV4ForCausalLM"]), "sglang")
        self.assertEqual(report["checks"]["architecture"]["status"], "UNSUPPORTED")

    def test_transformers_fallback_is_unknown_not_verified(self):
        model = discover_model(FIXTURES / "DeepSeek-V4.1-Flash-1017")
        report = assess_model_compatibility(model, environment(architectures=["TransformersForCausalLM"]), "sglang")
        self.assertEqual(report["checks"]["architecture"]["status"], "UNKNOWN")
        self.assertFalse(report["runtime_verified"])

    def test_native_config_architecture_conversion_is_respected(self):
        model = discover_model(FIXTURES / "DeepSeek-V4.1-Flash-1017")
        snapshot = environment(architectures=["InstalledNativeArchitecture"])
        snapshot.container["model_support"]["config"]["architectures"] = ["InstalledNativeArchitecture"]
        report = assess_model_compatibility(model, snapshot, "sglang")
        self.assertEqual(report["checks"]["architecture"]["status"], "SUPPORTED")

    def test_unavailable_parser_registry_does_not_claim_incompatibility(self):
        model = discover_model(FIXTURES / "DeepSeek-V4.1-Flash-1017")
        snapshot = environment()
        capabilities = snapshot.container["capabilities"]
        capabilities.update(tool_parsers=[], reasoning_parsers=[], parser_registry={
            "tool_parsers": {"status": "unknown"}, "reasoning_parsers": {"status": "unknown"}})
        report = assess_model_compatibility(model, snapshot, "sglang")
        require_model_compatibility(report)
        self.assertEqual(report["checks"]["tool_call_parser"]["status"], "UNKNOWN")
        self.assertEqual(report["parsers"]["tool_call_parser"], "deepseekv41")

    def test_doctor_writes_actionable_report_before_rejecting_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workload = root / "input.jsonl"
            workload.write_text(json.dumps({"api": "openai_chat_completions", "request_id": "one",
                "source_message_id": 1, "captured_at": "2026-10-21T00:00:00Z", "request": {
                    "model": "captured", "messages": [{"role": "user", "content": "hello"}]}}) + "\n")
            config = root / "experiment.json"
            config.write_text(json.dumps({"model_path": str(FIXTURES / "DeepSeek-V4.1-Flash-1017"),
                "input_path": str(workload), "output_dir": str(root / "results"),
                "image": "test/image@sha256:" + "a" * 64}))
            with self.assertRaisesRegex(ConfigError, "compatibility failed"):
                run_workflow(config, stop_after="doctor", probe=lambda _: environment(architectures=["OldModel"]))
            report_path = next((root / "results").glob("*/model-compatibility.json"))
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["checks"]["runtime"]["status"], "NOT_RUN")
