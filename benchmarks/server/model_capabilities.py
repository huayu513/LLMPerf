"""Inspect trusted, installed engine registries and a config-only mount."""
from __future__ import annotations

import contextlib
import importlib
import io
from pathlib import Path


def collect_model_support(engine, model_dir=None):
    result = {"version": 1, "engine": engine, "runtime_verified": False,
              "registry": {"status": "unknown", "architectures": []},
              "config": {"status": "unknown", "reason": "no config-only mount"},
              "parsers": {}, "errors": []}
    if engine not in {"sglang", "vllm"}:
        return result
    registry_module = ("sglang.srt.models.registry" if engine == "sglang"
                       else "vllm.model_executor.models.registry")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            plugins = importlib.import_module("sglang.srt.plugins" if engine == "sglang" else "vllm.plugins")
            loader = getattr(plugins, "load_plugins" if engine == "sglang" else "load_general_plugins", None)
            if callable(loader):
                loader()
        except ImportError:
            pass
        except Exception as exc:
            result["errors"].append("plugins: " + type(exc).__name__ + ": " + str(exc)[:500])
        try:
            registry = importlib.import_module(registry_module).ModelRegistry
            architectures = sorted(set(registry.get_supported_archs()))
            if not architectures or any(not isinstance(item, str) for item in architectures):
                raise ValueError("registry did not return architecture names")
            result["registry"] = {"status": "discovered", "architectures": architectures,
                                  "source": registry_module + ":ModelRegistry.get_supported_archs"}
        except Exception as exc:
            result["registry"]["reason"] = type(exc).__name__ + ": " + str(exc)[:500]
        if model_dir is not None and (Path(model_dir) / "config.json").is_file():
            try:
                config_module = ("sglang.srt.hf_transformers_utils" if engine == "sglang"
                                 else "vllm.transformers_utils.config")
                getter = importlib.import_module(config_module).get_config
                config = getter(str(model_dir), trust_remote_code=False)
                result["config"] = {"status": "accepted", "source": config_module + ":get_config",
                                    "model_type": getattr(config, "model_type", None),
                                    "architectures": list(getattr(config, "architectures", None) or []),
                                    "trust_remote_code": False, "weights_loaded": False}
            except Exception as exc:
                result["config"] = {"status": "unknown", "reason": type(exc).__name__ + ": " + str(exc)[:1000],
                                    "trust_remote_code": False, "weights_loaded": False}
        if engine == "sglang":
            parsers = (
                ("tool_parsers", "sglang.srt.function_call.function_call_parser", "FunctionCallParser", "ToolCallParserEnum"),
                ("reasoning_parsers", "sglang.srt.parser.reasoning_parser", "ReasoningParser", "DetectorMap"),
            )
        else:
            parsers = (
                ("tool_parsers", "vllm.tool_parsers", "ToolParserManager", "tool_parsers"),
                ("reasoning_parsers", "vllm.reasoning", "ReasoningParserManager", "reasoning_parsers"),
            )
        for name, module_name, class_name, attribute in parsers:
            try:
                manager = getattr(importlib.import_module(module_name), class_name)
                names = set(getattr(manager, attribute, {}))
                for extra in ("lazy_parsers", "_lazy_parsers"):
                    names.update(getattr(manager, extra, {}))
                if not names:
                    raise ValueError("parser registry is not enumerable")
                result["parsers"][name] = {"status": "discovered", "choices": sorted(names),
                                          "source": module_name + ":" + class_name}
            except Exception as exc:
                result["parsers"][name] = {"status": "unknown", "choices": [],
                                          "reason": type(exc).__name__ + ": " + str(exc)[:500]}
    return result
