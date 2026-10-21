"""Engine-specific candidate capabilities without changing search scheduling."""
from __future__ import annotations

from .errors import ConfigError
from .model_compatibility import resolve_parsers


def vllm_plan_settings(config, model, capabilities):
    options = set(capabilities.get("options", []))
    native = {
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": config.search.concurrency_max,
        "max_num_batched_tokens": 4096,
        "enforce_eager": False,
        "enable_expert_parallel": False,
        "speculative_config": None,
    }
    native.update({key: value for key, value in getattr(config, "engine_options", {}).items()
                   if key not in {"tensor_parallel_size", "data_parallel_size", "pipeline_parallel_size"}})
    discovered = capabilities.get("schema", {}).get("parameters", {})
    if "enable_expert_parallel" not in getattr(config, "engine_options", {}):
        default = discovered.get("enable_expert_parallel", {}).get("default")
        if type(default) is bool:
            native["enable_expert_parallel"] = default
    if native["max_num_batched_tokens"] < native["max_num_seqs"]:
        raise ConfigError("vLLM max_num_batched_tokens must be >= max_num_seqs")
    required = {"--served-model-name", "--tensor-parallel-size", "--data-parallel-size",
                "--pipeline-parallel-size", "--trust-remote-code", "--host", "--port",
                "--default-chat-template-kwargs"}
    required.update("--" + key.replace("_", "-") for key, value in native.items()
                    if value is not False and value is not None)
    parsers, parser_checks = resolve_parsers(model, "vllm", capabilities)
    provenance = model.raw.get("provenance", {})
    for key, catalog in (("tool_call_parser", "tool_parsers"), ("reasoning_parser", "reasoning_parsers")):
        value = parsers[key]
        if parser_checks[key]["status"] == "UNSUPPORTED":
            raise ConfigError(parser_checks[key]["reason"])
        if value:
            required.add("--" + key.replace("_", "-"))
            advertised = capabilities.get(catalog, [])
            if advertised and value not in advertised:
                raise ConfigError(f"configured vLLM image does not advertise {key} {value!r}; use model_overrides")
        parsers[key] = value
    if parsers["tool_call_parser"]:
        required.add("--enable-auto-tool-choice")
    if provenance.get("quantization") == "model_overrides.quantization":
        required.add("--quantization")
    requested = config.search.backends
    backend = native.pop("moe_backend", None)
    if requested and backend is not None:
        raise ConfigError("set either search.backends or engine_options.moe_backend, not both")
    backends = list(requested or ([backend] if backend else ["auto"]))
    if any(value != "auto" for value in backends):
        if not model.raw.get("is_moe"):
            raise ConfigError("vLLM MoE backend requires a MoE model")
        required.add("--moe-backend")
        advertised = capabilities.get("runner_backends", [])
        if advertised and any(value not in advertised for value in backends):
            raise ConfigError("vLLM backend not advertised by the configured image")
    missing = sorted(required - options)
    if missing:
        raise ConfigError("configured image lacks required vLLM launch options: " + ", ".join(missing))
    negative_flags = {name: next(flag for flag in item["flags"] if flag.startswith("--no-"))
                      for name, item in discovered.items() if item.get("type") == "boolean"
                      and any(flag.startswith("--no-") for flag in item["flags"])}
    if negative_flags:
        native["native_boolean_flags"] = negative_flags
    return native, [None if value == "auto" else value for value in backends], parsers
