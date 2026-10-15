"""Engine-specific candidate capabilities without changing search scheduling."""
from __future__ import annotations

from .errors import ConfigError


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
    if native["max_num_batched_tokens"] < native["max_num_seqs"]:
        raise ConfigError("vLLM max_num_batched_tokens must be >= max_num_seqs")
    required = {"--served-model-name", "--tensor-parallel-size", "--data-parallel-size",
                "--pipeline-parallel-size", "--trust-remote-code", "--host", "--port",
                "--default-chat-template-kwargs"}
    required.update("--" + key.replace("_", "-") for key, value in native.items()
                    if value is not False and value is not None)
    parsers = {}
    mapping = {"deepseekv4": "deepseek_v4", "deepseek-v4": "deepseek_v4",
               "deepseekv3": "deepseek_v3", "deepseek-v3": "deepseek_v3",
               "deepseekv31": "deepseek_v31", "deepseekv32": "deepseek_v32"}
    provenance = model.raw.get("provenance", {})
    for key, catalog in (("tool_call_parser", "tool_parsers"), ("reasoning_parser", "reasoning_parsers")):
        value = getattr(model, key)
        if provenance.get(key) != f"model_overrides.{key}":
            value = mapping.get(value, value)
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
    return native, [None if value == "auto" else value for value in backends], parsers
