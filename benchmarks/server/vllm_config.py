"""vLLM launch arguments and independently captured runtime configuration."""
from __future__ import annotations

import json


NATIVE_OPTIONS = {
    "gpu_memory_utilization", "max_num_seqs", "max_num_batched_tokens",
    "max_model_len", "enforce_eager", "enable_expert_parallel",
    "all2all_backend", "moe_backend", "speculative_config",
}
ENGINE_FIELDS = {
    "model_path": ("model_config", "model"),
    "tp": ("parallel_config", "tensor_parallel_size"),
    "dp": ("parallel_config", "data_parallel_size"),
    "pp": ("parallel_config", "pipeline_parallel_size"),
    "gpu_memory_utilization": ("cache_config", "gpu_memory_utilization"),
    "max_num_seqs": ("scheduler_config", "max_num_seqs"),
    "max_num_batched_tokens": ("scheduler_config", "max_num_batched_tokens"),
    "max_model_len": ("model_config", "max_model_len"),
    "enforce_eager": ("model_config", "enforce_eager"),
    "enable_expert_parallel": ("parallel_config", "enable_expert_parallel"),
    "all2all_backend": ("parallel_config", "all2all_backend"),
    "moe_backend": ("kernel_config", "moe_backend"),
    "quantization": ("model_config", "quantization"),
}


def normalized_speculation(value):
    if isinstance(value, dict) and value.get("method") == "deepseek_mtp":
        return {**value, "method": "mtp"}
    return value


def launch_config(metadata: dict, static: dict, max_num_seqs: int) -> dict:
    config = {key: value for key, value in static.items() if key in NATIVE_OPTIONS}
    config.update({
        "model_path": "/model", "served_model_name": metadata["served_model_name"],
        "tp": static.get("tp", 1), "dp": static.get("dp", 1), "pp": static.get("pp", 1),
        "max_num_seqs": max_num_seqs,
        "chat_template_kwargs": metadata.get("chat_template_kwargs", {}),
    })
    if static.get("native_boolean_flags"):
        config["native_boolean_flags"] = static["native_boolean_flags"]
    for key in ("tool_call_parser", "reasoning_parser", "quantization"):
        if metadata.get(key):
            config[key] = metadata[key]
    return config


def requested_parameters(config: dict) -> dict:
    keys = set(ENGINE_FIELDS) | {"served_model_name", "speculative_config"}
    return {key: value for key, value in config.items() if key in keys}


def command_args(config: dict, port: int, executable: str = "vllm") -> list[str]:
    command = [executable, "serve", config["model_path"],
               "--served-model-name", config["served_model_name"],
               "--host", "127.0.0.1", "--port", str(port), "--trust-remote-code"]
    names = {"tp": "tensor-parallel-size", "dp": "data-parallel-size",
             "pp": "pipeline-parallel-size", "chat_template_kwargs": "default-chat-template-kwargs"}
    for key in ("tp", "dp", "pp", *sorted(NATIVE_OPTIONS), "tool_call_parser",
                "reasoning_parser", "quantization", "chat_template_kwargs"):
        value = config.get(key)
        if value is None:
            continue
        flag = "--" + names.get(key, key.replace("_", "-"))
        if isinstance(value, bool):
            if value:
                command.append(flag)
            elif config.get("native_boolean_flags", {}).get(key):
                command.append(config["native_boolean_flags"][key])
        else:
            command += [flag, json.dumps(value, sort_keys=True) if isinstance(value, dict) else str(value)]
    if config.get("tool_call_parser"):
        command.append("--enable-auto-tool-choice")
    return command


def server_parameters(info: dict) -> dict:
    if not isinstance(info, dict):
        return {}
    config = info.get("vllm_config")
    if not isinstance(config, dict):
        return {}
    values = {}
    for key, (section, name) in ENGINE_FIELDS.items():
        source = config.get(section)
        if isinstance(source, dict) and name in source:
            values[key] = source[name]
    models = info.get("models")
    if isinstance(models, dict) and isinstance(models.get("data"), list):
        aliases = [item.get("id") for item in models["data"] if isinstance(item, dict)]
        if len(aliases) == 1:
            values["served_model_name"] = aliases[0]
    if "speculative_config" in config:
        spec = config["speculative_config"]
        if isinstance(spec, dict):
            values["speculative_config"] = {key: value for key, value in spec.items()
                                            if value is None or type(value) in (str, int, float, bool)}
        else:
            values["speculative_config"] = spec
    return values
