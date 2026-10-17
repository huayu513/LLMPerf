"""Evidence-backed native feature contracts, independent of engine versions."""
from __future__ import annotations

import copy
import math
import re


FEATURES = {
    "vllm": {
        "enable_expert_parallel": "boolean", "all2all_backend": "string",
        "moe_backend": "string", "speculative_config": "object",
    },
    "sglang": {
        "speculative_algorithm": "string", "speculative_num_draft_tokens": "integer",
        "speculative_dspark_block_size": "integer", "speculative_num_steps": "integer",
        "speculative_eagle_topk": "integer", "ep_size": "integer",
        "moe_runner_backend": "string", "moe_a2a_backend": "string",
    },
}


def engine_catalog(engine, capabilities):
    schema = capabilities.get("schema", {})
    parameters = copy.deepcopy(schema.get("parameters", {}))
    for flag in capabilities.get("options", []):
        name = flag.removeprefix("--").replace("-", "_")
        parameters.setdefault(name, {"flags": [flag], "type": "unknown", "choices": None,
                                     "source": "serve-help", "status": "discovered"})
    for name, item in parameters.items():
        item.update(id="parameter:" + name, integrated=name in FEATURES.get(engine, {}),
                    runtime_verified=False)
        if name in FEATURES.get(engine, {}) and item.get("type") == "unknown":
            item["type"] = FEATURES[engine][name]
    return {"version": 1, "engine": engine, "parameters": parameters,
            "nested": copy.deepcopy(schema.get("nested", {})),
            "documents": copy.deepcopy(schema.get("documents", {})),
            "errors": list(schema.get("errors", [])),
            "semantics": "discovered is not runtime verified; unintegrated is not unsupported"}


def validate_value(name, value, catalog):
    parameter = catalog["parameters"].get(name)
    if parameter is None:
        raise ValueError(f"{name}: not discovered in this image")
    if not parameter.get("integrated"):
        raise ValueError(f"{name}: discovered but launch/evidence adapter is not integrated")
    if value is None and name == "speculative_config":
        return
    datatype = parameter.get("type")
    expected = {"boolean": bool, "integer": int, "number": (int, float), "string": str, "object": dict}
    if datatype not in expected or not isinstance(value, expected[datatype]) or (
            isinstance(value, bool) and datatype != "boolean"):
        raise ValueError(f"{name}: invalid {datatype} value")
    if isinstance(value, (int, float)) and (not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name}: requires a positive finite value")
    if isinstance(value, str) and (not value or len(value) > 200 or not re.fullmatch(r"[A-Za-z0-9_.:+-]+", value)):
        raise ValueError(f"{name}: invalid native identifier")
    choices = parameter.get("choices")
    if choices and value not in choices:
        raise ValueError(f"{name}: value not advertised by image")
    if name == "speculative_config":
        fields = catalog.get("nested", {}).get(name, {})
        if not fields:
            raise ValueError("speculative_config: nested schema unknown; request additional evidence")
        allowed = {key: item for key, item in fields.items()
                   if item.get("type") in {"integer", "number", "boolean", "string"} or item.get("choices")}
        if not value or "method" not in value or set(value) - set(allowed):
            raise ValueError("speculative_config: undiscovered or non-scalar fields")
        for key, child in value.items():
            if re.search(r"model|path|revision|quantization|tokenizer|device|cache", key):
                raise ValueError(f"speculative_config.{key}: changes a protected test condition")
            field = allowed[key]
            if field.get("choices") and child not in field["choices"]:
                raise ValueError(f"speculative_config.{key}: value not advertised by image")
            if type(child) not in (int, float, bool, str) or (
                    type(child) in (int, float) and (not math.isfinite(child) or child <= 0)):
                raise ValueError(f"speculative_config.{key}: invalid scalar")
            if isinstance(child, str) and not re.fullmatch(r"[A-Za-z0-9_.:+-]{1,200}", child):
                raise ValueError(f"speculative_config.{key}: invalid identifier")
            declared = field.get("type")
            if declared in expected and (not isinstance(child, expected[declared]) or
                                         isinstance(child, bool) and declared != "boolean"):
                raise ValueError(f"speculative_config.{key}: incorrect type")


def validate_combination(static, catalog):
    for name in FEATURES.get(catalog["engine"], {}):
        if name in static and static[name] is not None:
            validate_value(name, static[name], catalog)
    world = int(static.get("world_size", 1))
    if "ep_size" in static and (static["ep_size"] > world or world % static["ep_size"]):
        raise ValueError("ep_size must divide the candidate's allocated device count")
    if static.get("max_num_batched_tokens", 1) < static.get("max_num_seqs", 1):
        raise ValueError("max_num_batched_tokens must be >= max_num_seqs")


def native_summary(static):
    engine = static.get("engine", "sglang")
    return {key: copy.deepcopy(static[key]) for key in FEATURES.get(engine, {}) if key in static}
