"""Evidence-backed native feature contracts, independent of engine versions."""
from __future__ import annotations

import copy
import json
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

PLANNER_PARAMETERS = {
    "vllm": {
        "tensor_parallel_size": ("tp", "integer"),
        "data_parallel_size": ("dp", "integer"),
        "pipeline_parallel_size": ("pp", "integer"),
    },
    "sglang": {
        "tp_size": ("tp", "integer"), "dp_size": ("dp", "integer"),
        "dp": ("dp", "integer"), "pp_size": ("pp", "integer"),
        "attn_dp_size": ("attn_dp_size", "integer"),
        "enable_dp_attention": ("dp_attention", "boolean"),
    },
}


def planner_field(name, catalog):
    entry = PLANNER_PARAMETERS.get(catalog["engine"], {}).get(name)
    return entry[0] if entry else None


def engine_catalog(engine, capabilities):
    schema = capabilities.get("schema", {})
    parameters = copy.deepcopy(schema.get("parameters", {}))
    for flag in capabilities.get("options", []):
        name = flag.removeprefix("--").replace("-", "_")
        parameters.setdefault(name, {"flags": [flag], "type": "unknown", "choices": None,
                                     "source": "serve-help", "status": "discovered"})
    for name, item in parameters.items():
        mapped = PLANNER_PARAMETERS.get(engine, {}).get(name)
        integrated = name in FEATURES.get(engine, {}) or mapped is not None
        item.update(id="parameter:" + name, integrated=integrated,
                    runtime_verified=False)
        if integrated and item.get("type", "unknown") == "unknown":
            item["type"] = mapped[1] if mapped else FEATURES[engine][name]
        if mapped:
            item.update(integration="existing_planner", planner_field=mapped[0],
                        semantics="select an existing legal topology; never resize an allocated candidate")
        if name == "speculative_config" and engine == "vllm":
            item["type"] = "object"
    return {"version": 1, "engine": engine, "parameters": parameters,
            "nested": copy.deepcopy(schema.get("nested", {})),
            "documents": copy.deepcopy(schema.get("documents", {})),
            "errors": list(schema.get("errors", [])),
            "semantics": "discovered is not runtime verified; unintegrated is not unsupported"}


def proposal_matches(static, params, catalog):
    return all(static.get(planner_field(name, catalog) or name,
                          1 if planner_field(name, catalog) == "attn_dp_size" else None) == value
               for name, value in params.items())


def missing_native_fields(static, catalog):
    """Controller requirements backed by fields discovered in this image.

    DSPARK experiments must state their draft length when the image provides no
    usable default. Other methods may infer it from model metadata, so do not
    impose this condition on every speculative method.
    """
    spec = static.get("speculative_config")
    fields = catalog.get("nested", {}).get("speculative_config", {})
    if not isinstance(spec, dict):
        return []
    missing = [name for name, item in fields.items() if item.get("required") and name not in spec
               and item.get("type") in {"integer", "number", "boolean", "string"}
               and not re.search(r"model|path|revision|quantization|tokenizer|device|cache", name)]
    count = fields.get("num_speculative_tokens")
    if str(spec.get("method", "")).lower() == "dspark" and count is not None:
        default = count.get("default")
        if ("num_speculative_tokens" not in spec and not (type(default) is int and default > 0)
                and "num_speculative_tokens" not in missing):
            missing.append("num_speculative_tokens")
    return ["speculative_config." + name for name in missing]


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
    if type(value) in (int, float) and (not math.isfinite(value) or value <= 0):
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
        missing = missing_native_fields({name: value}, catalog)
        if missing:
            raise ValueError("missing required native fields: " + ", ".join(missing))
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
            if static[name] is False and name not in catalog["parameters"]:
                continue
            validate_value(name, static[name], catalog)
    world = int(static.get("world_size", 1))
    if "ep_size" in static and (static["ep_size"] > world or world % static["ep_size"]):
        raise ValueError("ep_size must divide the candidate's allocated device count")
    if static.get("max_num_batched_tokens", 1) < static.get("max_num_seqs", 1):
        raise ValueError("max_num_batched_tokens must be >= max_num_seqs")


def native_summary(static):
    engine = static.get("engine", "sglang")
    return {key: copy.deepcopy(static[key]) for key in FEATURES.get(engine, {}) if key in static}


def adapt_native_static(static, catalog):
    static = copy.deepcopy(static)
    if catalog is None:
        return static
    validate_combination(static, catalog)
    if catalog["engine"] == "sglang":
        if "speculative_algorithm" in static:
            static["dspark"] = static["speculative_algorithm"].upper() == "DSPARK"
        if "moe_runner_backend" in static:
            static["backend"] = static["moe_runner_backend"]
    else:
        flags = {name: next((flag for flag in item["flags"]
            if flag.startswith("--no-")), None) for name, item in catalog["parameters"].items()
            if item.get("type") == "boolean" and any(flag.startswith("--no-") for flag in item["flags"])}
        if flags:
            static["native_boolean_flags"] = flags
    return static


def native_contract(static, catalog):
    requested = {}
    removed = {}
    append = []
    parameters = catalog["parameters"]
    for name in FEATURES.get(catalog["engine"], {}):
        if name not in static or static[name] is None or name not in parameters:
            continue
        value = static[name]
        validate_value(name, value, catalog)
        item = parameters[name]
        flags = item["flags"]
        boolean = FEATURES[catalog["engine"]][name] == "boolean"
        removed.update({flag: not boolean for flag in flags})
        if boolean:
            if value:
                append.append(next(flag for flag in flags if not flag.startswith("--no-")))
            elif item.get("default") is True:
                negative = next((flag for flag in flags if flag.startswith("--no-")), None)
                if negative is None:
                    raise ValueError(f"{name}: no safe native disable flag discovered")
                append.append(negative)
        else:
            append.extend((flags[0], json.dumps(value, sort_keys=True) if isinstance(value, dict) else str(value)))
        requested[name] = copy.deepcopy(value)
    drop = []
    if catalog["engine"] == "sglang":
        if "speculative_algorithm" in requested:
            removed.update({"--speculative-algorithm": True, "--speculative-dspark-block-size": True})
            drop.append("dspark")
        if "attn_dp_size" in parameters:
            removed["--enable-dp-attention"] = False
            append.extend((parameters["attn_dp_size"]["flags"][0], str(static.get("attn_dp_size", 1))))
            requested["attn_dp_size"] = static.get("attn_dp_size", 1)
            drop.append("dp_attention")
        if "enable_dp_lm_head" not in parameters:
            removed["--enable-dp-lm-head"] = False
            drop.append("dp_lm_head")
        for name, legacy in (("dp_size", "--dp"), ("tp_size", "--tp-size"), ("pp_size", "--pp-size")):
            item = parameters.get(name)
            if item and legacy not in item["flags"]:
                removed[legacy] = True
                append.extend((item["flags"][0], str(static.get({"dp_size": "dp", "tp_size": "tp", "pp_size": "pp"}[name], 1))))
    return {"append": append, "remove_flags": removed, "requested": requested, "drop_requested": drop}
