"""Normalize static model facts without importing checkpoint code."""
from __future__ import annotations

import copy

from .errors import ConfigError


_TEXT_FIELDS = (
    "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "hidden_size", "head_dim",
    "n_routed_experts", "num_experts", "num_local_experts", "num_experts_per_tok",
    "linear_num_key_heads", "linear_num_value_heads", "linear_attn_config",
    "max_position_embeddings", "dtype", "torch_dtype", "num_nextn_predict_layers",
    "mtp_num_hidden_layers", "mtp", "dspark_block_size", "dspark_target_layer_ids",
    "engram_layer_ids", "engram_num_embeddings", "ple_layer_ids",
)


def normalize_metadata(metadata, overrides):
    text = metadata.get("text_config", {})
    vision = metadata.get("vision_config", {})
    if text is None:
        text = {}
    if vision is None:
        vision = {}
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise ConfigError("model metadata text_config and vision_config must be objects")
    structural = {**metadata, **text}
    values = {}
    provenance = {}
    unknowns = []
    for key in _TEXT_FIELDS:
        source = "config.json:text_config." if key in text else "config.json:"
        if key in overrides:
            value, source = overrides[key], "model_overrides."
        elif key in structural:
            value = structural[key]
        else:
            continue
        if key in {"num_attention_heads", "num_hidden_layers", "num_key_value_heads",
                   "linear_num_key_heads", "linear_num_value_heads"}:
            if type(value) is not int or value <= 0:
                raise ConfigError(f"model metadata {key} must be a positive integer")
        values[key] = copy.deepcopy(value)
        provenance[key] = source + key
    for key in ("num_attention_heads", "num_hidden_layers"):
        if key not in values:
            unknowns.append(key + "_not_declared; related topology constraints are unknown")
    quantizations = {}
    for component, document in (("model", metadata), ("text", text), ("vision", vision)):
        config = document.get("quantization_config")
        if config is not None:
            if not isinstance(config, dict):
                raise ConfigError(f"model metadata {component}.quantization_config must be an object")
            quantizations[component] = {key: copy.deepcopy(value) for key, value in config.items()
                                       if key not in {"modules_to_not_convert", "ignored_layers"}}
    effective_quantization = quantizations.get("text", quantizations.get("model", {}))
    expert_dtype = effective_quantization.get("expert_dtype")
    expert_layout = effective_quantization.get("routed_experts_quant_method")
    if expert_dtype and not expert_layout:
        unknowns.append("expert_storage_layout_requires_image_resolution")
    layer_types = structural.get("layer_types", [])
    if not isinstance(layer_types, list) or any(not isinstance(item, str) for item in layer_types):
        raise ConfigError("model metadata layer_types must be an array of strings")
    linear = any("linear" in item or "mamba" in item for item in layer_types)
    compressed = "compress_ratios" in structural
    attention = {"kind": "hybrid" if linear else "compressed" if compressed else "unknown",
                 "layer_types": copy.deepcopy(layer_types)}
    if compressed:
        attention["compress_ratios"] = copy.deepcopy(structural["compress_ratios"])
    return {"version": 1, "text": values, "has_vision": bool(vision),
            "text_model_type": text.get("model_type"), "attention": attention,
            "quantization": {"components": quantizations, "expert_dtype": expert_dtype,
                             "expert_layout": expert_layout,
                             "requires_native_resolution": bool(expert_dtype or expert_layout)},
            "provenance": provenance, "unknowns": unknowns}
