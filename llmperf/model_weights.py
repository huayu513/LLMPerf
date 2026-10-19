"""Read checkpoint metadata without importing a model or reading tensor payloads."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath


_MAX_HEADER_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_HEADER_BYTES = 256 * 1024 * 1024
_DTYPE_BITS = {
    "BOOL": 8, "U8": 8, "I8": 8, "I16": 16, "U16": 16,
    "I32": 32, "U32": 32, "I64": 64, "U64": 64,
    "F16": 16, "BF16": 16, "F32": 32, "F64": 64,
    "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8,
}


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON key: " + str(name)[:160])
        result[name] = value
    return result


def _json_file(path):
    if path.stat().st_size > _MAX_HEADER_BYTES:
        raise ValueError("checkpoint index exceeds metadata size limit")
    with path.open(encoding="utf-8") as source:
        return json.load(source, object_pairs_hook=_unique_object)


def _checkpoint_files(root):
    for name, suffix in (("model.safetensors.index.json", ".safetensors"),
                         ("pytorch_model.bin.index.json", ".bin")):
        index = root / name
        if not index.is_file():
            continue
        document = _json_file(index)
        mapping = document.get("weight_map") if isinstance(document, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("checkpoint index has no weight_map")
        files = {}
        for tensor, relative in mapping.items():
            if not isinstance(tensor, str) or not isinstance(relative, str):
                raise ValueError("checkpoint index has invalid tensor or shard name")
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative or path.suffix != suffix:
                raise ValueError("checkpoint index references an unsafe or unexpected shard")
            files.setdefault(relative, set()).add(tensor)
        size = document.get("metadata", {}).get("total_size")
        return files, name, size if type(size) is int and size >= 0 else None, True
    if (root / "model.safetensors").is_file():
        return {"model.safetensors": None}, None, None, True
    files = sorted(path.name for path in root.glob("*.safetensors")
                   if path.name != "adapter_model.safetensors")
    if files:
        shards = [re.fullmatch(r"model-(\d+)-of-(\d+)\.safetensors", name) for name in files]
        complete = (all(shards) and len({int(match[2]) for match in shards}) == 1
                    and {int(match[1]) for match in shards} == set(range(1, len(shards) + 1))
                    and int(shards[0][2]) == len(shards))
        return dict.fromkeys(files), None, None, bool(complete)
    files = {path.name: None for path in root.glob("*.bin")
             if path.name.startswith("pytorch_model")}
    return files, None, None, False


def _read_header(path, remaining):
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors length prefix")
        size = int.from_bytes(prefix, "little")
        if not 1 <= size <= min(_MAX_HEADER_BYTES, remaining):
            raise ValueError("safetensors header exceeds metadata size limit")
        if 8 + size > path.stat().st_size:
            raise ValueError("truncated safetensors header")
        data = source.read(size)
    header = json.loads(data, object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    intervals = []
    tensors = {}
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict):
            raise ValueError("invalid tensor descriptor")
        shape, dtype, offsets = tensor.get("shape"), tensor.get("dtype"), tensor.get("data_offsets")
        if (not isinstance(shape, list) or not all(type(dimension) is int and dimension >= 0 for dimension in shape)
                or not isinstance(dtype, str) or not isinstance(offsets, list) or len(offsets) != 2
                or not all(type(offset) is int for offset in offsets) or not 0 <= offsets[0] <= offsets[1]):
            raise ValueError("invalid tensor shape, dtype or offsets")
        elements = math.prod(shape)
        data_bytes = offsets[1] - offsets[0]
        if dtype in _DTYPE_BITS and (elements * _DTYPE_BITS[dtype] + 7) // 8 != data_bytes:
            raise ValueError("tensor shape does not match its data size")
        intervals.append(tuple(offsets))
        tensors[name] = {"dtype": dtype, "elements": elements, "bytes": data_bytes}
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("tensor offsets overlap or contain unaccounted data")
        cursor = end
    if cursor != path.stat().st_size - 8 - size:
        raise ValueError("tensor data is truncated or has unaccounted trailing bytes")
    return tensors, size, hashlib.sha256(data).hexdigest()


def _resident_parameter(name, metadata):
    if not name.endswith((".weight", ".bias")) or name.endswith(("lm_head.weight", "output.weight")):
        return False
    if re.search(r"(?:^|[._])(mtp|draft|speculative|nextn|eagle)(?:[._]|$)", name, re.I):
        return False
    layer = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    layers = metadata.get("num_hidden_layers")
    return not (layer and type(layers) is int and int(layer[1]) >= layers)


def inspect_model_weights(root, metadata, quantization=None):
    root = Path(root)
    report = {"version": 1, "status": "unknown", "format": "unknown", "file_count": 0,
              "tensor_count": 0, "stored_tensor_elements": None, "parameter_count": None,
              "parameter_count_kind": "unknown", "logical_parameter_count": None,
              "stored_weight_bytes": None, "weight_file_bytes": 0, "dtype_summary": {},
              "quantization": quantization, "minimum_resident_weight_bytes": None,
              "files": [], "unknowns": [], "errors": [],
              "semantics": "stored tensor elements are not unique logical parameters; disk bytes are not GPU usage"}
    try:
        files, index, declared, selected = _checkpoint_files(root)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        report["errors"].append(str(exc)[:500])
        return report
    report.update(file_count=len(files), index=index, index_declared_weight_bytes=declared)
    if not files:
        report["unknowns"].append("no_supported_checkpoint_found; GGUF, adapters and remote weights are not decoded")
        return report
    dtype_stats = {}
    names = set()
    remaining = _MAX_TOTAL_HEADER_BYTES
    payload_bytes = 0
    elements = 0
    minimum = 0
    all_floating = True
    for relative, expected in sorted(files.items()):
        path = root / relative
        try:
            size = path.stat().st_size
            report["weight_file_bytes"] += size
            if path.suffix != ".safetensors":
                report["unknowns"].append("binary_checkpoint_not_deserialized")
                continue
            tensors, header_bytes, digest = _read_header(path, remaining)
            remaining -= header_bytes
            if expected is not None:
                if expected - tensors.keys():
                    raise ValueError("indexed tensors missing from shard header")
                tensors = {name: tensors[name] for name in expected}
            if names & tensors.keys():
                raise ValueError("duplicate tensors across checkpoint shards")
            names.update(tensors)
            for name, tensor in tensors.items():
                dtype = tensor["dtype"]
                floating = dtype in _DTYPE_BITS and dtype.startswith(("F", "BF"))
                all_floating &= floating
                stats = dtype_stats.setdefault(dtype, {"tensor_count": 0, "elements": 0, "data_bytes": 0})
                stats["tensor_count"] += 1
                stats["elements"] += tensor["elements"]
                stats["data_bytes"] += tensor["bytes"]
                payload_bytes += tensor["bytes"]
                elements += tensor["elements"]
                if floating and _resident_parameter(name, metadata):
                    minimum += tensor["elements"] * min(2, _DTYPE_BITS[dtype] // 8)
            report["files"].append({"path": relative, "file_bytes": size, "header_bytes": header_bytes,
                                    "header_sha256": digest, "tensor_count": len(tensors)})
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            if len(report["errors"]) < 16:
                report["errors"].append(relative + ": " + str(exc)[:500])
    complete = selected and len(report["files"]) == len(files) and not report["errors"] and bool(names)
    report.update(format="safetensors" if report["files"] else "binary_or_unknown",
                  status="complete" if complete else "partial" if report["files"] else "unknown",
                  tensor_count=len(names), dtype_summary=dtype_stats)
    if report["files"]:
        report.update(stored_tensor_elements=elements, stored_weight_bytes=payload_bytes)
    if not selected:
        report["unknowns"].append("checkpoint_selection_not_confirmed_by_index_or_canonical_filename")
    structural = {**metadata, **(metadata.get("text_config") or {})}
    quantized = bool(quantization) or any(document.get(name) is not None
        for document in (metadata, structural) for name in ("quantization_config", "compression_config"))
    report["quantization_declared"] = quantized
    declared_dtype = structural.get("dtype", structural.get("torch_dtype"))
    report["model_dtype"] = declared_dtype
    standard_dtype = declared_dtype is None or str(declared_dtype).removeprefix("torch.") in {
        "auto", "float16", "bfloat16", "float32", "float64", "half", "float", "double"}
    if complete and all_floating and not quantized:
        report.update(parameter_count=elements, parameter_count_kind="unquantized_checkpoint_tensor_elements")
        if minimum and not metadata.get("text_config") and standard_dtype:
            report["minimum_resident_weight_bytes"] = minimum
            report["minimum_resident_basis"] = "unquantized floating weight/bias tensors at <=16 bits; omit output heads and draft layers"
        else:
            report["unknowns"].append("GPU_residency_not_proven_for_partial_model_loading_or_nonstandard_dtype")
    else:
        report["unknowns"].append("logical_parameter_count_and_GPU_residency_not_proven; packed tensors, scales or incomplete evidence")
    report["unknowns"] = list(dict.fromkeys(report["unknowns"]))
    return report


def weight_summary(report):
    return {name: value for name, value in report.items() if name != "files"}


def screen_weight_capacity(candidates, model, environment):
    weights = model.raw.get("weights", {})
    minimum = weights.get("minimum_resident_weight_bytes")
    if model.quantization or model.profile_env:
        minimum = None
    gpus = {gpu.index: gpu.memory_total_mb for gpu in environment.gpus}
    rows = []
    kept = []
    for candidate in candidates:
        static = candidate.static_config
        deployment = static.get("deployment", {})
        instances = deployment.get("instances") or [{"gpu_indexes": list(candidate.gpu_indexes)}]
        capacities = []
        for instance in instances:
            indexes = instance.get("gpu_indexes", [])
            memory = [gpus.get(index) for index in indexes]
            if not memory or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in memory):
                capacities = []
                break
            shards = len(memory) if model.raw.get("is_moe") else min(len(memory), candidate.tp * candidate.pp)
            capacities.append(int(sum(sorted(memory, reverse=True)[:shards]) * 1024 * 1024))
        capacity = min(capacities) if capacities else None
        status, reason = "unknown", "resident_weight_lower_bound_or_GPU_capacity_unknown"
        if minimum is not None and capacity is not None:
            status = "excluded" if minimum > capacity else "retained"
            reason = "weight_lower_bound_exceeds_replica_GPU_capacity" if status == "excluded" else "weight_lower_bound_fits; runtime_memory_not_verified"
        row = {"candidate_id": candidate.id, "deployment": deployment.get("label"),
               "instance_count": deployment.get("instance_count", 1),
               "gpus_per_instance": deployment.get("gpus_per_instance", len(candidate.gpu_indexes)),
               "tp": candidate.tp, "dp": candidate.dp, "pp": candidate.pp,
               "status": status, "reason": reason, "optimistic_replica_capacity_bytes": capacity,
               "minimum_resident_weight_bytes": minimum}
        rows.append(row)
        if status != "excluded":
            kept.append(candidate)
    counts = Counter(row["status"] for row in rows)
    report = {"version": 1, "scope": "necessary_weight_capacity_only", "runtime_verified": False,
              "status": "unknown" if counts["unknown"] == len(rows) else "partial" if counts["unknown"] else "screened",
              "input_candidates": len(rows), "candidate_scope": "base_planner_candidates_before_native_variants",
              "excluded_count": counts["excluded"], "retained_count": len(kept),
              "unknown_count": counts["unknown"], "candidates": rows,
              "assumptions": ["no CPU/disk offload or runtime quantization; custom profile_env disables hard exclusions",
                              "independent instances do not pool VRAM; dense DP replicas do not shard weights",
                              "MoE uses optimistic whole-instance capacity, never active-parameter count",
                              "no KV cache, activations, CUDA graphs, kernel workspace or uneven-shard guarantee"]}
    if not minimum:
        report["reason"] = "unknown_weight_residency_or_custom_loading; no_hard_exclusions"
    return tuple(kept), report


def capacity_summary(report):
    groups = {}
    for row in report.get("candidates", []):
        key = (row["instance_count"], row["gpus_per_instance"], row["status"], row["reason"])
        if key not in groups:
            groups[key] = {name: row[name] for name in ("deployment", "instance_count", "gpus_per_instance", "status", "reason")}
            groups[key]["candidate_count"] = 0
        groups[key]["candidate_count"] += 1
    return {**{name: value for name, value in report.items() if name != "candidates"},
            "deployment_summary": list(groups.values())[:64]}


def write_model_analysis(plan, root):
    from .artifacts import write_json_atomic
    from .errors import ConfigError

    weights = plan.metadata.get("model_snapshot", {}).get("raw", {}).get("weights", {})
    write_json_atomic(root / "model-weights.json", weights)
    write_json_atomic(root / "weight-capacity.json", plan.metadata["weight_capacity"])
    if not plan.candidates:
        raise ConfigError("all candidates exceed proven weight capacity; inspect " + str(root / "weight-capacity.json"))
