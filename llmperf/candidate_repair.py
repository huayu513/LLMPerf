"""Typed runtime candidate proposals; never mutate a frozen plan or workload."""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, replace

from .adapters import _effective_max_running_requests
from .artifacts import sha256_file
from .errors import ConfigError
from .feature_catalog import FEATURES, adapt_native_static, missing_native_fields
from .planner import fingerprint
from .types import CandidateConfig, PlanTask


TUNABLES = {
    "vllm": {"gpu_memory_utilization": float, "max_num_seqs": int, "max_num_batched_tokens": int},
    "sglang": {"mem_fraction_static": float, "max_running_requests": int, "chunked_prefill_size": int},
}


def proposal_candidate(proposal, candidates, metadata, allowed_bases, evidence_ids):
    if (not isinstance(proposal, dict) or set(proposal) - {"base_id", "patch", "reason", "evidence_ids"}
            or not isinstance(proposal.get("base_id"), str) or proposal["base_id"] not in allowed_bases):
        raise ValueError("proposal must reference a supplied base candidate")
    references = proposal.get("evidence_ids")
    if (not isinstance(references, list) or not 1 <= len(references) <= 8 or
            any(not isinstance(ident, str) or ident not in evidence_ids for ident in references)):
        raise ValueError("proposal needs supplied evidence IDs")
    reason = proposal.get("reason", "")
    if not isinstance(reason, str) or len(reason) > 500:
        raise ValueError("invalid proposal reason")
    catalog = metadata.get("engine_catalog")
    if not catalog:
        raise ValueError("runtime proposal requires an image capability catalog")
    engine = catalog["engine"]
    patch = proposal.get("patch")
    allowed = set(FEATURES.get(engine, {})) | set(TUNABLES.get(engine, {}))
    if not isinstance(patch, dict) or not 1 <= len(patch) <= 8 or set(patch) - allowed:
        raise ValueError("patch changes a protected or unintegrated parameter")
    for name, value in patch.items():
        if name not in catalog["parameters"]:
            raise ValueError(f"{name}: not discovered in this image")
        datatype = TUNABLES.get(engine, {}).get(name)
        if datatype:
            valid_type = type(value) is int if datatype is int else type(value) in (int, float)
            if not valid_type or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}: requires a positive finite {datatype.__name__}")
            if datatype is float and value >= 1:
                raise ValueError(f"{name}: must be < 1")
    base = candidates[proposal["base_id"]]
    static = copy.deepcopy(base["static_config"])
    for name, value in patch.items():
        if isinstance(value, dict) and isinstance(static.get(name), dict):
            static[name].update(copy.deepcopy(value))
        else:
            static[name] = copy.deepcopy(value)
    restrictions = metadata.get("configuration", {}).get("engine_options", {})
    for name, value in restrictions.items():
        if name in patch and static[name] != value:
            raise ValueError(f"{name}: conflicts with explicit user restriction")
    backends = metadata.get("search", {}).get("backends")
    backend_name = "moe_backend" if engine == "vllm" else "moe_runner_backend"
    if backends and backend_name in patch and static[backend_name] not in backends:
        raise ValueError("backend conflicts with search.backends")
    if engine == "vllm" and backend_name in patch:
        static["backend"] = None if static[backend_name] == "auto" else static[backend_name]
    capacity = "max_num_seqs" if engine == "vllm" else "max_running_requests"
    if capacity in patch:
        task = PlanTask("runtime-proposal", 0, base["id"], concurrency=metadata["scenario"]["target_concurrency"])
        static[capacity] = _effective_max_running_requests(task, static)
    static = adapt_native_static(static, catalog)
    if static == base["static_config"]:
        raise ValueError("proposal does not change the base configuration")
    digest = fingerprint(static)
    candidate = CandidateConfig(**{**base, "gpu_indexes": tuple(base.get("gpu_indexes", ())),
                                   "reasons": tuple(base.get("reasons", ()))})
    candidate = replace(candidate, id="repair-" + digest[:16], static_config=static, config_hash=digest,
                        backend=static.get("backend", candidate.backend),
                        dspark=static.get("dspark", candidate.dspark), reasons=(reason,))
    return json.loads(json.dumps(asdict(candidate)))


def configuration_defects(candidate, catalog, summary=None):
    """Only quarantine reproducible missing-field defects, never an entire backend."""
    static = candidate["static_config"]
    missing = missing_native_fields(static, catalog or {})
    spec = static.get("speculative_config")
    if isinstance(spec, dict) and spec.get("method") and not spec.get("num_speculative_tokens"):
        # Older saved catalogs may not carry the nested schema. Require an
        # explicit native validation error before learning the same dependency.
        text = "\n".join(row.get("tail", "") for row in (summary or {}).get("log_excerpts", []))
        if ("SpeculativeConfig" in text and re.search(
                r"num_speculative_tokens.{0,180}(?:must|require|cannot|can't|not.{0,10}None|missing)",
                text, re.IGNORECASE | re.DOTALL)):
            missing.append("speculative_config.num_speculative_tokens")
    return [{"parameter": "speculative_config", "method": spec.get("method"), "field": field}
            for field in sorted(set(missing))] if isinstance(spec, dict) else []


def matches_defect(candidate, defect):
    spec = candidate["static_config"].get(defect["parameter"])
    field = defect["field"].split(".", 1)[1]
    return isinstance(spec, dict) and spec.get("method") == defect["method"] and field not in spec


def runtime_candidate_records(root):
    """Read committed, hash-checked proposals for display and reproduction."""
    state_path = root / "tune-state.json"
    if not state_path.is_file():
        return []
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        result = []
        for entry in state.get("adaptive", {}).get("records", []):
            if not entry.get("sha256"):
                continue
            path = root / entry["manifest"]
            if (not path.resolve().is_relative_to((root / "decisions").resolve()) or
                    sha256_file(path) != entry["sha256"]):
                raise ConfigError("recorded runtime decision changed")
            document = json.loads(path.read_text(encoding="utf-8"))
            if document["plan_hash"] != state["plan_hash"]:
                raise ConfigError("runtime candidate belongs to another plan")
            for proposal in document.get("proposals", []):
                result.append({**proposal, "decision": document["id"]})
        return result
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigError("invalid runtime candidate journal") from exc
