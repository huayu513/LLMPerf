"""Capability-derived experiments and bounded, GPU-free native parser preflight."""
from __future__ import annotations

import copy
import itertools
import json
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from .artifacts import write_json_atomic
from .docker_runtime import DockerRuntime, DockerTaskSpec, Mount
from .feature_catalog import adapt_native_static, native_contract, native_summary, planner_field, proposal_matches
from .planner import fingerprint


def proposal_knobs(analysis, config):
    knobs = {}
    rejected = []
    restrictions = dict(getattr(config, "engine_options", {}))
    backends = getattr(config.search, "backends", None)
    if backends:
        restrictions["moe_backend" if config.engine == "vllm" else "moe_runner_backend"] = list(backends)
    for row in sorted(analysis.get("features", []), key=lambda item: item.get("priority", 0), reverse=True):
        name = row["parameter"]
        allowed = restrictions.get(name)
        values = [value for value in row["values"] if name not in restrictions or
                  (value in allowed if isinstance(allowed, list) else value == allowed)]
        if values:
            knobs[name] = values
        if len(values) != len(row["values"]):
            rejected.append({"parameter": name, "reason": "explicit_user_restriction"})
    return knobs, rejected


def observed_knobs(catalog):
    knobs = {}
    for name, item in catalog["parameters"].items():
        if not item.get("integrated"):
            continue
        if item.get("type") == "boolean" and type(item.get("default")) is bool:
            knobs[name] = [item["default"], not item["default"]]
        elif item.get("choices") and name != "speculative_config":
            knobs[name] = [value for value in item["choices"] if isinstance(value, str)][:8]
    spec = catalog.get("nested", {}).get("speculative_config", {})
    methods = spec.get("method", {}).get("choices", [])
    if "speculative_config" in catalog["parameters"] and methods:
        count = spec.get("num_speculative_tokens", {}).get("default")
        knobs["speculative_config"] = [{"method": method, **({"num_speculative_tokens": count}
            if type(count) is int and count > 0 else {})} for method in methods if isinstance(method, str)][:8]
    return knobs


def feature_controls(plan, catalog, knobs, limit, experiments=()):
    bases = list(plan.candidates)
    variants = []
    def identity(static):
        return fingerprint({key: value for key, value in static.items()
                            if key not in {"feature_variant", "native_boolean_flags"}})
    seen = {identity(candidate.static_config) for candidate in bases}
    existing = {identity(candidate.static_config): candidate for candidate in bases}
    promoted = {}
    rejections = []
    attempts = 0
    proposals = [row["params"] for row in sorted(experiments, key=lambda item: item.get("priority", 0), reverse=True)]
    proposals += [{name: value} for name, values in knobs.items() for value in values]
    first = {name: values[0] for name, values in knobs.items() if values}
    proposals.extend(dict(pair) for pair in itertools.combinations(first.items(), 2))
    def cases(params):
        nonlocal attempts
        topology = {name: value for name, value in params.items() if planner_field(name, catalog)}
        native = {name: value for name, value in params.items() if not planner_field(name, catalog)}
        matched = [base for base in bases if proposal_matches(base.static_config, topology, catalog)]
        if not matched and len(rejections) < 64:
            rejections.append({"params": params, "reason": "no_existing_legal_topology"})
        for base in matched:
            if attempts >= max(1, limit) * 16:
                return
            attempts += 1
            try:
                static = adapt_native_static({**base.static_config, **native}, catalog)
            except ValueError as exc:
                if len(rejections) < 64:
                    rejections.append({"base_id": base.id, "params": params, "reason": str(exc)[:500]})
                continue
            yield base, static, bool(topology)

    for group in itertools.zip_longest(*(cases(params) for params in proposals)):
        for case in group:
            if case is None:
                continue
            base, static, has_topology = case
            key = identity(static)
            if key in seen:
                if has_topology and key in existing:
                    promoted[existing[key].id] = existing[key]
                continue
            seen.add(key)
            static["feature_variant"] = json.dumps(native_summary(static), ensure_ascii=False, sort_keys=True)
            digest = fingerprint(static)
            variants.append(replace(base, id="native-" + digest[:16], static_config=static,
                backend=static.get("backend", base.backend), dspark=static.get("dspark", base.dspark),
                config_hash=digest, reasons=("镜像证据与顾问建议；待运行验证",)))
            if len(variants) >= limit:
                break
        if len(variants) >= limit:
            break
    ordered = bases[:1] + [candidate for candidate in promoted.values() if candidate.id != bases[0].id]
    ordered += variants + [candidate for candidate in bases[1:] if candidate.id not in promoted]
    metadata = {**plan.metadata, "candidates": {candidate.id: asdict(candidate) for candidate in ordered}}
    metadata["native_generation"] = {"attempts": attempts, "created": len(variants),
        "promoted_topologies": list(promoted), "rejections": rejections, "variant_limit": limit}
    groups = {}
    for candidate in ordered:
        group = candidate.static_config.get("comparison_group")
        if group:
            groups.setdefault(group, []).append(candidate.id)
    metadata["comparison_groups"] = groups
    return replace(plan, candidates=tuple(ordered), metadata=metadata)


def proposal_outcomes(analysis, candidates, catalog):
    proposals = [{"source": "feature", "params": {row["parameter"]: value}}
                 for row in analysis.get("features", []) for value in row["values"]]
    proposals += [{"source": "experiment", "params": row["params"]}
                  for row in analysis.get("experiments", [])]
    return [{**proposal, "candidate_ids": [candidate.id for candidate in candidates
               if proposal_matches(candidate.static_config, proposal["params"], catalog)]}
            for proposal in proposals]


def _preflight_results(results):
    counts = {status: sum(row["status"] == status for row in results)
              for status in ("preflight_passed", "parser_rejected", "unknown")}
    reasons = Counter(row.get("reason", "native parser unavailable")[:2000]
                      for row in results if row["status"] == "unknown")
    status = "unknown" if not results or counts["unknown"] == len(results) else (
        "partial" if counts["unknown"] else "completed")
    return {"status": status, "counts": counts,
            "unknown_reasons": [{"reason": reason, "count": count} for reason, count in reasons.most_common(8)],
            "reason": "; ".join(reason for reason, _ in reasons.most_common(2))}


def _needs_device_context(report):
    reasons = [report.get("reason", "")] + [item["reason"] for item in report.get("unknown_reasons", [])]
    return any(marker in reason.lower() for reason in reasons for marker in
               ("failed to infer device type", "no cuda gpus are available", "no cuda-capable device"))


def preflight_candidates(plan, config, root, *, runner=subprocess.run):
    catalog = plan.metadata["engine_catalog"]
    schema = plan.metadata["environment"].get("container", {}).get("capabilities", {}).get("schema", {})
    report = {"version": 2, "scope": "native_parser_only", "runtime_verified": False,
              "results": [], "attempts": [], "timeout_seconds": 180}
    if schema.get("status") != "discovered":
        report["status"] = "unknown"
        report["reason"] = "structured_parser_not_discovered"
    else:
        cases = []
        for candidate in plan.candidates:
            static = candidate.static_config
            contract = native_contract(static, catalog)
            argv = list(contract["append"])
            names = (("tp_size", "tp"), ("dp_size", "dp"), ("pp_size", "pp")) if config.engine == "sglang" else (
                ("tensor_parallel_size", "tp"), ("data_parallel_size", "dp"), ("pipeline_parallel_size", "pp"))
            for name, key in names:
                item = catalog["parameters"].get(name)
                if item and not any(flag in argv for flag in item["flags"]):
                    argv += [item["flags"][0], str(static.get(key, 1))]
            if config.engine == "sglang" and "attn_dp_size" not in catalog["parameters"] and static.get("dp_attention"):
                argv.append("--enable-dp-attention")
            model = catalog["parameters"].get("model_path" if config.engine == "sglang" else "model")
            if model:
                argv += [model["flags"][0], "/model"]
            cases.append({"id": candidate.id, "argv": argv})
        path = root / "native-preflight-input.json"
        write_json_atomic(path, cases)
        source = (Path(__file__).resolve().parents[1] / "benchmarks/server/image_capabilities.py").read_text()
        script = source + f'\nfrom pathlib import Path\nprint("NATIVE_PREFLIGHT="+json.dumps(preflight({config.engine!r}, json.loads(Path("/cases.json").read_text()))))\n'
        selected = getattr(config, "gpu_indexes", None)
        if selected is None:
            selected = sorted({index for candidate in plan.candidates
                               for index in candidate.static_config.get("gpu_indexes", [])})
        contexts = [(), tuple(selected)] if selected else [()]
        deadline = time.monotonic() + report["timeout_seconds"]
        for indexes in contexts:
            remaining = deadline - time.monotonic() if report["attempts"] else report["timeout_seconds"]
            if remaining <= 0:
                report.update(status="unknown", reason="native_preflight_time_budget_exhausted")
                break
            device_context = "selected_gpus" if indexes else "no_gpu"
            env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, range(len(indexes)))),
                   "NVIDIA_DRIVER_CAPABILITIES": "compute,utility"} if indexes else {
                       "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void"}
            spec = DockerTaskSpec(config.image, "capability-preflight-" + uuid.uuid4().hex[:12],
                gpu_indexes=indexes, mounts=(Mount(path, "/cases.json", True),), env=env,
                entrypoint="python3", command=("-c", script), network_mode="none")
            attempt = {"attempt": len(report["attempts"]) + 1, "device_context": device_context,
                       "gpu_indexes": list(indexes), "timeout_seconds": remaining}
            try:
                completed = runner(DockerRuntime().build_run_command(spec), capture_output=True, text=True, timeout=remaining)
                rows = [line.removeprefix("NATIVE_PREFLIGHT=") for line in completed.stdout.splitlines()
                        if line.startswith("NATIVE_PREFLIGHT=")]
                results = json.loads(rows[-1]) if rows else None
                if (completed.returncode or not isinstance(results, list) or len(results) != len(plan.candidates)
                        or not all(isinstance(row, dict) and isinstance(row.get("id"), str) and row.get("status") in
                                   {"preflight_passed", "parser_rejected", "unknown"} for row in results)
                        or {row["id"] for row in results} != {candidate.id for candidate in plan.candidates}):
                    raise ValueError("native parser probe incomplete: " + str(completed.stderr)[-1000:])
                results = [{**row, "runtime_verified": False} for row in results]
                report.update(results=results, **_preflight_results(results))
                attempt.update({key: report[key] for key in ("status", "counts", "unknown_reasons", "reason")})
            except subprocess.TimeoutExpired as exc:
                report.update(status="unknown", reason=str(exc)[-1000:])
                attempt.update(status="unknown", reason=report["reason"])
                try:
                    runner(["docker", "rm", "-f", spec.name], capture_output=True, text=True, timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            except (OSError, ValueError) as exc:
                report.update(status="unknown", reason=str(exc)[-1000:])
                attempt.update(status="unknown", reason=report["reason"])
            report["attempts"].append(attempt)
            report["device_context"] = device_context
            if indexes or not _needs_device_context(report):
                break
            if not selected:
                report["retry_unavailable"] = "no_selected_gpu_context"
    write_json_atomic(root / "native-preflight.json", report)
    return report
