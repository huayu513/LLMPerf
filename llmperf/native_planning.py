"""Capability-derived experiments and bounded, GPU-free native parser preflight."""
from __future__ import annotations

import copy
import itertools
import json
import subprocess
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from .artifacts import write_json_atomic
from .docker_runtime import DockerRuntime, DockerTaskSpec, Mount
from .feature_catalog import adapt_native_static, native_contract, native_summary
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
    seen = {candidate.config_hash for candidate in bases}
    proposals = [row["params"] for row in sorted(experiments, key=lambda item: item.get("priority", 0), reverse=True)]
    proposals += [{name: value} for name, values in knobs.items() for value in values]
    first = {name: values[0] for name, values in knobs.items() if values}
    proposals.extend(dict(pair) for pair in itertools.combinations(first.items(), 2))
    for base, params in itertools.islice(itertools.chain.from_iterable(
            [(base, params) for params in proposals] for base in bases), max(1, limit) * 16):
        try:
            static = adapt_native_static({**base.static_config, **params}, catalog)
        except ValueError:
            continue
        static["feature_variant"] = json.dumps(native_summary(static), ensure_ascii=False, sort_keys=True)
        digest = fingerprint(static)
        if digest in seen:
            continue
        seen.add(digest)
        variants.append(replace(base, id="native-" + digest[:16], static_config=static,
            backend=static.get("backend", base.backend), dspark=static.get("dspark", base.dspark),
            config_hash=digest, reasons=("镜像证据与顾问建议；待运行验证",)))
        if len(variants) >= limit:
            break
    ordered = bases[:1] + variants + bases[1:]
    metadata = {**plan.metadata, "candidates": {candidate.id: asdict(candidate) for candidate in ordered}}
    groups = {}
    for candidate in ordered:
        group = candidate.static_config.get("comparison_group")
        if group:
            groups.setdefault(group, []).append(candidate.id)
    metadata["comparison_groups"] = groups
    return replace(plan, candidates=tuple(ordered), metadata=metadata)


def preflight_candidates(plan, config, root, *, runner=subprocess.run):
    catalog = plan.metadata["engine_catalog"]
    schema = plan.metadata["environment"].get("container", {}).get("capabilities", {}).get("schema", {})
    report = {"version": 1, "scope": "native_parser_only", "runtime_verified": False, "results": []}
    if schema.get("status") != "discovered":
        report["status"] = "unknown"
        report["reason"] = "structured_parser_not_discovered"
    else:
        cases = []
        for candidate in plan.candidates:
            contract = native_contract(candidate.static_config, catalog)
            argv = list(contract["append"])
            model = catalog["parameters"].get("model_path" if config.engine == "sglang" else "model")
            if model:
                argv += [model["flags"][0], "/model"]
            cases.append({"id": candidate.id, "argv": argv})
        path = root / "native-preflight-input.json"
        write_json_atomic(path, cases)
        source = (Path(__file__).resolve().parents[1] / "benchmarks/server/image_capabilities.py").read_text()
        script = source + f'\nfrom pathlib import Path\nprint("NATIVE_PREFLIGHT="+json.dumps(preflight({config.engine!r}, json.loads(Path("/cases.json").read_text()))))\n'
        spec = DockerTaskSpec(config.image, "capability-preflight-" + uuid.uuid4().hex[:12],
            mounts=(Mount(path, "/cases.json", True),), entrypoint="python3", command=("-c", script),
            network_mode="none")
        try:
            completed = runner(DockerRuntime().build_run_command(spec), capture_output=True, text=True, timeout=180)
            rows = [line.removeprefix("NATIVE_PREFLIGHT=") for line in completed.stdout.splitlines()
                    if line.startswith("NATIVE_PREFLIGHT=")]
            results = json.loads(rows[-1]) if rows else None
            if completed.returncode or not isinstance(results, list) or not all(isinstance(row, dict) for row in results) or {
                    row.get("id") for row in results} != {candidate.id for candidate in plan.candidates}:
                raise ValueError("native parser probe incomplete: " + str(completed.stderr)[-1000:])
            report.update(status="completed", results=results)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            report.update(status="unknown", reason=str(exc)[-1000:])
    write_json_atomic(root / "native-preflight.json", report)
    return report
