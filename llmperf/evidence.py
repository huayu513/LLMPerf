"""Bounded, credential-redacted evidence for adaptive experiment decisions."""
from __future__ import annotations

import re
import math
from pathlib import Path


def redact(value, secrets=()):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: "[REDACTED]" if key.lower() in
                {"authorization", "api_key", "password", "secret", "access_token"}
                else redact(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)\b(?:Bearer|Basic)\s+[A-Za-z0-9+/._~=-]+", "[REDACTED]", value)
        value = re.sub(
            r'''(?i)(["']?(?:authorization|api[_-]?key|access[_-]?token|password|secret)["']?\s*[:=]\s*)("[^"]*"|'[^']*'|[^\s,;]+)''',
            r"\1[REDACTED]", value)
        return value[:4096]
    return value


def candidate_summary(candidate: dict) -> dict:
    static = candidate["static_config"]
    deployment = static.get("deployment", {})
    return {"candidate_id": candidate["id"], "config_hash": candidate["config_hash"],
            "gpu_indexes": static.get("gpu_indexes", candidate.get("gpu_indexes", [])),
            "params": {key: static.get(key) for key in
                       ("tp", "dp", "pp", "dp_attention", "backend", "dspark",
                        "mem_fraction_static", "chunked_prefill_size", "max_running_requests")},
            "deployment": {key: deployment.get(key) for key in
                           ("label", "instance_count", "gpus_per_instance", "total_gpu_count")}}


def environment_summary(metadata: dict) -> dict:
    model = metadata.get("model_snapshot", {})
    raw = model.get("raw", {})
    weights = raw.get("inventory", {}).get("weight_files", [])
    environment = metadata.get("environment", {})
    return redact({
        "model": {"quantization": model.get("quantization"),
                  **{key: raw.get(key) for key in
                     ("model_type", "is_moe", "num_attention_heads", "num_hidden_layers")},
                  "weight_bytes": sum(row.get("size", 0) for row in weights)},
        "gpus": environment.get("gpus", []),
        "topology": environment.get("topology", {}),
        "engine_versions": environment.get("container", {}).get("versions", {}),
        "capabilities": environment.get("container", {}).get("capabilities", {}),
        "image": metadata.get("image"),
    })


def summarize_trial(result: dict, candidate: dict, root: Path) -> dict:
    root = root.resolve()
    summary = {key: result[key] for key in
               ("task_id", "candidate_id", "status", "score", "output_tokens_per_second",
                "latency_seconds", "ttft_seconds", "tpot_seconds", "error_rate",
                "scene_lengths", "failure_kind", "failure_phase", "container_oom_killed",
                "backend_unsupported", "reasons", "measured_seconds", "trial_seconds")
               if key in result}
    summary["planned_configuration"] = candidate_summary(candidate)
    effective = result.get("effective_static_config")
    if isinstance(effective, dict):
        summary["effective_configuration"] = candidate_summary({**candidate, "static_config": effective})
        summary["effective_differences"] = {
            key: {"planned": candidate["static_config"].get(key), "effective": value}
            for key, value in effective.items()
            if key in summary["planned_configuration"]["params"]
            and candidate["static_config"].get(key) != value}
    excerpts = []
    attempt = Path(result.get("result_path", root / "absent")).resolve()
    if result.get("status") != "VALID" and attempt.is_relative_to((root / "trials").resolve()):
        for path in sorted(attempt.rglob("*.log")):
            if len(excerpts) >= 2:
                break
            if not path.resolve().is_relative_to(attempt):
                continue
            try:
                with path.open("rb") as source:
                    source.seek(0, 2)
                    source.seek(max(0, source.tell() - 2048))
                    text = source.read(2048).decode("utf-8", errors="replace")
                excerpts.append({"path": str(path.relative_to(root)), "tail": text})
            except OSError:
                continue
    summary["log_excerpts"] = excerpts
    return redact(summary)
