"""Docker replay adapter and normalized benchmark-attempt reader."""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .docker_runtime import DockerRuntime, DockerTaskSpec, Mount
from .types import DockerConfig, PlanTask


_DEFAULT_CHAT_TEMPLATE_KWARGS = {
    "enable_thinking": True,
    "reasoning_effort": "high",
    "thinking": True,
}


def _chat_template_provenance(metadata: Mapping[str, Any]) -> Any:
    snapshot = metadata.get("model_snapshot", {})
    raw = snapshot.get("raw", {}) if isinstance(snapshot, Mapping) else {}
    provenance = raw.get("provenance", {}) if isinstance(raw, Mapping) else {}
    return provenance.get("chat_template_kwargs") if isinstance(provenance, Mapping) else None


def _effective_chat_template_kwargs(metadata: Mapping[str, Any]) -> Any:
    """Return high-thinking defaults for plans written before that default.

    An explicit model override, including an intentionally empty mapping,
    remains authoritative.  Plans written by the old discovery path carry the
    ``safe empty default`` provenance marker and are upgraded on replay.
    """

    if "chat_template_kwargs" not in metadata:
        return dict(_DEFAULT_CHAT_TEMPLATE_KWARGS)
    value = metadata["chat_template_kwargs"]
    if value == {} and _chat_template_provenance(metadata) == "safe empty default":
        return dict(_DEFAULT_CHAT_TEMPLATE_KWARGS)
    return value


def _candidate(metadata: Mapping[str, Any], task: PlanTask) -> Mapping[str, Any]:
    candidates = metadata.get("candidates", {})
    if not isinstance(candidates, Mapping):
        return {}
    value = candidates.get(task.candidate_id, {})
    return value if isinstance(value, Mapping) else {}


def _static_config(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    value = candidate.get("static_config", {})
    return value if isinstance(value, Mapping) else {}


def _effective_max_running_requests(task: PlanTask, static: Mapping[str, Any]) -> int:
    configured = static.get("max_running_requests")
    configured_int = max(1, int(configured)) if configured is not None else None
    if str(task.mode).replace("_", "-") == "open-loop":
        return configured_int or max(1, int(task.concurrency or 1))
    concurrency = max(1, int(task.concurrency or 1))
    deployment = static.get("deployment", {})
    instance_count = 1
    if isinstance(deployment, Mapping):
        try:
            instance_count = max(1, int(deployment.get("instance_count", 1)))
        except (TypeError, ValueError):
            instance_count = 1
    per_instance = max(1, math.ceil(concurrency / instance_count))
    return min(configured_int, per_instance) if configured_int is not None else per_instance


def _task_request_limit(task: PlanTask, metadata: Mapping[str, Any]) -> int | None:
    expected = metadata.get("expected_request_count", metadata.get("request_count"))
    expected_int = int(expected) if expected is not None else None
    if task.run_class == "smoke":
        return min(8, expected_int) if expected_int is not None else 8
    if task.run_class in {"concurrency", "tuning", "diagnostic"}:
        search = metadata.get("search", {})
        raw_limit = search.get("explore_request_limit") if isinstance(search, Mapping) else None
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            limit = 0
        if limit > 0:
            return min(limit, expected_int) if expected_int is not None else limit
    return None


def _controller_class(task: PlanTask, metadata: Mapping[str, Any]) -> str:
    if task.run_class in {"smoke", "formal", "diagnostic"}:
        return task.run_class
    return "diagnostic" if _task_request_limit(task, metadata) is not None else "formal"


def _expected_request_count_for_task(task: PlanTask, metadata: Mapping[str, Any]) -> int | None:
    expected = metadata.get("expected_request_count", metadata.get("request_count"))
    if expected is None:
        return None
    result = int(expected)
    limit = _task_request_limit(task, metadata)
    if limit is not None:
        result = min(result, int(limit))
    return result


def _deployment_with_logical_gpus(
    deployment: Mapping[str, Any],
    gpu_indexes: tuple[int, ...],
) -> dict[str, Any] | None:
    if not deployment:
        return None
    logical_by_physical = {physical: offset for offset, physical in enumerate(gpu_indexes)}
    instances = []
    for raw in deployment.get("instances", ()):
        if not isinstance(raw, Mapping):
            return None
        physical = tuple(int(value) for value in raw.get("gpu_indexes", ()))
        if any(index not in logical_by_physical for index in physical):
            return None
        item = dict(raw)
        item["gpu_indexes"] = list(physical)
        item["logical_gpu_indexes"] = [logical_by_physical[index] for index in physical]
        instances.append(item)
    result = dict(deployment)
    result["gpu_indexes"] = list(gpu_indexes)
    result["instances"] = instances
    return result


_SERVER_INFO_ALIASES = {
    "model_path": ("model_path",),
    "served_model_name": ("served_model_name", "served_model"),
    "tool_call_parser": ("tool_call_parser",),
    "reasoning_parser": ("reasoning_parser",),
    "chat_template_kwargs": ("chat_template_kwargs", "default_chat_template_kwargs"),
    "quantization": ("quantization",),
    "tp": ("tp", "tp_size", "tensor_parallel_size"),
    "dp": ("dp", "dp_size", "data_parallel_size"),
    "pp": ("pp", "pp_size", "pipeline_parallel_size"),
    "dp_attention": ("dp_attention", "enable_dp_attention"),
    "dp_lm_head": ("dp_lm_head", "enable_dp_lm_head"),
    "backend": ("moe_runner_backend", "moe_runner", "backend"),
    "moe_a2a_backend": ("moe_a2a_backend", "a2a_backend"),
    "dspark": ("dspark", "enable_dspark", "speculative_algorithm"),
    "mem_fraction_static": ("mem_fraction_static",),
    "max_running_requests": ("max_running_requests",),
    "chunked_prefill_size": ("chunked_prefill_size",),
}


def _normalized_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _server_info_leaves(value: Any) -> dict[str, Any]:
    # ServerArgs contains unrelated nested backends (for example CUDA graphs).
    # Only direct argument fields can establish the launch parameter values.
    if not isinstance(value, Mapping):
        return {}
    arguments = value.get("server_args", value)
    if not isinstance(arguments, Mapping):
        return {}
    return {_normalized_key(key): child for key, child in arguments.items()}


def _values_match(key: str, expected: Any, actual: Any) -> bool:
    if key in {"backend", "moe_a2a_backend"} and isinstance(expected, str) and expected.lower() == "auto":
        unresolved = {"", "auto"} if key == "moe_a2a_backend" else {"", "auto", "none"}
        return actual is not None and str(actual).strip().lower() not in unresolved
    if isinstance(expected, bool):
        if key == "dspark" and isinstance(actual, str):
            actual = actual.strip().upper() == "DSPARK"
        elif key == "dspark" and actual is None:
            actual = False
        elif isinstance(actual, str):
            lowered = actual.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                actual = True
            elif lowered in {"0", "false", "no", "off", "none", ""}:
                actual = False
        elif isinstance(actual, (int, float)) and actual in (0, 1):
            actual = bool(actual)
        return actual is expected
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    if isinstance(expected, Mapping) and isinstance(actual, str):
        try:
            actual = json.loads(actual)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _compare_server_parameters(requested: Mapping[str, Any], server_info: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    leaves = _server_info_leaves(server_info)
    unsupported: list[str] = []
    mismatched: list[str] = []
    for key, expected in requested.items():
        found = False
        actual = None
        for alias in _SERVER_INFO_ALIASES.get(key, (key,)):
            if alias in leaves:
                found = True
                actual = leaves[alias]
                break
        if not found:
            unsupported.append(key)
        elif not _values_match(key, expected, actual):
            derived_chunked_prefill = (
                key == "chunked_prefill_size"
                and requested.get("dp_attention") is True
                and isinstance(expected, int)
                and isinstance(requested.get("dp"), int)
                and requested["dp"] > 0
                and _values_match(key, expected // requested["dp"], actual)
            )
            if not derived_chunked_prefill:
                mismatched.append(key)
    return unsupported, mismatched


def _evidence_fingerprint(artifact_dir: Path, summary_path: Path | None, attempt_path: Path) -> str | None:
    paths = [
        summary_path,
        artifact_dir / "server.requested.json",
        artifact_dir / "server.info.json",
        artifact_dir / "server.evidence.json",
        artifact_dir / "run_manifest.json",
        artifact_dir / "server.command.sh",
        attempt_path / "container-exit-code",
    ]
    if any(path is None or not path.is_file() for path in paths[:5]):
        return None
    digest = hashlib.sha256()
    for role, path in enumerate(paths):
        if path is not None and path.is_file():
            digest.update(f"artifact-{role}\0".encode("ascii"))
            contents = path.read_bytes()
            digest.update(contents)
            digest.update(b"\0")
    return digest.hexdigest()


def _summary_metric_fields() -> tuple[str, ...]:
    return (
        "request_count", "successes", "failures", "error_rate",
        "server_usage_available", "server_usage_missing",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "input_tokens_per_second", "total_tokens_per_second",
        "successful_requests_per_second", "completion_tokens_per_request",
        "latency_seconds", "ttft_seconds", "tpot_seconds",
        "dispatch_lag_seconds", "finish_reasons", "http_statuses",
        "failed_request_ids", "missing_usage_request_ids",
        "base_url", "base_urls", "endpoints", "endpoint_summaries",
    )


_STARTUP_FAILURE_REASONS = frozenset({
    "readiness_missing",
    "server_evidence_missing",
    "server_info_missing",
    "server_parameter_artifacts_missing",
    "unsupported_server_parameters",
    "server_parameter_mismatch",
    "requested_server_parameters_mismatch",
    "server_evidence_inconsistent",
    "resolved_parameters_missing",
})


def _failure_phase(
    status: str,
    reasons: list[str],
    summary: Mapping[str, Any] | None,
    evidence: Mapping[str, Any] | None,
) -> str | None:
    """Classify failures before search applies backend quarantine.

    A replay can fail after a healthy server has started (for example because
    an individual request failed or the load caused an OOM).  Those failures
    must stop only the current candidate branch.  Missing readiness/evidence
    is the stronger signal that the requested server configuration itself did
    not start, so only that phase is eligible to block a backend.
    """

    if status == "UNSUPPORTED" or any(reason in _STARTUP_FAILURE_REASONS for reason in reasons):
        return "startup"
    if summary is None and evidence is None:
        return "startup"
    if summary is None and evidence is not None and evidence.get("readiness") is not True:
        return "startup"
    if summary is not None:
        return "replay"
    return None


def _local_loopback_url(value: Any, allowed_ports: set[int]) -> bool:
    try:
        address = urlsplit(value) if isinstance(value, str) else None
        return bool(
            address and address.scheme == "http"
            and address.hostname in {"127.0.0.1", "localhost", "::1"}
            and address.port in allowed_ports
            and address.username is None and address.password is None
            and address.path in {"", "/"} and not address.query and not address.fragment
        )
    except (ValueError, TypeError):
        return False


def _expected_requested_parameters(metadata: Mapping[str, Any], task: PlanTask) -> dict[str, Any]:
    candidate = _candidate(metadata, task)
    static = _static_config(candidate)
    expected: dict[str, Any] = {
        "model_path": "/model",
        "max_running_requests": _effective_max_running_requests(task, static),
    }
    snapshot = metadata.get("model_snapshot", {})
    raw = snapshot.get("raw", {}) if isinstance(snapshot, Mapping) else {}
    is_moe = isinstance(raw, Mapping) and bool(raw.get("is_moe"))
    for key in (
        "tp", "dp", "pp", "dp_attention", "dp_lm_head", "backend",
        "moe_a2a_backend", "dspark", "mem_fraction_static", "chunked_prefill_size",
    ):
        value = static.get(key, candidate.get(key))
        if value is not None:
            expected[key] = value
    if is_moe:
        expected["backend"] = static.get("backend") or "auto"
        expected["moe_a2a_backend"] = static.get("moe_a2a_backend") or "auto"
    for key in ("served_model_name", "tool_call_parser", "reasoning_parser", "quantization"):
        value = metadata.get(key)
        if value not in (None, ""):
            expected[key] = value
    if "chat_template_kwargs" in metadata:
        expected["chat_template_kwargs"] = _effective_chat_template_kwargs(metadata)
    return expected


class ReplayAdapter:
    """Build and execute a replay task in an immutable Docker image."""

    def __init__(self, metadata: Mapping[str, Any], run_root: Path, runtime: DockerRuntime | None = None):
        self.metadata = dict(metadata)
        candidates = self.metadata.get("candidates", {})
        self.candidates = candidates if isinstance(candidates, Mapping) else {}
        self.run_root = Path(run_root)
        self.runtime = runtime or DockerRuntime()

    def _value(self, *keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in self.metadata and self.metadata[key] not in (None, ""):
                return self.metadata[key]
        return default

    def build_spec(self, task: PlanTask, attempt: int, attempt_dir: Path | None = None) -> DockerTaskSpec:
        candidate = _candidate(self.metadata, task)
        static = _static_config(candidate)
        image = str(self._value("image", "docker_image", default=""))
        model_host = Path(str(self._value("model_host", default="/tmp/model"))).resolve()
        jsonl_host = Path(str(self._value("jsonl_host", default="/tmp/input.jsonl"))).resolve()
        index_host = Path(str(self._value("index_host", default=self.run_root / "replay_index.json"))).resolve()
        result_dir = Path(attempt_dir or (self.run_root / task.id / f"attempt-{attempt:03d}")).resolve()
        result_dir.mkdir(parents=True, exist_ok=True)
        benchmark_dir = Path(str(self._value("benchmark_dir", default=Path(__file__).resolve().parents[1] / "benchmarks"))).resolve()
        profile = str(self._value("profile", default="AUTO"))
        controller_class = _controller_class(task, self.metadata)
        mode = str(task.mode).replace("_", "-")
        command_values = [
            "bash", "/opt/s1slow/benchmarks/run_point.sh", controller_class,
            "--profile", profile, "--mode", mode,
            "--concurrency", str(task.concurrency or 1),
            "--arrival-rate-scale", str(task.scale or 1.0),
            "--run", str(attempt),
            "--warmup", str(self._value("warmup", default=0)),
            "--request-timeout", str(self._value("request_timeout", default=3600)),
            "--ready-timeout", str(self._value("ready_timeout", default=3600)),
            "--jsonl", "/run/workload/input.jsonl",
            "--index", "/run/workload/replay_index.json",
            "--no-gpu-monitor",
        ]

        gpu_values = candidate.get("gpu_indexes", self._value("gpu_indexes", default=()))
        gpu_indexes = tuple(int(x) for x in (gpu_values or ()))
        service_port = int(self._value("service_port", default=DockerConfig().service_port))
        deployment = _deployment_with_logical_gpus(static.get("deployment", {}), gpu_indexes)
        if deployment and int(deployment.get("instance_count", 1)) > 1:
            command_values[0:2] = ["python3", "/opt/s1slow/benchmarks/run_deployment_point.py"]
        limit = _task_request_limit(task, self.metadata)
        if limit is not None:
            command_values.extend(("--limit", str(max(0, int(limit)))))
        command = tuple(command_values)
        if task.run_class == "concurrency_rejected":
            command = ("bash", "-c", "exit 2")

        env = {
            "MODEL_PATH": "/model",
            "SERVED_MODEL_NAME": str(self._value("served_model_name", default="model")),
            "S1_CONFIG_FILE": "/opt/s1slow/benchmarks/config/portable.env",
            "S1_PROFILE_DIR": "/opt/s1slow/benchmarks/server/profiles",
            "S1_SERVER_STATE_DIR": "/run/results/server",
            "S1_JSONL_PATH": "/run/workload/input.jsonl",
            "S1_INDEX_PATH": "/run/workload/replay_index.json",
            "S1_RESULTS_ROOT": "/run/results",
            "S1_PYTHON_BIN": "python3",
            "SGLANG_BIN": "sglang",
            "SERVER_PORT": str(service_port),
            "S1_BASE_URL": f"http://127.0.0.1:{service_port}",
        }
        if deployment:
            deployment["service_port"] = service_port
            deployment["base_urls"] = [
                f"http://127.0.0.1:{service_port + offset}"
                for offset in range(int(deployment.get("instance_count", 1)))
            ]
            env["S1_DEPLOYMENT"] = json.dumps(
                deployment, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        model_snapshot = self._value("model_snapshot", default={})
        model_raw = model_snapshot.get("raw", {}) if isinstance(model_snapshot, Mapping) else {}
        env["MODEL_IS_MOE"] = "1" if isinstance(model_raw, Mapping) and bool(model_raw.get("is_moe")) else "0"
        for source, target in {
            "tool_call_parser": "TOOL_CALL_PARSER",
            "reasoning_parser": "REASONING_PARSER",
            "quantization": "QUANTIZATION",
        }.items():
            value = self._value(source)
            if value is not None:
                env[target] = str(value)
        template_kwargs = _effective_chat_template_kwargs(self.metadata)
        if not isinstance(template_kwargs, Mapping):
            raise ValueError("chat_template_kwargs must be an object")
        env["DEFAULT_CHAT_TEMPLATE_KWARGS"] = json.dumps(
            dict(template_kwargs), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        profile_env = self._value("profile_env", default={})
        if not isinstance(profile_env, Mapping):
            raise ValueError("profile_env must be an object")
        reserved_profile_keys = set(env) | {
            "TP_SIZE", "DP_SIZE", "PP_SIZE", "ENABLE_DP_ATTENTION",
            "ENABLE_DP_LM_HEAD", "ENABLE_DSPARK", "MOE_RUNNER_BACKEND",
            "MOE_A2A_BACKEND", "MEM_FRACTION_STATIC", "MAX_RUNNING_REQUESTS",
            "CHUNKED_PREFILL_SIZE", "DEFAULT_CHAT_TEMPLATE_KWARGS",
            "TOOL_CALL_PARSER", "REASONING_PARSER", "QUANTIZATION",
        }
        for key, value in profile_env.items():
            if str(key) in reserved_profile_keys:
                raise ValueError(f"profile_env may not override runtime parameter {key}")
            env[str(key)] = str(value)
        static_env = (
            ("tp", "TP_SIZE"), ("dp", "DP_SIZE"), ("pp", "PP_SIZE"),
            ("dp_attention", "ENABLE_DP_ATTENTION"),
            ("dp_lm_head", "ENABLE_DP_LM_HEAD"),
            ("dspark", "ENABLE_DSPARK"),
            ("backend", "MOE_RUNNER_BACKEND"),
            ("moe_a2a_backend", "MOE_A2A_BACKEND"),
            ("mem_fraction_static", "MEM_FRACTION_STATIC"),
            ("chunked_prefill_size", "CHUNKED_PREFILL_SIZE"),
        )
        for source, target in static_env:
            value = static.get(source, candidate.get(source))
            if value is not None:
                if target.startswith("ENABLE_"):
                    value = int(bool(value))
                env[target] = str(value)
        env["MAX_RUNNING_REQUESTS"] = str(_effective_max_running_requests(task, static))
        logical_gpu_count = len(gpu_indexes) if gpu_indexes else 1
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in range(logical_gpu_count))
        mounts = (
            Mount(model_host, "/model", True),
            Mount(jsonl_host, "/run/workload/input.jsonl", True),
            Mount(index_host, "/run/workload/replay_index.json", True),
            Mount(result_dir, "/run/results", False),
            Mount(benchmark_dir, "/opt/s1slow/benchmarks", True),
        )
        return DockerTaskSpec(
            image=image, name=str(self._value("name_prefix", default="s1slow")) + "-" + uuid.uuid4().hex[:12],
            gpu_indexes=gpu_indexes, mounts=mounts, env=env, command=command,
            internal_port=service_port,
            network_mode=str(self._value("network_mode", default="bridge")),
            shm_size=str(self._value("shm_size", default="16g")),
            ipc=str(self._value("ipc", default="host")),
        )

    build_task_spec = build_spec

    def run(self, task: PlanTask, attempt: int):
        return self(task, attempt)

    def __call__(self, task: PlanTask, attempt: int):
        attempt_dir = self.run_root / task.id / f"attempt-{attempt:03d}"
        spec = self.build_spec(task, attempt, attempt_dir)
        result = self.runtime.run(spec, attempt_dir / "docker.log")
        exit_path = attempt_dir / "container-exit-code"
        exit_temp = attempt_dir / ".container-exit-code.tmp"
        exit_temp.write_text(f"{result.exit_code}\n", encoding="utf-8")
        exit_temp.replace(exit_path)
        return str(attempt_dir)


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def read_attempt(
    attempt_dir: Path,
    task: PlanTask,
    metadata: Mapping[str, Any],
    exit_code: int = 0,
) -> dict[str, Any]:
    """Normalize one attempt, requiring workload, usage, and server proof."""

    attempt_path = Path(attempt_dir)
    recorded_exit = attempt_path / "container-exit-code"
    if exit_code == 0 and recorded_exit.is_file():
        try:
            exit_code = int(recorded_exit.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            exit_code = 1
    summaries = sorted(attempt_path.rglob("*.summary.json"))
    summary_path = summaries[0] if len(summaries) == 1 else None
    summary = _load_json_object(summary_path) if summary_path is not None else None
    evidence_paths = sorted(attempt_path.rglob("server.evidence.json"))
    artifact_dir = (
        summary_path.parent if summary_path is not None
        else evidence_paths[0].parent if len(evidence_paths) == 1
        else attempt_path
    )
    evidence_path = evidence_paths[0] if len(evidence_paths) == 1 else artifact_dir / "server.evidence.json"
    evidence = _load_json_object(evidence_path)
    requested = _load_json_object(artifact_dir / "server.requested.json")
    server_info = _load_json_object(artifact_dir / "server.info.json")
    controller = _load_json_object(artifact_dir / "run_manifest.json")
    static = _static_config(_candidate(metadata, task))
    effective_static = dict(static)
    effective_static["max_running_requests"] = _effective_max_running_requests(task, static)
    server_command_path = artifact_dir / "server.command.sh"
    result = {
        "status": "INCONCLUSIVE",
        "reasons": [],
        "output_tokens_per_second": 0.0,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "candidate_id": task.candidate_id,
        "concurrency": task.concurrency,
        "max_running_requests": _effective_max_running_requests(task, static),
        "effective_static_config": effective_static,
        "launch_configuration": requested,
        "server_command_path": str(server_command_path) if server_command_path.is_file() else None,
        "evidence_fingerprint": _evidence_fingerprint(artifact_dir, summary_path, attempt_path),
    }
    reasons: list[str] = result["reasons"]
    if exit_code != 0:
        reasons.append(f"process_exit_{exit_code}")
    if len(summaries) > 1:
        reasons.append("summary_ambiguous")
    elif summary is None:
        reasons.append("summary_missing" if summary_path is None else "summary_invalid")
    if controller is None or controller.get("state") != "complete" or controller.get("exit_code") != 0:
        reasons.append("controller_completion_missing")
    computed_unsupported: list[str] = []
    computed_mismatched: list[str] = []
    if evidence is None:
        reasons.append("server_evidence_missing")
    else:
        unsupported = evidence.get("unsupported_parameters")
        mismatched = evidence.get("mismatched_parameters")
        if evidence.get("readiness") is not True:
            reasons.append("readiness_missing")
        if evidence.get("server_info_captured") is not True:
            reasons.append("server_info_missing")
        if requested is None or server_info is None:
            reasons.append("server_parameter_artifacts_missing")
        else:
            computed_unsupported, computed_mismatched = _compare_server_parameters(requested, server_info)
            if computed_unsupported:
                reasons.append("unsupported_server_parameters")
            if computed_mismatched:
                reasons.append("server_parameter_mismatch")
            expected_requested = _expected_requested_parameters(metadata, task)
            if any(key not in requested or requested[key] != value
                   for key, value in expected_requested.items()):
                reasons.append("requested_server_parameters_mismatch")
            if evidence.get("requested_server_parameters") != requested:
                reasons.append("server_evidence_inconsistent")
        if unsupported or computed_unsupported:
            reasons.append("unsupported_server_parameters")
        if mismatched or computed_mismatched:
            reasons.append("server_parameter_mismatch")
        if evidence.get("resolved") is not True and not unsupported and not mismatched:
            reasons.append("resolved_parameters_missing")
        # Derive exported values from the native server information, rather
        # than trusting the writer's duplicate resolved-value mapping.
        if requested is not None and server_info is not None:
            leaves = _server_info_leaves(server_info)
            for key in effective_static:
                if key not in requested:
                    continue
                for alias in _SERVER_INFO_ALIASES.get(key, (key,)):
                    if alias in leaves:
                        actual = leaves[alias]
                        if isinstance(requested[key], bool):
                            actual = requested[key] if _values_match(key, requested[key], actual) else actual
                        effective_static[key] = actual
                        break
        if evidence.get("resolved") is True and (computed_unsupported or computed_mismatched):
            reasons.append("server_evidence_inconsistent")

    formal = task.run_class not in {"smoke", "diagnostic"}
    if summary is not None:
        expected_sha = metadata.get("expected_source_sha256", metadata.get("source_sha256"))
        actual_sha = summary.get("source_sha256", summary.get("jsonl_sha256"))
        if expected_sha is None:
            reasons.append("expected_source_sha256_missing")
        elif actual_sha != expected_sha:
            reasons.append("source_sha256_mismatch")
        expected_count = metadata.get("expected_request_count", metadata.get("request_count"))
        expected_for_task = _expected_request_count_for_task(task, metadata)
        if expected_for_task is None:
            reasons.append("expected_request_count_missing")
        else:
            expected = expected_for_task
            successes = summary.get("successes")
            failures = summary.get("failures")
            if summary.get("request_count") != expected:
                reasons.append("request_count_mismatch")
            if formal and (
                isinstance(successes, bool) or not isinstance(successes, int)
                or isinstance(failures, bool) or not isinstance(failures, int)
                or successes + failures != expected
            ):
                reasons.append("request_set_incomplete")
        failures = summary.get("failures")
        if isinstance(failures, bool) or not isinstance(failures, int) or failures != 0:
            reasons.append("request_failures")
        successes = summary.get("successes")
        usage_available = summary.get("server_usage_available")
        usage_missing = summary.get("server_usage_missing", summary.get("usage_missing"))
        completion_tokens = summary.get("completion_tokens")
        if (
            isinstance(successes, bool) or not isinstance(successes, int)
            or isinstance(usage_available, bool) or not isinstance(usage_available, int)
            or usage_available < successes or usage_missing != 0
            or isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, (int, float))
            or not math.isfinite(completion_tokens) or completion_tokens <= 0
        ):
            reasons.append("usage_invalid")
        elapsed = summary.get("measured_seconds")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
            reasons.append("measured_seconds_invalid")
        score = summary.get("output_tokens_per_second")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score <= 0:
            reasons.append("output_tokens_per_second_invalid")
        else:
            result["output_tokens_per_second"] = float(score)
            if (
                isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
                and math.isfinite(elapsed) and elapsed > 0
                and isinstance(completion_tokens, (int, float)) and not isinstance(completion_tokens, bool)
                and math.isfinite(completion_tokens)
                and not math.isclose(score, completion_tokens / elapsed, rel_tol=1e-6, abs_tol=1e-9)
            ):
                reasons.append("throughput_inconsistent")
        for key in _summary_metric_fields():
            if key in summary:
                result[key] = summary[key]
        base_urls = summary.get("base_urls")
        if isinstance(base_urls, list):
            urls = base_urls
        else:
            urls = [summary.get("base_url")]
        deployment = static.get("deployment", {})
        instance_count = 1
        if isinstance(deployment, Mapping):
            try:
                instance_count = max(1, int(deployment.get("instance_count", 1)))
            except (TypeError, ValueError):
                instance_count = 1
        service_port = int(metadata.get("service_port", DockerConfig().service_port))
        allowed_ports = {service_port + offset for offset in range(instance_count)}
        if not urls or any(not _local_loopback_url(url, allowed_ports) for url in urls):
            reasons.append("formal_url_not_loopback")

    result["reasons"] = list(dict.fromkeys(reasons))
    if evidence is not None and (evidence.get("unsupported_parameters") or computed_unsupported):
        result["status"] = "UNSUPPORTED"
    elif exit_code != 0:
        result["status"] = "FAILED"
    elif not result["reasons"]:
        result["status"] = "VALID"
    if result["status"] != "VALID":
        phase = _failure_phase(result["status"], result["reasons"], summary, evidence)
        if phase is not None:
            result["failure_phase"] = phase
    return result


__all__ = ["ReplayAdapter", "read_attempt"]
