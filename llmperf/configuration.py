"""Strict loader for the single-file LLMPerf configuration."""
from __future__ import annotations

import json
import math
import os
from urllib.parse import urlsplit
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .docker_runtime import DockerRuntime, DockerTaskSpec
from .errors import ConfigError
from .types import DockerConfig


@dataclass(frozen=True)
class AdaptiveConfig:
    candidate_pool_limit: int = 256
    batch_size: int = 3
    max_advisor_calls: int = 12
    final_top_k: int = 2


@dataclass(frozen=True)
class SearchConfig:
    concurrency_max: int = 64
    start_concurrency: int | None = None
    explore_request_limit: int = 256
    promotion_tolerance: float = 0.05
    repetitions: int = 3
    max_trials: int = 64
    max_seconds: float = 14400
    backends: tuple[str, ...] | None = None
    open_loop_scales: tuple[float, ...] | None = None
    strategy: str = "strict"
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)


@dataclass(frozen=True)
class RunConfig:
    model_path: Path
    input_path: Path | None
    image: str
    output_dir: Path | None = None
    gpu_indexes: tuple[int, ...] | None = None
    search: SearchConfig = field(default_factory=SearchConfig)
    smoke: bool = False
    warmup: int = 0
    request_timeout: float = 6666.0
    ready_timeout: int = 3600
    model_overrides: dict[str, Any] = field(default_factory=dict)
    docker: DockerConfig = field(default_factory=DockerConfig)
    scenario: dict[str, Any] | None = None
    advisor: dict[str, Any] | None = None
    source_path: Path = Path()
    engine: str = "sglang"
    engine_options: dict[str, Any] = field(default_factory=dict)


_TOP_KEYS = {
    "model_path", "input_path", "image", "output_dir", "gpu_indexes",
    "search", "smoke", "warmup", "request_timeout", "ready_timeout",
    "model_overrides", "docker", "scenario", "advisor", "engine", "engine_options",
}
_SEARCH_KEYS = {
    "concurrency_max", "start_concurrency", "explore_request_limit",
    "promotion_tolerance", "repetitions", "max_trials", "max_seconds",
    "backends", "open_loop_scales", "strategy", "adaptive",
}
_DOCKER_KEYS = {"name_prefix", "network_mode", "service_port", "shm_size", "ipc"}
_LINUX_EPHEMERAL_PORT_RANGE = (32768, 60999)
ADVISOR_TIMEOUT_SECONDS = 2000


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite number {value}")


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file does not exist: {path}")
    if path.suffix.lower() != ".json":
        raise ConfigError(f"configuration must be a JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ConfigError(f"invalid configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"configuration root must be an object: {path}")
    return value


def _unknown(data: dict[str, Any], allowed: set[str], label: str) -> None:
    keys = sorted(set(data) - allowed)
    if keys:
        raise ConfigError(f"{label}: unknown keys: {', '.join(keys)}")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{label} keys must be strings")
    return value


def _string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ConfigError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, minimum: float, *, strict: bool = False) -> float:
    if type(value) not in (int, float):
        raise ConfigError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (strict and result == minimum):
        relation = ">" if strict else ">="
        raise ConfigError(f"{label} must be a finite number {relation} {minimum}")
    return result


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{label} must be a boolean")
    return value


def _path(value: Any, base: Path, label: str) -> Path:
    text = _string(value, label)
    result = Path(os.path.expandvars(text)).expanduser()
    if not result.is_absolute():
        result = base / result
    return result.resolve()


def _optional_strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{label} must be an array")
    result = tuple(_string(item, f"{label} item") for item in value)
    if len(set(result)) != len(result):
        raise ConfigError(f"{label} must not contain duplicates")
    return result


def _search(value: Any, label: str, *, default_strategy: str = "strict",
            strategy_override: str | None = None) -> SearchConfig:
    data = _object(value, label)
    _unknown(data, _SEARCH_KEYS, label)
    repetitions = _integer(data.get("repetitions", 3), f"{label}.repetitions", 1)
    max_trials = _integer(data.get("max_trials", 64), f"{label}.max_trials", 1)
    if max_trials < repetitions + 2:
        raise ConfigError(f"{label}.max_trials must be at least repetitions + 2")
    backends = None
    if "backends" in data:
        backends = _optional_strings(data["backends"], f"{label}.backends")
    scales = None
    if "open_loop_scales" in data:
        raw_scales = data["open_loop_scales"]
        if not isinstance(raw_scales, list):
            raise ConfigError(f"{label}.open_loop_scales must be an array")
        scales = tuple(
            _number(item, f"{label}.open_loop_scales item", 0.0, strict=True)
            for item in raw_scales
        )
        if len(set(scales)) != len(scales):
            raise ConfigError(f"{label}.open_loop_scales must not contain duplicates")
    concurrency_max = _integer(
        data.get("concurrency_max", 64), f"{label}.concurrency_max", 1
    )
    start_concurrency = None
    if data.get("start_concurrency") is not None:
        start_concurrency = _integer(data["start_concurrency"], f"{label}.start_concurrency", 1)
        if start_concurrency > concurrency_max:
            raise ConfigError(
                f"{label}.start_concurrency must be <= {label}.concurrency_max"
            )
    promotion_tolerance = _number(
        data.get("promotion_tolerance", 0.05), f"{label}.promotion_tolerance", 0.0
    )
    if promotion_tolerance >= 1.0:
        raise ConfigError(f"{label}.promotion_tolerance must be < 1.0")
    strategy = _string(data.get("strategy", default_strategy), f"{label}.strategy")
    if strategy not in {"strict", "adaptive"}:
        raise ConfigError(f"{label}.strategy must be strict or adaptive")
    if strategy_override is not None:
        strategy = _string(strategy_override, f"{label}.strategy")
        if strategy not in {"strict", "adaptive"}:
            raise ConfigError(f"{label}.strategy must be strict or adaptive")
    adaptive_data = _object(data.get("adaptive", {}), f"{label}.adaptive")
    _unknown(adaptive_data, set(AdaptiveConfig.__dataclass_fields__), f"{label}.adaptive")
    adaptive = AdaptiveConfig(**{
        key: _integer(adaptive_data.get(key, value), f"{label}.adaptive.{key}",
                      0 if key == "max_advisor_calls" else 1)
        for key, value in vars(AdaptiveConfig()).items()
    })
    if adaptive.candidate_pool_limit < 2:
        raise ConfigError(f"{label}.adaptive.candidate_pool_limit must be >= 2")
    if adaptive.batch_size > 8 or adaptive.final_top_k > 5:
        raise ConfigError(f"{label}.adaptive requires batch_size <= 8 and final_top_k <= 5")
    if strategy == "adaptive" and max_trials < (adaptive.final_top_k + 1) * repetitions + 2:
        raise ConfigError(f"{label}.max_trials must reserve finalist and baseline repeats plus two explorations")
    return SearchConfig(
        concurrency_max=concurrency_max,
        start_concurrency=start_concurrency,
        explore_request_limit=_integer(
            data.get("explore_request_limit", 256), f"{label}.explore_request_limit", 0
        ),
        promotion_tolerance=promotion_tolerance,
        repetitions=repetitions,
        max_trials=max_trials,
        max_seconds=_number(
            data.get("max_seconds", 14400), f"{label}.max_seconds", 0.0, strict=True
        ),
        backends=backends,
        open_loop_scales=scales,
        strategy=strategy,
        adaptive=adaptive,
    )


def _docker(value: Any, image: str, label: str) -> DockerConfig:
    data = _object(value, label)
    _unknown(data, _DOCKER_KEYS, label)
    defaults = DockerConfig()
    config = DockerConfig(
        name_prefix=_string(data.get("name_prefix", defaults.name_prefix), f"{label}.name_prefix"),
        network_mode=_string(data.get("network_mode", defaults.network_mode), f"{label}.network_mode"),
        service_port=_integer(data.get("service_port", defaults.service_port), f"{label}.service_port", 1),
        shm_size=_string(data.get("shm_size", defaults.shm_size), f"{label}.shm_size"),
        ipc=_string(data.get("ipc", defaults.ipc), f"{label}.ipc"),
    )
    if config.service_port > 65535:
        raise ConfigError(f"{label}.service_port must be <= 65535")
    ephemeral_start, ephemeral_end = _LINUX_EPHEMERAL_PORT_RANGE
    if ephemeral_start <= config.service_port <= ephemeral_end:
        raise ConfigError(
            f"{label}.service_port must be outside the Linux ephemeral port range "
            f"{ephemeral_start}-{ephemeral_end}; SGLang allocates internal ports "
            "from that range during startup"
        )
    if len(config.name_prefix) > 114:
        raise ConfigError(f"{label}.name_prefix must be at most 114 characters")
    try:
        DockerRuntime._validate_name(config.name_prefix)
        DockerRuntime._validate_runtime_options(DockerTaskSpec(
            image=image,
            name=config.name_prefix,
            network_mode=config.network_mode,
            shm_size=config.shm_size,
            ipc=config.ipc,
        ))
    except ValueError as exc:
        raise ConfigError(f"invalid {label}: {exc}") from exc
    return config


def validate_scenario(value: Any) -> dict[str, Any]:
    data = _object(value, "scenario")
    allowed = {"max_input_tokens", "max_output_tokens", "target_concurrency", "priority"}
    _unknown(data, allowed, "scenario")
    result = {
        "max_input_tokens": (_integer(data["max_input_tokens"], "scenario.max_input_tokens", 1)
                             if data.get("max_input_tokens") is not None else None),
        "max_output_tokens": (_integer(data["max_output_tokens"], "scenario.max_output_tokens", 1)
                              if data.get("max_output_tokens") is not None else None),
        "target_concurrency": _integer(data.get("target_concurrency"), "scenario.target_concurrency", 1),
        "priority": _string(data.get("priority"), "scenario.priority"),
    }
    if result["priority"] not in {"latency", "throughput"}:
        raise ConfigError("scenario.priority must be latency or throughput")
    return result


def resolve_scenario(value: Any, search: SearchConfig,
                     gpu_indexes: tuple[int, ...] | None = None) -> dict[str, Any]:
    data = dict(_object(value, "scenario"))
    if data.get("target_concurrency") is None:
        data["target_concurrency"] = search.start_concurrency or min(
            search.concurrency_max, max(8, 8 * len(gpu_indexes or (0,))))
    if data.get("priority") is None:
        data["priority"] = "throughput"
    return validate_scenario(data)


def validate_advisor(value: Any) -> dict[str, Any]:
    data = _object(value, "advisor")
    _unknown(data, {"base_url", "model", "model_pool", "authorization", "api_key_env", "timeout_seconds", "thinking", "reasoning_effort", "task_max_tokens"}, "advisor")
    url = _string(data.get("base_url"), "advisor.base_url").rstrip("/")
    try:
        parts = urlsplit(url)
        valid = (parts.scheme in {"http", "https"} and bool(parts.hostname)
                 and parts.port != 0 and not parts.username and not parts.password
                 and not parts.query and not parts.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise ConfigError("advisor.base_url must be an HTTP(S) endpoint without credentials or query")
    budgets = _object(data.get("task_max_tokens", {}), "advisor.task_max_tokens")
    _unknown(budgets, {"scene", "capabilities", "candidates", "explain", "next_experiments"}, "advisor.task_max_tokens")
    thinking = _boolean(data.get("thinking", False), "advisor.thinking")
    effort = _string(data.get("reasoning_effort", "max" if thinking else "none"), "advisor.reasoning_effort")
    if effort not in {"none", "high", "max"} or thinking == (effort == "none"):
        raise ConfigError("advisor.reasoning_effort must be high/max with thinking enabled, or none with thinking disabled")
    return {
        "base_url": url,
        "model": _string(data.get("model"), "advisor.model"),
        "model_pool": _string(data.get("model_pool", ""), "advisor.model_pool", nonempty=False),
        "authorization": _string(data.get("authorization", ""), "advisor.authorization", nonempty=False),
        "api_key_env": _string(data.get("api_key_env", ""), "advisor.api_key_env", nonempty=False),
        "timeout_seconds": _number(data.get("timeout_seconds", ADVISOR_TIMEOUT_SECONDS), "advisor.timeout_seconds", 0, strict=True),
        "thinking": thinking,
        "reasoning_effort": effort,
        "task_max_tokens": {key: _integer(value, f"advisor.task_max_tokens.{key}", 1)
                            for key, value in budgets.items()},
    }


def _engine_options(value: Any, engine: str) -> dict[str, Any]:
    data = _object(value, "engine_options")
    if engine == "sglang":
        _unknown(data, set(), "engine_options for sglang")
        return {}
    allowed = {
        "max_model_len", "gpu_memory_utilization", "max_num_seqs",
        "max_num_batched_tokens", "enforce_eager", "enable_expert_parallel",
        "all2all_backend", "moe_backend", "speculative_config",
        "tensor_parallel_size", "data_parallel_size", "pipeline_parallel_size",
    }
    _unknown(data, allowed, "engine_options")
    result = {}
    for key, value in data.items():
        label = f"engine_options.{key}"
        if key in {"max_model_len", "max_num_seqs", "max_num_batched_tokens",
                   "tensor_parallel_size", "data_parallel_size", "pipeline_parallel_size"}:
            result[key] = _integer(value, label, 1)
        elif key == "gpu_memory_utilization":
            result[key] = _number(value, label, 0.0, strict=True)
            if result[key] >= 1:
                raise ConfigError(f"{label} must be < 1")
        elif key in {"enforce_eager", "enable_expert_parallel"}:
            result[key] = _boolean(value, label)
        elif key == "speculative_config":
            spec = _object(value, label)
            _unknown(spec, {"method", "num_speculative_tokens"}, label)
            if set(spec) != {"method", "num_speculative_tokens"}:
                raise ConfigError(f"{label} requires method and num_speculative_tokens")
            result[key] = {
                "method": _string(spec["method"], f"{label}.method"),
                "num_speculative_tokens": _integer(
                    spec["num_speculative_tokens"], f"{label}.num_speculative_tokens", 1),
            }
        else:
            result[key] = _string(value, label)
    return result


def load_config(path: Path, *, tuning: bool = False,
                default_strategy: str = "strict", strategy: str | None = None) -> RunConfig:
    """Load and validate one JSON run configuration."""
    source_path = Path(path).expanduser().resolve()
    data = _read(source_path)
    _unknown(data, _TOP_KEYS, str(source_path))
    for required in (("model_path", "image") if tuning else ("model_path", "input_path", "image")):
        if required not in data:
            raise ConfigError(f"{source_path}: missing required field {required}")

    base = source_path.parent
    model_path = _path(data["model_path"], base, f"{source_path}: model_path")
    input_path = (_path(data["input_path"], base, f"{source_path}: input_path")
                  if data.get("input_path") is not None else None)
    if not model_path.is_dir():
        raise ConfigError(f"model_path is not a directory: {model_path}")
    if input_path is not None and not input_path.is_file():
        raise ConfigError(f"input_path is not a file: {input_path}")
    image = _string(data["image"], f"{source_path}: image")
    try:
        DockerRuntime._validate_image(image)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    output_dir = None
    if data.get("output_dir") is not None:
        output_dir = _path(data["output_dir"], base, f"{source_path}: output_dir")

    gpu_indexes = None
    if data.get("gpu_indexes") is not None:
        raw_indexes = data["gpu_indexes"]
        if not isinstance(raw_indexes, list):
            raise ConfigError(f"{source_path}: gpu_indexes must be an array")
        if not raw_indexes:
            raise ConfigError(f"{source_path}: gpu_indexes must not be empty")
        gpu_indexes = tuple(
            _integer(item, f"{source_path}: gpu_indexes item", 0) for item in raw_indexes
        )
        if len(set(gpu_indexes)) != len(gpu_indexes):
            raise ConfigError(f"{source_path}: gpu_indexes must be unique")

    overrides = _object(data.get("model_overrides", {}), f"{source_path}: model_overrides")
    search = _search(data.get("search", {}), f"{source_path}: search",
                     default_strategy=default_strategy, strategy_override=strategy)
    scenario = resolve_scenario(data["scenario"], search, gpu_indexes) if "scenario" in data else None
    advisor = validate_advisor(data["advisor"]) if "advisor" in data else None
    engine = _string(data.get("engine", "sglang"), "engine")
    if engine not in {"sglang", "vllm"}:
        raise ConfigError("engine must be sglang or vllm")
    return RunConfig(
        model_path=model_path,
        input_path=input_path,
        image=image,
        output_dir=output_dir,
        gpu_indexes=gpu_indexes,
        smoke=_boolean(data.get("smoke", False), f"{source_path}: smoke"),
        search=search,
        warmup=_integer(data.get("warmup", 0), f"{source_path}: warmup", 0),
        request_timeout=_number(
            data.get("request_timeout", 6666.0), f"{source_path}: request_timeout",
            0.0, strict=True,
        ),
        ready_timeout=_integer(
            data.get("ready_timeout", 3600), f"{source_path}: ready_timeout", 1
        ),
        model_overrides=dict(overrides),
        docker=_docker(data.get("docker", {}), image, f"{source_path}: docker"),
        scenario=scenario,
        advisor=advisor,
        source_path=source_path,
        engine=engine,
        engine_options=_engine_options(data.get("engine_options", {}), engine),
    )
