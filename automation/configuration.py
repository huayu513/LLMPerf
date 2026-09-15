"""Strict loader for the single-file automation configuration."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .docker_runtime import DockerRuntime, DockerTaskSpec
from .errors import ConfigError
from .types import DockerConfig


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


@dataclass(frozen=True)
class RunConfig:
    model_path: Path
    input_path: Path
    image: str
    output_dir: Path | None = None
    gpu_indexes: tuple[int, ...] | None = None
    search: SearchConfig = field(default_factory=SearchConfig)
    smoke: bool = False
    warmup: int = 0
    request_timeout: float = 3600.0
    ready_timeout: int = 3600
    model_overrides: dict[str, Any] = field(default_factory=dict)
    docker: DockerConfig = field(default_factory=DockerConfig)
    source_path: Path = Path()


_TOP_KEYS = {
    "model_path", "input_path", "image", "output_dir", "gpu_indexes",
    "search", "smoke", "warmup", "request_timeout", "ready_timeout",
    "model_overrides", "docker",
}
_SEARCH_KEYS = {
    "concurrency_max", "start_concurrency", "explore_request_limit",
    "promotion_tolerance", "repetitions", "max_trials", "max_seconds",
    "backends", "open_loop_scales",
}
_DOCKER_KEYS = {"name_prefix", "network_mode", "service_port", "shm_size", "ipc"}
_LINUX_EPHEMERAL_PORT_RANGE = (32768, 60999)


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


def _search(value: Any, label: str) -> SearchConfig:
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
    start_concurrency = None
    if data.get("start_concurrency") is not None:
        start_concurrency = _integer(data["start_concurrency"], f"{label}.start_concurrency", 1)
    promotion_tolerance = _number(
        data.get("promotion_tolerance", 0.05), f"{label}.promotion_tolerance", 0.0
    )
    if promotion_tolerance >= 1.0:
        raise ConfigError(f"{label}.promotion_tolerance must be < 1.0")
    return SearchConfig(
        concurrency_max=_integer(
            data.get("concurrency_max", 64), f"{label}.concurrency_max", 1
        ),
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


def load_config(path: Path) -> RunConfig:
    """Load and validate one JSON run configuration."""
    source_path = Path(path).expanduser().resolve()
    data = _read(source_path)
    _unknown(data, _TOP_KEYS, str(source_path))
    for required in ("model_path", "input_path", "image"):
        if required not in data:
            raise ConfigError(f"{source_path}: missing required field {required}")

    base = source_path.parent
    model_path = _path(data["model_path"], base, f"{source_path}: model_path")
    input_path = _path(data["input_path"], base, f"{source_path}: input_path")
    if not model_path.is_dir():
        raise ConfigError(f"model_path is not a directory: {model_path}")
    if not input_path.is_file():
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
    return RunConfig(
        model_path=model_path,
        input_path=input_path,
        image=image,
        output_dir=output_dir,
        gpu_indexes=gpu_indexes,
        smoke=_boolean(data.get("smoke", False), f"{source_path}: smoke"),
        search=_search(data.get("search", {}), f"{source_path}: search"),
        warmup=_integer(data.get("warmup", 0), f"{source_path}: warmup", 0),
        request_timeout=_number(
            data.get("request_timeout", 3600.0), f"{source_path}: request_timeout",
            0.0, strict=True,
        ),
        ready_timeout=_integer(
            data.get("ready_timeout", 3600), f"{source_path}: ready_timeout", 1
        ),
        model_overrides=dict(overrides),
        docker=_docker(data.get("docker", {}), image, f"{source_path}: docker"),
        source_path=source_path,
    )
