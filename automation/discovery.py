"""Read-only checkpoint and captured-workload discovery."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from .errors import ConfigError
from .types import ModelManifest, WorkloadManifest


_OVERRIDE_KEYS = {
    "model_type", "architectures", "num_attention_heads", "num_hidden_layers",
    "is_moe", "tool_call_parser", "reasoning_parser", "chat_template_kwargs",
    "quantization", "profile_env",
}
_QWEN2_TYPES = {"qwen", "qwen2", "qwen2_moe"}
_QWEN3_TYPES = {
    "qwen3", "qwen3_moe", "qwen3_next", "qwen3_5", "qwen3_5_text",
    "qwen3_5_moe", "qwen3_5_moe_text",
}
_GLM_TYPES = {"glm4_moe", "glm4_moe_lite", "glm_moe_dsa"}
_DEEPSEEK3_TYPES = {"deepseek_v3"}
_DEEPSEEK4_TYPES = {"deepseek_v4"}
_QWEN2_ARCHS = {"QWenLMHeadModel", "Qwen2ForCausalLM", "Qwen2MoeForCausalLM"}
_QWEN3_ARCHS = {
    "Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "Qwen3NextForCausalLM",
    "Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM",
}
_GLM_ARCHS = {
    "Glm4MoeForCausalLM", "Glm4MoeLiteForCausalLM", "GlmMoeDsaForCausalLM",
}
_DEEPSEEK3_ARCHS = {"DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM"}
_DEEPSEEK4_ARCHS = {"DeepseekV4ForCausalLM"}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite number {token}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ConfigError(f"invalid {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must contain a JSON object: {path}")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"model metadata {name} must be a positive integer")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError(f"model override {name} must be a non-empty string or null")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ConfigError(f"model override {name} must be an object with string keys")
    return copy.deepcopy(value)


def _architectures(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ConfigError(
            "model metadata architectures must be an array of non-empty strings"
        )
    return tuple(value)


def _template_strings(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for name, template in value.items():
            result.extend(_template_strings(template, f"{label}.{name}"))
        return result
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            if isinstance(item, dict) and "template" in item:
                result.extend(
                    _template_strings(item["template"], f"{label}[{index}].template")
                )
            else:
                result.extend(_template_strings(item, f"{label}[{index}]"))
        return result
    raise ConfigError(f"{label} must contain a string chat template")


def _static_chat_templates(
    model_path: Path, metadata: dict[str, Any]
) -> tuple[str, tuple[str, ...]]:
    templates: list[str] = []
    sources: list[str] = []

    def add(value: Any, source: str) -> None:
        found = _template_strings(value, source)
        templates.extend(found)
        sources.extend(source for _ in found)

    if "chat_template" in metadata:
        add(metadata["chat_template"], "config.json:chat_template")
    tokenizer_config_path = model_path / "tokenizer_config.json"
    if tokenizer_config_path.is_file():
        tokenizer_config = _read_json(tokenizer_config_path, "tokenizer metadata")
        if "chat_template" in tokenizer_config:
            add(
                tokenizer_config["chat_template"],
                "tokenizer_config.json:chat_template",
            )
    root_template = model_path / "chat_template.jinja"
    if root_template.is_file():
        try:
            templates.append(root_template.read_text(encoding="utf-8"))
        except (OSError, UnicodeError) as exc:
            raise ConfigError(f"invalid chat template {root_template}: {exc}") from exc
        sources.append("chat_template.jinja")
    template_dir = model_path / "chat_templates"
    if template_dir.is_dir():
        for template_path in sorted(template_dir.glob("*.jinja")):
            try:
                templates.append(template_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError) as exc:
                raise ConfigError(
                    f"invalid chat template {template_path}: {exc}"
                ) from exc
            sources.append(template_path.relative_to(model_path).as_posix())
    return "\n".join(templates), tuple(sources)


def _infer_tool_parser_from_text(
    family: str, template: str, sources: tuple[str, ...]
) -> tuple[str, str]:
    source = ", ".join(sources) if sources else "no static chat template metadata"
    if family == "deepseek-v3":
        formats: list[str] = []
        if "<｜DSML｜function_calls>" in template:
            formats.append("deepseekv32")
        if (
            "<｜tool▁calls▁begin｜>" in template
            and "<｜tool▁sep｜>" in template
        ):
            if (
                "function<｜tool▁sep｜>" in template
                or chr(96) * 3 + "json" in template
            ):
                formats.append("deepseekv3")
            else:
                formats.append("deepseekv31")
        formats = list(dict.fromkeys(formats))
        if len(formats) == 1:
            return formats[0], f"static chat template markers ({source})"
        if len(formats) > 1:
            source = f"conflicting static chat template markers ({source})"
    elif family == "qwen3":
        if "<function=" in template and "<parameter=" in template:
            return "qwen3_coder", f"static chat template markers ({source})"
        if "<tool_call>" in template and "</tool_call>" in template:
            return "qwen", f"static chat template markers ({source})"

    raise ConfigError(
        f"cannot infer tool_call_parser for {family} from {source}; "
        "set model_overrides.tool_call_parser explicitly"
    )


def _classify(
    model_type: str, architectures: tuple[str, ...]
) -> tuple[str, str | None, str | None]:
    arch_set = set(architectures)
    if model_type in _DEEPSEEK4_TYPES or arch_set & _DEEPSEEK4_ARCHS:
        return "deepseek-v4", "deepseekv4", "deepseek-v4"
    if model_type in _DEEPSEEK3_TYPES or arch_set & _DEEPSEEK3_ARCHS:
        tool_parser = (
            "deepseekv32"
            if "DeepseekV32ForCausalLM" in arch_set
            else None
        )
        return "deepseek-v3", tool_parser, "deepseek-v3"
    if model_type in _GLM_TYPES or arch_set & _GLM_ARCHS:
        return "glm", "glm", "glm45"
    if model_type in _QWEN3_TYPES or arch_set & _QWEN3_ARCHS:
        return "qwen3", None, "qwen3"
    if model_type in _QWEN2_TYPES or arch_set & _QWEN2_ARCHS:
        return "qwen2", "qwen", None
    details = f"model_type={model_type!r}, architectures={list(architectures)!r}"
    raise ConfigError(
        "unsupported model metadata (" + details + "); provide explicit "
        "model_overrides for a supported Qwen, GLM, or DeepSeek architecture"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_inventory(root: Path) -> dict[str, Any]:
    metadata_names = {
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
        "preprocessor_config.json", "processor_config.json", "chat_template.jinja",
    }
    metadata_files: dict[str, dict[str, Any]] = {}
    weight_files: list[dict[str, Any]] = []
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(root).as_posix()
        stat = item.stat()
        if item.suffix in {".safetensors", ".bin"}:
            weight_files.append({
                "path": relative,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
        elif item.name in metadata_names or item.name.endswith(".index.json"):
            metadata_files[relative] = {
                "size": stat.st_size,
                "sha256": _sha256_file(item),
            }
    payload = {
        "metadata_files": metadata_files,
        "weight_files": weight_files,
        "metadata_identity": "sha256",
        "weight_identity": "filename-size-mtime_ns",
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _quantization(metadata: dict[str, Any]) -> tuple[str | None, str]:
    value = metadata.get("quantization")
    if isinstance(value, str) and value:
        return value, "config.json:quantization"
    config = metadata.get("quantization_config")
    if config is None:
        return None, "not present"
    if not isinstance(config, dict):
        raise ConfigError("model metadata quantization_config must be an object")
    for key in ("quant_method", "format", "quantization_method"):
        method = config.get(key)
        if isinstance(method, str) and method:
            return method, f"config.json:quantization_config.{key}"
    return None, "config.json:quantization_config (method unspecified)"


def discover_model(
    model_path: Path, overrides: dict[str, Any] | None = None
) -> ModelManifest:
    """Discover a supported checkpoint from static config.json metadata."""
    original = str(model_path)
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise ConfigError(f"model checkpoint is not a directory: {path}")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ConfigError(f"model checkpoint has no config.json: {path}")
    metadata = _read_json(config_path, "model metadata")
    supplied = {} if overrides is None else overrides
    if not isinstance(supplied, dict):
        raise ConfigError("model_overrides must be an object")
    unknown = sorted(set(supplied) - _OVERRIDE_KEYS)
    if unknown:
        raise ConfigError(f"unknown model override keys: {', '.join(unknown)}")
    override = copy.deepcopy(supplied)

    text_config = metadata.get("text_config")
    if text_config is not None and not isinstance(text_config, dict):
        raise ConfigError("model metadata text_config must be an object")
    structural = {**metadata, **(text_config or {})}
    provenance: dict[str, str] = {}

    model_type_value = override.get("model_type", metadata.get("model_type"))
    if not isinstance(model_type_value, str) or not model_type_value:
        raise ConfigError("model metadata model_type must be a non-empty string")
    model_type = model_type_value.lower()
    provenance["model_type"] = (
        "model_overrides.model_type"
        if "model_type" in override
        else "config.json:model_type"
    )
    architectures = _architectures(
        override.get("architectures", metadata.get("architectures"))
    )
    provenance["architectures"] = (
        "model_overrides.architectures"
        if "architectures" in override
        else "config.json:architectures"
    )
    family, default_tool_parser, default_reasoning_parser = _classify(
        model_type, architectures
    )
    if "tool_call_parser" in override:
        tool_call_parser = _optional_string(
            override["tool_call_parser"], "tool_call_parser"
        )
        provenance["tool_call_parser"] = "model_overrides.tool_call_parser"
    else:
        template_metadata = {**metadata}
        template, sources = _static_chat_templates(path, template_metadata)
        if default_tool_parser is not None:
            tool_call_parser = default_tool_parser
            tool_parser_source = "unambiguous architecture mapping"
        else:
            tool_call_parser, tool_parser_source = _infer_tool_parser_from_text(
                family, template, sources
            )
        provenance["tool_call_parser"] = tool_parser_source

    heads = _positive_int(
        override.get("num_attention_heads", structural.get("num_attention_heads")),
        "num_attention_heads",
    )
    layers = _positive_int(
        override.get("num_hidden_layers", structural.get("num_hidden_layers")),
        "num_hidden_layers",
    )
    provenance["num_attention_heads"] = (
        "model_overrides.num_attention_heads"
        if "num_attention_heads" in override
        else "config.json:num_attention_heads"
    )
    provenance["num_hidden_layers"] = (
        "model_overrides.num_hidden_layers"
        if "num_hidden_layers" in override
        else "config.json:num_hidden_layers"
    )

    expert_keys = ("n_routed_experts", "num_experts", "num_local_experts")
    inferred_moe = (
        family in {"deepseek-v3", "deepseek-v4", "glm"}
        or any(
            type(structural.get(key)) is int and structural[key] > 0
            for key in expert_keys
        )
        or "moe" in model_type
        or any("Moe" in arch for arch in architectures)
    )
    is_moe = override.get("is_moe", inferred_moe)
    if type(is_moe) is not bool:
        raise ConfigError("model override is_moe must be a boolean")
    provenance["is_moe"] = (
        "model_overrides.is_moe"
        if "is_moe" in override
        else "derived from model_type, architectures, and expert metadata"
    )

    quantization, quantization_source = _quantization(metadata)
    if "quantization" in override:
        quantization = _optional_string(override["quantization"], "quantization")
        quantization_source = "model_overrides.quantization"
    provenance["quantization"] = quantization_source

    reasoning_parser = _optional_string(
        override.get("reasoning_parser", default_reasoning_parser),
        "reasoning_parser",
    )
    provenance["reasoning_parser"] = (
        "model_overrides.reasoning_parser"
        if "reasoning_parser" in override
        else "architecture mapping"
    )

    chat_template_kwargs = _mapping(
        override.get("chat_template_kwargs", {}), "chat_template_kwargs"
    )
    provenance["chat_template_kwargs"] = (
        "model_overrides.chat_template_kwargs"
        if "chat_template_kwargs" in override
        else "safe empty default"
    )
    profile_env = _mapping(override.get("profile_env", {}), "profile_env")
    if any(not isinstance(value, str) for value in profile_env.values()):
        raise ConfigError("model override profile_env values must be strings")

    raw = {
        "metadata": copy.deepcopy(metadata),
        "inventory": _model_inventory(path),
        "model_type": model_type,
        "architectures": list(architectures),
        "family": family,
        "num_attention_heads": heads,
        "num_hidden_layers": layers,
        "is_moe": is_moe,
        "provenance": provenance,
    }
    return ModelManifest(
        id=path.name,
        checkpoint=str(path),
        checkpoint_original=original,
        checkpoint_kind="directory",
        version=str(
            metadata.get("_commit_hash")
            or metadata.get("transformers_version")
            or ""
        ),
        served_model_name=path.name,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        chat_template_kwargs=chat_template_kwargs,
        quantization=quantization,
        profile_env=profile_env,
        raw=raw,
    )


def _replay_script() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "replay"
        / "prepare_jsonl_replay.py"
    )


@lru_cache(maxsize=1)
def _replay_module() -> ModuleType:
    script = _replay_script()
    spec = importlib.util.spec_from_file_location(
        "s1slow_prepare_jsonl_replay", script
    )
    if spec is None or spec.loader is None:
        raise ConfigError(f"cannot load bundled replay validator: {script}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ConfigError(f"cannot load bundled replay validator: {exc}") from exc
    return module


def inspect_workload(input_path: Path) -> WorkloadManifest:
    """Validate captured JSONL without rewriting it and return immutable facts."""
    original = str(input_path)
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"captured JSONL does not exist: {path}")
    validator = _replay_module().validate_request
    digest = hashlib.sha256()
    count = 0
    models: Counter[str] = Counter()
    request_ids: set[str] = set()
    source_message_ids: set[str] = set()
    try:
        with path.open("rb") as source:
            for line_number, line in enumerate(source, 1):
                digest.update(line)
                if not line.strip():
                    raise ValueError(f"line {line_number}: empty line")
                try:
                    payload = json.loads(line)
                except Exception as exc:
                    raise ValueError(
                        f"line {line_number}: invalid JSON: {exc}"
                    ) from exc
                request, _ = validator(
                    payload, line_number, require_stream_usage=False
                )
                request_id = payload["request_id"]
                source_message_id = str(payload["source_message_id"])
                if request_id in request_ids:
                    raise ValueError(
                        f"line {line_number}: duplicate request_id {request_id}"
                    )
                if source_message_id in source_message_ids:
                    raise ValueError(
                        f"line {line_number}: duplicate source_message_id "
                        f"{source_message_id}"
                    )
                request_ids.add(request_id)
                source_message_ids.add(source_message_id)
                models[request["model"]] += 1
                count += 1
    except (OSError, ValueError) as exc:
        raise ConfigError(f"invalid captured JSONL {path}: {exc}") from exc
    if count == 0:
        raise ConfigError(
            f"invalid captured JSONL {path}: JSONL has no records"
        )
    if len(models) != 1:
        aliases = ", ".join(sorted(models))
        raise ConfigError(
            f"captured JSONL uses mixed request model aliases ({aliases}); "
            "all requests must use one model alias"
        )
    model_alias = next(iter(models))
    raw = {
        "count": count,
        "models": dict(models),
        "model": model_alias,
        "validation": (
            "benchmarks/replay/prepare_jsonl_replay.py:validate_request"
        ),
        "source_bytes_preserved": True,
    }
    return WorkloadManifest(
        id=path.stem,
        jsonl=str(path),
        jsonl_original=original,
        jsonl_resolved=str(path),
        sha256=digest.hexdigest(),
        request_mode="raw_request",
        order="fixed",
        raw=raw,
    )


def prepare_workload(input_path: Path, index_path: Path) -> dict[str, Any]:
    """Create the canonical token-free replay index and return its contents."""
    workload = inspect_workload(input_path)
    source_path = Path(workload.jsonl_resolved or workload.jsonl)
    output_path = Path(index_path).expanduser().resolve()
    if source_path == output_path:
        raise ConfigError(
            "replay index path must differ from the captured JSONL path"
        )
    command = [
        sys.executable,
        "-B",
        str(_replay_script()),
        "--jsonl",
        str(source_path),
        "--output",
        str(output_path),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, shell=False
    )
    if result.returncode != 0:
        reason = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise ConfigError(f"failed to prepare replay index: {reason}")
    index = _read_json(output_path, "replay index")
    expected = {
        "format": "s1_jsonl_chat_requests",
        "jsonl_sha256": workload.sha256,
        "count": workload.raw["count"],
        "models": workload.raw["models"],
    }
    for key, value in expected.items():
        if index.get(key) != value:
            raise ConfigError(
                f"replay index {key} does not match inspected workload"
            )
    if index.get("model_path") is not None or index.get("input_tokens") is not None:
        raise ConfigError("replay index unexpectedly contains tokenizer statistics")
    return index
