"""Separate static recognition, engine evidence, and actual runtime validation."""
from __future__ import annotations

from .errors import ConfigError
from .model_protocols import parser_candidates


def resolve_parsers(model, engine, capabilities):
    resolved = {}
    checks = {}
    provenance = model.raw.get("provenance", {})
    for name, catalog in (("tool_call_parser", "tool_parsers"), ("reasoning_parser", "reasoning_parsers")):
        value = getattr(model, name)
        explicit = provenance.get(name) == "model_overrides." + name
        candidates = (value,) if explicit and value else parser_candidates(value, engine, name)
        advertised = capabilities.get(catalog, [])
        selected = next((item for item in candidates if item in advertised), None)
        if not candidates:
            checks[name] = {"status": "NOT_REQUESTED", "explicit": explicit}
        elif selected is not None:
            checks[name] = {"status": "SUPPORTED", "value": selected, "evidence": "image parser choices"}
        elif advertised or (engine == "sglang" and
                            capabilities.get("parser_registry", {}).get(catalog, {}).get("status") != "unknown"):
            checks[name] = {"status": "UNSUPPORTED", "requested": list(candidates),
                            "reason": "configured image does not advertise parser " + repr(value) +
                                      "; use a compatible image or model_overrides"}
        else:
            selected = candidates[0]
            checks[name] = {"status": "UNKNOWN", "value": selected,
                            "reason": "image parser choices unavailable; native and runtime validation required"}
        resolved[name] = selected
    return resolved, checks


def assess_model_compatibility(model, environment, engine, workload=None):
    container = environment.container
    capabilities = container.get("capabilities", {})
    support = container.get("model_support", {})
    registry = support.get("registry", {})
    declared = model.raw.get("metadata", {})
    architectures = declared.get("architectures", model.raw.get("architectures", [])) or []
    config = support.get("config", {})
    native_architectures = config.get("architectures") if config.get("status") == "accepted" else None
    effective_architectures = native_architectures or architectures
    matches = sorted(set(effective_architectures) & set(registry.get("architectures", [])))
    architecture_check = {"status": "UNKNOWN", "requested": architectures,
                          "reason": "image model registry unavailable or model architecture unspecified"}
    if matches:
        architecture_check = {"status": "SUPPORTED", "matched": matches, "source": registry.get("source")}
    elif effective_architectures and registry.get("status") == "discovered":
        fallback = any("Transformers" in name for name in registry.get("architectures", []))
        architecture_check = {"status": "UNKNOWN" if fallback else "UNSUPPORTED", "requested": architectures,
                              "reason": "architecture absent from native registry" +
                                        ("; Transformers fallback requires runtime validation" if fallback else "")}
    resolved, parser_checks = resolve_parsers(model, engine, capabilities)
    requirements = workload.raw.get("requirements", {}) if workload is not None else {}
    checks = {"architecture": architecture_check,
              "config": {**config, "status": "SUPPORTED" if config.get("status") == "accepted" else "UNKNOWN"},
              **parser_checks,
              "encoding": {"status": "UNKNOWN", "protocol": model.raw.get("protocol", {}).get("id", "generic"),
                           "static_template_present": model.raw.get("protocol", {}).get("static_template_present"),
                           "reason": "template presence and parser names do not prove prompt encoding; smoke required"},
              "quantization": {"status": "UNKNOWN", "declared": model.raw.get("normalized", {}).get("quantization", {}),
                               "reason": "config parsing does not prove weight layout or GPU kernel support"},
              "runtime": {"status": "NOT_RUN", "weights_loaded": False}}
    errors = [name + ": " + check.get("reason", "unsupported")
              for name, check in checks.items() if check.get("status") == "UNSUPPORTED"]
    return {"version": 1, "status": "FAIL" if errors else "WARN", "engine": engine,
            "identity": {"checkpoint": model.checkpoint,
                         "model_type": declared.get("model_type", model.raw.get("model_type")),
                         "image": environment.image.to_dict() if hasattr(environment.image, "to_dict") else dict(environment.image),
                         "versions": container.get("versions", {})},
            "runtime_verified": False, "parsers": resolved, "requirements": requirements,
            "checks": checks, "errors": errors,
            "unknowns": list(model.raw.get("normalized", {}).get("unknowns", []))}


def require_model_compatibility(report):
    if report["errors"]:
        raise ConfigError("model/image compatibility failed: " + "; ".join(report["errors"]) +
                          "; inspect model-compatibility.json")
