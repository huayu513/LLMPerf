"""Bounded advisor interpretation of immutable image evidence, never shell commands."""
from __future__ import annotations

import copy
import time

from .advisor import LocalAdvisor, ask_advisor
from .artifacts import write_json_atomic
from .configuration import ADVISOR_TIMEOUT_SECONDS
from .evidence import redact
from .errors import ConfigError
from .feature_catalog import validate_value


def analyze_capabilities(catalog, context, client, root, *, max_rounds=3):
    started = time.monotonic()
    timeout = min(float(getattr(client, "settings", {}).get("timeout_seconds", ADVISOR_TIMEOUT_SECONDS)),
                  ADVISOR_TIMEOUT_SECONDS)
    parameters = catalog["parameters"]
    evidence = {item["id"]: item for item in parameters.values()}
    evidence.update({"source:" + name: document for name, document in catalog["documents"].items()})
    evidence.update({"nested:" + name: document for name, document in catalog["nested"].items()})
    visible = {key: value for key, value in evidence.items()
               if key.startswith("nested:") or key.startswith("parameter:") and value.get("integrated")}
    report = {"version": 1, "status": "FALLBACK", "engine": catalog["engine"], "rounds": [],
              "features": [], "experiments": [], "hypotheses": [], "unknowns": [], "rejections": [],
              "budget": {"max_rounds": max_rounds, "timeout_seconds": timeout},
              "semantics": "advisor hypotheses are not native validation or runtime verification"}
    secrets = ()
    if isinstance(client, LocalAdvisor):
        authorization = client.settings.get("authorization", "")
        secrets = (authorization, authorization.removeprefix("Bearer "))

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        write_json_atomic(root / "capability-analysis.json", redact(report, secrets))

    for index in range(max_rounds):
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            report["reason"] = "capability_analysis_time_budget_exhausted"
            break
        request = {"engine": catalog["engine"], "context": context,
                   "available_evidence": list(evidence), "evidence": visible,
                   "previous_rejections": report["rejections"][-20:],
                   "remaining_rounds": max_rounds - index, "remaining_seconds": remaining}
        entry = {"round": index + 1, "input": request, "status": "REQUESTED"}
        report["rounds"].append(entry)
        save()
        active = LocalAdvisor({**client.settings, "timeout_seconds": remaining}) if isinstance(client, LocalAdvisor) else client
        try:
            def validate(reply):
                if set(reply) - {"read_evidence", "features", "experiments", "hypotheses", "unknowns"}:
                    raise ValueError("unknown capability analysis fields")
                if not any(key in reply for key in ("read_evidence", "features", "experiments", "unknowns")):
                    raise ValueError("capability analysis must request evidence or return findings")
                for key in ("read_evidence", "features", "experiments", "hypotheses", "unknowns"):
                    if key in reply and (not isinstance(reply[key], list) or len(reply[key]) > 32):
                        raise ValueError(f"{key} must be a bounded list")

            reply = ask_advisor(active, "capabilities", request, validate)
            validate(reply)
            entry.update(status="PASS", response=reply)
            reads = reply.get("read_evidence", [])
            for ident in reads:
                if isinstance(ident, str) and ident in evidence:
                    visible[ident] = evidence[ident]
                else:
                    report["rejections"].append({"reason": "unknown_read_only_evidence_id", "id": str(ident)[:200]})
            for row in reply.get("features", []):
                try:
                    if not isinstance(row, dict) or set(row) - {"parameter", "values", "priority", "reason", "evidence_ids"}:
                        raise ValueError("invalid feature proposal")
                    name = row["parameter"]
                    if not isinstance(name, str) or not isinstance(row.get("values"), list) or not 1 <= len(row["values"]) <= 8:
                        raise ValueError("feature requires a parameter and 1..8 values")
                    references = row.get("evidence_ids", [])
                    if not references or not all(isinstance(ident, str) and ident in visible for ident in references):
                        raise ValueError("feature needs references to evidence already provided")
                    for value in row["values"]:
                        validate_value(name, value, catalog)
                    if type(row.get("priority", 0)) is not int or not isinstance(row.get("reason", ""), str):
                        raise ValueError("invalid priority or reason")
                    report["features"] = [item for item in report["features"] if item["parameter"] != name]
                    report["features"].append({**copy.deepcopy(row), "status": "proposed", "runtime_verified": False})
                except (ValueError, KeyError, TypeError) as exc:
                    report["rejections"].append({"reason": str(exc)[:500]})
            for key in ("hypotheses", "unknowns"):
                report[key].extend(item[:2000] for item in reply.get(key, []) if isinstance(item, str))
            for row in reply.get("experiments", []):
                try:
                    if not isinstance(row, dict) or set(row) - {"params", "priority", "reason", "evidence_ids"}:
                        raise ValueError("invalid experiment proposal")
                    params = row.get("params")
                    if not isinstance(params, dict) or not 1 <= len(params) <= 8:
                        raise ValueError("experiment needs 1..8 native parameters")
                    if not row.get("evidence_ids") or not all(ident in visible for ident in row["evidence_ids"]):
                        raise ValueError("experiment needs provided evidence references")
                    for name, value in params.items():
                        validate_value(name, value, catalog)
                    if type(row.get("priority", 0)) is not int or not isinstance(row.get("reason", ""), str):
                        raise ValueError("invalid experiment priority or reason")
                    report["experiments"].append({**copy.deepcopy(row), "runtime_verified": False})
                except (ValueError, KeyError, TypeError) as exc:
                    report["rejections"].append({"reason": str(exc)[:500]})
            report["status"] = "PASS"
            if not reads:
                break
        except (ConfigError, ValueError, TypeError, KeyError, OSError) as exc:
            entry.update(status="FALLBACK", reason=str(exc)[:1000])
            report["reason"] = "capability_advisor_unavailable_or_invalid"
            break
        finally:
            if isinstance(active, LocalAdvisor):
                entry["request_attempts"] = active.last_call
            save()
    save()
    return redact(report, secrets)
