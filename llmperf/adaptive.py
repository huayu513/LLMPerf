"""Validated candidate pools and deterministic adaptive experiment selection."""
from __future__ import annotations

import copy
import itertools
import json
import os
from dataclasses import replace
from pathlib import Path

from .adapters import _effective_max_running_requests
from .advisor import LocalAdvisor, load_runtime_advisor
from .configuration import ADVISOR_TIMEOUT_SECONDS
from .artifacts import sha256_file, write_json_atomic
from .errors import ConfigError
from .evidence import candidate_summary, environment_summary, redact, summarize_trial
from .planner import fingerprint
from .types import CandidateConfig, PlanTask


def launch_identity(static: dict) -> str:
    canonical = copy.deepcopy(static)
    for key in ("comparison_group", "feature_variant", "logical_gpu_count"):
        canonical.pop(key, None)
    deployment = canonical.get("deployment", {})
    for key in ("label", "ascii_label", "base_urls", "service_port", "service_port_stride"):
        deployment.pop(key, None)
    return fingerprint(canonical)


def normalize_candidate(candidate: CandidateConfig, target: int) -> CandidateConfig:
    static = copy.deepcopy(candidate.static_config)
    task = PlanTask("normalize", 0, candidate.id, concurrency=target)
    static["max_running_requests"] = _effective_max_running_requests(task, static)
    return replace(candidate, static_config=static, config_hash=fingerprint(static))


def _subset_variations(subset):
    keys = tuple(key for key, _ in subset)
    for values in itertools.product(*(options for _, options in subset)):
        yield dict(zip(keys, values))


def _parameter_variations(knobs: dict):
    items = list(knobs.items())
    for width in range(1, len(items) + 1):
        streams = []
        for subset in itertools.combinations(items, width):
            streams.append(_subset_variations(subset))
        for group in itertools.zip_longest(*streams):
            yield from (params for params in group if params is not None)


def build_candidate_pool(bases: list[CandidateConfig], knobs: dict, target: int,
                         limit: int, seeds=()) -> tuple[CandidateConfig, ...]:
    selected = []
    identities = set()

    def add(candidate):
        normalized = normalize_candidate(candidate, target)
        identity = launch_identity(normalized.static_config)
        if identity not in identities and len(selected) < limit:
            identities.add(identity)
            selected.append(normalized)

    def variants(base):
        yield base
        for params in _parameter_variations(knobs):
            static = {**copy.deepcopy(base.static_config), **params}
            digest = fingerprint(static)
            yield replace(base, id=f"scene-{digest[:16]}", static_config=static,
                          config_hash=digest, reasons=("rule catalog",))

    for candidate in seeds:
        add(candidate)
    for group in itertools.zip_longest(*(variants(base) for base in bases)):
        for candidate in group:
            if candidate is not None:
                add(candidate)
            if len(selected) >= limit:
                return tuple(selected)
    return tuple(selected)


def _family(candidate):
    static = candidate["static_config"]
    return tuple(static.get(key) for key in
                 ("tp", "dp", "pp", "dp_attention", "backend", "dspark")) + (
        static.get("deployment", {}).get("instance_count", 1),
        tuple(static.get("gpu_indexes", candidate.get("gpu_indexes", []))))


class AdaptiveScheduler:
    def __init__(self, metadata: dict, candidates: dict, root: Path, state: dict,
                 save, advisor=None):
        self.metadata = metadata
        self.candidates = candidates
        self.root = root.resolve()
        self.state = state
        self.save = save
        self.policy = metadata["adaptive_policy"]
        if self.policy.get("version") != 1:
            raise ConfigError("unsupported adaptive policy version")
        self.progress = state.setdefault("adaptive", {
            "records": [], "pending": [], "advisor_calls": 0, "failure_signatures": [],
        })
        self.advisor = advisor
        self.advisor_error = None
        if advisor is None and self.policy["max_advisor_calls"]:
            try:
                self.advisor = load_runtime_advisor(metadata["runtime_advisor"], ADVISOR_TIMEOUT_SECONDS)
            except (ConfigError, KeyError, TypeError, ValueError):
                self.advisor_error = "runtime_advisor_unavailable"
        settings = getattr(self.advisor, "settings", {})
        authorization = settings.get("authorization", "")
        self.secrets = (authorization, authorization.removeprefix("Bearer "),
                        os.environ.get(settings.get("api_key_env", "LLMPERF_ADVISOR_KEY"), ""))
        self.audit()

    def audit(self):
        for entry in self.progress["records"]:
            if entry.get("sha256"):
                self._read_decision(entry)
        if self.progress["pending"]:
            if not self.progress["records"]:
                raise ConfigError("adaptive pending tasks have no decision")
            document = self._read_decision(self.progress["records"][-1])
            if not set(self.progress["pending"]) <= set(document["selected"]):
                raise ConfigError("adaptive pending tasks changed since decision")

    def _read_decision(self, entry):
        path = self.root / entry["manifest"]
        if not path.resolve().is_relative_to(self.root / "decisions"):
            raise ConfigError("adaptive decision path is outside its journal")
        try:
            if entry.get("sha256") and sha256_file(path) != entry["sha256"]:
                raise ConfigError("recorded adaptive decision changed")
            document = json.loads(path.read_text(encoding="utf-8"))
            if (document["plan_hash"] != self.state["plan_hash"] or
                    document["id"] != entry["id"] or
                    fingerprint(document["input"]) != entry["input_hash"]):
                raise ConfigError("adaptive decision identity changed")
            allowed = {row["candidate_id"] for row in document["input"]["shortlist"]}
            selected = document["selected"]
            if (not selected or len(selected) > document["input"]["batch_size"] or
                    len(set(selected)) != len(selected) or not set(selected) <= allowed):
                raise ConfigError("adaptive decision contains invalid selections")
            for evidence in document["input"]["evidence"]:
                artifact = self.root / evidence["manifest"]
                if (not artifact.resolve().is_relative_to(self.root / "trials") or
                        sha256_file(artifact) != evidence["sha256"]):
                    raise ConfigError("adaptive decision evidence changed")
            return document
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ConfigError("invalid adaptive decision journal") from exc

    def summary(self, result, candidate_id):
        return redact(summarize_trial(result, self.candidates[candidate_id], self.root), self.secrets)

    def remaining(self):
        tested = {row["candidate_id"] for row in self.state["observations"]}
        effective = {launch_identity(row["effective_static_config"])
                     for row in self.state["observations"]
                     if row.get("score") is not None and isinstance(row.get("effective_static_config"), dict)}
        return [ident for ident, candidate in self.candidates.items()
                if ident not in tested and launch_identity(candidate["static_config"]) not in effective]

    def _shortlist(self, remaining):
        observations = self.state["observations"]
        seen = {_family(self.candidates[row["candidate_id"]]) for row in observations}
        novel = []
        for ident in remaining:
            family = _family(self.candidates[ident])
            if family not in seen:
                novel.append(ident)
                seen.add(family)
        valid = [row for row in observations if row.get("score") is not None]
        nearby = []
        if valid:
            best = self.candidates[max(valid, key=lambda row: row["score"])["candidate_id"]]
            nearby = [ident for ident in remaining if _family(self.candidates[ident]) == _family(best)]
            nearby.sort(key=lambda ident: sum(
                self.candidates[ident]["static_config"].get(key) != best["static_config"].get(key)
                for key in ("mem_fraction_static", "chunked_prefill_size", "max_running_requests")))
        offset = (len(self.progress["records"]) * self.policy["batch_size"]) % len(remaining)
        rotated = remaining[offset:] + remaining[:offset]
        ordered = []
        for group in itertools.zip_longest(novel, nearby, rotated, remaining):
            for ident in group:
                if ident is not None and ident not in ordered:
                    ordered.append(ident)
        if not observations:
            baseline = self.metadata["baseline_candidate_id"]
            ordered = [baseline] + [ident for ident in ordered if ident != baseline]
        return ordered[:24]

    def _payload(self, shortlist, batch_size, remaining_trials, remaining_seconds):
        observations = self.state["observations"]
        ranked = sorted((row for row in observations if row.get("score") is not None),
                        key=lambda row: row["score"], reverse=True)
        return redact({
            "environment": environment_summary(self.metadata),
            "scenario": self.metadata["scenario"],
            "workload": {"sha256": self.metadata["expected_source_sha256"],
                         "requests": self.metadata["expected_request_count"],
                         "source": self.metadata["workload_source"]},
            "tested": [{key: row.get(key) for key in
                        ("candidate_id", "task_id", "status", "score", "failure_kind", "failure_phase")}
                       for row in observations],
            "recent": [row["evidence_summary"] for row in observations[-6:]],
            "best": [row["evidence_summary"] for row in ranked[:3]],
            "evidence": [row["evidence"] for row in observations],
            "shortlist": [candidate_summary(self.candidates[ident]) for ident in shortlist],
            "remaining_trials": remaining_trials,
            "remaining_seconds": remaining_seconds,
            "remaining_advisor_calls": self.policy["max_advisor_calls"] - self.progress["advisor_calls"],
            "batch_size": batch_size,
        }, self.secrets)

    def _validate_reply(self, reply, payload):
        if not isinstance(reply, dict) or set(reply) - {"diagnosis", "next_candidates"}:
            raise ValueError("unsupported feedback fields")
        rows = reply.get("next_candidates")
        if not isinstance(rows, list) or len(rows) > payload["batch_size"]:
            raise ValueError("feedback must return a bounded candidate list")
        allowed = {row["candidate_id"] for row in payload["shortlist"]}
        selected, rejections = [], []
        for row in rows:
            if (not isinstance(row, dict) or set(row) - {"candidate_id", "reason", "expected_effect"}
                    or not isinstance(row.get("candidate_id"), str)
                    or row["candidate_id"] not in allowed or row["candidate_id"] in selected
                    or any(not isinstance(row.get(key, ""), str) or len(row.get(key, "")) > 500
                           for key in ("reason", "expected_effect"))):
                rejections.append("invalid_or_duplicate_candidate")
            else:
                selected.append(row["candidate_id"])
        diagnosis = reply.get("diagnosis")
        if diagnosis is not None:
            known = {row["task_id"] for row in payload["tested"]}
            if (not isinstance(diagnosis, dict) or
                    set(diagnosis) - {"kind", "evidence_ids", "hypothesis"} or
                    any(not isinstance(diagnosis.get(key, ""), str) or len(diagnosis.get(key, "")) > 1000
                        for key in ("kind", "hypothesis")) or
                    not isinstance(diagnosis.get("evidence_ids", []), list) or
                    len(diagnosis.get("evidence_ids", [])) > 8 or
                    any(not isinstance(ident, str) or ident not in known
                        for ident in diagnosis.get("evidence_ids", []))):
                rejections.append("invalid_diagnosis_evidence")
                diagnosis = None
        return selected, rejections, diagnosis

    def _finish(self, entry, document):
        write_json_atomic(self.root / entry["manifest"], document)
        entry["sha256"] = sha256_file(self.root / entry["manifest"])
        entry.pop("input", None)
        self.progress["pending"] = list(document["selected"])
        self.save()
        print(json.dumps({"event": "adaptive_decision", "decision": entry["id"],
                          "source": document["source"], "status": document["status"],
                          "selected": document["selected"]}), flush=True)
        return self.progress["pending"]

    def next_batch(self, remaining_trials: int, timeout_seconds: float, remaining_seconds=None):
        remaining = self.remaining()
        self.progress["pending"] = [ident for ident in self.progress["pending"] if ident in remaining]
        if self.progress["pending"]:
            return self.progress["pending"]
        if self.progress["records"] and not self.progress["records"][-1].get("sha256"):
            entry = self.progress["records"][-1]
            if (self.root / entry["manifest"]).exists():
                document = self._read_decision(entry)
            else:
                document = {"id": entry["id"], "plan_hash": self.state["plan_hash"],
                            "input": entry["input"], "source": "rule", "status": "FALLBACK",
                            "reason": "interrupted_advisor_call", "response": None,
                            "selected": [row["candidate_id"] for row in
                                         entry["input"]["shortlist"][:entry["input"]["batch_size"]]]}
            return self._finish(entry, document)
        if not remaining or remaining_trials <= 0:
            return []
        shortlist = self._shortlist(remaining)
        batch_size = min(self.policy["batch_size"], remaining_trials, len(shortlist))
        payload = self._payload(shortlist, batch_size, remaining_trials,
                                timeout_seconds if remaining_seconds is None else remaining_seconds)
        bootstrap = not self.state["observations"]
        can_ask = (not bootstrap and self.progress["advisor_calls"] < self.policy["max_advisor_calls"])
        entry = {"id": len(self.progress["records"]) + 1, "sha256": None,
                 "input": payload, "input_hash": fingerprint(payload)}
        entry["manifest"] = f"decisions/{entry['id']:06d}.json"
        self.progress["records"].append(entry)
        if can_ask:
            self.progress["advisor_calls"] += 1
        self.save()
        document = {"id": entry["id"], "plan_hash": self.state["plan_hash"], "input": payload,
                    "source": "bootstrap" if bootstrap else "rule", "status": "PASS",
                    "response": None, "reason": "bootstrap" if bootstrap else "advisor_call_budget_exhausted"}
        selected = []
        if can_ask:
            try:
                if self.advisor is None:
                    raise ConfigError(self.advisor_error or "runtime_advisor_unavailable")
                client = self.advisor
                if isinstance(client, LocalAdvisor):
                    client = LocalAdvisor({**client.settings, "timeout_seconds": min(
                        float(client.settings["timeout_seconds"]), max(0.01, timeout_seconds))})
                try:
                    reply = client.ask("next_experiments", payload)
                finally:
                    self.save()
                selected, rejected, diagnosis = self._validate_reply(reply, payload)
                document.update(response=redact(reply, self.secrets), validation_rejections=rejected,
                                diagnosis=redact(diagnosis, self.secrets))
                if not selected:
                    raise ValueError("no_valid_feedback_candidates")
                document.update(source="advisor", reason="measured_feedback")
            except (ConfigError, ValueError, TypeError, KeyError, OSError) as exc:
                document.update(status="FALLBACK", reason=redact(str(exc), self.secrets))
        selected.extend(ident for ident in shortlist if ident not in selected)
        document["selected"] = selected[:batch_size]
        return self._finish(entry, document)

    def consume(self, candidate_id, result):
        if not self.progress["pending"] or self.progress["pending"][0] != candidate_id:
            raise ConfigError("trial does not match its accepted adaptive decision")
        self.progress["pending"].pop(0)
        if result.get("score") is None:
            signature = [result.get("failure_kind"), result.get("failure_phase"), result.get("status")]
            if signature not in self.progress["failure_signatures"]:
                self.progress["failure_signatures"].append(signature)
                if self.progress["pending"]:
                    self.state["decisions"].append({"reason": "new_failure_feedback",
                        "cancelled_candidates": list(self.progress["pending"])})
                    self.progress["pending"] = []
