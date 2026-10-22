"""Validated candidate pools and deterministic adaptive experiment selection."""
from __future__ import annotations

import copy
import itertools
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from .adapters import _effective_max_running_requests
from .advisor import LocalAdvisor, ask_advisor, load_runtime_advisor
from .configuration import ADVISOR_TIMEOUT_SECONDS
from .artifacts import sha256_file, write_json_atomic
from .errors import ConfigError
from .candidate_repair import TUNABLES, configuration_defects, matches_defect, proposal_candidate
from .feature_catalog import adapt_native_static, native_summary
from .evidence import candidate_summary, environment_summary, redact, summarize_trial
from .planner import fingerprint
from .native_planning import preflight_candidates
from .types import CandidateConfig, Plan, PlanTask


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
    capacity_key = "max_num_seqs" if static.get("engine") == "vllm" else "max_running_requests"
    static[capacity_key] = _effective_max_running_requests(task, static)
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
                         limit: int, seeds=(), feature_catalog=None) -> tuple[CandidateConfig, ...]:
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
            try:
                static = adapt_native_static(static, feature_catalog)
            except ValueError:
                continue
            digest = fingerprint(static)
            yield replace(base, id=f"scene-{digest[:16]}", static_config=static,
                          backend=static.get("backend", base.backend), dspark=static.get("dspark", base.dspark),
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
        tuple(static.get("gpu_indexes", candidate.get("gpu_indexes", []))),
        static.get("attn_dp_size"), json.dumps(native_summary(static), sort_keys=True))


class AdaptiveScheduler:
    def __init__(self, metadata: dict, candidates: dict, root: Path, state: dict,
                 save, advisor=None):
        self.metadata = metadata
        self.candidates = candidates
        self.root = root.resolve()
        self.state = state
        self.save = save
        self.policy = metadata["adaptive_policy"]
        if self.policy.get("version") not in {1, 2}:
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

    @property
    def repairs_enabled(self):
        return self.policy["version"] >= 2

    def audit(self):
        for entry in self.progress["records"]:
            if entry.get("sha256") or (self.root / entry["manifest"]).exists():
                document = self._read_decision(entry)
                self._register_proposals(document)
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
            proposals = document.get("proposals", [])
            if proposals and not self.repairs_enabled:
                raise ConfigError("frozen policy does not allow runtime proposals")
            if not isinstance(proposals, list) or len(proposals) > min(
                    document["input"]["batch_size"], document["input"].get("remaining_proposals", 0)):
                raise ConfigError("runtime proposal allowance exceeded")
            bases, evidence = self._proposal_context(document["input"])
            for proposal in proposals:
                candidate = proposal_candidate(proposal["request"], self.candidates, self.metadata, bases, evidence)
                if fingerprint(candidate) != fingerprint(proposal["candidate"]):
                    raise ConfigError("runtime candidate differs from its recorded patch")
                artifact = self.root / proposal["preflight_manifest"]
                if (not artifact.resolve().is_relative_to(self.root / "repairs") or
                        sha256_file(artifact) != proposal["preflight_sha256"]):
                    raise ConfigError("runtime candidate preflight evidence changed")
                report = json.loads(artifact.read_text(encoding="utf-8"))
                if (report != proposal["preflight"] or len(report["results"]) != 1 or
                        report["results"][0]["id"] != candidate["id"] or
                        report["results"][0]["status"] != "preflight_passed"):
                    raise ConfigError("runtime candidate has no successful native preflight")
                allowed.add(candidate["id"])
            selected = document["selected"]
            if ((not selected and not self.repairs_enabled) or len(selected) > document["input"]["batch_size"] or
                    len(set(selected)) != len(selected) or not set(selected) <= allowed):
                raise ConfigError("adaptive decision contains invalid selections")
            for evidence in document["input"]["evidence"]:
                artifact = self.root / evidence["manifest"]
                if (not artifact.resolve().is_relative_to(self.root / "trials") or
                        sha256_file(artifact) != evidence["sha256"]):
                    raise ConfigError("adaptive decision evidence changed")
            return document
        except ConfigError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ConfigError("invalid adaptive decision journal") from exc

    def _register_proposals(self, document):
        for proposal in document.get("proposals", []):
            candidate = proposal["candidate"]
            previous = self.candidates.get(candidate["id"])
            if previous is not None and fingerprint(previous) != fingerprint(candidate):
                raise ConfigError("runtime candidate identity collision")
            self.candidates[candidate["id"]] = candidate

    def _defects(self):
        defects = []
        for row in self.state["observations"]:
            if row.get("score") is not None or row.get("status") == "VALID":
                continue
            candidate = self.candidates[row["candidate_id"]]
            for defect in configuration_defects(candidate, self.metadata.get("engine_catalog"),
                                                row.get("evidence_summary")):
                if defect not in defects:
                    defects.append(defect)
        return defects

    def blocked(self, candidate):
        return bool(configuration_defects(candidate, self.metadata.get("engine_catalog")) or
                    any(matches_defect(candidate, defect) for defect in self._defects()))

    def summary(self, result, candidate_id):
        return redact(summarize_trial(result, self.candidates[candidate_id], self.root), self.secrets)

    def remaining(self):
        tested = {row["candidate_id"] for row in self.state["observations"]}
        effective = {launch_identity(row["effective_static_config"])
                     for row in self.state["observations"]
                     if row.get("score") is not None and isinstance(row.get("effective_static_config"), dict)}
        return [ident for ident, candidate in self.candidates.items()
                if ident not in tested and launch_identity(candidate["static_config"]) not in effective
                and not self.blocked(candidate)]

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
                for key in ("mem_fraction_static", "chunked_prefill_size", "max_running_requests",
                            "gpu_memory_utilization", "max_num_batched_tokens", "max_num_seqs")))
        offset = (len(self.progress["records"]) * self.policy["batch_size"]) % max(1, len(remaining))
        rotated = remaining[offset:] + remaining[:offset]
        ordered = []
        for group in itertools.zip_longest(novel, nearby, rotated, remaining):
            for ident in group:
                if ident is not None and ident not in ordered:
                    ordered.append(ident)
        if not observations:
            baseline = self.metadata["baseline_candidate_id"]
            if baseline in remaining:
                ordered = [baseline] + [ident for ident in ordered if ident != baseline]
        return ordered[:24]

    def _payload(self, shortlist, batch_size, remaining_trials, remaining_seconds):
        observations = self.state["observations"]
        ranked = sorted((row for row in observations if row.get("score") is not None),
                        key=lambda row: row["score"], reverse=True)
        payload = {
            "engine": self.metadata.get("engine", "sglang"),
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
            "capability_analysis": {key: self.metadata.get("capability_analysis", {}).get(key)
                                    for key in ("status", "features", "hypotheses", "unknowns")},
            "remaining_seconds": remaining_seconds,
            "remaining_advisor_calls": self.policy["max_advisor_calls"] - self.progress["advisor_calls"],
            "batch_size": batch_size,
        }
        if self.repairs_enabled:
            bases = list(dict.fromkeys([row["candidate_id"] for row in observations[-6:]] +
                [row["candidate_id"] for row in ranked[:3]] + shortlist +
                [ident for ident, candidate in self.candidates.items() if self.blocked(candidate)][:8]))
            catalog = self.metadata.get("engine_catalog", {})
            payload.update(proposal_bases=[candidate_summary(self.candidates[ident]) for ident in bases],
                configuration_defects=self._defects(),
                proposal_catalog={"parameters": {name: item for name, item in catalog.get("parameters", {}).items()
                    if item.get("integrated") or name in TUNABLES.get(catalog.get("engine"), {})},
                    "nested": catalog.get("nested", {})},
                user_restrictions=self.metadata.get("configuration", {}).get("engine_options", {}),
                remaining_proposals=self.policy["max_generated_candidates"] - self._generated_count())
            payload["proposal_evidence_ids"] = sorted(self._proposal_context(payload)[1])
            previous = [self._read_decision(entry) for entry in self.progress["records"][-3:] if entry.get("sha256")]
            payload["previous_proposal_rejections"] = [rejection for document in previous
                for rejection in (document.get("validation_rejections") or
                    ([document["reason"]] if document["status"] == "FALLBACK" else []))][-8:]
        return redact(payload, self.secrets)

    def _generated_count(self):
        return sum(len(self._read_decision(entry).get("proposals", []))
                   for entry in self.progress["records"] if entry.get("sha256"))

    def can_propose(self):
        return (self.repairs_enabled and self.progress["advisor_calls"] < self.policy["max_advisor_calls"]
                and self._generated_count() < self.policy["max_generated_candidates"])

    def _proposal_context(self, payload):
        bases = {row["candidate_id"] for row in payload.get("proposal_bases", [])}
        evidence = {row["task_id"] for row in payload["tested"]}
        catalog = self.metadata.get("engine_catalog", {})
        evidence.update(item["id"] for item in catalog.get("parameters", {}).values() if "id" in item)
        evidence.update("nested:" + name for name in catalog.get("nested", {}))
        return bases, evidence

    def _validate_proposals(self, reply, payload):
        proposals = reply.get("candidate_proposals", [])
        if (not isinstance(proposals, list) or len(proposals) > min(
                payload["batch_size"], max(0, payload.get("remaining_proposals", 0)))):
            raise ValueError("too many runtime candidate proposals")
        bases, evidence = self._proposal_context(payload)
        accepted, rejected = [], []
        identities = {launch_identity(candidate["static_config"]) for candidate in self.candidates.values()}
        for proposal in proposals:
            try:
                candidate = proposal_candidate(proposal, self.candidates, self.metadata, bases, evidence)
                identity = launch_identity(candidate["static_config"])
                if identity in identities or self.blocked(candidate):
                    raise ValueError("duplicate candidate or unresolved configuration defect")
                identities.add(identity)
                accepted.append({"request": proposal, "candidate": candidate})
            except (ValueError, TypeError, KeyError) as exc:
                rejected.append(str(exc))
        return accepted, rejected

    def _validate_reply(self, reply, payload):
        fields = {"diagnosis", "next_candidates"} | ({"candidate_proposals"} if self.repairs_enabled else set())
        if not isinstance(reply, dict) or set(reply) - fields:
            raise ValueError("unsupported feedback fields")
        rows = reply.get("next_candidates", [])
        if not isinstance(rows, list) or len(rows) > payload["batch_size"]:
            raise ValueError("feedback must return a bounded candidate list")
        proposals = reply.get("candidate_proposals", [])
        if not isinstance(proposals, list) or len(proposals) + len(rows) > payload["batch_size"]:
            raise ValueError("feedback selections and proposals exceed batch_size")
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
        self._register_proposals(document)
        entry.pop("input", None)
        self.progress["pending"] = list(document["selected"])
        self.save()
        print(json.dumps({"event": "adaptive_decision", "decision": entry["id"],
                          "source": document["source"], "status": document["status"],
                          "selected": document["selected"]}), flush=True)
        return self.progress["pending"]

    def _preflight_proposals(self, proposals, entry, deadline):
        accepted, rejected = [], []
        for index, proposal in enumerate(proposals):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                rejected.append("runtime_proposal_time_budget_exhausted")
                break
            candidate = proposal["candidate"]
            root = self.root / "repairs" / f"{entry['id']:06d}" / f"{index + 1:02d}"
            root.mkdir(parents=True, exist_ok=False)
            draft = Plan("runtime-proposal", candidates=(CandidateConfig(**candidate),), metadata=self.metadata)
            config = SimpleNamespace(engine=self.metadata["engine"], image=self.metadata["image"],
                                     gpu_indexes=self.metadata.get("configuration", {}).get("gpu_indexes"),
                                     model_path=self.metadata.get("model_host"))
            report = preflight_candidates(draft, config, root, timeout_seconds=remaining)
            if (len(report["results"]) != 1 or report["results"][0]["id"] != candidate["id"] or
                    report["results"][0]["status"] != "preflight_passed"):
                rejected.append({"base_id": proposal["request"]["base_id"], "reason": "native_preflight_not_passed",
                                 "preflight": report})
                continue
            artifact = root / "native-preflight.json"
            accepted.append({**proposal, "preflight": report,
                "preflight_manifest": str(artifact.relative_to(self.root)), "preflight_sha256": sha256_file(artifact)})
        return accepted, rejected

    def next_batch(self, remaining_trials: int, timeout_seconds: float, remaining_seconds=None):
        deadline = time.monotonic() + timeout_seconds
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
                                         entry["input"]["shortlist"] if row["candidate_id"] in remaining
                                         ][:entry["input"]["batch_size"]]}
            return self._finish(entry, document)
        can_repair = self.can_propose()
        if remaining_trials <= 0 or not remaining and not can_repair:
            return []
        shortlist = self._shortlist(remaining)
        batch_size = min(self.policy["batch_size"], remaining_trials, max(len(shortlist), int(can_repair)))
        payload = self._payload(shortlist, batch_size, remaining_trials,
                                timeout_seconds if remaining_seconds is None else remaining_seconds)
        bootstrap = not self.state["observations"] and bool(shortlist)
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

                def validate(reply):
                    selected, rejected, _ = self._validate_reply(reply, payload)
                    proposals, proposal_rejections = self._validate_proposals(reply, payload) if self.repairs_enabled else ([], [])
                    if not selected and not proposals:
                        detail = "; ".join(rejected + proposal_rejections)[:1500]
                        raise ValueError("no_valid_feedback_candidates" + (": " + detail if detail else ""))

                try:
                    reply = ask_advisor(client, "next_experiments", payload, validate)
                finally:
                    if isinstance(client, LocalAdvisor):
                        document["request_attempts"] = client.last_call
                    self.save()
                selected, rejected, diagnosis = self._validate_reply(reply, payload)
                proposals, proposal_rejections = self._validate_proposals(reply, payload) if self.repairs_enabled else ([], [])
                proposals, preflight_rejections = self._preflight_proposals(proposals, entry, deadline)
                rejected.extend(proposal_rejections + preflight_rejections)
                selected = [row["candidate"]["id"] for row in proposals] + selected
                document.update(response=redact(reply, self.secrets), validation_rejections=rejected,
                                diagnosis=redact(diagnosis, self.secrets), proposals=redact(proposals, self.secrets))
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
        # Re-evaluate even an existing pending batch after every failure. This
        # also covers a defect learned from an old image's native error text.
        self.progress["pending"] = [ident for ident in self.progress["pending"]
                                     if not self.blocked(self.candidates[ident])]
        if result.get("score") is None:
            signature = [result.get("failure_kind"), result.get("failure_phase"), result.get("status")]
            if signature not in self.progress["failure_signatures"]:
                self.progress["failure_signatures"].append(signature)
                if self.progress["pending"]:
                    self.state["decisions"].append({"reason": "new_failure_feedback",
                        "cancelled_candidates": list(self.progress["pending"])})
                    self.progress["pending"] = []
