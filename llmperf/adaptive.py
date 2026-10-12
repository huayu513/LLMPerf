"""Validated candidate pools and deterministic adaptive experiment selection."""
from __future__ import annotations

import copy
import itertools
from dataclasses import replace

from .adapters import _effective_max_running_requests
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
