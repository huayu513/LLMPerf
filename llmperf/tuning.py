"""Scene-driven, measured tuning with a local advisor and rule fallback."""
from __future__ import annotations

import copy
import itertools
import json
import math
import statistics
import time
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import ReplayAdapter, read_attempt
from .advisor import LocalAdvisor, runtime_advisor_spec
from .adaptive import AdaptiveScheduler, build_candidate_pool
from .evidence import redact
from .artifacts import sha256_file, write_json_atomic
from .backend_resolution import apply_backend_resolution, expand_auto_backends, resolve_in_image
from .capabilities import probe_environment
from .configuration import load_config
from .discovery import discover_model, inspect_workload, prepare_workload
from .docker_runtime import DockerRuntime, DockerTaskSpec, Mount
from .errors import ConfigError
from .planner import _jsonable, create_search_plan, fingerprint, plan_to_dict, write_plan
from .scene_workload import create_scene_workload
from .search import collect_results
from .types import CandidateConfig, Plan, PlanTask
from .workflow import _bundle_fingerprint, _probe_config


def _metric(result: dict, priority: str) -> float | None:
    if result.get("status") != "VALID":
        return None
    if priority == "throughput":
        value = result.get("output_tokens_per_second")
        return float(value) if type(value) in (int, float) and math.isfinite(value) and value > 0 else None
    latency = result.get("latency_seconds")
    value = latency.get("p95") if isinstance(latency, dict) else None
    return -float(value) if type(value) in (int, float) and math.isfinite(value) and value > 0 else None


def _catalog(plan: Plan, scenario: dict) -> dict:
    options = set(plan.metadata.get("environment", {}).get("container", {})
                  .get("capabilities", {}).get("options", []))
    target = scenario["target_concurrency"]
    knobs = {}
    if "--mem-fraction-static" in options:
        knobs["mem_fraction_static"] = [0.75, 0.8, 0.85, 0.9]
    if "--chunked-prefill-size" in options:
        knobs["chunked_prefill_size"] = [2048, 4096, 8192, 16384]
    if "--max-running-requests" in options:
        knobs["max_running_requests"] = sorted({max(1, target // 2), target})
    return {"version": 1, "base_candidates": [asdict(c) for c in plan.candidates],
            "knobs": knobs, "semantics": "omitted means keep the base value; false is an explicit value"}


def _visible_catalog(catalog: dict, batch: int) -> dict:
    bases = catalog["base_candidates"]
    if len(bases) > 40:
        offset = (batch * 40) % len(bases)
        bases = (bases[offset:] + bases[:offset])[:40]
    return {"knobs": catalog["knobs"], "base_candidates": [
        {"base_id": row["id"], "tp": row["tp"], "dp": row["dp"],
         "pp": row["pp"], "dp_attention": row["dp_attention"],
         "backend": row["backend"], "dspark": row["dspark"],
         "deployment": row["static_config"].get("deployment", {}).get("label")}
        for row in bases]}


def _from_suggestion(item: Any, catalog: dict) -> CandidateConfig:
    if not isinstance(item, dict) or set(item) - {"base_id", "params", "reason"}:
        raise ValueError("candidate has unsupported fields")
    base_id = item.get("base_id")
    bases = {row["id"]: row for row in catalog["base_candidates"]}
    if base_id not in bases:
        raise ValueError("base_id is not in the capability catalog")
    params = item.get("params", {})
    if not isinstance(params, dict) or set(params) - set(catalog["knobs"]):
        raise ValueError("candidate params are outside the capability catalog")
    for key, value in params.items():
        expected_type = int if key in {"chunked_prefill_size", "max_running_requests"} else float
        if type(value) is not expected_type or value not in catalog["knobs"][key]:
            raise ValueError(f"unsupported value for {key}")
    reason = item.get("reason", "")
    if not isinstance(reason, str) or len(reason) > 500:
        raise ValueError("candidate reason is invalid")
    base = bases[base_id]
    static = copy.deepcopy(base["static_config"])
    static.update(params)
    digest = fingerprint(static)
    return CandidateConfig(
        id=f"scene-{digest[:16]}", tp=base["tp"], dp=base["dp"], pp=base["pp"],
        dp_attention=base["dp_attention"], backend=base["backend"],
        dspark=base["dspark"], gpu_indexes=tuple(base["gpu_indexes"]),
        status="CANDIDATE", reasons=(reason,) if reason else (),
        static_config=static, config_hash=digest,
    )


def _result_length_evidence(result: dict, scenario: dict) -> None:
    """Attach actual server usage; never present max_tokens as achieved length."""
    result_path = result.get("result_path")
    if not result_path:
        return
    files = list(Path(result_path).rglob("*.requests.jsonl"))
    if len(files) != 1:
        result["status"] = "INCONCLUSIVE"
        result.setdefault("reasons", []).append("scene_request_usage_missing")
        return
    prompts, completions = [], []
    usage_missing = False
    try:
        with files[0].open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row.get("ok"):
                    prompt = row.get("prompt_tokens")
                    completion = row.get("completion_tokens")
                    if (type(prompt) is not int or prompt < 0 or
                            type(completion) is not int or completion < 0):
                        usage_missing = True
                    else:
                        prompts.append(prompt)
                        completions.append(completion)
    except (OSError, ValueError):
        result["status"] = "INCONCLUSIVE"
        result.setdefault("reasons", []).append("scene_request_usage_invalid")
        return
    result["scene_lengths"] = {
        "observed_prompt_tokens_max": max(prompts) if prompts else None,
        "observed_completion_tokens_max": max(completions) if completions else None,
        "verified_request_count": len(prompts),
        "requested_max_input_tokens": scenario["max_input_tokens"],
        "requested_max_output_tokens": scenario["max_output_tokens"],
    }
    if usage_missing or not prompts:
        result["status"] = "INCONCLUSIVE"
        result.setdefault("reasons", []).append("scene_request_usage_missing")
    if prompts and max(prompts) > scenario["max_input_tokens"]:
        result["status"] = "INCONCLUSIVE"
        result.setdefault("reasons", []).append("scene_max_input_exceeded")
    if completions and max(completions) > scenario["max_output_tokens"]:
        result["status"] = "INCONCLUSIVE"
        result.setdefault("reasons", []).append("scene_max_output_exceeded")


def _calibrate_scene(root: Path, config, model, scene: dict) -> dict:
    data_dir = root / "data"
    input_path = data_dir / "calibration-input.json"
    output_path = data_dir / "calibration-result.json"
    write_json_atomic(input_path, {
        "max_input_tokens": scene["max_input_tokens"],
        "chat_template_kwargs": model.chat_template_kwargs,
    })
    spec = DockerTaskSpec(
        image=config.image, name="llmperf-scene-" + uuid.uuid4().hex[:12],
        mounts=(Mount(config.model_path, "/model", True),
                Mount(data_dir, "/run/scene", False),
                Mount(Path(__file__).resolve().parents[1] / "benchmarks", "/opt/benchmarks", True)),
        command=("python3", "/opt/benchmarks/replay/calibrate_scene.py",
                 "--model", "/model", "--input", "/run/scene/calibration-input.json",
                 "--output", "/run/scene/calibration-result.json"),
        network_mode=config.docker.network_mode, shm_size=config.docker.shm_size,
        ipc=config.docker.ipc,
    )
    try:
        run = DockerRuntime().run(spec, data_dir / "calibration.log")
        if run.exit_code != 0:
            raise RuntimeError(f"tokenizer container exited with code {run.exit_code}")
        result = json.loads(output_path.read_text(encoding="utf-8"))
        counts = result.get("repeat_counts")
        if (not isinstance(counts, list) or len(counts) != 3
                or any(type(n) is not int or n < 0 for n in counts)):
            raise ValueError("invalid tokenizer calibration result")
        return result
    except (OSError, RuntimeError, ValueError) as exc:
        result = {"status": "FALLBACK", "reason": str(exc),
                  "method": "conservative character proxy; validate actual server usage"}
        write_json_atomic(output_path, result)
        return result


def _prepare_new_run(config_path: Path, *, use_captured: bool, probe,
                     calibrate: bool) -> tuple[Path, Plan, dict]:
    config = load_config(config_path, tuning=True)
    if config.scenario is None:
        raise ConfigError("scenario is missing; describe the scene before tune")
    if config.advisor is None:
        raise ConfigError("advisor settings are required for tune")
    if use_captured and config.input_path is None:
        raise ConfigError("--use-captured requires input_path")
    scene = config.scenario
    base = config.output_dir or config.source_path.parent / "results"
    ident = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    root = Path(base) / ident
    root.mkdir(parents=True, exist_ok=False)
    model = discover_model(config.model_path, config.model_overrides)
    source = config.input_path if use_captured else root / "data" / "scene-requests.jsonl"
    if not use_captured:
        create_scene_workload(source, scene)
    workload = inspect_workload(source)
    effective_search = replace(config.search,
                               concurrency_max=scene["target_concurrency"],
                               start_concurrency=scene["target_concurrency"])
    effective_config = replace(config, input_path=source, search=effective_search)
    env = probe(_probe_config(effective_config, root))
    write_json_atomic(root / "environment.json", env.to_dict())
    if env.status == "FAIL":
        raise ConfigError(f"environment checks failed; see {root / 'environment.json'}")
    calibration = None
    if not use_captured and calibrate:
        calibration = _calibrate_scene(root, config, model, scene)
        if calibration["status"] == "PASS":
            source.unlink()
            source.with_suffix(".manifest.json").unlink()
            create_scene_workload(source, scene, repeat_counts=calibration["repeat_counts"])
            workload = inspect_workload(source)
    index = prepare_workload(source, root / "data" / "replay_index.json")
    if index["jsonl_sha256"] != workload.sha256:
        raise ConfigError("workload changed during preparation")
    plan = create_search_plan(effective_config, model, workload, env, root)
    if model.raw.get("is_moe"):
        image = env.image.to_dict() if hasattr(env.image, "to_dict") else dict(env.image)
        if image.get("image_id"):
            resolutions = resolve_in_image(plan, effective_config, root)
            source_label = "configured Docker image"
        else:
            resolutions = {}
            source_label = "unavailable: image identity missing from environment probe"
        plan, added = expand_auto_backends(plan, resolutions)
        if added and image.get("image_id"):
            resolutions.update(resolve_in_image(plan, effective_config, root,
                                               candidate_ids=set(added)))
        plan = apply_backend_resolution(plan, resolutions, source=source_label, added=added)
        write_json_atomic(root / "backend-resolution.json", plan.metadata["backend_resolution"])
    plan.metadata.update({
        "workflow_mode": "tune", "scenario": scene, "workload_source": "captured" if use_captured else "synthetic",
        "scene_calibration": calibration,
        "config_path": str(config_path),
        "configuration": {"model_path": str(config.model_path), "image": config.image,
                          "gpu_indexes": config.gpu_indexes, "scenario": scene},
        "bundle_fingerprint": _bundle_fingerprint(),
        "index_fingerprint": sha256_file(root / "data" / "replay_index.json"),
        "objective": f"best {scene['priority']} at concurrency {scene['target_concurrency']}",
    })
    catalog = _catalog(plan, scene)
    write_json_atomic(root / "catalog.json", catalog)
    write_json_atomic(root / "scenario.json", scene)
    write_json_atomic(root / "resolved-config.json", {
        "version": 3, "workflow_mode": "tune", "scenario": scene,
        "model": asdict(model), "input": {**workload.raw, "path": str(source), "sha256": workload.sha256},
        "served_model_name": plan.metadata["served_model_name"],
        "workload_source": plan.metadata["workload_source"], "result_dir": str(root),
        "scene_calibration": calibration,
    })
    return root, plan, catalog



def plan_tuning(config_path: str | Path, *, use_captured: bool = False,
                probe=probe_environment, advisor: LocalAdvisor | None = None,
                calibrate: bool = True) -> dict:
    """Freeze a scene workload and all search candidates without benchmarking."""
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path, tuning=True)
    source_hash = sha256_file(config_path)
    if config.scenario is None or config.advisor is None:
        raise ConfigError("intelligent planning requires scenario and advisor in experiment.json")
    adaptive = config.search.strategy == "adaptive"
    reserve = ((config.search.adaptive.final_top_k + 1) if adaptive else 2) * config.search.repetitions
    exploration_capacity = config.search.max_trials - reserve
    capacity = config.search.adaptive.candidate_pool_limit if adaptive else exploration_capacity
    if exploration_capacity < 2:
        raise ConfigError("search.max_trials must allow two explorations plus winner and baseline repeats")
    root, template, catalog = _prepare_new_run(config_path, use_captured=use_captured,
                                                probe=probe, calibrate=calibrate)
    if sha256_file(config_path) != source_hash:
        raise ConfigError("experiment.json changed during intelligent planning; generate a new plan")
    print(json.dumps({"event": "run_created", "result_dir": str(root)}), flush=True)
    scene = config.scenario
    selected: dict[str, CandidateConfig] = {}
    hashes: set[str] = set()
    decisions: list[dict] = []
    advisor_calls: list[dict] = []

    def add(candidate: CandidateConfig) -> None:
        if len(selected) < capacity and candidate.config_hash not in hashes:
            selected[candidate.id] = candidate
            hashes.add(candidate.config_hash)

    baseline = catalog["base_candidates"][0]
    add(CandidateConfig(**{**baseline, "gpu_indexes": tuple(baseline["gpu_indexes"]),
                           "reasons": tuple(baseline.get("reasons", ())) }))
    try:
        reply = (advisor or LocalAdvisor(config.advisor)).ask("candidates", {
            "scenario": scene, "catalog": _visible_catalog(catalog, 0),
            "tested": [], "remaining_trials": exploration_capacity - 1,
            "request_count": template.metadata["expected_request_count"],
        })
        rows = reply.get("candidates")
        if not isinstance(rows, list) or len(rows) > 5:
            raise ValueError("advisor must return at most 5 candidates")
        accepted = 0
        for row in rows:
            try:
                before = len(selected)
                add(_from_suggestion(row, catalog))
                accepted += len(selected) - before
            except ValueError as exc:
                decisions.append({"reason": "advisor_candidate_rejected", "detail": str(exc)})
        advisor_calls.append({"task": "candidates", "status": "PASS", "accepted": accepted})
    except (ConfigError, ValueError) as exc:
        advisor_calls.append({"task": "candidates", "status": "FALLBACK", "reason": str(exc)})

    if adaptive:
        bases = [CandidateConfig(**{**row, "gpu_indexes": tuple(row["gpu_indexes"]),
                                    "reasons": tuple(row.get("reasons", ()))})
                 for row in catalog["base_candidates"]]
        pool = build_candidate_pool(bases, catalog["knobs"], scene["target_concurrency"],
                                    capacity, seeds=selected.values())
        selected = {candidate.id: candidate for candidate in pool}
        template.metadata["adaptive_policy"] = {"version": 1, **asdict(config.search.adaptive)}
        template.metadata["runtime_advisor"] = runtime_advisor_spec(config.advisor, config_path)
        secrets = (config.advisor.get("authorization", "").removeprefix("Bearer "),)
        selected = {ident: replace(candidate, reasons=tuple(redact(candidate.reasons, secrets)))
                    for ident, candidate in selected.items()}
        advisor_calls = redact(advisor_calls, secrets)
        decisions = redact(decisions, secrets)
    else:
        for row in catalog["base_candidates"]:
            add(CandidateConfig(**{**row, "gpu_indexes": tuple(row["gpu_indexes"]),
                                   "reasons": tuple(row.get("reasons", ())) }))
        knob_items = list(catalog["knobs"].items())
        for width in range(1, len(knob_items) + 1):
            for subset in itertools.combinations(knob_items, width):
                for values in itertools.product(*(options for _, options in subset)):
                    if len(selected) >= capacity:
                        break
                    params = dict(zip((name for name, _ in subset), values))
                    for base in catalog["base_candidates"]:
                        if len(selected) >= capacity:
                            break
                        if all(base["static_config"].get(key) == value
                               for key, value in params.items()):
                            continue
                        add(_from_suggestion({"base_id": base["id"], "params": params,
                                              "reason": "rule catalog"}, catalog))
                if len(selected) >= capacity:
                    break
            if len(selected) >= capacity:
                break

    source = Path(template.metadata["jsonl_host"])
    effective_search = replace(config.search, concurrency_max=scene["target_concurrency"],
                               start_concurrency=scene["target_concurrency"])
    effective = replace(config, input_path=source, search=effective_search)
    configuration = _jsonable(asdict(effective))
    configuration.pop("advisor", None)
    if sha256_file(config_path) != source_hash:
        raise ConfigError("experiment.json changed during intelligent planning; generate a new plan")
    template.metadata.update({
        "configuration": configuration,
        "config_sha256_at_planning": source_hash,
        "candidate_source": "advisor_with_validated_rule_catalog",
        "baseline_candidate_id": baseline["id"],
        "planning": {"advisor_calls": advisor_calls, "decisions": decisions,
                     "exploration_capacity": exploration_capacity, "planned_candidates": len(selected),
                     "reserved_final_trials": reserve},
        "candidates": {candidate.id: asdict(candidate) for candidate in selected.values()},
    })
    plan = Plan(root.name, candidates=tuple(selected.values()), metadata=template.metadata)
    write_plan(root / "plan.json", plan)
    write_json_atomic(root / "plan-lock.json", {"sha256": sha256_file(root / "plan.json")})
    resolved_path = root / "resolved-config.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved["candidate_count"] = len(plan.candidates)
    resolved["search_scope"] = template.metadata["search_scope"]
    write_json_atomic(resolved_path, resolved)
    return {"command": "tune-plan", "status": "PASS", "result_dir": str(root),
            "plan": str(root / "plan.json"), "candidates": len(plan.candidates),
            "requests": template.metadata["expected_request_count"],
            "scenario": scene, "advisor_calls": advisor_calls}


def execute_tuning_plan(plan: Plan, run_root: str | Path, *, executor=None,
                        resume: bool = False, advisor=None) -> dict:
    """Measure a frozen scene plan using its approved search strategy."""
    if plan.metadata.get("workflow_mode") != "tune" or not plan.candidates:
        raise ConfigError("saved plan is not a complete intelligent tuning plan")
    root = Path(run_root)
    adaptive = plan.metadata["search"].get("strategy", "strict") == "adaptive"
    identity = fingerprint(plan_to_dict(plan))
    state_path = root / "tune-state.json"
    if state_path.exists():
        if not resume:
            raise FileExistsError("search already exists; use resume for this plan")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("plan_hash") != identity:
            raise ConfigError("saved plan changed since tuning started")
        if not adaptive and state.get("phase") in {"complete", "inconclusive"} and (root / "best.json").is_file():
            return json.loads((root / "best.json").read_text(encoding="utf-8"))
    else:
        state = {"version": 2, "plan_hash": identity, "phase": "explore",
                 "next_index": 0, "trials": {}, "observations": [], "decisions": [],
                 "elapsed_seconds": 0.0, "stagnant_batches": 0}
        write_json_atomic(state_path, state)
    scene = plan.metadata["scenario"]
    limits = plan.metadata["search"]
    limit = int(limits["max_trials"])
    repetitions = int(limits["repetitions"])
    max_seconds = float(limits["max_seconds"])
    target = scene["target_concurrency"]
    candidates = {candidate.id: asdict(candidate) for candidate in plan.candidates}
    candidate_ids = list(candidates)
    baseline_id = plan.metadata["baseline_candidate_id"]
    if baseline_id not in candidates:
        raise ConfigError("plan baseline is missing from candidates")
    started = time.monotonic()
    previous_elapsed = float(state["elapsed_seconds"])

    def save() -> None:
        state["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
        write_json_atomic(state_path, state)

    def elapsed() -> float:
        return previous_elapsed + time.monotonic() - started

    def used() -> int:
        return sum(1 for entries in state["trials"].values() for entry in entries
                   if adaptive or entry.get("completed") is True)

    scheduler = None
    if adaptive:
        for entries in state["trials"].values():
            for entry in entries:
                if not (root / entry["manifest"]).resolve().is_relative_to((root / "trials").resolve()):
                    raise ConfigError("adaptive trial path is outside its run")
                if entry.get("completed"):
                    path = root / entry["manifest"]
                    if (not path.resolve().is_relative_to((root / "trials").resolve()) or
                            not path.is_file() or sha256_file(path) != entry["sha256"]):
                        raise ConfigError("recorded adaptive trial evidence changed")
        scheduler = AdaptiveScheduler(plan.metadata, candidates, root, state, save, advisor=advisor)
        if state["phase"] in {"complete", "inconclusive"}:
            report_path = root / "best.json"
            if not report_path.is_file() or sha256_file(report_path) != state.get("report_sha256"):
                raise ConfigError("recorded adaptive report changed")
            return json.loads(report_path.read_text(encoding="utf-8"))

    def one_trial(candidate_id: str, run_class: str, suffix: str = "") -> dict | None:
        task_id = f"{candidate_id}-{run_class}-c{target}{suffix}"
        entries = state["trials"].setdefault(task_id, [])
        for entry in reversed(entries):
            if entry.get("completed"):
                path = root / entry["manifest"]
                if path.is_file() and sha256_file(path) == entry["sha256"]:
                    return json.loads(path.read_text(encoding="utf-8"))
                raise ConfigError(f"recorded trial evidence changed: {path}")
        if adaptive:
            for entry in entries:
                path = root / entry["manifest"]
                if path.is_file():
                    result = json.loads(path.read_text(encoding="utf-8"))
                    if (result.get("task_id") != task_id or result.get("candidate_id") != candidate_id
                            or result.get("candidate_hash") != candidates[candidate_id]["config_hash"]
                            or result.get("concurrency") != target or result.get("run_class") != run_class
                            or result.get("score") != _metric(result, scene["priority"])):
                        raise ConfigError("interrupted trial has invalid result identity")
                    entry.update(completed=True, sha256=sha256_file(path))
                    save()
                    return result
            if used() >= limit or elapsed() >= max_seconds:
                return None
        attempt = len(entries) + 1
        task = PlanTask(task_id, 5 if run_class == "final_repeat" else 3,
                        candidate_id, run_class=run_class, concurrency=target,
                        candidate_hash=candidates[candidate_id]["config_hash"])
        path = root / "trials" / task_id / f"attempt-{attempt:03d}" / "trial.json"
        entries.append({"manifest": str(path.relative_to(root)), "completed": False})
        save()
        trial_started = time.monotonic()
        metadata = copy.deepcopy(plan.metadata)
        metadata["candidates"] = candidates
        try:
            if executor is None:
                adapter = ReplayAdapter(metadata, root / "trials")
                try:
                    adapter(task, attempt)
                    exit_code = 0
                except RuntimeError:
                    exit_code = 1
                result = read_attempt(path.parent, task, metadata, exit_code=exit_code)
                result["result_path"] = str(path.parent)
            else:
                result = dict(executor(task, attempt))
        except Exception as exc:
            result = {"status": "FAILED", "reasons": [f"{type(exc).__name__}: {exc}"]}
        finally:
            if adaptive:
                save()
        result.update(task_id=task_id, candidate_id=candidate_id, concurrency=target,
                      run_class=run_class, candidate_hash=task.candidate_hash,
                      attempt=attempt)
        _result_length_evidence(result, scene)
        result["score"] = _metric(result, scene["priority"])
        if adaptive:
            result["trial_seconds"] = time.monotonic() - trial_started
            result = redact(result, scheduler.secrets)
        write_json_atomic(path, result)
        entries[-1] = {"manifest": str(path.relative_to(root)),
                       "sha256": sha256_file(path), "completed": True}
        save()
        collect_results(root)
        return result

    def observe(result):
        candidate_id = result["candidate_id"]
        if not any(row["candidate_id"] == candidate_id for row in state["observations"]):
            entry = next(entry for entry in reversed(state["trials"][result["task_id"]]) if entry["completed"])
            observation = {key: result[key] for key in
                           ("candidate_id", "task_id", "score", "status", "output_tokens_per_second",
                            "latency_seconds", "ttft_seconds", "tpot_seconds", "reasons",
                            "failure_kind", "failure_phase", "effective_static_config", "trial_seconds")
                           if key in result}
            observation.update(evidence={"manifest": entry["manifest"], "sha256": entry["sha256"]},
                               evidence_summary=scheduler.summary(result, candidate_id))
            state["observations"].append(observation)
            scheduler.consume(candidate_id, result)
            save()

    def estimated_repeat_seconds():
        durations = [row["trial_seconds"] for row in state["observations"] if row.get("trial_seconds", 0) > 0]
        request_limit = limits.get("explore_request_limit", 0)
        ratio = max(1, plan.metadata["expected_request_count"] / request_limit) if request_limit else 1
        return max(durations[-6:], default=0) * ratio

    if adaptive and state["phase"] == "explore":
        for task_id, entries in list(state["trials"].items()):
            for entry in entries:
                path = root / entry["manifest"]
                if not entry.get("completed") and path.is_file():
                    orphan = json.loads(path.read_text(encoding="utf-8"))
                    candidate_id = orphan.get("candidate_id")
                    run_class = orphan.get("run_class")
                    prefix = f"{candidate_id}-{run_class}-c{target}"
                    if (candidate_id not in candidates or run_class not in {"tuning", "final_repeat"}
                            or orphan.get("task_id") != task_id or not task_id.startswith(prefix)):
                        raise ConfigError("interrupted trial has invalid task identity")
                    one_trial(candidate_id, run_class, task_id[len(prefix):])
                if entry.get("completed"):
                    result = json.loads((root / entry["manifest"]).read_text(encoding="utf-8"))
                    if result["run_class"] == "tuning":
                        observe(result)
        reserve = plan.metadata["planning"]["reserved_final_trials"]
        while state["phase"] == "explore":
            available = limit - reserve - used()
            if available <= 0:
                state["decisions"].append({"reason": "exploration_trial_budget_exhausted"})
                break
            time_available = max_seconds - elapsed() - reserve * estimated_repeat_seconds()
            if time_available <= 0:
                state["decisions"].append({"reason": "time_reserved_for_final_verification"})
                break
            pending = scheduler.next_batch(available, min(30, time_available), time_available)
            if not pending:
                state["decisions"].append({"reason": "candidate_pool_exhausted"})
                break
            if max_seconds - elapsed() <= reserve * estimated_repeat_seconds():
                state["decisions"].append({"reason": "time_reserved_for_final_verification"})
                break
            result = one_trial(pending[0], "tuning")
            if result is None:
                break
            observe(result)

    while not adaptive and state["phase"] == "explore" and state["next_index"] < len(candidate_ids):
        if used() >= limit - 2 * repetitions:
            state["decisions"].append({"reason": "exploration_trial_budget_exhausted"})
            break
        if elapsed() >= max_seconds:
            state["decisions"].append({"reason": "time_budget_exhausted"})
            break
        candidate_id = candidate_ids[state["next_index"]]
        result = one_trial(candidate_id, "tuning")
        if not any(row["candidate_id"] == candidate_id for row in state["observations"]):
            state["observations"].append({"candidate_id": candidate_id,
                                          "score": result["score"], "status": result["status"],
                                          "output_tokens_per_second": result.get("output_tokens_per_second"),
                                          "latency_seconds": result.get("latency_seconds"),
                                          "reasons": result.get("reasons", [])})
        state["next_index"] += 1
        save()
        if state["next_index"] % 5 == 0:
            current = max((row["score"] for row in state["observations"]
                           if row["score"] is not None), default=None)
            previous = state.get("last_batch_best")
            if current is not None and previous is not None:
                gain = (current - previous) / max(abs(previous), 1e-9)
                state["stagnant_batches"] = 0 if gain > 0.01 else state["stagnant_batches"] + 1
                if state["stagnant_batches"] >= 2:
                    state["decisions"].append({"reason": "converged_no_improvement"})
                    break
            state["last_batch_best"] = current
            save()
    state["phase"] = "final"
    save()
    ranked = sorted((row for row in state["observations"] if row["score"] is not None),
                    key=lambda row: row["score"], reverse=True)

    def verified(candidate_id: str, repeats: list[dict]) -> dict | None:
        scores = [_metric(row, scene["priority"]) for row in repeats]
        effective = repeats[0].get("effective_static_config") if repeats else None
        if not (len(repeats) == repetitions and all(score is not None for score in scores)
                and all(row.get("effective_static_config") == effective for row in repeats)):
            return None
        if adaptive and any(_metric(row, "throughput") is None for row in repeats):
            return None
        median = statistics.median(scores)
        planned = candidates[candidate_id]
        configuration = copy.deepcopy(planned)
        if effective is not None:
            configuration["static_config"] = effective
            configuration["config_hash"] = fingerprint(effective)
        return {"candidate_id": candidate_id, "concurrency": target,
                "priority": scene["priority"], "score": median,
                "metric_value": -median if scene["priority"] == "latency" else median,
                "output_tokens_per_second": statistics.median(
                    row["output_tokens_per_second"] for row in repeats),
                "repetitions": repetitions, "configuration": configuration,
                "planned_configuration": planned, "effective_static_config": effective,
                "repetition_scores": scores,
                "server_reproduce_paths": [row.get("server_reproduce_path") for row in repeats
                                           if row.get("server_reproduce_path")],
                "scene_lengths": [row.get("scene_lengths") for row in repeats]}

    best = None
    if adaptive:
        if "finalist_ids" not in state:
            tolerance = float(limits.get("promotion_tolerance", 0.05))
            finalists = [row for row in ranked if
                         (ranked[0]["score"] - row["score"]) <= max(abs(ranked[0]["score"]), 1e-9) * tolerance]
            state["finalist_ids"] = [row["candidate_id"] for row in
                                     finalists[:plan.metadata["adaptive_policy"]["final_top_k"]]]
            save()
        for candidate_id in state["finalist_ids"]:
            repeats = [result for index in range(repetitions)
                       if (result := one_trial(candidate_id, "final_repeat", f"-adaptive-r{index + 1}")) is not None]
            candidate_best = verified(candidate_id, repeats)
            if candidate_best and (best is None or candidate_best["score"] > best["score"]):
                best = candidate_best
    elif ranked:
        finalist = ranked[0]
        repeats = [one_trial(finalist["candidate_id"], "final_repeat", f"-r{i + 1}")
                   for i in range(repetitions) if used() < limit and elapsed() < max_seconds]
        best = verified(finalist["candidate_id"], repeats)
    baseline = next((row for row in state["observations"] if row["candidate_id"] == baseline_id), None)
    baseline_validated = None
    if baseline and baseline["score"] is not None:
        if best and baseline_id == best["candidate_id"]:
            baseline_validated = {"candidate_id": baseline_id,
                                  "metric_value": best["metric_value"], "repetitions": repetitions}
        else:
            if adaptive:
                repeats = [result for index in range(repetitions)
                           if (result := one_trial(baseline_id, "final_repeat", f"-adaptive-r{index + 1}")) is not None]
            else:
                repeats = [one_trial(baseline_id, "final_repeat", f"-baseline-r{i + 1}")
                           for i in range(repetitions) if used() < limit and elapsed() < max_seconds]
            baseline_best = verified(baseline_id, repeats)
            if baseline_best:
                baseline_validated = {"candidate_id": baseline_id,
                                      "metric_value": baseline_best["metric_value"],
                                      "repetitions": repetitions}
                if best is None or baseline_best["score"] > best["score"]:
                    best = baseline_best
    comparison = {"baseline": baseline, "best": best,
                  "baseline_validated": baseline_validated,
                  "metric": "output_tokens_per_second" if scene["priority"] == "throughput" else "latency_seconds.p95",
                  "target_concurrency": target}
    summary = ("按指定业务并发完成复验；详细指标、实际输出长度与复现脚本见 best.json。"
               if best else "未得到通过重复复验的推荐配置；请检查试验失败原因。")
    report = {"version": 4, "workflow_mode": "tune", "status": "PASS" if best else "INCONCLUSIVE",
              "result_dir": str(root), "scenario": scene,
              "workload_source": plan.metadata["workload_source"],
              "scene_calibration": plan.metadata.get("scene_calibration"),
              "best": best, "comparison": comparison, "summary": summary,
              "trials": used(), "advisor_calls": plan.metadata["planning"]["advisor_calls"],
              "stop_reasons": state["decisions"],
              "length_note": "max_tokens is an upper bound; compare observed server usage in each trial"}
    if adaptive:
        baseline_metric = baseline_validated["metric_value"] if baseline_validated else None
        improvement = None
        if best and baseline_metric:
            improvement = ((best["metric_value"] - baseline_metric) / baseline_metric
                           if scene["priority"] == "throughput" else
                           (baseline_metric - best["metric_value"]) / baseline_metric)
        comparison["relative_improvement"] = improvement
        report.update(strategy="adaptive", adaptive={
            "advisor_calls": scheduler.progress["advisor_calls"],
            "max_advisor_calls": plan.metadata["adaptive_policy"]["max_advisor_calls"],
            "decision_records": scheduler.progress["records"],
            "finalist_ids": state["finalist_ids"],
            "baseline_verified": baseline_validated is not None,
            "elapsed_seconds": elapsed(),
        })
        report = _jsonable(report)
    write_json_atomic(root / "best.json", report)
    collect_results(root)
    state["phase"] = "complete" if best else "inconclusive"
    if adaptive:
        state["report_sha256"] = sha256_file(root / "best.json")
    save()
    return report
