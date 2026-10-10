"""Scene-driven, measured tuning with a local advisor and rule fallback."""
from __future__ import annotations

import copy
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
from .advisor import LocalAdvisor
from .artifacts import sha256_file, write_json_atomic
from .backend_resolution import apply_backend_resolution, expand_auto_backends, resolve_in_image
from .capabilities import probe_environment
from .configuration import load_config
from .discovery import discover_model, inspect_workload, prepare_workload
from .docker_runtime import DockerRuntime, DockerTaskSpec, Mount
from .errors import ConfigError
from .planner import create_search_plan, fingerprint, plan_to_dict
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
                          "gpu_indexes": config.gpu_indexes,
                          "advisor": {key: value for key, value in config.advisor.items()
                                      if key != "authorization"},
                          "scenario": scene},
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


def _save_view_plan(root: Path, template: Plan, state: dict) -> None:
    candidates = tuple(CandidateConfig(**{**row, "gpu_indexes": tuple(row["gpu_indexes"]),
                                          "reasons": tuple(row.get("reasons", ()))})
                       for row in state["candidates"].values())
    metadata = copy.deepcopy(template.metadata)
    metadata["candidates"] = {c.id: asdict(c) for c in candidates}
    write_json_atomic(root / "plan.json", plan_to_dict(Plan(root.name, candidates=candidates, metadata=metadata)))


def run_tuning(config_path: str | Path, *, use_captured: bool = False,
               resume_run: str | Path | None = None, probe=probe_environment,
               advisor: LocalAdvisor | None = None, executor=None) -> dict:
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path, tuning=True)
    if config.scenario is None or config.advisor is None:
        raise ConfigError("tune requires scenario and advisor in experiment.json")
    if config.search.max_trials < 2 * config.search.repetitions + 2:
        raise ConfigError("tune search.max_trials must reserve baseline and winner repeats plus two trials")
    if resume_run is None:
        root, template, catalog = _prepare_new_run(config_path, use_captured=use_captured,
                                                   probe=probe, calibrate=executor is None)
        state = {"version": 1, "phase": "explore", "scenario_hash": fingerprint(config.scenario),
                 "config_sha256": sha256_file(config_path),
                 "workload_sha256": template.metadata["expected_source_sha256"],
                 "bundle_fingerprint": template.metadata["bundle_fingerprint"],
                 "candidates": {}, "pending": [], "trials": {}, "observations": [],
                 "advisor_calls": [], "decisions": [], "elapsed_seconds": 0.0, "batch": 0}
        write_json_atomic(root / "tune-state.json", state)
        _save_view_plan(root, template, state)
    else:
        root = Path(resume_run).expanduser().resolve()
        state = json.loads((root / "tune-state.json").read_text())
        template_doc = json.loads((root / "plan.json").read_text())
        template = Plan(root.name, metadata=template_doc["metadata"])
        catalog = json.loads((root / "catalog.json").read_text())
        if root != Path(template.metadata["results_host"]).resolve():
            raise ConfigError("tune run moved away from its recorded result directory")
        if state["scenario_hash"] != fingerprint(config.scenario):
            raise ConfigError("scene changed since planning; start a new tune run")
        if state["config_sha256"] != sha256_file(config_path):
            raise ConfigError("experiment.json changed since planning; start a new tune run")
        if state["bundle_fingerprint"] != _bundle_fingerprint():
            raise ConfigError("benchmark runtime changed since planning; start a new tune run")
        if sha256_file(Path(template.metadata["jsonl_host"])) != state["workload_sha256"]:
            raise ConfigError("workload changed since planning; start a new tune run")
        if sha256_file(Path(template.metadata["index_host"])) != template.metadata["index_fingerprint"]:
            raise ConfigError("replay index changed since planning")
        model = discover_model(config.model_path, config.model_overrides)
        if fingerprint(asdict(model)) != template.metadata["model_fingerprint"]:
            raise ConfigError("model metadata or weight inventory changed since planning")
        source = Path(template.metadata["jsonl_host"])
        current = probe(_probe_config(replace(config, input_path=source), root))
        if current.status == "FAIL":
            raise ConfigError("current environment checks failed; cannot resume tune")
        previous = template.metadata["environment"]
        now = current.to_dict()
        selected = {i for row in catalog["base_candidates"] for i in row["gpu_indexes"]}
        def devices(snapshot):
            return [{k: gpu.get(k) for k in ("index", "name", "uuid", "compute_capability", "memory_total_mb", "mig_mode")}
                    for gpu in snapshot["gpus"] if gpu["index"] in selected]
        if (devices(now) != devices(previous)
                or now.get("image", {}).get("image_id") != previous.get("image", {}).get("image_id")
                or now.get("container", {}).get("capabilities") != previous.get("container", {}).get("capabilities")
                or now.get("container", {}).get("versions") != previous.get("container", {}).get("versions")
                or now.get("docker", {}).get("facts", {}).get("driver") != previous.get("docker", {}).get("facts", {}).get("driver")):
            raise ConfigError("GPU, image, engine or driver changed since planning")
        if state.get("phase") in {"complete", "inconclusive"} and (root / "best.json").is_file():
            return json.loads((root / "best.json").read_text())
    scene = config.scenario
    local_advisor = advisor or LocalAdvisor(config.advisor)
    state_path = root / "tune-state.json"
    started = time.monotonic()
    previous_elapsed = float(state["elapsed_seconds"])
    limit = config.search.max_trials
    repetitions = config.search.repetitions
    reserve = 2 * repetitions
    target = scene["target_concurrency"]

    def save():
        state["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
        write_json_atomic(state_path, state)
        _save_view_plan(root, template, state)

    def used():
        return sum(1 for entries in state["trials"].values() for entry in entries
                   if entry.get("completed") is True)

    def propose(first: bool):
        visible = _visible_catalog(catalog, state["batch"])
        prior = sorted(state["observations"],
                       key=lambda row: row.get("score") if row.get("score") is not None else -math.inf,
                       reverse=True)[:15]
        prior = [{**row, "configuration": {
            key: value for key, value in state["candidates"][row["candidate_id"]]["static_config"].items()
            if key in {"tp", "dp", "pp", "dp_attention", "backend", "dspark",
                       "mem_fraction_static", "chunked_prefill_size", "max_running_requests"}}}
                 for row in prior]
        payload = {"scenario": scene, "catalog": visible,
                   "tested": prior, "remaining_trials": max(0, limit - used()),
                   "request_count": template.metadata["expected_request_count"]}
        suggestions = []
        try:
            reply = local_advisor.ask("candidates", payload)
            rows = reply.get("candidates")
            if not isinstance(rows, list) or len(rows) > 5:
                raise ValueError("advisor must return at most 5 candidates")
            for row in rows:
                try:
                    suggestions.append(_from_suggestion(row, catalog))
                except ValueError as exc:
                    state["decisions"].append({"reason": "advisor_candidate_rejected", "detail": str(exc)})
            state["advisor_calls"].append({"batch": state["batch"], "status": "PASS",
                                           "accepted": len(suggestions), "first": first})
        except (ConfigError, ValueError) as exc:
            state["advisor_calls"].append({"batch": state["batch"], "status": "FALLBACK",
                                           "reason": str(exc), "first": first})
        seen_hashes = {row["config_hash"] for row in state["candidates"].values()}
        if first and not state["candidates"]:
            baseline = catalog["base_candidates"][0]
            state["candidates"][baseline["id"]] = baseline
            state["pending"].append(baseline["id"])
            seen_hashes.add(baseline["config_hash"])
        for candidate in suggestions:
            if candidate.config_hash not in seen_hashes:
                state["candidates"][candidate.id] = asdict(candidate)
                state["pending"].append(candidate.id)
                seen_hashes.add(candidate.config_hash)
        # Rule candidates guarantee progress when a small local model is slow,
        # unavailable, or repeatedly suggests duplicates.
        needed = max(0, 5 - len(state["pending"]))
        for row in catalog["base_candidates"]:
            if needed <= 0:
                break
            if row["config_hash"] in seen_hashes:
                continue
            state["candidates"][row["id"]] = row
            state["pending"].append(row["id"])
            seen_hashes.add(row["config_hash"])
            needed -= 1
        if needed:
            for base in catalog["base_candidates"]:
                for key, values in catalog["knobs"].items():
                    for value in values:
                        if needed <= 0:
                            break
                        if base["static_config"].get(key) == value:
                            continue
                        candidate = _from_suggestion({"base_id": base["id"],
                            "params": {key: value}, "reason": "rule fallback"}, catalog)
                        if candidate.config_hash in seen_hashes:
                            continue
                        state["candidates"][candidate.id] = asdict(candidate)
                        state["pending"].append(candidate.id)
                        seen_hashes.add(candidate.config_hash)
                        needed -= 1
                    if needed <= 0:
                        break
                if needed <= 0:
                    break
        state["batch"] += 1
        save()

    def one_trial(candidate_id: str, run_class: str, suffix: str = "") -> dict:
        task_id = f"{candidate_id}-{run_class}-c{target}{suffix}"
        entries = state["trials"].setdefault(task_id, [])
        for entry in reversed(entries):
            if entry.get("completed"):
                path = root / entry["manifest"]
                if path.is_file() and sha256_file(path) == entry["sha256"]:
                    return json.loads(path.read_text())
                raise ConfigError(f"recorded trial evidence changed: {path}")
        attempt = len(entries) + 1
        task = PlanTask(task_id, 5 if run_class == "final_repeat" else 3,
                        candidate_id, run_class=run_class, concurrency=target,
                        candidate_hash=state["candidates"][candidate_id]["config_hash"])
        path = root / "trials" / task_id / f"attempt-{attempt:03d}" / "trial.json"
        entries.append({"manifest": str(path.relative_to(root)), "completed": False})
        save()
        metadata = copy.deepcopy(template.metadata)
        metadata["candidates"] = state["candidates"]
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
        result.update(task_id=task_id, candidate_id=candidate_id, concurrency=target,
                      run_class=run_class, candidate_hash=task.candidate_hash,
                      attempt=attempt)
        _result_length_evidence(result, scene)
        result["score"] = _metric(result, scene["priority"])
        write_json_atomic(path, result)
        entries[-1] = {"manifest": str(path.relative_to(root)),
                       "sha256": sha256_file(path), "completed": True}
        save()
        collect_results(root)
        return result

    if not state["candidates"]:
        propose(first=True)
    while state["phase"] == "explore" and used() < limit - reserve:
        if previous_elapsed + time.monotonic() - started >= config.search.max_seconds:
            state["decisions"].append({"reason": "time_budget_exhausted"})
            break
        if not state["pending"]:
            current = max((row["score"] for row in state["observations"]
                           if row["score"] is not None), default=None)
            previous = state.get("last_batch_best")
            if current is not None and previous is not None:
                improvement = (current - previous) / max(abs(previous), 1e-9)
                state["stagnant_batches"] = 0 if improvement > 0.01 else state.get("stagnant_batches", 0) + 1
                if state["stagnant_batches"] >= 2:
                    state["decisions"].append({"reason": "converged_no_improvement"})
                    break
            state["last_batch_best"] = current
            propose(first=False)
            if not state["pending"]:
                state["decisions"].append({"reason": "no_new_candidates"})
                break
            if previous_elapsed + time.monotonic() - started >= config.search.max_seconds:
                state["decisions"].append({"reason": "time_budget_exhausted_after_advisor"})
                break
        candidate_id = state["pending"].pop(0)
        save()
        result = one_trial(candidate_id, "tuning")
        if not any(row["candidate_id"] == candidate_id for row in state["observations"]):
            state["observations"].append({"candidate_id": candidate_id,
                                          "score": result["score"],
                                          "status": result["status"],
                                          "output_tokens_per_second": result.get("output_tokens_per_second"),
                                          "latency_seconds": result.get("latency_seconds"),
                                          "reasons": result.get("reasons", [])})
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
        median = statistics.median(scores)
        planned = state["candidates"][candidate_id]
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
    if ranked:
        finalist = ranked[0]
        repeats = [one_trial(finalist["candidate_id"], "final_repeat", f"-r{i + 1}")
                   for i in range(repetitions) if (used() < limit and
                       previous_elapsed + time.monotonic() - started < config.search.max_seconds)]
        best = verified(finalist["candidate_id"], repeats)
    baseline = next((row for row in state["observations"] if row["candidate_id"] == catalog["base_candidates"][0]["id"]), None)
    baseline_validated = None
    if baseline and baseline["score"] is not None:
        if best and baseline["candidate_id"] == best["candidate_id"]:
            baseline_validated = {"candidate_id": baseline["candidate_id"],
                                  "metric_value": best["metric_value"], "repetitions": repetitions}
        else:
            baseline_repeats = [one_trial(baseline["candidate_id"], "final_repeat", f"-baseline-r{i + 1}")
                                for i in range(repetitions) if (used() < limit and
                                    previous_elapsed + time.monotonic() - started < config.search.max_seconds)]
            baseline_best = verified(baseline["candidate_id"], baseline_repeats)
            if baseline_best:
                baseline_validated = {"candidate_id": baseline["candidate_id"],
                                      "metric_value": baseline_best["metric_value"],
                                      "repetitions": repetitions}
                if best is None or baseline_best["score"] > best["score"]:
                    best = baseline_best
    comparison = {"baseline": baseline, "best": best,
                  "baseline_validated": baseline_validated,
                  "metric": "output_tokens_per_second" if scene["priority"] == "throughput" else "latency_seconds.p95",
                  "target_concurrency": target}
    summary = None
    if best:
        try:
            explanation = local_advisor.ask("explain", {"scenario": scene, "comparison": comparison})
            if isinstance(explanation.get("summary"), str):
                summary = explanation["summary"][:2000]
        except ConfigError as exc:
            state["advisor_calls"].append({"task": "explain", "status": "FALLBACK", "reason": str(exc)})
    if summary is None:
        summary = ("按指定业务并发完成复验；详细指标、实际输出长度与复现脚本见 best.json。"
                   if best else "未得到通过重复复验的推荐配置；请检查试验失败原因。")
    report = {"version": 3, "workflow_mode": "tune", "status": "PASS" if best else "INCONCLUSIVE",
              "result_dir": str(root), "scenario": scene, "workload_source": template.metadata["workload_source"],
              "scene_calibration": template.metadata.get("scene_calibration"),
              "best": best, "comparison": comparison, "summary": summary,
              "trials": used(), "advisor_calls": state["advisor_calls"],
              "stop_reasons": state["decisions"],
              "length_note": "max_tokens is an upper bound; compare observed server usage in each trial"}
    write_json_atomic(root / "best.json", report)
    collect_results(root)
    state["phase"] = "complete" if best else "inconclusive"
    save()
    return report
