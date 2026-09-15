#!/usr/bin/env python3
"""Single-server web console for Automation runs.

The server intentionally uses only the Python standard library so the
Automation directory remains copyable to benchmark hosts.  Its HTTP API is
kept small and REST-shaped; it can be moved to FastAPI later without changing
the static frontend contract.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import mimetypes
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

AUTOMATION_ROOT = Path(__file__).resolve().parents[1]
if str(AUTOMATION_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOMATION_ROOT))

from automation.adapters import ReplayAdapter, read_attempt  # noqa: E402
from automation.artifacts import write_json_atomic  # noqa: E402
from automation.planner import fingerprint, load_plan, plan_to_dict  # noqa: E402
from automation.search import _candidate_gpu_count, _start_concurrency, collect_results  # noqa: E402
from automation.types import PlanTask  # noqa: E402
from automation.workflow import _bundle_fingerprint  # noqa: E402

DEFAULT_RESULT_ROOT = Path(os.environ.get("AUTOMATION_RESULT_ROOT", "/data/hjh/Automation/results"))
DEFAULT_CONFIG_PATH = Path(os.environ.get("AUTOMATION_CONFIG", str(AUTOMATION_ROOT / "configs" / "experiment.json")))
MAX_ARTIFACT_BYTES = int(os.environ.get("AUTOMATION_WEB_MAX_ARTIFACT_BYTES", str(2 * 1024 * 1024)))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_load(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(value, encoding="utf-8")
        tmp.replace(path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def safe_result_root(raw: str | None) -> Path:
    text = raw or str(DEFAULT_RESULT_ROOT)
    return Path(os.path.expandvars(text)).expanduser().resolve()


def safe_run_dir(result_root: Path, run_id: str) -> Path:
    if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise ValueError("invalid run id")
    root = result_root.resolve()
    path = (root / run_id).resolve()
    if path != root and root in path.parents:
        return path
    raise ValueError("run path escapes result root")


def safe_artifact_path(run_dir: Path, raw_path: str) -> Path:
    if not raw_path:
        raise ValueError("artifact path is required")
    decoded = unquote(raw_path)
    pure = PurePosixPath(decoded)
    if pure.is_absolute() or any(part in {"..", ""} for part in pure.parts):
        raise ValueError("invalid artifact path")
    path = (run_dir / Path(*pure.parts)).resolve()
    root = run_dir.resolve()
    if path == root or root not in path.parents:
        raise ValueError("artifact path escapes run directory")
    return path


def file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def summarize_best(best_doc: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(best_doc, dict):
        return {}
    best = best_doc.get("best")
    return {
        "status": best_doc.get("status"),
        "trials": best_doc.get("trials"),
        "elapsed_seconds": best_doc.get("elapsed_seconds"),
        "best": {
            "candidate_id": best.get("candidate_id"),
            "concurrency": best.get("concurrency"),
            "output_tokens_per_second": best.get("output_tokens_per_second"),
            "repetitions": best.get("repetitions"),
        } if isinstance(best, dict) else None,
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    resolved = json_load(run_dir / "resolved-config.json", {})
    best_doc = json_load(run_dir / "best.json", {})
    plan = json_load(run_dir / "plan.json", {})
    state = json_load(run_dir / "search-state.json", {})
    rows_doc = json_load(run_dir / "results-index.json", {})
    files = {
        "resolved_config": (run_dir / "resolved-config.json").is_file(),
        "environment": (run_dir / "environment.json").is_file(),
        "plan": (run_dir / "plan.json").is_file(),
        "search_state": (run_dir / "search-state.json").is_file(),
        "results_index": (run_dir / "results-index.json").is_file(),
        "leaderboard": (run_dir / "leaderboard.json").is_file(),
        "best": (run_dir / "best.json").is_file(),
    }
    row_count = len(rows_doc.get("rows", [])) if isinstance(rows_doc, dict) else 0
    candidates = plan.get("candidates", []) if isinstance(plan, dict) else []
    status = None
    if isinstance(best_doc, dict):
        status = best_doc.get("status")
    if status is None and isinstance(state, dict) and state:
        status = "RUNNING_OR_INTERRUPTED"
    if status is None and files["plan"]:
        status = "PLANNED"
    if status is None:
        status = "UNKNOWN"
    return {
        "id": run_dir.name,
        "path": str(run_dir),
        "mtime": file_mtime(run_dir),
        "status": status,
        "files": files,
        "candidate_count": len(candidates) if isinstance(candidates, list) else None,
        "row_count": row_count,
        "best_summary": summarize_best(best_doc if isinstance(best_doc, dict) else None),
        "served_model_name": (
            resolved.get("served_model_name")
            if isinstance(resolved, dict)
            else None
        ),
        "created_event_result_dir": (
            resolved.get("result_dir")
            if isinstance(resolved, dict)
            else None
        ),
    }


def scan_runs(result_root: Path) -> list[dict[str, Any]]:
    if not result_root.is_dir():
        return []
    runs = []
    for child in result_root.iterdir():
        if not child.is_dir():
            continue
        if any((child / name).exists() for name in (
            "plan.json", "best.json", "resolved-config.json", "results-index.json", "search-state.json"
        )):
            runs.append(summarize_run(child))
    runs.sort(key=lambda item: item.get("mtime", 0), reverse=True)
    return runs


def candidate_static(candidate: dict[str, Any]) -> dict[str, Any]:
    static = candidate.get("static_config")
    return static if isinstance(static, dict) else {}


def plan_candidates(run_dir: Path) -> list[dict[str, Any]]:
    plan = json_load(run_dir / "plan.json", {})
    raw_candidates = plan.get("candidates", []) if isinstance(plan, dict) else []
    rows_doc = json_load(run_dir / "results-index.json", {})
    rows = rows_doc.get("rows", []) if isinstance(rows_doc, dict) else []
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("candidate_id"):
            by_candidate.setdefault(str(row["candidate_id"]), []).append(row)
    result = []
    for plan_order, candidate in enumerate(raw_candidates if isinstance(raw_candidates, list) else []):
        if not isinstance(candidate, dict):
            continue
        static = candidate_static(candidate)
        deployment = static.get("deployment", {}) if isinstance(static.get("deployment"), dict) else {}
        attempts = by_candidate.get(str(candidate.get("id")), [])
        valid = [r for r in attempts if r.get("status") == "VALID" and isinstance(r.get("output_tokens_per_second"), (int, float))]
        latest = max(attempts, key=lambda r: str(r.get("manifest", "")), default=None)
        best = max(valid, key=lambda r: float(r.get("output_tokens_per_second", 0)), default=None)
        result.append({
            "id": candidate.get("id"),
            "plan_order": plan_order,
            "tp": candidate.get("tp", static.get("tp")),
            "dp": candidate.get("dp", static.get("dp")),
            "pp": candidate.get("pp", static.get("pp")),
            "dp_attention": candidate.get("dp_attention", static.get("dp_attention")),
            "backend": candidate.get("backend", static.get("backend")) or "auto",
            "moe_a2a_backend": static.get("moe_a2a_backend"),
            "dspark": candidate.get("dspark", static.get("dspark")),
            "gpu_indexes": candidate.get("gpu_indexes", static.get("gpu_indexes", [])),
            "deployment_label": deployment.get("label"),
            "instance_count": deployment.get("instance_count"),
            "gpus_per_instance": deployment.get("gpus_per_instance"),
            "mem_fraction_static": static.get("mem_fraction_static"),
            "max_running_requests": static.get("max_running_requests"),
            "chunked_prefill_size": static.get("chunked_prefill_size"),
            "attempt_count": len(attempts),
            "latest_status": latest.get("status") if isinstance(latest, dict) else "NOT_RUN",
            "latest_reasons": latest.get("reasons") if isinstance(latest, dict) else [],
            "best_output_tokens_per_second": best.get("output_tokens_per_second") if isinstance(best, dict) else None,
            "best_concurrency": best.get("concurrency") if isinstance(best, dict) else None,
        })
    return result


def plan_order_map(run_dir: Path) -> dict[str, int]:
    plan_doc = json_load(run_dir / "plan.json", {})
    raw_candidates = plan_doc.get("candidates", []) if isinstance(plan_doc, dict) else []
    result: dict[str, int] = {}
    for idx, candidate in enumerate(raw_candidates if isinstance(raw_candidates, list) else []):
        if isinstance(candidate, dict) and candidate.get("id") is not None:
            result[str(candidate["id"])] = idx
    return result


def trial_order_map(run_dir: Path) -> dict[str, int]:
    state = json_load(run_dir / "search-state.json", {})
    raw_trials = state.get("trials", {}) if isinstance(state, dict) else {}
    if not isinstance(raw_trials, dict):
        return {}
    return {str(task_id): idx for idx, task_id in enumerate(raw_trials.keys())}


def _decorate_trial_rows(run_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_order = plan_order_map(run_dir)
    task_order = trial_order_map(run_dir)
    decorated = []
    for row in rows:
        item = dict(row)
        candidate_id = item.get("candidate_id")
        task_id = item.get("task_id") or item.get("id")
        if candidate_id is not None and str(candidate_id) in candidate_order:
            item["plan_order"] = candidate_order[str(candidate_id)]
        if task_id is not None and str(task_id) in task_order:
            item["trial_order"] = task_order[str(task_id)]
        decorated.append(item)
    decorated.sort(key=lambda item: (
        item.get("plan_order") is None,
        int(item.get("plan_order") if item.get("plan_order") is not None else 1_000_000),
        item.get("trial_order") is None,
        int(item.get("trial_order") if item.get("trial_order") is not None else 1_000_000),
        str(item.get("task_id") or item.get("id") or ""),
        int(item.get("attempt") or 0),
    ))
    return decorated


def trial_rows(run_dir: Path, include_debug: bool = True) -> list[dict[str, Any]]:
    rows_doc = json_load(run_dir / "results-index.json", {})
    raw_rows = rows_doc.get("rows", []) if isinstance(rows_doc, dict) and isinstance(rows_doc.get("rows"), list) else []
    rows = [row for row in raw_rows if isinstance(row, dict)]
    if include_debug:
        debug = json_load(run_dir / "debug-trials" / "debug-index.json", {})
        debug_rows = debug.get("rows", []) if isinstance(debug, dict) else []
        for row in debug_rows if isinstance(debug_rows, list) else []:
            if isinstance(row, dict):
                item = dict(row)
                item["debug"] = True
                rows.append(item)
    return _decorate_trial_rows(run_dir, rows)


def candidate_detail(run_dir: Path, candidate_id: str) -> dict[str, Any]:
    plan_doc = json_load(run_dir / "plan.json", {})
    candidates = plan_doc.get("candidates", []) if isinstance(plan_doc, dict) else []
    candidate = next((c for c in candidates if isinstance(c, dict) and c.get("id") == candidate_id), None)
    if candidate is None:
        raise ValueError("candidate not found")
    rows = [row for row in trial_rows(run_dir) if row.get("candidate_id") == candidate_id]
    rows.sort(key=lambda row: str(row.get("manifest") or row.get("result_path") or ""))
    static = candidate_static(candidate)
    metadata = plan_doc.get("metadata", {}) if isinstance(plan_doc, dict) else {}
    concurrency = default_debug_concurrency(metadata, {candidate_id: candidate}, candidate, rows)
    preview = preview_launch(metadata, candidate, concurrency)
    return {
        "candidate": candidate,
        "metadata": {
            "served_model_name": metadata.get("served_model_name"),
            "tool_call_parser": metadata.get("tool_call_parser"),
            "reasoning_parser": metadata.get("reasoning_parser"),
            "image": metadata.get("image"),
            "service_port": metadata.get("service_port"),
            "search": metadata.get("search"),
        },
        "static_config": static,
        "default_debug": {
            "concurrency": concurrency,
            "run_class": default_debug_run_class(rows),
            "mode": default_debug_mode(rows),
        },
        "launch_preview": preview,
        "trials": rows,
        "artifacts": list_candidate_artifacts(run_dir, rows),
    }


def default_debug_run_class(rows: list[dict[str, Any]]) -> str:
    failed = [row for row in rows if row.get("status") != "VALID"]
    if failed:
        return str(failed[-1].get("run_class") or "concurrency")
    return "concurrency"


def default_debug_mode(rows: list[dict[str, Any]]) -> str:
    failed = [row for row in rows if row.get("status") != "VALID"]
    if failed:
        mode = str(failed[-1].get("mode") or "closed-loop")
        return mode.replace("_", "-")
    return "closed-loop"


def default_debug_concurrency(
    metadata: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
) -> int:
    failed = [row for row in rows if row.get("status") != "VALID" and row.get("concurrency")]
    if failed:
        try:
            return max(1, int(failed[-1]["concurrency"]))
        except (TypeError, ValueError):
            pass
    search = metadata.get("search", {}) if isinstance(metadata.get("search"), dict) else {}
    try:
        maximum = int(search.get("concurrency_max") or metadata.get("expected_request_count") or 64)
    except (TypeError, ValueError):
        maximum = 64
    maximum = max(1, maximum)
    try:
        return int(_start_concurrency(search, candidates, maximum))
    except Exception:
        gpu_count = _candidate_gpu_count(candidate)
        return min(maximum, max(8, 8 * gpu_count))


def preview_launch(metadata: dict[str, Any], candidate: dict[str, Any], concurrency: int) -> dict[str, Any]:
    static = candidate_static(candidate)
    snapshot = metadata.get("model_snapshot", {})
    raw = snapshot.get("raw", {}) if isinstance(snapshot, dict) else {}
    is_moe = bool(raw.get("is_moe")) if isinstance(raw, dict) else False
    deployment = static.get("deployment", {}) if isinstance(static.get("deployment"), dict) else {}
    instance_count = 1
    try:
        instance_count = max(1, int(deployment.get("instance_count", 1)))
    except (TypeError, ValueError):
        pass
    configured = static.get("max_running_requests")
    try:
        configured_int = max(1, int(configured)) if configured is not None else None
    except (TypeError, ValueError):
        configured_int = None
    per_instance = max(1, (max(1, int(concurrency)) + instance_count - 1) // instance_count)
    effective_max = min(configured_int, per_instance) if configured_int is not None else per_instance
    params: dict[str, Any] = {
        "model_path": "/model",
        "served_model_name": metadata.get("served_model_name"),
        "tp": static.get("tp", candidate.get("tp")),
        "dp": static.get("dp", candidate.get("dp")),
        "pp": static.get("pp", candidate.get("pp")),
        "dp_attention": static.get("dp_attention", candidate.get("dp_attention")),
        "dp_lm_head": static.get("dp_lm_head"),
        "dspark": static.get("dspark", candidate.get("dspark")),
        "mem_fraction_static": static.get("mem_fraction_static"),
        "max_running_requests": effective_max,
        "chunked_prefill_size": static.get("chunked_prefill_size"),
    }
    if is_moe:
        params["backend"] = static.get("backend") or "auto"
        params["moe_a2a_backend"] = static.get("moe_a2a_backend") or "auto"
    for key in ("tool_call_parser", "reasoning_parser", "quantization", "chat_template_kwargs"):
        if metadata.get(key) not in (None, ""):
            params[key] = metadata.get(key)
    flags = [
        "sglang", "serve",
        "--trust-remote-code",
        "--model-path", str(params["model_path"]),
        "--served-model-name", str(params.get("served_model_name") or "model"),
        "--enable-metrics",
        "--enable-cache-report",
        "--tp-size", str(params.get("tp") or 1),
        "--dp", str(params.get("dp") or 1),
        "--pp-size", str(params.get("pp") or 1),
        "--mem-fraction-static", str(params.get("mem_fraction_static") or 0.85),
        "--max-running-requests", str(params.get("max_running_requests") or 1),
        "--host", "0.0.0.0",
        "--port", str(metadata.get("service_port") or 25080),
        "--chunked-prefill-size", str(params.get("chunked_prefill_size") or 8192),
        "--default-chat-template-kwargs", json.dumps(params.get("chat_template_kwargs") or {}, ensure_ascii=False, sort_keys=True),
    ]
    optional = {
        "tool_call_parser": "--tool-call-parser",
        "reasoning_parser": "--reasoning-parser",
        "quantization": "--quantization",
        "backend": "--moe-runner-backend",
        "moe_a2a_backend": "--moe-a2a-backend",
    }
    for key, flag in optional.items():
        value = params.get(key)
        if value not in (None, ""):
            flags += [flag, str(value)]
    if params.get("dp_attention"):
        flags.append("--enable-dp-attention")
    if params.get("dp_lm_head"):
        flags.append("--enable-dp-lm-head")
    if params.get("dspark"):
        flags += ["--speculative-algorithm", "DSPARK"]
    return {
        "concurrency": concurrency,
        "instance_count": instance_count,
        "requested_parameters": params,
        "sglang_command_preview": flags,
        "deployment": deployment or None,
        "note": "Preview is computed from plan metadata. After a real attempt, server.command.sh and server.info.json are the source of truth.",
    }


def list_candidate_artifacts(run_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        base = row.get("result_path")
        manifest = row.get("manifest")
        if not base and isinstance(manifest, str):
            if manifest.endswith("/trial.json"):
                base = str((run_dir / manifest).parent)
        if not base:
            continue
        try:
            base_path = Path(base)
            if not base_path.is_absolute():
                base_path = (run_dir / base_path).resolve()
            base_path = base_path.resolve()
            base_path.relative_to(run_dir.resolve())
        except Exception:
            continue
        for item in list_artifacts_for_base(run_dir, base_path):
            item["trial"] = row.get("task_id")
            result.append(item)
    return result


def list_artifacts_for_base(run_dir: Path, base_path: Path) -> list[dict[str, Any]]:
    root = run_dir.resolve()
    base = base_path.resolve()
    try:
        base.relative_to(root)
    except ValueError:
        raise ValueError("artifact base escapes run directory")
    if base.is_file():
        base = base.parent
    if not base.is_dir():
        return []
    suffixes = {".log", ".sh", ".json", ".jsonl"}
    paths: list[Path] = []
    try:
        iterator = base.iterdir() if base == root else base.rglob("*")
        for candidate in iterator:
            if candidate.is_file() and candidate.suffix.lower() in suffixes:
                paths.append(candidate)
    except OSError:
        pass
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for path in paths:
        try:
            rel = path.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        if rel in seen:
            continue
        seen.add(rel)
        result.append({
            "name": path.name,
            "path": rel,
            "size": path.stat().st_size,
            "mtime": path.stat().st_mtime,
        })
    result.sort(key=lambda item: item["path"])
    return result


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}

    def create_subprocess(self, name: str, argv: list[str], cwd: Path, target_run: Path | None = None) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "name": name,
            "kind": "subprocess",
            "argv": argv,
            "cwd": str(cwd),
            "target_run": str(target_run.resolve()) if target_run is not None else None,
            "status": "running",
            "returncode": None,
            "pid": None,
            "stop_requested": False,
            "started_at": utc_now(),
            "finished_at": None,
            "lines": [],
            "events": [],
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(target=self._run_subprocess, args=(job_id,), daemon=True)
        thread.start()
        return self.get(job_id)

    def create_function(
        self,
        name: str,
        func,
        *args: Any,
        target_run: Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "name": name,
            "kind": "function",
            "argv": [],
            "cwd": str(AUTOMATION_ROOT),
            "target_run": str(target_run.resolve()) if target_run is not None else None,
            "status": "running",
            "returncode": None,
            "started_at": utc_now(),
            "finished_at": None,
            "lines": [],
            "events": [],
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(target=self._run_function, args=(job_id, func, args, kwargs), daemon=True)
        thread.start()
        return self.get(job_id)

    def _append_line(self, job_id: str, line: str) -> None:
        line = line.rstrip("\n")
        event = None
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                event = parsed
        except json.JSONDecodeError:
            pass
        with self._lock:
            job = self._jobs[job_id]
            job["lines"].append(line)
            if len(job["lines"]) > 4000:
                job["lines"] = job["lines"][-4000:]
            if event is not None:
                job["events"].append(event)
                if len(job["events"]) > 1000:
                    job["events"] = job["events"][-1000:]

    def _run_subprocess(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            argv = list(job["argv"])
            cwd = Path(job["cwd"])
        try:
            process = subprocess.Popen(
                argv,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                shell=False,
                start_new_session=(os.name != "nt"),
            )
            with self._lock:
                self._processes[job_id] = process
                job = self._jobs[job_id]
                job["pid"] = process.pid
            assert process.stdout is not None
            for line in process.stdout:
                self._append_line(job_id, line)
            returncode = process.wait()
            result = None
            with self._lock:
                lines = list(self._jobs[job_id]["lines"])
            for line in reversed(lines):
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    result = parsed
                    break
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "exited"
                job["returncode"] = returncode
                job["finished_at"] = utc_now()
                job["result"] = result
                self._processes.pop(job_id, None)
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "error"
                job["returncode"] = 1
                job["finished_at"] = utc_now()
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["lines"].append(traceback.format_exc())
                self._processes.pop(job_id, None)

    def _run_function(self, job_id: str, func, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        try:
            result = func(lambda line: self._append_line(job_id, line), *args, **kwargs)
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "exited"
                job["returncode"] = 0
                job["finished_at"] = utc_now()
                job["result"] = result
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "error"
                job["returncode"] = 1
                job["finished_at"] = utc_now()
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["lines"].append(traceback.format_exc())

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return copy.deepcopy(self._jobs[job_id])

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [copy.deepcopy(job) for job in self._jobs.values()]
        jobs.sort(key=lambda job: job.get("started_at") or "", reverse=True)
        return jobs

    def running_for_run(self, run_dir: Path) -> list[dict[str, Any]]:
        target = str(run_dir.resolve())
        plan_path = str((run_dir / "plan.json").resolve())
        with self._lock:
            jobs = []
            for job in self._jobs.values():
                if job.get("status") != "running":
                    continue
                argv = [str(arg) for arg in job.get("argv", [])]
                if job.get("target_run") == target or plan_path in argv:
                    jobs.append(copy.deepcopy(job))
            return jobs

    def stop(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
            process = self._processes.get(job_id)
            if job.get("status") != "running" or process is None:
                return copy.deepcopy(job)
            job["stop_requested"] = True
            job["lines"].append("stop requested from web console")
            pid = process.pid
        try:
            if os.name != "nt":
                os.killpg(pid, signal.SIGINT)
            else:
                process.send_signal(signal.CTRL_BREAK_EVENT if hasattr(signal, "CTRL_BREAK_EVENT") else signal.SIGINT)
        except ProcessLookupError:
            pass
        except Exception as exc:
            with self._lock:
                self._jobs[job_id]["lines"].append(f"failed to send stop signal: {type(exc).__name__}: {exc}")
        return self.get(job_id)

    def stop_all(self) -> None:
        with self._lock:
            ids = list(self._processes)
        for job_id in ids:
            try:
                self.stop(job_id)
            except Exception:
                pass


JOBS = JobManager()


def python_executable() -> str:
    return sys.executable


def benchctl_argv(command: str, *args: str) -> list[str]:
    return [python_executable(), str(AUTOMATION_ROOT / "benchctl.py"), command, *args]


def run_plan_job(config_path: Path) -> dict[str, Any]:
    return JOBS.create_subprocess("plan", benchctl_argv("plan", "--config", str(config_path)), AUTOMATION_ROOT)


def run_saved_plan_job(run_dir: Path, resume: bool = False) -> dict[str, Any]:
    argv = benchctl_argv("run", "--plan", str(run_dir / "plan.json"))
    if resume:
        argv.append("--resume")
    return JOBS.create_subprocess("resume" if resume else "run", argv, AUTOMATION_ROOT, target_run=run_dir)


def run_auto_job(config_path: Path) -> dict[str, Any]:
    return JOBS.create_subprocess("auto", benchctl_argv("auto", "--config", str(config_path)), AUTOMATION_ROOT)


def collect_run(run_dir: Path) -> dict[str, Any]:
    collected = collect_results(run_dir)
    return {"status": "PASS", "rows": len(collected.get("rows", [])), "result_dir": str(run_dir)}


def code_state(run_dir: Path | None = None) -> dict[str, Any]:
    current = _bundle_fingerprint()
    result = {"current_bundle_fingerprint": current}
    if run_dir is not None:
        plan = json_load(run_dir / "plan.json", {})
        planned = None
        if isinstance(plan, dict):
            planned = (plan.get("metadata") or {}).get("bundle_fingerprint") if isinstance(plan.get("metadata"), dict) else None
        result.update({
            "planned_bundle_fingerprint": planned,
            "matches": planned == current if planned else None,
        })
    return result


def adopt_current_runtime(run_dir: Path, note: str = "") -> dict[str, Any]:
    plan_path = run_dir / "plan.json"
    state_path = run_dir / "search-state.json"
    if not plan_path.is_file():
        raise ValueError("plan.json does not exist")
    new_bundle = _bundle_fingerprint()
    plan = load_plan(plan_path)
    old_bundle = plan.metadata.get("bundle_fingerprint")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    adoption_dir = run_dir / "runtime-adoptions" / (timestamp + "-" + uuid.uuid4().hex[:8])
    adoption_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(plan_path, adoption_dir / "plan.json.before")
    if state_path.is_file():
        shutil.copy2(state_path, adoption_dir / "search-state.json.before")
    plan.metadata["bundle_fingerprint"] = new_bundle
    plan_data = plan_to_dict(plan)
    write_json_atomic(plan_path, plan_data)
    state_changed = False
    if state_path.is_file():
        state = json_load(state_path, {})
        if not isinstance(state, dict):
            raise ValueError("search-state.json is not a JSON object")
        state["plan_hash"] = fingerprint(plan_data)
        write_json_atomic(state_path, state)
        state_changed = True
    audit = {
        "created_at": utc_now(),
        "action": "adopt_current_runtime",
        "note": note,
        "old_bundle_fingerprint": old_bundle,
        "new_bundle_fingerprint": new_bundle,
        "plan_path": str(plan_path),
        "state_path": str(state_path) if state_path.is_file() else None,
        "state_changed": state_changed,
    }
    write_json_atomic(adoption_dir / "adoption.json", audit)
    return {"status": "PASS", "adoption": audit, "adoption_dir": str(adoption_dir)}


def debug_index_path(run_dir: Path) -> Path:
    return run_dir / "debug-trials" / "debug-index.json"


def append_debug_row(run_dir: Path, row: dict[str, Any]) -> None:
    path = debug_index_path(run_dir)
    document = json_load(path, {})
    rows = document.get("rows", []) if isinstance(document, dict) and isinstance(document.get("rows"), list) else []
    rows.append(row)
    write_json_atomic(path, {"version": 1, "run_id": run_dir.name, "rows": rows})


def find_candidate(plan_doc: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in plan_doc.get("candidates", []):
        if isinstance(candidate, dict) and candidate.get("id") == candidate_id:
            return candidate
    raise ValueError("candidate not found")


def find_candidate_optional(plan_doc: dict[str, Any], candidate_id: str) -> dict[str, Any] | None:
    try:
        return find_candidate(plan_doc, candidate_id)
    except ValueError:
        return None


def task_id_from_row(row: dict[str, Any]) -> str:
    raw = row.get("task_id") or row.get("id")
    if raw:
        return str(raw)
    manifest = row.get("manifest")
    if isinstance(manifest, str):
        parts = PurePosixPath(manifest).parts
        if len(parts) >= 4 and parts[0] == "trials":
            return parts[1]
    raise ValueError("trial task_id is required")


def trial_row_matches(row: dict[str, Any], *, task_id: str, attempt: str, manifest: str) -> bool:
    if manifest and str(row.get("manifest") or "") == manifest:
        return True
    row_task = str(row.get("task_id") or row.get("id") or "")
    if task_id and row_task == task_id:
        return not attempt or str(row.get("attempt") or "") == str(attempt)
    return False


def find_official_trial_row(run_dir: Path, body: dict[str, Any]) -> dict[str, Any]:
    task_id = str(body.get("task_id") or "")
    attempt = str(body.get("attempt") or "")
    manifest = str(body.get("manifest") or "")
    for row in trial_rows(run_dir, include_debug=False):
        if trial_row_matches(row, task_id=task_id, attempt=attempt, manifest=manifest):
            return row
    raise ValueError("official trial not found")


def repair_candidate_config(plan_doc: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(row.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError("trial candidate_id is required")
    planned = find_candidate_optional(plan_doc, candidate_id)
    if planned is not None:
        return copy.deepcopy(planned)

    for key in ("planned_configuration", "configuration"):
        value = row.get(key)
        if isinstance(value, dict) and value.get("id") == candidate_id:
            return copy.deepcopy(value)

    if "-tune-" in candidate_id:
        base_id = candidate_id.split("-tune-", 1)[0]
        base = find_candidate_optional(plan_doc, base_id)
        static = row.get("effective_static_config")
        if base is not None and isinstance(static, dict):
            tuned = copy.deepcopy(base)
            tuned["id"] = candidate_id
            tuned["static_config"] = copy.deepcopy(static)
            tuned["config_hash"] = str(row.get("candidate_hash") or fingerprint(static))
            return tuned

    raise ValueError("candidate config for this trial is not available")


def plan_task_from_trial_row(row: dict[str, Any], candidate: dict[str, Any]) -> PlanTask:
    task_id = task_id_from_row(row)
    candidate_id = str(row.get("candidate_id") or candidate.get("id") or "")
    if not candidate_id:
        raise ValueError("trial candidate_id is required")
    run_class = str(row.get("run_class") or "concurrency")
    if run_class not in {"smoke", "concurrency", "tuning", "final_repeat", "open_loop", "diagnostic", "formal"}:
        raise ValueError("invalid run_class")
    mode = str(row.get("mode") or ("open-loop" if row.get("scale") is not None else "closed-loop")).replace("_", "-")
    if mode not in {"closed-loop", "open-loop"}:
        raise ValueError("mode must be closed-loop or open-loop")
    concurrency = row.get("concurrency")
    if concurrency is None:
        raise ValueError("trial concurrency is required")
    scale = row.get("scale")
    return PlanTask(
        id=task_id,
        stage=5 if run_class == "final_repeat" else 3,
        candidate_id=candidate_id,
        run_class=run_class,
        mode=mode,
        concurrency=max(1, int(concurrency)),
        scale=float(scale) if scale is not None else None,
        candidate_hash=str(row.get("candidate_hash") or candidate.get("config_hash") or ""),
    )


def normalize_trial_result(result: dict[str, Any]) -> dict[str, Any]:
    value = result.get("output_tokens_per_second")
    if (
        result.get("status") == "VALID"
        and not (
            type(value) in (int, float)
            and math.isfinite(float(value))
            and float(value) > 0
        )
    ):
        result.update(
            status="INCONCLUSIVE",
            output_tokens_per_second=None,
            reasons=["invalid_throughput_measurement"],
        )
    return result


def next_repair_attempt(run_dir: Path, task_id: str, history: list[Any]) -> tuple[int, Path]:
    attempt = len(history) + 1
    while True:
        attempt_dir = run_dir / "trials" / task_id / f"attempt-{attempt:03d}"
        if not attempt_dir.exists():
            return attempt, attempt_dir
        attempt += 1


def official_repair_worker(
    emit,
    run_dir: Path,
    selector: dict[str, Any],
) -> dict[str, Any]:
    row = find_official_trial_row(run_dir, selector)
    plan = load_plan(run_dir / "plan.json")
    plan_doc = plan_to_dict(plan)
    metadata = copy.deepcopy(plan.metadata)
    candidates = {c["id"]: c for c in plan_doc.get("candidates", []) if isinstance(c, dict) and c.get("id")}
    candidate = repair_candidate_config(plan_doc, row)
    candidates[str(candidate["id"])] = candidate
    metadata["candidates"] = candidates
    metadata["benchmark_dir"] = str(AUTOMATION_ROOT / "benchmarks")
    metadata["repair_current_bundle_fingerprint"] = _bundle_fingerprint()

    task = plan_task_from_trial_row(row, candidate)
    state_path = run_dir / "search-state.json"
    state = json_load(state_path, {})
    if not isinstance(state, dict):
        raise ValueError("search-state.json is not a JSON object")
    if "trials" not in state:
        state["trials"] = {}
    if not isinstance(state["trials"], dict):
        raise ValueError("search-state.json.trials is not a JSON object")
    history = state["trials"].setdefault(task.id, [])
    if not isinstance(history, list):
        raise ValueError(f"search-state trial history for {task.id} is not a list")
    if "plan_hash" not in state:
        state["plan_hash"] = fingerprint(plan_doc)

    attempt, attempt_dir = next_repair_attempt(run_dir, task.id, history)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    trial_path = attempt_dir / "trial.json"
    entry = {
        "manifest": str(trial_path.relative_to(run_dir)),
        "task": asdict(task),
        "fingerprint": None,
        "repair": {
            "created_at": utc_now(),
            "source_manifest": row.get("manifest"),
            "source_attempt": row.get("attempt"),
        },
    }
    history.append(entry)
    write_json_atomic(state_path, state)

    emit(json.dumps({
        "event": "official_repair_start",
        "task": task.id,
        "attempt": attempt,
        "candidate_id": task.candidate_id,
        "run_class": task.run_class,
        "concurrency": task.concurrency,
    }, ensure_ascii=False))

    adapter = ReplayAdapter(metadata, run_dir / "trials")
    exit_code = 0
    result: dict[str, Any] | None = None
    try:
        attempt_dir = Path(adapter(task, attempt))
    except RuntimeError:
        exit_code = 1
    except Exception as exc:
        result = {
            "status": "FAILED",
            "reasons": [f"{type(exc).__name__}: {exc}"],
            "output_tokens_per_second": None,
        }
        result.update({
            "task_id": task.id,
            "candidate_id": task.candidate_id,
            "concurrency": task.concurrency,
            "mode": task.mode,
            "scale": task.scale,
            "run_class": task.run_class,
            "attempt": attempt,
            "candidate_hash": task.candidate_hash,
            "result_path": str(attempt_dir),
            "manifest": str(trial_path.relative_to(run_dir)),
            "repair_current_bundle_fingerprint": metadata["repair_current_bundle_fingerprint"],
        })
        write_json_atomic(trial_path, result)
    if result is None:
        result = read_attempt(attempt_dir, task, metadata, exit_code=exit_code)
        result = normalize_trial_result(dict(result))
        result.update({
            "task_id": task.id,
            "candidate_id": task.candidate_id,
            "concurrency": task.concurrency,
            "mode": task.mode,
            "scale": task.scale,
            "run_class": task.run_class,
            "attempt": attempt,
            "candidate_hash": task.candidate_hash,
            "result_path": str(attempt_dir),
            "manifest": str(trial_path.relative_to(run_dir)),
            "repair_current_bundle_fingerprint": metadata["repair_current_bundle_fingerprint"],
        })
        write_json_atomic(trial_path, result)

    entry["fingerprint"] = fingerprint(result)
    entry["repair"]["finished_at"] = utc_now()
    entry["repair"]["status"] = result.get("status")
    write_json_atomic(state_path, state)
    collected = collect_results(run_dir)
    emit(json.dumps({
        "event": "official_repair_done",
        "task": task.id,
        "attempt": attempt,
        "status": result.get("status"),
        "rows": len(collected.get("rows", [])),
    }, ensure_ascii=False))
    return result


def start_official_repair(run_dir: Path, body: dict[str, Any]) -> dict[str, Any]:
    conflicts = JOBS.running_for_run(run_dir)
    if conflicts:
        names = ", ".join(f"{job.get('name')}:{job.get('id')}" for job in conflicts)
        raise ValueError(f"run has active jobs; stop or wait first: {names}")
    selector = {
        "task_id": body.get("task_id"),
        "attempt": body.get("attempt"),
        "manifest": body.get("manifest"),
    }
    return JOBS.create_function(
        "repair-trial",
        official_repair_worker,
        run_dir,
        selector,
        target_run=run_dir,
    )


def debug_rerun_worker(
    emit,
    run_dir: Path,
    candidate_id: str,
    concurrency: int | None,
    run_class: str | None,
    mode: str | None,
    scale: float | None,
) -> dict[str, Any]:
    emit(json.dumps({"event": "debug_rerun_start", "candidate_id": candidate_id}, ensure_ascii=False))
    plan = load_plan(run_dir / "plan.json")
    plan_doc = plan_to_dict(plan)
    metadata = copy.deepcopy(plan.metadata)
    candidates = {c["id"]: c for c in plan_doc.get("candidates", []) if isinstance(c, dict) and c.get("id")}
    candidate = find_candidate(plan_doc, candidate_id)
    rows = [row for row in trial_rows(run_dir) if row.get("candidate_id") == candidate_id]
    if concurrency is None:
        concurrency = default_debug_concurrency(metadata, candidates, candidate, rows)
    if run_class is None:
        run_class = default_debug_run_class(rows)
    if mode is None:
        mode = default_debug_mode(rows)
    mode = str(mode).replace("_", "-")
    if mode not in {"closed-loop", "open-loop"}:
        raise ValueError("mode must be closed-loop or open-loop")
    if run_class not in {"smoke", "concurrency", "tuning", "final_repeat", "open_loop", "diagnostic", "formal"}:
        raise ValueError("invalid run_class")
    metadata["candidates"] = candidates
    metadata["benchmark_dir"] = str(AUTOMATION_ROOT / "benchmarks")
    metadata["debug_current_bundle_fingerprint"] = _bundle_fingerprint()
    task_id = "debug-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    task = PlanTask(
        id=task_id,
        stage=3,
        candidate_id=candidate_id,
        run_class=run_class,
        mode=mode,
        concurrency=max(1, int(concurrency)),
        scale=scale,
        candidate_hash=str(candidate.get("config_hash") or ""),
    )
    debug_root = run_dir / "debug-trials" / candidate_id
    adapter = ReplayAdapter(metadata, debug_root)
    emit(json.dumps({"event": "debug_trial", "task": task.id, "attempt": 1, "run_class": run_class, "concurrency": task.concurrency}, ensure_ascii=False))
    attempt_dir = Path(adapter(task, 1))
    result = read_attempt(attempt_dir, task, metadata)
    result.update({
        "task_id": task.id,
        "candidate_id": candidate_id,
        "concurrency": task.concurrency,
        "mode": mode,
        "scale": scale,
        "run_class": run_class,
        "attempt": 1,
        "candidate_hash": task.candidate_hash,
        "debug": True,
        "result_path": str(attempt_dir),
        "manifest": str((attempt_dir / "trial.json").relative_to(run_dir)),
        "current_bundle_fingerprint": metadata["debug_current_bundle_fingerprint"],
    })
    write_json_atomic(attempt_dir / "trial.json", result)
    append_debug_row(run_dir, result)
    emit(json.dumps({"event": "debug_rerun_done", "status": result.get("status"), "result_path": str(attempt_dir)}, ensure_ascii=False))
    return result


def start_debug_rerun(run_dir: Path, body: dict[str, Any]) -> dict[str, Any]:
    candidate_id = body.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id is required")
    concurrency = body.get("concurrency")
    if concurrency is not None:
        concurrency = int(concurrency)
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
    scale = body.get("scale")
    if scale is not None:
        scale = float(scale)
        if scale <= 0:
            raise ValueError("scale must be > 0")
    return JOBS.create_function(
        "debug-rerun",
        debug_rerun_worker,
        run_dir,
        candidate_id,
        concurrency,
        body.get("run_class"),
        body.get("mode"),
        scale,
        target_run=run_dir,
    )


class AutomationHandler(BaseHTTPRequestHandler):
    server_version = "AutomationWeb/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"error": message, "status": int(status)}, status)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def route_parts(self) -> tuple[list[str], dict[str, list[str]]]:
        parsed = urlparse(self.path)
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        query = parse_qs(parsed.query)
        return parts, query

    def result_root_from_query(self, query: dict[str, list[str]]) -> Path:
        return safe_result_root(query.get("result_root", [None])[0])

    def do_GET(self) -> None:
        try:
            parts, query = self.route_parts()
            if not parts:
                return self.serve_static("index.html")
            if parts[0] != "api":
                return self.serve_static("/".join(parts))
            return self.handle_get_api(parts[1:], query)
        except ValueError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except KeyError as exc:
            self.send_error_json(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:
        try:
            parts, query = self.route_parts()
            if not parts or parts[0] != "api":
                return self.send_error_json(HTTPStatus.NOT_FOUND, "unknown endpoint")
            body = self.read_json_body()
            return self.handle_post_api(parts[1:], query, body)
        except ValueError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except KeyError as exc:
            self.send_error_json(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def handle_get_api(self, parts: list[str], query: dict[str, list[str]]) -> None:
        if parts == ["settings"]:
            return self.send_json({
                "automation_root": str(AUTOMATION_ROOT),
                "default_result_root": str(DEFAULT_RESULT_ROOT),
                "default_config_path": str(DEFAULT_CONFIG_PATH),
                "python": sys.executable,
                "code_state": code_state(),
            })
        if parts == ["jobs"]:
            return self.send_json({"jobs": JOBS.list()})
        if len(parts) == 2 and parts[0] == "jobs":
            return self.send_json(JOBS.get(parts[1]))
        if parts == ["runs"]:
            root = self.result_root_from_query(query)
            return self.send_json({"result_root": str(root), "runs": scan_runs(root)})
        if len(parts) >= 2 and parts[0] == "runs":
            root = self.result_root_from_query(query)
            run_dir = safe_run_dir(root, parts[1])
            if len(parts) == 2:
                return self.send_json(run_detail(run_dir))
            if parts[2:] == ["candidates"]:
                return self.send_json({"run_id": parts[1], "candidates": plan_candidates(run_dir)})
            if len(parts) == 4 and parts[2] == "candidates":
                return self.send_json(candidate_detail(run_dir, parts[3]))
            if parts[2:] == ["trials"]:
                return self.send_json({"run_id": parts[1], "rows": trial_rows(run_dir)})
            if parts[2:] == ["code-state"]:
                return self.send_json(code_state(run_dir))
            if parts[2:] == ["artifacts"]:
                base_raw = query.get("base", [""])[0]
                base_path = safe_artifact_path(run_dir, base_raw) if base_raw else run_dir
                return self.send_json({
                    "run_id": parts[1],
                    "base": str(base_path.resolve().relative_to(run_dir.resolve())) if base_path != run_dir else "",
                    "artifacts": list_artifacts_for_base(run_dir, base_path),
                })
            if parts[2:] == ["artifact"]:
                artifact = safe_artifact_path(run_dir, query.get("path", [""])[0])
                tail = int(query.get("tail", [str(MAX_ARTIFACT_BYTES)])[0])
                return self.send_artifact(run_dir, artifact, max(1, min(tail, 20 * 1024 * 1024)))
        self.send_error_json(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def handle_post_api(self, parts: list[str], query: dict[str, list[str]], body: dict[str, Any]) -> None:
        if parts == ["jobs", "plan"]:
            config_path = Path(os.path.expandvars(str(body.get("config_path") or DEFAULT_CONFIG_PATH))).expanduser().resolve()
            return self.send_json(run_plan_job(config_path), HTTPStatus.ACCEPTED)
        if parts == ["jobs", "auto"]:
            config_path = Path(os.path.expandvars(str(body.get("config_path") or DEFAULT_CONFIG_PATH))).expanduser().resolve()
            return self.send_json(run_auto_job(config_path), HTTPStatus.ACCEPTED)
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "stop":
            return self.send_json(JOBS.stop(parts[1]), HTTPStatus.ACCEPTED)
        if len(parts) >= 2 and parts[0] == "runs":
            root = self.result_root_from_query(query)
            run_dir = safe_run_dir(root, parts[1])
            if parts[2:] == ["run"]:
                return self.send_json(run_saved_plan_job(run_dir, resume=False), HTTPStatus.ACCEPTED)
            if parts[2:] == ["resume"]:
                return self.send_json(run_saved_plan_job(run_dir, resume=True), HTTPStatus.ACCEPTED)
            if parts[2:] == ["collect"]:
                return self.send_json(collect_run(run_dir))
            if parts[2:] == ["adopt-runtime"]:
                return self.send_json(adopt_current_runtime(run_dir, str(body.get("note") or "")))
            if parts[2:] == ["debug-rerun"]:
                return self.send_json(start_debug_rerun(run_dir, body), HTTPStatus.ACCEPTED)
            if parts[2:] == ["repair-trial"]:
                return self.send_json(start_official_repair(run_dir, body), HTTPStatus.ACCEPTED)
        self.send_error_json(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def serve_static(self, raw_path: str) -> None:
        static_root = Path(__file__).resolve().parent / "static"
        target = (static_root / raw_path).resolve()
        if target.is_dir():
            target = target / "index.html"
        if static_root.resolve() not in target.parents and target != static_root.resolve():
            return self.send_error(HTTPStatus.NOT_FOUND)
        if not target.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        data = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if target.suffix == ".js":
            content_type = "text/javascript"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith(("text/", "application/javascript", "text/javascript")) else ""))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_artifact(self, run_dir: Path, artifact: Path, tail_bytes: int) -> None:
        if not artifact.is_file():
            return self.send_error_json(HTTPStatus.NOT_FOUND, "artifact does not exist")
        size = artifact.stat().st_size
        with artifact.open("rb") as source:
            if size > tail_bytes:
                source.seek(max(0, size - tail_bytes))
                data = source.read()
                truncated = True
            else:
                data = source.read()
                truncated = False
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        rel = artifact.resolve().relative_to(run_dir.resolve()).as_posix()
        parsed = json_load(artifact, None) if artifact.suffix == ".json" and not truncated else None
        self.send_json({
            "path": rel,
            "size": size,
            "truncated_to_tail": truncated,
            "tail_bytes": tail_bytes,
            "text": text,
            "json": parsed,
        })


def run_detail(run_dir: Path) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise KeyError("run not found")
    files = {
        "resolved_config": json_load(run_dir / "resolved-config.json", None),
        "environment": json_load(run_dir / "environment.json", None),
        "plan": json_load(run_dir / "plan.json", None),
        "search_state": json_load(run_dir / "search-state.json", None),
        "results_index": json_load(run_dir / "results-index.json", None),
        "leaderboard": json_load(run_dir / "leaderboard.json", None),
        "best_so_far": json_load(run_dir / "best-so-far.json", None),
        "best": json_load(run_dir / "best.json", None),
        "debug_index": json_load(debug_index_path(run_dir), None),
    }
    return {
        "summary": summarize_run(run_dir),
        "code_state": code_state(run_dir),
        **files,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the local Automation web console")
    parser.add_argument("--host", default=os.environ.get("AUTOMATION_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AUTOMATION_WEB_PORT", "18080")))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), AutomationHandler)
    print(json.dumps({
        "event": "automation_web_started",
        "url": f"http://{args.host}:{args.port}/",
        "default_result_root": str(DEFAULT_RESULT_ROOT),
        "automation_root": str(AUTOMATION_ROOT),
        "python": sys.executable,
    }, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        JOBS.stop_all()
        print("stopping Automation web console", file=sys.stderr)
        return 130
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
