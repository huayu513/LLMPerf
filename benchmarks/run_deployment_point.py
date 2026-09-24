#!/usr/bin/env python3
"""Run one benchmark point against a multi-instance local deployment."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
LAUNCHER = SCRIPT_DIR / "server" / "launch_server.sh"
REPLAY = SCRIPT_DIR / "replay" / "replay_jsonl_sglang.py"

# SGLang derives DP-attention TCP/ZMQ endpoints from the HTTP service port.
# Reserve a separate block for every instance instead of using adjacent ports.
DEPLOYMENT_PORT_STRIDE = 256


ALIASES = {
    "model_path": ("model_path",),
    "served_model_name": ("served_model_name", "served_model"),
    "tool_call_parser": ("tool_call_parser",),
    "reasoning_parser": ("reasoning_parser",),
    "chat_template_kwargs": ("chat_template_kwargs", "default_chat_template_kwargs"),
    "quantization": ("quantization",),
    "tp": ("tp", "tp_size", "tensor_parallel_size"),
    "dp": ("dp", "dp_size", "data_parallel_size"),
    "pp": ("pp", "pp_size", "pipeline_parallel_size"),
    "dp_attention": ("dp_attention", "enable_dp_attention"),
    "dp_lm_head": ("dp_lm_head", "enable_dp_lm_head"),
    "backend": ("moe_runner_backend", "moe_runner", "backend"),
    "moe_a2a_backend": ("moe_a2a_backend", "a2a_backend"),
    "dspark": ("dspark", "enable_dspark", "speculative_algorithm"),
    "mem_fraction_static": ("mem_fraction_static",),
    "max_running_requests": ("max_running_requests",),
    "chunked_prefill_size": ("chunked_prefill_size",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_class", choices=("smoke", "formal", "diagnostic"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--mode", choices=("closed-loop", "open-loop"), default="closed-loop")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--arrival-rate-scale", type=float, default=1.0)
    parser.add_argument("--max-in-flight", type=int, default=0)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--ready-timeout", type=int, default=3600)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--jsonl", default=os.environ.get("S1_JSONL_PATH", "/run/workload/input.jsonl"))
    parser.add_argument("--index", default=os.environ.get("S1_INDEX_PATH", "/run/workload/replay_index.json"))
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--no-gpu-monitor", action="store_true")
    parser.add_argument("--no-verify-source", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def startup_log_has_failure(log_file: Path) -> bool:
    if not log_file.is_file():
        return False
    try:
        text = log_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "Traceback (most recent call last):":
            return True
        if stripped.endswith(("Error:", "Exception:")):
            return True
        if ": " in stripped:
            prefix = stripped.split(": ", 1)[0]
            if prefix.endswith(("Error", "Exception")) and prefix.replace("_", "").isalnum():
                return True
    return False


def load_deployment() -> dict[str, Any]:
    raw = os.environ.get("S1_DEPLOYMENT")
    if not raw:
        die("S1_DEPLOYMENT is required for deployment runs")
    try:
        deployment = json.loads(raw)
    except json.JSONDecodeError as exc:
        die(f"S1_DEPLOYMENT is invalid JSON: {exc}")
    if not isinstance(deployment, dict):
        die("S1_DEPLOYMENT must be a JSON object")
    instances = deployment.get("instances")
    if not isinstance(instances, list) or len(instances) < 2:
        die("S1_DEPLOYMENT.instances must contain at least two instances")
    return deployment


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temp), str(path))


def manifest(path: Path, args: argparse.Namespace, deployment: dict[str, Any], state: str, exit_code: int | None) -> None:
    data = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
    data.update({
        "schema_version": 1,
        "profile": args.profile,
        "class": args.run_class,
        "mode": args.mode,
        "concurrency": args.concurrency,
        "arrival_rate_scale": args.arrival_rate_scale,
        "run": args.run,
        "deployment": deployment,
        "request_sender_location": "target_pod",
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    if "started_at" not in data:
        data["started_at"] = datetime.now(timezone.utc).isoformat()
    if exit_code is not None:
        data["exit_code"] = int(exit_code)
    atomic_json(path, data)


def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except (OSError, urllib.error.URLError, ValueError):
        return False


def http_json(url: str, timeout: float = 10.0) -> tuple[bool, dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
        data = json.loads(payload.decode("utf-8"))
        return True, data if isinstance(data, dict) else {}
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError):
        return False, {}


def normalized(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def server_info_leaves(server_info: dict[str, Any]) -> dict[str, Any]:
    arguments = server_info.get("server_args", server_info)
    if not isinstance(arguments, dict):
        return {}
    return {normalized(key): value for key, value in arguments.items()}


def values_match(key: str, expected: Any, actual: Any) -> bool:
    if key in {"backend", "moe_a2a_backend"} and isinstance(expected, str) and expected.lower() == "auto":
        unresolved = {"", "auto"} if key == "moe_a2a_backend" else {"", "auto", "none"}
        return actual is not None and str(actual).strip().lower() not in unresolved
    if isinstance(expected, bool):
        if key == "dspark" and isinstance(actual, str):
            actual = actual.strip().upper() == "DSPARK"
        elif key == "dspark" and actual is None:
            actual = False
        elif isinstance(actual, str):
            lowered = actual.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                actual = True
            elif lowered in {"0", "false", "no", "off", "none", ""}:
                actual = False
        elif isinstance(actual, (int, float)) and actual in (0, 1):
            actual = bool(actual)
        return actual is expected
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    if isinstance(expected, dict) and isinstance(actual, str):
        try:
            actual = json.loads(actual)
        except (TypeError, ValueError):
            return False
    return actual == expected


def compare_parameters(requested: dict[str, Any], server_info: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    leaves = server_info_leaves(server_info)
    unsupported: list[str] = []
    mismatched: list[str] = []
    checks: dict[str, Any] = {}
    for key, expected in requested.items():
        found = False
        actual = None
        for alias in ALIASES.get(key, (key,)):
            if alias in leaves:
                found = True
                actual = leaves[alias]
                break
        matches = found and values_match(key, expected, actual)
        if not found:
            unsupported.append(key)
        elif not matches:
            derived_chunked_prefill = (
                key == "chunked_prefill_size"
                and requested.get("dp_attention") is True
                and isinstance(expected, int)
                and isinstance(requested.get("dp"), int)
                and requested["dp"] > 0
                and values_match(key, expected // requested["dp"], actual)
            )
            if derived_chunked_prefill:
                matches = True
            else:
                mismatched.append(key)
        checks[key] = {
            "requested": expected,
            "actual": actual,
            "supported": found,
            "matches": matches,
        }
    return unsupported, mismatched, checks


def instance_env(base: dict[str, str], results_root: Path, instance: dict[str, Any], port: int) -> dict[str, str]:
    env = dict(base)
    env["S1_SERVER_STATE_DIR"] = str(results_root / "instances" / str(instance["id"]) / "server")
    env["SERVER_PORT"] = str(port)
    env["S1_BASE_URL"] = f"http://127.0.0.1:{port}"
    logical = instance.get("logical_gpu_indexes")
    if not isinstance(logical, list) or not logical:
        die(f"deployment instance {instance.get('id')} has no logical_gpu_indexes")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(index)) for index in logical)
    return env


def main() -> int:
    args = parse_args()
    deployment = load_deployment()
    results_root = Path(os.environ.get("S1_RESULTS_ROOT", "/run/results"))
    results_root.mkdir(parents=True, exist_ok=True)
    started: list[tuple[dict[str, Any], dict[str, str]]] = []
    final_exit = 2
    final_state = "failed"
    manifest_path = results_root / "run_manifest.json"
    manifest(manifest_path, args, deployment, "running", None)

    base_port = int(os.environ.get("SERVER_PORT", "25080"))
    instances = list(deployment["instances"])
    try:
        port_stride = int(deployment.get("service_port_stride", DEPLOYMENT_PORT_STRIDE))
    except (TypeError, ValueError):
        die("deployment service_port_stride must be an integer")
    if port_stride < 1:
        die("deployment service_port_stride must be positive")
    if os.environ.get("ENABLE_DP_ATTENTION", "0") == "1" and port_stride < DEPLOYMENT_PORT_STRIDE:
        die(
            "DP Attention multi-instance services require service_port_stride >= "
            f"{DEPLOYMENT_PORT_STRIDE}; got {port_stride}"
        )
    last_reserved_port = base_port + (len(instances) - 1) * port_stride + port_stride - 1
    if base_port < 1 or last_reserved_port > 65535:
        die(
            "multi-instance service port plan exceeds TCP port range: "
            f"base={base_port} instances={len(instances)} stride={port_stride}"
        )
    service_ports = [base_port + offset * port_stride for offset in range(len(instances))]
    base_urls = [f"http://127.0.0.1:{port}" for port in service_ports]
    endpoint_ids = [str(instance.get("id", f"i{offset}")) for offset, instance in enumerate(instances)]
    base_env = dict(os.environ)

    try:
        for url in base_urls:
            if http_ok(url.rstrip("/") + "/health"):
                die(f"a service is already responding at {url}; refuse to mix server state")

        if args.dry_run:
            for offset, instance in enumerate(instances):
                env = instance_env(base_env, results_root, instance, service_ports[offset])
                command = [str(LAUNCHER), "dry-run", args.profile]
                print(" ".join(command), "CUDA_VISIBLE_DEVICES=" + env["CUDA_VISIBLE_DEVICES"])
            final_exit = 0
            final_state = "complete"
            return final_exit

        for offset, instance in enumerate(instances):
            env = instance_env(base_env, results_root, instance, service_ports[offset])
            instance_dir = results_root / "instances" / str(instance["id"])
            instance_dir.mkdir(parents=True, exist_ok=True)
            log_file = instance_dir / "server.log"
            # Invoke through bash so an uploaded bundle does not depend on
            # executable-bit preservation or a shebang being honored by the
            # host filesystem.
            command = ["bash", str(LAUNCHER), "start", args.profile, "--log-file", str(log_file)]
            subprocess.run(command, env=env, check=True)
            started.append((instance, env))

        deadline = time.monotonic() + args.ready_timeout
        for offset, (instance, env) in enumerate(started):
            instance_dir = results_root / "instances" / str(instance["id"])
            log_file = instance_dir / "server.log"
            url = base_urls[offset].rstrip("/") + "/health"
            while not http_ok(url, timeout=5.0):
                status = subprocess.run(
                    ["bash", str(LAUNCHER), "status"],
                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=False,
                )
                if status.returncode != 0:
                    die(f"SGLang instance {instance.get('id')} exited before becoming ready; inspect {log_file}")
                if startup_log_has_failure(log_file):
                    die(f"SGLang instance {instance.get('id')} logged a startup exception before becoming ready; inspect {log_file}")
                if time.monotonic() >= deadline:
                    die(f"SGLang instance {instance.get('id')} did not become ready within {args.ready_timeout}s")
                time.sleep(2)
            print(f"server ready: {base_urls[offset]}", flush=True)

        root_requested: dict[str, Any] | None = None
        root_info: dict[str, Any] | None = None
        root_command_parts: list[str] = []
        instance_evidence = []
        unsupported_all: list[str] = []
        mismatched_all: list[str] = []
        for offset, (instance, env) in enumerate(started):
            state_dir = Path(env["S1_SERVER_STATE_DIR"])
            instance_dir = results_root / "instances" / str(instance["id"])
            requested = json.loads((state_dir / "server.requested.json").read_text(encoding="utf-8"))
            captured, info = http_json(base_urls[offset].rstrip("/") + "/get_server_info")
            atomic_json(instance_dir / "server.info.json", info)
            for name in ("server.command.sh", "server.meta", "server.requested.json"):
                source = state_dir / name
                if source.is_file():
                    shutil.copy2(source, instance_dir / name)
            command_text = (state_dir / "server.command.sh").read_text(encoding="utf-8")
            root_command_parts.append(f"# instance {instance['id']} {base_urls[offset]}\n{command_text}")
            unsupported, mismatched, checks = compare_parameters(requested, info)
            unsupported_all.extend(unsupported)
            mismatched_all.extend(mismatched)
            instance_evidence.append({
                "id": instance["id"],
                "base_url": base_urls[offset],
                "readiness": True,
                "server_info_captured": captured,
                "requested_server_parameters": requested,
                "unsupported_parameters": unsupported,
                "mismatched_parameters": mismatched,
                "parameter_checks": checks,
                "resolved": captured and not unsupported and not mismatched,
            })
            if root_requested is None:
                root_requested = requested
                root_info = info

        if root_requested is None:
            die("no instance server evidence was captured")
        atomic_json(results_root / "server.requested.json", root_requested)
        atomic_json(results_root / "server.info.json", root_info or {})
        (results_root / "server.command.sh").write_text("\n".join(root_command_parts), encoding="utf-8")
        evidence = {
            "readiness": True,
            "server_info_captured": all(item["server_info_captured"] for item in instance_evidence),
            "resolved": all(item["resolved"] for item in instance_evidence),
            "requested_server_parameters": root_requested,
            "unsupported_parameters": sorted(set(unsupported_all)),
            "mismatched_parameters": sorted(set(mismatched_all)),
            "instances": instance_evidence,
        }
        atomic_json(results_root / "server.evidence.json", evidence)
        if not evidence["resolved"]:
            print("server parameter evidence did not resolve; skipping replay", file=sys.stderr)
            final_exit = 0
            final_state = "complete"
        else:
            run_decimal = int(args.run)
            run_pad = f"{run_decimal:03d}"
            load_tag = f"c{args.concurrency}" if args.mode == "closed-loop" else f"trace_{str(args.arrival_rate_scale).replace('.', 'p')}x"
            run_name = f"deployment_{deployment.get('ascii_label', 'multi')}_{load_tag}_run{run_pad}"
            python_bin = os.environ.get("S1_PYTHON_BIN", sys.executable)
            command = [
                python_bin, str(REPLAY),
                "--jsonl", args.jsonl,
                "--index", args.index,
                "--base-url", base_urls[0],
                "--base-urls", ",".join(base_urls),
                "--endpoint-ids", ",".join(endpoint_ids),
                "--mode", args.mode,
                "--concurrency", str(args.concurrency),
                "--arrival-rate-scale", str(args.arrival_rate_scale),
                "--max-in-flight", str(args.max_in_flight),
                "--output-dir", str(results_root),
                "--run-name", run_name,
                "--warmup", str(args.warmup),
                "--limit", str(args.limit),
                "--request-timeout", str(args.request_timeout),
            ]
            command.append("--no-verify-source" if args.no_verify_source else "--verify-source")
            (results_root / "replay.command.sh").write_text(
                "#!/usr/bin/env bash\nexec " + " ".join(shlex_quote(part) for part in command) + "\n",
                encoding="utf-8",
            )
            os.chmod(results_root / "replay.command.sh", 0o755)
            final_exit = subprocess.run(command).returncode
            final_state = "complete" if final_exit == 0 else "failed"
    finally:
        stop_exit = 0
        if not args.keep_server:
            for instance, env in reversed(started):
                try:
                    subprocess.run(["bash", str(LAUNCHER), "stop"], env=env, check=True)
                except subprocess.CalledProcessError as exc:
                    stop_exit = stop_exit or int(exc.returncode)
                    print(f"error: stopping instance {instance.get('id')} failed with {exc.returncode}", file=sys.stderr)
        if stop_exit and final_exit == 0:
            final_exit = stop_exit
            final_state = "failed"
        manifest(manifest_path, args, deployment, final_state, final_exit)
    return final_exit


def shlex_quote(value: str) -> str:
    if not value:
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_@%+=:,./-"
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
