"""Process-group lifecycle for a loopback-only vLLM benchmark server."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

if __package__:
    from .vllm_config import command_args, requested_parameters
else:
    from vllm_config import command_args, requested_parameters


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def managed_pid(state: Path) -> int | None:
    try:
        record = json.loads((state / "process.json").read_text())
        pid = int(record["pid"])
        if pid < 2:
            return None
        current = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if current[0] == "Z" or current[19] != record["start_time"]:
            return None
        if os.getpgid(pid) != pid:
            return None
        return pid
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def stop(state: Path) -> None:
    pid = managed_pid(state)
    if pid is None:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 20
    while managed_pid(state) is not None and time.monotonic() < deadline:
        time.sleep(0.2)
    if managed_pid(state) is not None:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "stop", "status", "dry-run", "argv-json"))
    parser.add_argument("profile", nargs="?", default="AUTO")
    parser.add_argument("--log-file")
    args = parser.parse_args()
    state = Path(os.environ["S1_SERVER_STATE_DIR"])
    if args.action == "status":
        return 0 if managed_pid(state) is not None else 1
    if args.action == "stop":
        stop(state)
        return 0
    if args.profile != "AUTO":
        parser.error("vLLM only supports the AUTO profile")
    config = json.loads(os.environ["S1_VLLM_CONFIG"])
    command = command_args(config, int(os.environ["SERVER_PORT"]))
    environment = json.loads(os.environ.get("S1_VLLM_ENV", "{}"))
    if not isinstance(environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items()
    ):
        parser.error("S1_VLLM_ENV must contain string environment values")
    environment.update(VLLM_SERVER_DEV_MODE="1", CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"])
    if args.action == "argv-json":
        print(json.dumps({"argv": command, "env": [f"{key}={value}" for key, value in environment.items()]}))
        return 0
    shell_command = "env " + shlex.join([f"{key}={value}" for key, value in environment.items()] + command)
    if args.action == "dry-run":
        print(shell_command)
        return 0
    if managed_pid(state) is not None:
        parser.error("a managed vLLM process is already running")
    executable = shutil.which("vllm", path=environment.get("PATH", os.environ.get("PATH")))
    if executable is None:
        parser.error("vLLM executable is missing from PATH")
    command[0] = executable
    shell_command = "env " + shlex.join([f"{key}={value}" for key, value in environment.items()] + command)
    state.mkdir(parents=True, exist_ok=True)
    atomic_json(state / "server.requested.json", requested_parameters(config))
    (state / "server.command.sh").write_text(shell_command + "\n")
    (state / "server.meta").write_text("engine=vllm\nprofile=AUTO\n")
    log = Path(args.log_file or state / "server.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    process = None
    try:
        with log.open("ab") as output:
            process = subprocess.Popen(command, env={**os.environ, **environment},
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        stat = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
        atomic_json(state / "process.json", {"pid": process.pid, "start_time": stat[19]})
    except BaseException:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
