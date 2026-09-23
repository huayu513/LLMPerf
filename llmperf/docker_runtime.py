"""Safe Docker command construction and container lifecycle helpers.

The runtime deliberately deals in argument vectors (never shell strings), and
only accepts immutable image references and well-formed Docker identifiers.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class Mount:
    src: Path
    dst: str
    read_only: bool = False


@dataclass(frozen=True)
class DockerTaskSpec:
    image: str
    name: str
    gpu_indexes: tuple[int, ...] = ()
    mounts: tuple[Mount, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    command: tuple[str, ...] = ()
    internal_port: int = 25080
    host_port: int | None = None
    # Keep options after host_port for backwards compatibility with positional callers.
    network_mode: str = "bridge"
    shm_size: str = "16g"
    ipc: str = "host"


@dataclass(frozen=True)
class DockerRunResult:
    exit_code: int
    started_at: str
    finished_at: str
    inspect: dict[str, object] = field(default_factory=dict)
    container_name: str = ""
    command: tuple[str, ...] = ()


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_IMAGE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:\-]*$")
_DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NETWORK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHM_RE = re.compile(r"^[1-9][0-9]*(?:[kKmMgGtTpP])?$")


def _attrs(result: Any):
    return (
        int(getattr(result, "returncode", 1)),
        str(getattr(result, "stdout", "") or ""),
        str(getattr(result, "stderr", "") or ""),
    )


class DockerRuntime:
    def __init__(self, runner: Callable[..., Any] | None = None):
        self._custom_runner = runner is not None
        self.runner = runner or self._subprocess_runner

    @staticmethod
    def _subprocess_runner(args: Sequence[str], **kwargs: Any):
        kwargs.pop("shell", None)
        return subprocess.run(list(args), shell=False, capture_output=True, text=True, **kwargs)

    @staticmethod
    def _validate_name(name: str):
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise ValueError("invalid container name")

    @staticmethod
    def _validate_image(image: str):
        if not isinstance(image, str) or image.count("@sha256:") != 1:
            raise ValueError("image must be an immutable sha256 digest reference")
        ref, digest = image.split("@sha256:", 1)
        if (not _IMAGE_REF_RE.fullmatch(ref) or not _DIGEST_RE.fullmatch(digest)
                or ref.endswith("/") or "//" in ref):
            raise ValueError("image must be an immutable sha256 digest reference")
        # A colon is valid only as a numeric registry port (tags are mutable).
        for component in ref.split("/"):
            if ":" in component:
                host, port = component.rsplit(":", 1)
                if not host or not port.isdigit() or not 1 <= int(port) <= 65535:
                    raise ValueError("image must be an immutable sha256 digest reference")

    @staticmethod
    def _validate_command(command: Sequence[str]):
        if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
            raise ValueError("command must be a sequence of arguments")
        for arg in command:
            if not isinstance(arg, str) or not arg or "\x00" in arg:
                raise ValueError("invalid command argument")

    @staticmethod
    def _validate_env(env: dict[str, str]):
        if not isinstance(env, dict):
            raise ValueError("environment must be a mapping")
        for key, value in env.items():
            if not isinstance(key, str) or not _ENV_KEY_RE.fullmatch(key):
                raise ValueError(f"invalid environment variable name: {key!r}")
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError(f"invalid value for environment variable: {key}")

    @staticmethod
    def _validate_mount(mount: Mount):
        if not isinstance(mount, Mount) or not isinstance(mount.read_only, bool):
            raise ValueError("invalid mount")
        src = Path(mount.src)
        dst = mount.dst
        if not src.is_absolute():
            raise ValueError(f"mount source must be absolute: {src}")
        if not isinstance(dst, str) or not dst.startswith("/") or "\x00" in dst:
            raise ValueError(f"container destination must be absolute: {dst}")
        parts = dst.split("/")
        if dst == "/" or dst.endswith("/") or any(part == "" for part in parts[1:]):
            raise ValueError(f"invalid container destination: {dst}")
        if any(part in (".", "..") for part in parts[1:]) or "\\" in dst:
            raise ValueError(f"invalid container destination: {dst}")
        try:
            resolved = src.resolve(strict=True)
        except FileNotFoundError:
            # ``strict=True`` fails for a missing writable leaf and returning
            # the lexical path would miss symlinked parents (e.g.
            # ``/tmp/link/new`` where ``link -> /``). Resolve non-strictly and
            # reject that root escape before the runtime creates directories.
            resolved = src.resolve(strict=False)
            parent = src.parent
            try:
                parent_resolved = parent.resolve(strict=True)
            except FileNotFoundError:
                parent_resolved = parent.resolve(strict=False)
            if parent != Path("/") and parent_resolved == Path("/"):
                raise ValueError("mount source parent resolves to host root")
        if resolved == Path("/"):
            raise ValueError("refusing to mount host root")

    @staticmethod
    def _validate_runtime_options(spec: DockerTaskSpec):
        network = spec.network_mode
        if not isinstance(network, str) or not _NETWORK_RE.fullmatch(network):
            raise ValueError("invalid Docker network mode")
        if network.lower() == "host":
            raise ValueError("host network mode is not allowed for benchmark tasks")
        shm = spec.shm_size
        if not isinstance(shm, str) or not _SHM_RE.fullmatch(shm):
            raise ValueError("invalid shm size")
        ipc = spec.ipc
        if not isinstance(ipc, str) or not ipc:
            raise ValueError("invalid IPC mode")
        if ipc in {"host", "private", "shareable"}:
            return
        if ipc.startswith("container:") and _NAME_RE.fullmatch(ipc.split(":", 1)[1]):
            return
        raise ValueError("invalid IPC mode")

    def build_run_command(self, spec: DockerTaskSpec) -> list[str]:
        self._validate_image(spec.image)
        self._validate_name(spec.name)
        self._validate_command(spec.command)
        self._validate_env(spec.env)
        self._validate_runtime_options(spec)
        if not isinstance(spec.gpu_indexes, (tuple, list)):
            raise ValueError("GPU indexes must be a sequence")
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in spec.gpu_indexes):
            raise ValueError("invalid GPU index")
        if len(set(spec.gpu_indexes)) != len(spec.gpu_indexes):
            raise ValueError("duplicate GPU index")
        if not isinstance(spec.internal_port, int) or isinstance(spec.internal_port, bool) or not 1 <= spec.internal_port <= 65535:
            raise ValueError("invalid port")
        if spec.host_port is not None and (not isinstance(spec.host_port, int) or isinstance(spec.host_port, bool) or not 1 <= spec.host_port <= 65535):
            raise ValueError("invalid port")
        argv = ["docker", "run", "--rm", "--name", spec.name]
        if spec.gpu_indexes:
            # Docker parses this option as CSV even with shell=False. Keep a
            # multi-device list inside one CSV field.
            gpu_request = "device=" + ",".join(map(str, spec.gpu_indexes))
            argv += ["--gpus", '"' + gpu_request + '"' if len(spec.gpu_indexes) > 1 else gpu_request]
        argv += ["--ipc=" + spec.ipc, "--shm-size", spec.shm_size,
                 "--network", spec.network_mode]
        if spec.host_port is not None:
            argv += ["--publish", f"127.0.0.1:{spec.host_port}:{spec.internal_port}"]
        for key, value in sorted(spec.env.items()):
            argv += ["--env", f"{key}={value}"]
        for mount in spec.mounts:
            self._validate_mount(mount)
            src, dst = Path(mount.src).resolve(), mount.dst
            if not src.exists():
                if mount.read_only:
                    raise FileNotFoundError(src)
                src.mkdir(parents=True, exist_ok=True)
            argv += ["--mount", f"type=bind,src={src},dst={dst}{',readonly' if mount.read_only else ''}"]
        return argv + [spec.image, *spec.command]

    def _inspect_raw(self, name):
        self._validate_name(name)
        return _attrs(self.runner(["docker", "inspect", name], shell=False))

    @classmethod
    def _redact(cls, value, key=None):
        if key and any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            return "[REDACTED]"
        if isinstance(value, dict):
            return {k: cls._redact(v, str(k)) for k, v in value.items()}
        if isinstance(value, list):
            redacted = []
            for item in value:
                if isinstance(item, str) and "=" in item and key and key.lower() == "env":
                    k, _ = item.split("=", 1)
                    if any(token in k.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                        item = f"{k}=[REDACTED]"
                redacted.append(item)
            return redacted
        return value

    @classmethod
    def _redact_argv(cls, argv: Sequence[str]) -> tuple[str, ...]:
        result = list(argv)
        for i, item in enumerate(result[:-1]):
            if item in ("--env", "-e"):
                value = result[i + 1]
                if isinstance(value, str) and "=" in value:
                    key, _ = value.split("=", 1)
                    if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                        result[i + 1] = f"{key}=[REDACTED]"
        return tuple(result)

    def inspect(self, container_name):
        code, out, err = self._inspect_raw(container_name)
        if code != 0:
            raise RuntimeError("docker inspect failed")
        try:
            payload = json.loads(out)
        except Exception as exc:
            raise RuntimeError("invalid docker inspect output") from exc
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        if not isinstance(payload, dict):
            raise RuntimeError("invalid docker inspect payload")
        return self._redact(payload)

    def remove(self, container_name, force=False):
        self._validate_name(container_name)
        argv = ["docker", "rm"] + (["-f"] if force else []) + [container_name]
        code, _, err = _attrs(self.runner(argv, shell=False))
        if code != 0:
            raise RuntimeError("docker rm failed")

    def _stream_run(self, command, log_path):
        if self._custom_runner:
            process = self.runner(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        else:
            process = subprocess.Popen(list(command), shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        output = []
        stream = getattr(process, "stdout", None)
        if isinstance(stream, str):
            output.append(stream)
        elif stream is not None:
            try:
                with log_path.open("a", encoding="utf-8") as log:
                    for chunk in stream:
                        output.append(str(chunk))
                        log.write(str(chunk))
                        log.flush()
            except TypeError:
                pass
        if hasattr(process, "wait") and callable(process.wait):
            try:
                code = int(process.wait())
            except TypeError:
                code = int(getattr(process, "returncode", 1))
        else:
            code = int(getattr(process, "returncode", 1))
        if output and isinstance(stream, str):
            with log_path.open("a", encoding="utf-8") as log:
                log.write("".join(output))
        return code

    def run(self, spec, log_path):
        code, _, err = self._inspect_raw(spec.name)
        if code == 0:
            raise RuntimeError(f"container name already exists: {spec.name}")
        if code != 1:
            raise RuntimeError("unable to check container name")
        absent = ("no such object" in err.lower() or "no such container" in err.lower()
                  or "cannot find" in err.lower() or "not found" in err.lower() or not err.strip())
        if not absent:
            raise RuntimeError("unable to check container name")
        command = self.build_run_command(spec)
        started = datetime.now(timezone.utc).isoformat()
        data: dict[str, object] = {}
        exit_code = 1
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        cleanup_target: str = spec.name
        try:
            exit_code = self._stream_run(command, log_path)
        finally:
            try:
                data = self.inspect(spec.name)
                ident = data.get("Id") or data.get("ID")
                if isinstance(ident, str) and ident:
                    cleanup_target = ident
            except Exception:
                pass
            try:
                self.remove(cleanup_target, force=True)
            except Exception:
                pass
        return DockerRunResult(exit_code, started, datetime.now(timezone.utc).isoformat(), data, spec.name, self._redact_argv(command))


__all__ = ["Mount", "DockerTaskSpec", "DockerRunResult", "DockerRuntime"]
