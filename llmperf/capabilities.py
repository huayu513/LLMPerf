"""Read-only Docker, NVIDIA and container capability discovery."""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class CommandResult:
    args: Tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    def __call__(self, args: Sequence[str], **kwargs: Any) -> CommandResult: ...


def subprocess_runner(args: Sequence[str], **kwargs: Any) -> CommandResult:
    argv = tuple(str(x) for x in args)
    try:
        p = subprocess.run(list(argv), capture_output=True, text=True,
                           timeout=kwargs.get("timeout", 30), shell=False)
        return CommandResult(argv, p.returncode, p.stdout, p.stderr)
    except FileNotFoundError as exc:
        return CommandResult(argv, 127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes): out = out.decode(errors="replace")
        if isinstance(err, bytes): err = err.decode(errors="replace")
        return CommandResult(argv, 124, out, err or "command timed out")


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    memory_total_mb: float | None = None
    compute_capability: str | None = None
    uuid: str | None = None
    mig_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in {
            "index": self.index, "name": self.name,
            "memory_total_mb": self.memory_total_mb,
            "compute_capability": self.compute_capability,
            "uuid": self.uuid, "mig_mode": self.mig_mode,
        }.items() if v is not None}


@dataclass(frozen=True)
class ImageSnapshot:
    image: str
    status: str
    image_id: str | None = None
    digest: str | None = None
    created: str | None = None
    details: Tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"image": self.image, "status": self.status,
                "image_id": self.image_id, "digest": self.digest,
                "created": self.created, "details": list(self.details),
                "raw": dict(self.raw)}


@dataclass(frozen=True)
class EnvironmentSnapshot:
    status: str
    docker: Mapping[str, Any] = field(default_factory=dict)
    image: ImageSnapshot | Mapping[str, Any] = field(default_factory=dict)
    gpus: Tuple[GPUInfo, ...] = ()
    topology: Mapping[str, Any] = field(default_factory=dict)
    container: Mapping[str, Any] = field(default_factory=dict)
    host_paths: Mapping[str, Any] = field(default_factory=dict)
    checks: Tuple[Mapping[str, Any], ...] = ()
    probed_at: str = ""
    commands: Tuple[Tuple[str, ...], ...] = ()
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        image = self.image.to_dict() if isinstance(self.image, ImageSnapshot) else dict(self.image)
        return {"status": self.status, "docker": dict(self.docker), "image": image,
                "gpus": [g.to_dict() for g in self.gpus], "topology": dict(self.topology),
                "container": dict(self.container), "host_paths": dict(self.host_paths),
                "checks": [dict(c) for c in self.checks], "probed_at": self.probed_at,
                "commands": [list(c) for c in self.commands], "reasons": list(self.reasons)}


def _run(runner: CommandRunner, args: Sequence[str], *, timeout: int = 30) -> CommandResult:
    try:
        return runner(args, timeout=timeout)
    except TypeError:
        return runner(args)


def check_image(image: str, runner: CommandRunner = subprocess_runner) -> ImageSnapshot:
    result = _run(runner, ["docker", "image", "inspect", image])
    if result.returncode:
        return ImageSnapshot(image, "FAIL", details=(result.stderr.strip() or "image not found",))
    try:
        value = json.loads(result.stdout)
    except (TypeError, ValueError) as exc:
        return ImageSnapshot(image, "FAIL", details=(f"malformed image inspect JSON: {exc}",), raw={"stdout": result.stdout, "stderr": result.stderr})
    if not isinstance(value, list) or not value or not isinstance(value[0], dict):
        return ImageSnapshot(image, "FAIL", details=("image inspect returned no image metadata",))
    item = value[0]
    image_id = item.get("Id")
    if not isinstance(image_id, str) or not image_id:
        return ImageSnapshot(image, "FAIL", details=("image inspect omitted Id",), raw=item)
    digests = item.get("RepoDigests", [])
    digest = digests[0] if isinstance(digests, list) and digests and isinstance(digests[0], str) else None
    return ImageSnapshot(image, "PASS", image_id, digest, item.get("Created"), raw=item)


def _parse_gpus(text: str, *, has_compute_capability: bool = True) -> Tuple[GPUInfo, ...]:
    out = []
    for line in text.splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        memory = None
        if len(fields) > 2 and fields[2]:
            try: memory = float(fields[2])
            except ValueError: pass
        compute_capability = (
            fields[3] if has_compute_capability and len(fields) > 3 else None
        )
        uuid_index = 4 if has_compute_capability else 3
        uuid = fields[uuid_index] if len(fields) > uuid_index else None
        out.append(GPUInfo(index, fields[1], memory, compute_capability, uuid))
    return tuple(out)


_CONTAINER_PROBE = """import importlib, importlib.metadata, json, os, pathlib, platform, subprocess, sys
facts={"python":sys.version.split()[0],"kernel":platform.release(),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES","")}
errors=[]
try:
 import torch
 facts["torch"]=torch.__version__; facts["cuda"]=str(torch.version.cuda) if torch.version.cuda is not None else None
 facts["cuda_available"]=bool(torch.cuda.is_available()); facts["cuda_device_count"]=int(torch.cuda.device_count())
 try: facts["nccl"]=str(torch.cuda.nccl.version()) if facts["cuda_available"] else None
 except Exception as e: errors.append("nccl:"+str(e))
except Exception as e: errors.append("torch:"+str(e))
try:
 import sglang
 facts["sglang"]=getattr(sglang,"__version__",None) or importlib.metadata.version("sglang")
except Exception as e: errors.append("sglang:"+str(e))
facts["kernel_packages"]={}
for package in ("sgl_kernel","sglang_kernel"):
 try:
  module=importlib.import_module(package)
  facts["kernel_packages"][package]=getattr(module,"__version__",None) or importlib.metadata.version(package.replace("_","-"))
 except Exception: pass
facts["shm_bytes"]=pathlib.Path("/dev/shm").stat().st_size
facts["nvidia_devices"]=len(list(pathlib.Path("/dev").glob("nvidia*")))
facts["probe_errors"]=errors
help_result=None
try:
 candidate=subprocess.run(["sglang","serve","--help"],capture_output=True,text=True,timeout=30)
except Exception as e:
 errors.append("sglang serve --help unavailable: "+str(e))
else:
 if candidate.returncode==0:
  help_result=candidate
 else:
  detail=(candidate.stderr or candidate.stdout).strip()
  errors.append("sglang serve --help failed (exit "+str(candidate.returncode)+"): "+detail)
print("S1SLOW_PROBE_JSON="+json.dumps(facts,sort_keys=True))
if help_result is not None:
 print("S1SLOW_SGLANG_HELP_BEGIN"); print(help_result.stdout); print("S1SLOW_SGLANG_HELP_END")
required=("python","torch","cuda","nccl","sglang")
if errors or not facts.get("cuda_available") or any(not facts.get(key) for key in required) or help_result is None: sys.exit(1)
"""


_PROBE_PREFIX = "S1SLOW_PROBE_JSON="
_HELP_BEGIN = "S1SLOW_SGLANG_HELP_BEGIN\n"
_HELP_END = "S1SLOW_SGLANG_HELP_END"


def _parse_container_probe(text: str) -> tuple[dict[str, Any], str | None]:
    facts: dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith(_PROBE_PREFIX):
            try:
                value = json.loads(line[len(_PROBE_PREFIX):])
            except (TypeError, ValueError):
                break
            if isinstance(value, dict):
                facts = value
            break
    start = text.find(_HELP_BEGIN)
    if start < 0:
        return facts, None
    start += len(_HELP_BEGIN)
    end = text.find(_HELP_END, start)
    return facts, text[start:end] if end >= 0 else None


def _choice_values(help_text: str, option: str) -> list[str]:
    flattened = " ".join(help_text.split())
    match = re.search(rf"{re.escape(option)}(?:=|\s+)\{{([^}}]+)\}}", flattened)
    if not match:
        return []
    return [value.strip() for value in match.group(1).split(",") if value.strip()]


def _parse_sglang_capabilities(help_text: str) -> dict[str, list[str]]:
    usage = help_text.split("\n\n", 1)[0]
    option_lines = "\n".join(
        line for line in help_text.splitlines()
        if re.match(r"^\s+(?:-[A-Za-z0-9],\s*)?--[a-z0-9]", line)
    )
    options = sorted(set(re.findall(r"--[a-z0-9][a-z0-9-]*", usage + "\n" + option_lines)))
    return {
        "options": options,
        "runner_backends": _choice_values(help_text, "--moe-runner-backend"),
        "a2a_backends": _choice_values(help_text, "--moe-a2a-backend"),
        "tool_parsers": _choice_values(help_text, "--tool-call-parser"),
        "reasoning_parsers": _choice_values(help_text, "--reasoning-parser"),
    }


def probe_environment(config: Any, runner: CommandRunner = subprocess_runner) -> EnvironmentSnapshot:
    commands: list[Tuple[str, ...]] = []
    reasons: list[str] = []
    checks: list[Mapping[str, Any]] = []

    def call(args: Sequence[str], *, timeout: int = 30) -> CommandResult:
        commands.append(tuple(str(x) for x in args))
        return _run(runner, args, timeout=timeout)

    docker_version = call(["docker", "version", "--format", "{{json .}}"]); docker_ok = docker_version.returncode == 0
    if not docker_ok: reasons.append("Docker daemon unavailable")
    docker_info = call(["docker", "info", "--format", "{{json .Runtimes}}"]); runtime_ok = docker_info.returncode == 0 and "nvidia" in (docker_info.stdout + docker_info.stderr).lower()
    if not runtime_ok: reasons.append("NVIDIA container runtime unavailable")
    image = check_image(config.image, runner); commands.append(("docker", "image", "inspect", config.image))
    if image.status != "PASS": reasons.append("requested Docker image unavailable")

    gpu_result = call(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,uuid", "--format=csv,noheader,nounits"])
    if gpu_result.returncode == 0:
        gpus = _parse_gpus(gpu_result.stdout)
    else:
        fallback_gpu_result = call([
            "nvidia-smi", "--query-gpu=index,name,memory.total,uuid",
            "--format=csv,noheader,nounits",
        ])
        gpus = (
            _parse_gpus(fallback_gpu_result.stdout, has_compute_capability=False)
            if fallback_gpu_result.returncode == 0 else ()
        )
        checks.append({
            "name": "compute_capability",
            "status": "WARN",
            "detail": "nvidia-smi compute_cap unavailable; compute capability is unknown",
            "stdout": gpu_result.stdout,
            "stderr": gpu_result.stderr,
        })
    if not gpus: reasons.append("GPU enumeration failed")
    mig = call(["nvidia-smi", "--query-gpu=index,mig.mode.current", "--format=csv,noheader"]); mig_map={}
    for line in mig.stdout.splitlines():
        parts=[x.strip() for x in line.split(",",1)]
        if len(parts)==2:
            try: mig_map[int(parts[0])]=parts[1]
            except ValueError: pass
    gpus=tuple(GPUInfo(g.index,g.name,g.memory_total_mb,g.compute_capability,g.uuid,mig_map.get(g.index)) for g in gpus)
    checks.append({"name":"mig", "status":"PASS" if mig.returncode == 0 else "WARN", "values":mig_map, "stdout":mig.stdout, "stderr":mig.stderr})
    topo = call(["nvidia-smi", "topo", "-m"]); topology = {"text": topo.stdout} if topo.returncode == 0 and topo.stdout.strip() else {}
    if not topology: checks.append({"name":"topology", "status":"WARN", "detail":topo.stderr.strip() or "topology unavailable"})
    driver = call(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]); checks.append({"name":"driver", "status":"PASS" if driver.returncode == 0 else "WARN", "stdout":driver.stdout, "stderr":driver.stderr})
    visible = call(["printenv", "CUDA_VISIBLE_DEVICES"]); checks.append({"name":"cuda_visible_devices", "status":"PASS", "stdout":visible.stdout.strip()})
    numa = call(["numactl", "--hardware"]); checks.append({"name":"numa", "status":"PASS" if numa.returncode == 0 else "WARN", "stdout":numa.stdout, "stderr":numa.stderr})
    devices = call(["ls", "-l", "/dev/nvidiactl", "/dev/nvidia-uvm"]); checks.append({"name":"nvidia_devices", "status":"PASS" if devices.returncode == 0 else "WARN", "stdout":devices.stdout, "stderr":devices.stderr})
    free = call(["df", "-Pk", str(config.paths.results_host)]); checks.append({"name":"disk", "status":"PASS" if free.returncode == 0 else "WARN", "stdout":free.stdout, "stderr":free.stderr})

    paths: dict[str, Any] = {}
    for key in ("model_host", "jsonl_host", "results_host"):
        path = Path(getattr(config.paths, key)); exists = path.exists(); readable = os.access(path, os.R_OK) if exists else False
        writable = os.access(path, os.W_OK) if exists else (key == "results_host" and os.access(path.parent, os.W_OK))
        paths[key] = {"path": str(path), "exists": exists, "readable": readable, "writable": writable}
        if key != "results_host" and not (exists and readable): reasons.append(f"host path unavailable: {key}")
        if key == "results_host" and not writable: reasons.append("host path unavailable: results_host")

    container: dict[str, Any] = {}
    if docker_ok and image.status == "PASS":
        docker_cfg=getattr(config, "docker", None)
        ipc=str(getattr(docker_cfg, "ipc", "host")); shm=str(getattr(docker_cfg, "shm_size", "16g")); network=str(getattr(docker_cfg, "network_mode", "bridge"))
        selected = getattr(config, "gpu_indexes", None)
        gpu_request = (
            "all" if selected is None
            else '"device=' + ",".join(str(index) for index in selected) + '"'
        )
        probe_args=["docker", "run", "--rm", "--gpus", gpu_request, "--ipc="+ipc,
                    "--shm-size", shm, "--network", network, config.image,
                    "python3", "-c", _CONTAINER_PROBE]
        probe = call(probe_args, timeout=180)
        probe_facts, help_text = _parse_container_probe(probe.stdout)
        required = ("python", "torch", "cuda", "nccl", "sglang")
        versions_ok = all(probe_facts.get(key) for key in required)
        cuda_ok = (probe_facts.get("cuda_available") is True
                   and isinstance(probe_facts.get("cuda_device_count"), int)
                   and probe_facts["cuda_device_count"] > 0)
        allocation_ok = selected is None or probe_facts.get("cuda_device_count") == len(selected)
        capabilities = _parse_sglang_capabilities(help_text) if help_text is not None else {
            "options": [], "runner_backends": [], "a2a_backends": [],
            "tool_parsers": [], "reasoning_parsers": [],
        }
        probe_ok = (probe.returncode == 0 and versions_ok and cuda_ok and allocation_ok
                    and help_text is not None and not probe_facts.get("probe_errors"))
        versions = {key: probe_facts.get(key) for key in required}
        versions["kernel_packages"] = (
            probe_facts.get("kernel_packages")
            if isinstance(probe_facts.get("kernel_packages"), dict) else {}
        )
        container = {
            "status": "PASS" if probe_ok else "FAIL",
            "stdout": probe.stdout,
            "stderr": probe.stderr,
            "versions": versions,
            "cuda_available": probe_facts.get("cuda_available"),
            "cuda_device_count": probe_facts.get("cuda_device_count"),
            "capabilities": capabilities,
        }
        if not cuda_ok:
            reasons.append("container CUDA runtime unavailable")
        if not allocation_ok:
            reasons.append("container GPU allocation mismatch")
        if not versions_ok:
            reasons.append("container required package version unavailable")
        if help_text is None:
            reasons.append("SGLang serve capabilities unavailable")
        if probe_facts.get("probe_errors"):
            reasons.append("container capability probe reported errors")
        if probe.returncode and not any(
            reason.startswith(("container ", "SGLang ")) for reason in reasons
        ):
            reasons.append("container capability probe failed")

    facts={"driver":driver.stdout.strip(),"nccl":container.get("versions",{}).get("nccl"),"mig":mig_map,"cuda_visible_devices":visible.stdout.strip(),"numa":numa.stdout.strip(),"nvidia_devices":devices.stdout.strip(),"free_space":free.stdout.strip()}
    status = "FAIL" if reasons else ("WARN" if any(c.get("status") == "WARN" for c in checks) else "PASS")
    docker_meta={"status":"PASS" if docker_ok else "FAIL","version":docker_version.stdout,"info":docker_info.stdout,"runtime_nvidia":runtime_ok,"network_mode":getattr(getattr(config,"docker",None),"network_mode",None),"service_port":getattr(getattr(config,"docker",None),"service_port",None),"ipc":getattr(getattr(config,"docker",None),"ipc",None),"shm_size":getattr(getattr(config,"docker",None),"shm_size",None),"facts":facts}
    return EnvironmentSnapshot(status, docker_meta, image, gpus, topology, container, paths, tuple(checks), datetime.now(timezone.utc).isoformat(), tuple(commands), tuple(reasons))


def write_environment(path: Path, snapshot: EnvironmentSnapshot) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(snapshot.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        try: os.unlink(temporary_name)
        except FileNotFoundError: pass
