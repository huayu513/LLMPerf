from dataclasses import dataclass, field
from typing import Dict, Tuple
from typing import Optional
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class PathConfig:
    model_host: Path
    jsonl_host: Path
    results_host: Path
    model_host_original: str = ""
    jsonl_host_original: str = ""
    results_host_original: str = ""

@dataclass(frozen=True)
class DockerConfig:
    name_prefix: str = "s1slow-benchmark"
    network_mode: str = "bridge"
    service_port: int = 25080
    shm_size: str = "16g"
    ipc: str = "host"

@dataclass(frozen=True)
class ModelManifest:
    id: str
    checkpoint: str
    checkpoint_original: str = ""
    checkpoint_kind: str = "directory"
    version: str = ""
    served_model_name: str = ""
    tokenizer: Optional[str] = None
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    quantization: Optional[str] = None
    moe_backend: Optional[str] = None
    backend: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    profile_env: Dict[str, str] = field(default_factory=dict)
    speculative_decoding: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

@dataclass(frozen=True)
class WorkloadManifest:
    id: str
    jsonl: str = ""
    jsonl_original: str = ""
    jsonl_resolved: Optional[str] = None
    sha256: Optional[str] = None
    request_mode: str = "raw_request"
    order: str = "fixed"
    timeout_seconds: float = 600.0
    warmup: int = 0
    closed_loop_concurrency: Tuple[int, ...] = ()
    open_loop_scales: Tuple[float, ...] = (1.0, 2.0, 4.0)
    repetitions: int = 3
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

@dataclass(frozen=True)
class CandidateConfig:
    id: str
    tp: int = 1
    dp: int = 1
    pp: int = 1
    dp_attention: bool = False
    backend: Optional[str] = None
    dspark: bool = False
    gpu_indexes: Tuple[int, ...] = ()
    status: str = "SUPPORTED"
    reasons: Tuple[str, ...] = ()
    static_config: Dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""

@dataclass(frozen=True)
class PlanTask:
    id: str
    stage: int
    candidate_id: str
    depends_on: Tuple[str, ...] = ()
    run_class: str = "smoke"
    mode: str = "closed_loop"
    concurrency: Optional[int] = None
    scale: Optional[float] = None
    attempt_policy: str = "retry"
    candidate_hash: str = ""

@dataclass(frozen=True)
class Plan:
    id: str
    tasks: Tuple[PlanTask, ...] = ()
    candidates: Tuple[CandidateConfig, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
