"""Resolve planned SGLang launches in the pinned image before benchmarking."""
from __future__ import annotations

import json
import subprocess
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path

from .adapters import ReplayAdapter
from .artifacts import write_json_atomic
from .docker_runtime import DockerRuntime, DockerTaskSpec, Mount
from .planner import fingerprint
from .types import Plan, PlanTask


_BATCH_PREFIX = 'S1SLOW_RESOLVE_BATCH_JSON='
_CASE_PREFIX = 'S1SLOW_RESOLVE_CASE_JSON='
_VOLATILE_ENV = {
    'S1_DEPLOYMENT', 'S1_BASE_URL', 'SERVER_PORT', 'S1_RESULTS_ROOT',
    'S1_SERVER_STATE_DIR', 'S1_INDEX_PATH', 'S1_JSONL_PATH',
}


def apply_backend_resolution(plan: Plan, results: dict, *, source: str) -> Plan:
    """Remove AUTO only when an explicit candidate resolves identically."""
    signatures = {}
    for candidate in plan.candidates:
        result = results.get(candidate.id, {})
        if result.get('status') != 'resolved':
            continue
        effective_hash = result.get('effective_config_sha256')
        if not isinstance(effective_hash, str) or len(effective_hash) != 64:
            continue
        context = fingerprint({key: value for key, value in candidate.static_config.items()
                               if key != 'backend'})
        key = (context, effective_hash)
        signatures.setdefault(key, []).append(candidate)

    removed = {}
    for equivalent in signatures.values():
        explicit = [candidate for candidate in equivalent
                    if candidate.backend is not None
                    and results[candidate.id].get('effective_backend') == candidate.backend]
        if not explicit:
            continue
        representative = explicit[0]
        for candidate in equivalent:
            if (candidate.backend is None
                    and results[candidate.id].get('effective_backend') == representative.backend):
                removed[candidate.id] = representative.id

    candidates = tuple(candidate for candidate in plan.candidates if candidate.id not in removed)
    metadata = dict(plan.metadata)
    metadata['candidates'] = {candidate.id: metadata['candidates'][candidate.id]
                              for candidate in candidates}
    metadata['comparison_groups'] = {
        group: [candidate_id for candidate_id in ids if candidate_id not in removed]
        for group, ids in metadata.get('comparison_groups', {}).items()
    }
    metadata['backend_resolution'] = {
        'source': source,
        'results': results,
        'removed_auto_duplicates': removed,
    }
    return replace(plan, candidates=candidates, metadata=metadata)


def _instance_gpus(spec) -> tuple[int, ...]:
    raw = spec.env.get('S1_DEPLOYMENT')
    if not raw:
        return tuple(spec.gpu_indexes)
    deployment = json.loads(raw)
    return tuple(int(value) for value in deployment['instances'][0]['gpu_indexes'])


def _case_key(env: dict, gpu_indexes: tuple[int, ...]) -> str:
    return fingerprint({
        'gpus': gpu_indexes,
        'env': {key: value for key, value in env.items() if key not in _VOLATILE_ENV},
    })


def _read_resolution_output(output) -> dict:
    if isinstance(output, bytes):
        output = output.decode(errors='replace')
    if not isinstance(output, str):
        return {}
    results = {}
    for line in output.splitlines():
        try:
            if line.startswith(_CASE_PREFIX):
                item = json.loads(line[len(_CASE_PREFIX):])
                if isinstance(item, dict) and isinstance(item.get('key'), str) \
                        and isinstance(item.get('result'), dict):
                    results[item['key']] = item['result']
            elif line.startswith(_BATCH_PREFIX):
                batch = json.loads(line[len(_BATCH_PREFIX):])
                if isinstance(batch, dict):
                    results.update(batch)
        except (TypeError, ValueError):
            continue
    return results


def resolve_in_image(plan: Plan, config, root: Path, *, runtime=None) -> dict:
    """Run the benchmark's own launcher and installed SGLang resolver in Docker.

    Cases sharing the same per-instance launch environment reuse one result;
    all cases for a GPU allocation run in one container. The probe never
    starts a server or loads weights.
    """
    runtime = runtime or DockerRuntime()
    by_gpu = {}
    case_ids = {}
    with tempfile.TemporaryDirectory(prefix='backend-resolve-', dir=root) as temporary:
        scratch = Path(temporary)
        adapter = ReplayAdapter(plan.metadata, scratch, runtime=runtime)
        benchmark_dir = None
        for candidate in plan.candidates:
            if not plan.metadata.get('model_snapshot', {}).get('raw', {}).get('is_moe'):
                continue
            task = PlanTask('resolve-' + candidate.id, 0, candidate.id,
                            concurrency=config.search.concurrency_max)
            spec = adapter.build_spec(task, 1, scratch / 'attempts' / candidate.id)
            benchmark_dir = next(mount.src for mount in spec.mounts
                                 if mount.dst == '/opt/s1slow/benchmarks')
            indexes = _instance_gpus(spec)
            env = dict(spec.env)
            env['CUDA_VISIBLE_DEVICES'] = ','.join(str(index) for index in range(len(indexes)))
            key = _case_key(env, indexes)
            by_gpu.setdefault(indexes, {}).setdefault(key, {'env': env})
            case_ids[candidate.id] = key

        case_results = {}
        for indexes, cases in by_gpu.items():
            print(json.dumps({'event': 'backend_resolution_batch_start',
                              'gpus': list(indexes), 'cases': len(cases)}), flush=True)
            input_path = scratch / ('cases-' + fingerprint(indexes)[:12] + '.json')
            write_json_atomic(input_path, cases)
            spec = DockerTaskSpec(
                image=config.image,
                name='llmperf-resolve-' + uuid.uuid4().hex[:12],
                gpu_indexes=indexes,
                mounts=(Mount(config.model_path, '/model', True),
                        Mount(benchmark_dir, '/opt/s1slow/benchmarks', True),
                        Mount(input_path, '/run/resolution/cases.json', True)),
                env={'CUDA_VISIBLE_DEVICES': ','.join(str(index) for index in range(len(indexes)))},
                command=('python3', '/opt/s1slow/benchmarks/server/resolve_config.py',
                         '/run/resolution/cases.json'),
                network_mode=config.docker.network_mode,
                shm_size=config.docker.shm_size,
                ipc=config.docker.ipc,
            )
            try:
                command = runtime.build_run_command(spec)
                completed = subprocess.run(command, capture_output=True, text=True,
                                           timeout=max(180, min(1200, 30 * len(cases))))
                batch = _read_resolution_output(completed.stdout)
                if not batch:
                    raise ValueError((completed.stderr or completed.stdout or
                                      'image resolver emitted no batch result')[-2000:])
                case_results.update(batch)
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                if isinstance(exc, subprocess.TimeoutExpired):
                    case_results.update(_read_resolution_output(exc.stdout))
                for key in cases:
                    case_results.setdefault(key, {
                        'status': 'unavailable',
                        'reason': f'{type(exc).__name__}: image resolver did not finish',
                    })
            print(json.dumps({'event': 'backend_resolution_batch_done',
                              'gpus': list(indexes),
                              'resolved': sum(case_results.get(key, {}).get('status') == 'resolved'
                                              for key in cases),
                              'cases': len(cases)}), flush=True)
        return {candidate_id: case_results.get(key, {
            'status': 'unavailable', 'reason': 'resolver omitted candidate context',
        }) for candidate_id, key in case_ids.items()}
