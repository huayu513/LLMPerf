"""Hardware/model-aware candidate planning for the single-configuration workflow."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path

from .artifacts import write_json_atomic
from .errors import ConfigError
from .types import CandidateConfig, Plan, PlanTask


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def plan_to_dict(plan):
    return _jsonable(asdict(plan))


def write_plan(path, plan, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'refusing to overwrite plan: {path}')
    write_json_atomic(path, plan_to_dict(plan))


def load_plan(path):
    try:
        data = json.loads(Path(path).read_text())
        if data.get('metadata', {}).get('workflow_version') != 2:
            raise ConfigError('legacy plan is not supported; generate a new plan from the single configuration')
        candidates = tuple(CandidateConfig(**{**c, 'gpu_indexes': tuple(c['gpu_indexes']),
                            'reasons': tuple(c.get('reasons', ()))}) for c in data['candidates'])
        return Plan(data['id'], tuple(PlanTask(**t) for t in data.get('tasks', [])), candidates, data['metadata'])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigError(f'invalid plan {path}: {exc}') from exc


def _groups(config, env):
    available = {g.index: g for g in env.gpus}
    indexes = config.gpu_indexes if config.gpu_indexes is not None else tuple(available)
    if not indexes or len(set(indexes)) != len(indexes) or any(i not in available for i in indexes):
        raise ConfigError('gpu_indexes must select unique physical GPUs present on this host')
    groups = defaultdict(list)
    for index in indexes:
        gpu = available[index]
        if str(gpu.mig_mode or '').lower() == 'enabled':
            raise ConfigError('MIG devices are not supported by physical-index search; disable MIG or use another GPU')
        groups[(gpu.name, gpu.compute_capability, gpu.memory_total_mb)].append(index)
    return list(groups.values())


def _power_of_two_divisors(value):
    result = []
    size = 1
    while size <= value:
        if value % size == 0:
            result.append(size)
        size *= 2
    return result


def _deployment_layouts(gpu_group):
    gpus = tuple(gpu_group)
    total = len(gpus)
    layouts = []
    for gpus_per_instance in _power_of_two_divisors(total):
        instance_count = total // gpus_per_instance
        instances = []
        for offset in range(instance_count):
            start = offset * gpus_per_instance
            assigned = gpus[start:start + gpus_per_instance]
            instances.append({
                'id': f'i{offset}',
                'ordinal': offset,
                'gpu_indexes': assigned,
            })
        layouts.append({
            'total_gpu_count': total,
            'instance_count': instance_count,
            'gpus_per_instance': gpus_per_instance,
            'label': f'{total}卡{instance_count}实例',
            'ascii_label': f'{total}g{instance_count}i',
            'gpu_indexes': gpus,
            'instances': instances,
        })
    return layouts


def create_search_plan(config, model, workload, env, result_dir):
    if env.status == 'FAIL':
        raise ConfigError('environment checks failed; inspect environment.json')
    groups = _groups(config, env)
    caps = env.container.get('capabilities', {})
    options = set(caps.get('options', []))
    if not options:
        raise ConfigError('cannot discover SGLang launch options from the configured image')
    required = {
        '--trust-remote-code', '--model-path', '--served-model-name', '--enable-metrics',
        '--enable-cache-report', '--tp-size', '--dp', '--pp-size', '--mem-fraction-static',
        '--max-running-requests', '--host', '--port', '--chunked-prefill-size',
        '--default-chat-template-kwargs',
    }
    if model.tool_call_parser:
        required.add('--tool-call-parser')
    if model.reasoning_parser:
        required.add('--reasoning-parser')
    if model.raw.get('provenance', {}).get('quantization') == 'model_overrides.quantization':
        required.add('--quantization')
    if model.raw.get('is_moe'):
        required.update(('--moe-runner-backend', '--moe-a2a-backend'))
    missing = sorted(required - options)
    if missing:
        raise ConfigError('configured image lacks required SGLang launch options: ' + ', '.join(missing))
    for parser, key in ((model.tool_call_parser, 'tool_parsers'), (model.reasoning_parser, 'reasoning_parsers')):
        if parser and parser not in caps.get(key, []):
            raise ConfigError(f'configured image does not advertise parser {parser!r}; use a compatible image or model_overrides')
    requested = tuple(config.search.backends or ())
    advertised = set(caps.get('runner_backends', []))
    if any(b not in advertised for b in requested):
        raise ConfigError('search.backends contains a backend not advertised by the configured image')
    is_moe = bool(model.raw.get('is_moe'))
    quant = str(model.quantization or '').lower()
    if requested and not is_moe:
        raise ConfigError('search.backends selects MoE runners but this model is not identified as MoE')
    if requested:
        backends = [None if b == 'auto' else b for b in requested]
    elif is_moe:
        relevant = ['triton', 'deep_gemm', 'flashinfer_trtllm']
        if 'mxfp4' in quant:
            relevant = ['flashinfer_mxfp4', 'triton']
        backends = [None] + [b for b in relevant if b in advertised]
    else:
        backends = [None]
    # "auto" remains engine-resolved and is verified from server info at runtime.
    heads = model.raw.get('num_attention_heads')
    layers = model.raw.get('num_hidden_layers')
    dpa_supported = is_moe and {'--enable-dp-attention', '--enable-dp-lm-head'} <= options
    dspark_supported = (model.raw.get('model_type') == 'deepseek_v4' and
                        {'--speculative-algorithm', '--speculative-dspark-block-size'} <= options)
    candidates = []
    for gpu_group in groups:
        for deployment in _deployment_layouts(gpu_group):
            n = deployment['gpus_per_instance']
            pp_values = [1]
            if '--pp-size' in options and isinstance(layers, int):
                pp_values += [p for p in range(2, n + 1) if layers % p == 0]
            for pp in pp_values:
                for tp in range(1, n // pp + 1):
                    if isinstance(heads, int) and heads % tp:
                        continue
                    for dp in range(1, n // (tp * pp) + 1):
                        strategies = [(False, dp)]
                        if dp == 1 and dpa_supported and tp > 1:
                            strategies += [(True, d) for d in range(2, tp + 1) if tp % d == 0]
                        for dpa, actual_dp in strategies:
                            world = tp * pp if dpa else tp * actual_dp * pp
                            if world != n:
                                continue
                            for backend in backends:
                                for dspark in ([False, True] if dspark_supported and pp == 1 else [False]):
                                    static = dict(tp=tp, dp=actual_dp, pp=pp, dp_attention=dpa,
                                        dp_lm_head=dpa, backend=backend, dspark=dspark,
                                        world_size=world, gpu_indexes=deployment['gpu_indexes'],
                                        logical_gpu_count=deployment['total_gpu_count'],
                                        deployment=deployment,
                                        moe_a2a_backend='none' if is_moe else None,
                                        mem_fraction_static=0.85, max_running_requests=config.search.concurrency_max,
                                        chunked_prefill_size=8192)
                                    digest = fingerprint(static)
                                    ident = (
                                        f"{deployment['ascii_label']}-tp{tp}-dp{actual_dp}-pp{pp}-"
                                        f'a{int(dpa)}-{backend or "auto"}-ds{int(dspark)}-{digest[:8]}'
                                    )
                                    candidates.append(CandidateConfig(ident, tp, actual_dp, pp, dpa, backend,
                                        dspark, tuple(deployment['gpu_indexes']), 'CANDIDATE', (), static, digest))
    if not candidates:
        raise ConfigError('no candidate fits the selected GPUs and model metadata')
    candidates.sort(key=lambda c: (
        c.static_config.get('deployment', {}).get('gpus_per_instance', len(c.gpu_indexes) or 1),
        c.tp, c.dp, c.dspark, c.pp > 1, c.dp_attention,
        c.backend is not None, c.id,
    ))
    # Interleave strategy families so finite budgets include more than the
    # ordinary TP/DP auto-backend prefix. Within each family retain hardware order.
    families = defaultdict(deque)
    for candidate in candidates:
        deployment = candidate.static_config.get('deployment', {})
        key = (
            deployment.get('ascii_label'), candidate.backend, candidate.dspark,
            candidate.pp > 1, candidate.dp_attention,
        )
        families[key].append(candidate)
    candidates = []
    while any(families.values()):
        for family in families.values():
            if family:
                candidates.append(family.popleft())
    models = workload.raw.get('models', {})
    if len(models) != 1:
        raise ConfigError('input must contain one request.model alias for a single checkpoint')
    meta = dict(workflow_version=2, profile='AUTO', engine='sglang', image=config.image,
        smoke=bool(getattr(config, 'smoke', False)),
        model_host=str(config.model_path), jsonl_host=str(config.input_path), results_host=str(result_dir),
        index_host=str(Path(result_dir) / 'data' / 'replay_index.json'),
        benchmark_dir=str(Path(__file__).resolve().parents[1] / 'benchmarks'),
        served_model_name=next(iter(models)), tool_call_parser=model.tool_call_parser,
        reasoning_parser=model.reasoning_parser, chat_template_kwargs=model.chat_template_kwargs,
        profile_env=model.profile_env, warmup=config.warmup, request_timeout=config.request_timeout,
        ready_timeout=config.ready_timeout, expected_source_sha256=workload.sha256,
        expected_request_count=workload.raw['count'], model_fingerprint=fingerprint(asdict(model)),
        model_snapshot=asdict(model), environment=env.to_dict(), search=vars(config.search),
        name_prefix=config.docker.name_prefix, service_port=config.docker.service_port, network_mode=config.docker.network_mode,
        shm_size=config.docker.shm_size, ipc=config.docker.ipc,
        candidates={c.id: asdict(c) for c in candidates},
        objective='highest full-workload closed-loop output tokens/s after limited exploration and repeated validation',
        search_scope='homogeneous GPU groups, deployment topology, compatible per-instance TP/DP/PP, advertised MoE runners, supported DSpark; runtime validation required')
    if model.raw.get('provenance', {}).get('quantization') == 'model_overrides.quantization':
        meta['quantization'] = model.quantization
    return Plan(Path(result_dir).name, candidates=tuple(candidates), metadata=_jsonable(meta))
