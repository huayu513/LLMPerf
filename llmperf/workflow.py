"""Single-configuration lifecycle and immutable run identity checks."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .artifacts import sha256_file, write_json_atomic
from .capabilities import probe_environment
from .configuration import load_config
from .discovery import discover_model, inspect_workload, prepare_workload
from .errors import ConfigError
from .planner import _jsonable, create_search_plan, fingerprint, load_plan, write_plan
from .search import execute_search
from .types import DockerConfig, PathConfig


def _probe_config(config, root):
    return SimpleNamespace(image=config.image, docker=config.docker, gpu_indexes=config.gpu_indexes,
        paths=PathConfig(config.model_path, config.input_path, root))


def _bundle_fingerprint():
    root = Path(__file__).resolve().parents[1]
    sources = [root / 'benchctl.py']
    for name in ('llmperf', 'benchmarks'):
        sources.extend(p for p in (root / name).rglob('*')
                       if p.is_file() and p.suffix in {'.py', '.sh', '.env'})
    return fingerprint({str(p.relative_to(root)): sha256_file(p) for p in sorted(sources)})


def run_workflow(config_path, *, stop_after='auto', probe=probe_environment, executor=None):
    if stop_after not in {'doctor', 'prepare', 'plan', 'auto'}:
        raise ValueError('invalid workflow stage')
    config = load_config(Path(config_path))
    # Fail malformed metadata/requests before any container or GPU allocation.
    model = discover_model(config.model_path, config.model_overrides)
    workload = inspect_workload(config.input_path)
    base = config.output_dir or config.source_path.parent / 'results'
    ident = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    root = Path(base) / ident
    root.mkdir(parents=True, exist_ok=False)
    print(json.dumps({'event': 'run_created', 'result_dir': str(root)}), flush=True)
    configuration = _jsonable(asdict(config))
    resolved = {'version': 2, 'configuration': configuration, 'model': _jsonable(asdict(model)),
                'input': {**workload.raw, 'path': str(config.input_path), 'sha256': workload.sha256},
                'defaults': 'configuration schema defaults; request generation parameters are unchanged',
                'result_dir': str(root)}
    write_json_atomic(root / 'resolved-config.json', resolved)
    env = probe(_probe_config(config, root))
    write_json_atomic(root / 'environment.json', env.to_dict())
    if env.status == 'FAIL':
        raise ConfigError(f'environment checks failed; see {root / "environment.json"}')
    if stop_after == 'doctor':
        return {'command': stop_after, 'status': env.status, 'result_dir': str(root)}
    index = prepare_workload(config.input_path, root / 'data' / 'replay_index.json')
    if index.get('jsonl_sha256') != workload.sha256 or index.get('count') != workload.raw['count']:
        raise ConfigError('input changed while preparing the replay index; start a new run')
    if stop_after == 'prepare':
        return {'command': stop_after, 'status': 'PASS', 'result_dir': str(root), 'requests': index['count']}
    plan = create_search_plan(config, model, workload, env, root)
    plan.metadata['configuration'] = configuration
    plan.metadata['bundle_fingerprint'] = _bundle_fingerprint()
    plan.metadata['index_fingerprint'] = sha256_file(root / 'data' / 'replay_index.json')
    write_plan(root / 'plan.json', plan)
    resolved['candidate_count'] = len(plan.candidates)
    resolved['served_model_name'] = plan.metadata['served_model_name']
    resolved['search_scope'] = plan.metadata['search_scope']
    write_json_atomic(root / 'resolved-config.json', resolved)
    if stop_after == 'plan':
        return {'command': stop_after, 'status': 'PASS', 'result_dir': str(root),
                'plan': str(root / 'plan.json'), 'candidates': len(plan.candidates)}
    report = execute_search(plan, root, executor=executor)
    return {**report, 'result_dir': str(root)}


def run_saved_plan(path, *, resume=False, executor=None, probe=probe_environment):
    path = Path(path).resolve()
    plan = load_plan(path)
    metadata = plan.metadata
    root = path.parent
    if root != Path(metadata['results_host']).resolve():
        raise ConfigError('plan was moved away from its recorded result directory; generate a new plan')
    source = Path(metadata['jsonl_host'])
    if not source.is_file() or sha256_file(source) != metadata['expected_source_sha256']:
        raise ConfigError('input source changed since planning; generate a new plan')
    options = metadata['configuration'].get('model_overrides', {})
    model = discover_model(Path(metadata['model_host']), options)
    if fingerprint(asdict(model)) != metadata['model_fingerprint']:
        raise ConfigError('model metadata or weight inventory changed since planning; generate a new plan')
    if _bundle_fingerprint() != metadata['bundle_fingerprint']:
        raise ConfigError('benchmark runtime changed since planning; generate a new plan')
    index_path = Path(metadata['index_host'])
    if not index_path.is_file() or sha256_file(index_path) != metadata['index_fingerprint']:
        raise ConfigError('replay index changed since planning; generate a new plan')
    index = json.loads(index_path.read_text())
    if index.get('jsonl_sha256') != metadata['expected_source_sha256'] or index.get('count') != metadata['expected_request_count']:
        raise ConfigError('replay index does not match the saved plan')
    settings = metadata['configuration']
    config = SimpleNamespace(image=metadata['image'], model_path=Path(metadata['model_host']),
        input_path=source, gpu_indexes=settings.get('gpu_indexes'), docker=DockerConfig(**settings['docker']))
    current = probe(_probe_config(config, root))
    if current.status == 'FAIL':
        raise ConfigError('current environment checks failed; cannot run the saved plan')
    previous = metadata['environment']
    now = current.to_dict()
    selected = {i for c in plan.candidates for i in c.gpu_indexes}
    def devices(snapshot):
        return [{k: g.get(k) for k in ('index', 'name', 'uuid', 'compute_capability', 'memory_total_mb', 'mig_mode')}
                for g in snapshot['gpus'] if g['index'] in selected]
    if devices(now) != devices(previous) or now.get('image', {}).get('image_id') != previous.get('image', {}).get('image_id'):
        raise ConfigError('GPU inventory or image changed since planning; generate a new plan')
    for section, key in (('container', 'versions'), ('container', 'capabilities')):
        if now.get(section, {}).get(key) != previous.get(section, {}).get(key):
            raise ConfigError('engine environment changed since planning; generate a new plan')
    if now.get('docker', {}).get('facts', {}).get('driver') != previous.get('docker', {}).get('facts', {}).get('driver'):
        raise ConfigError('GPU driver changed since planning; generate a new plan')
    report = execute_search(plan, root, executor=executor, resume=resume)
    return {**report, 'result_dir': str(root)}
