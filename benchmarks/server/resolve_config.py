#!/usr/bin/env python3
"""Resolve the exact benchmark launch arguments with the installed SGLang.

The batch process stays in one Docker container. Each case uses a fresh Python
child so SGLang's process-wide platform and environment caches cannot leak
between candidate configurations. No server or model weights are loaded.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


LAUNCHER = Path(__file__).with_name('launch_server.sh')
PREFIX = 'S1SLOW_RESOLVE_JSON='
BATCH_PREFIX = 'S1SLOW_RESOLVE_BATCH_JSON='
CASE_PREFIX = 'S1SLOW_RESOLVE_CASE_JSON='

# ``resolved_dict`` is also used as a server-info payload and contains process
# wiring chosen while constructing a server. Those values are intentionally
# different for every probe and are not part of backend equivalence.
_VOLATILE_FIELDS = {
    'random_seed', 'nccl_port', 'dist_init_addr', 'gated_launch_port',
    'tokenizer_ipc_name', 'scheduler_input_ipc_name', 'detokenizer_ipc_name',
    'rpc_ipc_name', 'metrics_ipc_name', 'tokenizer_worker_ipc_name',
    'decoupled_spec_ipc_config', 'load_collector_ipc_name', 'instance_id',
}


def _comparison_projection(value):
    """Drop process wiring while retaining all resolved launch decisions."""
    if isinstance(value, dict):
        return {
            key: _comparison_projection(item)
            for key, item in value.items()
            if key not in _VOLATILE_FIELDS and not key.endswith('_ipc_name')
        }
    if isinstance(value, list):
        return [_comparison_projection(item) for item in value]
    return value


def _resolve_one(case: dict) -> dict:
    case_env = {**os.environ, **case['env']}
    launched = subprocess.run(
        ['bash', str(LAUNCHER), 'argv-json', 'AUTO'],
        env=case_env, text=True, capture_output=True, timeout=30, check=True,
    )
    command = json.loads(launched.stdout)
    argv = command['argv']
    if len(argv) < 2 or argv[1] != 'serve':
        raise ValueError('benchmark launcher did not produce sglang serve arguments')
    for item in command['env']:
        key, value = item.split('=', 1)
        os.environ[key] = value
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args

    load_plugins()
    server_args = prepare_server_args(argv[2:])
    if not hasattr(server_args, 'resolve_once') or not hasattr(server_args, 'resolved_dict'):
        return {'status': 'unavailable', 'reason': 'installed SGLang lacks resolve_once/resolved_dict'}
    server_args.resolve_once()
    effective = server_args.resolved_dict()
    canonical = json.dumps(effective, sort_keys=True, separators=(',', ':'), default=str)
    comparison = _comparison_projection(effective)
    comparison_canonical = json.dumps(comparison, sort_keys=True,
                                      separators=(',', ':'), default=str)
    backend = effective.get('moe_runner_backend')
    if not isinstance(backend, str) or backend in ('', 'auto'):
        return {'status': 'unresolved', 'reason': 'SGLang did not resolve a concrete MoE backend'}
    return {
        'status': 'resolved',
        'effective_backend': backend,
        'effective_config_sha256': hashlib.sha256(canonical.encode()).hexdigest(),
        'effective_config_comparison_sha256': hashlib.sha256(
            comparison_canonical.encode()).hexdigest(),
    }


def _child() -> None:
    try:
        case = json.load(sys.stdin)
        result = _resolve_one(case)
    except (ImportError, ModuleNotFoundError) as exc:
        result = {'status': 'unavailable', 'reason': f'{type(exc).__name__}: {exc}'}
    except Exception as exc:
        result = {'status': 'error', 'reason': f'{type(exc).__name__}: {exc}'}
    print(PREFIX + json.dumps(result, sort_keys=True), flush=True)


def _batch(path: Path) -> None:
    cases = json.loads(path.read_text())
    results = {}
    unsupported = None
    for key, case in cases.items():
        if unsupported is not None:
            results[key] = unsupported
            continue
        try:
            completed = subprocess.run(
                [sys.executable, __file__, '--one'], input=json.dumps(case),
                text=True, capture_output=True, timeout=90,
            )
            marker = next((line[len(PREFIX):] for line in reversed(completed.stdout.splitlines())
                           if line.startswith(PREFIX)), None)
            results[key] = json.loads(marker) if marker else {
                'status': 'unavailable',
                'reason': (completed.stderr or completed.stdout or 'resolver exited without a result')[-2000:],
            }
        except (subprocess.TimeoutExpired, ValueError) as exc:
            results[key] = {'status': 'unavailable', 'reason': f'{type(exc).__name__}: {exc}'}
        result = results[key]
        if result.get('status') == 'unavailable' and (
            'lacks resolve_once/resolved_dict' in result.get('reason', '')
            or 'No module named' in result.get('reason', '')
        ):
            unsupported = result
        print(CASE_PREFIX + json.dumps({'key': key, 'result': result}, sort_keys=True),
              flush=True)
    print(BATCH_PREFIX + json.dumps(results, sort_keys=True), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--one':
        _child()
    elif len(sys.argv) == 2:
        _batch(Path(sys.argv[1]))
    else:
        raise SystemExit('usage: resolve_config.py CASES_JSON | --one')
