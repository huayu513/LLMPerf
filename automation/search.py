"""Measured, resumable search with bounded trials and independent final repeats."""
from __future__ import annotations

import copy
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from .artifacts import write_json_atomic
from .planner import fingerprint, plan_to_dict
from .types import PlanTask


_CONCURRENCY_STEP = 16


def _next_concurrency(current, maximum):
    return min(current + _CONCURRENCY_STEP, maximum)


def _score(result):
    value = result.get('output_tokens_per_second')
    return (float(value) if result.get('status') == 'VALID' and type(value) in (int, float)
            and math.isfinite(value) and value > 0 else None)


def collect_results(run_root):
    """Collect controller-owned normalized results, excluding nested native manifests."""
    root = Path(run_root)
    rows = []
    for path in sorted((root / 'trials').glob('*/attempt-*/trial.json')):
        try:
            row = json.loads(path.read_text())
            row['manifest'] = str(path.relative_to(root))
            rows.append(row)
        except (OSError, ValueError):
            continue
    document = {'version': 2, 'run_id': root.name, 'rows': rows}
    write_json_atomic(root / 'results-index.json', document)
    return document


def _candidate_gpu_count(candidate):
    static = candidate.get('static_config', {}) if isinstance(candidate, dict) else {}
    deployment = static.get('deployment', {}) if isinstance(static, dict) else {}
    try:
        return max(1, int(deployment.get('total_gpu_count')))
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        return max(1, len(candidate.get('gpu_indexes') or ()))
    except (AttributeError, TypeError):
        return 1


def _start_concurrency(limits, candidates, concurrency_max):
    configured = limits.get('start_concurrency')
    if configured is not None:
        return min(max(1, int(configured)), concurrency_max)
    total_gpus = max((_candidate_gpu_count(candidate) for candidate in candidates.values()), default=1)
    return min(max(8, 8 * total_gpus), concurrency_max)


def _promoted_results(measured, tolerance):
    valid = [r for points in measured.values() for r in points.values() if _score(r) is not None]
    if not valid:
        return []
    top = max(_score(r) for r in valid)
    cutoff = top * (1.0 - max(0.0, float(tolerance)))
    promoted = [r for r in valid if _score(r) >= cutoff]
    promoted.sort(key=lambda r: (-_score(r), str(r.get('candidate_id')), int(r.get('concurrency') or 0)))
    return promoted


def _write_progress_indexes(root):
    collected = collect_results(root)
    valid = [row for row in collected['rows'] if _score(row) is not None]
    valid.sort(key=lambda row: (-_score(row), str(row.get('task_id', ''))))
    leaderboard = {
        'version': 2,
        'run_id': root.name,
        'rows': valid,
    }
    write_json_atomic(root / 'leaderboard.json', leaderboard)
    best = valid[0] if valid else None
    write_json_atomic(root / 'best-so-far.json', {
        'version': 2,
        'run_id': root.name,
        'best': best,
    })


def execute_search(plan, run_root, executor=None, resume=False):
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    identity = fingerprint(plan_to_dict(plan))
    state_path = root / 'search-state.json'
    if state_path.exists():
        if not resume:
            raise FileExistsError('search already exists; use --resume or a fresh result directory')
        state = json.loads(state_path.read_text())
        if state.get('plan_hash') != identity:
            raise ValueError('resume plan does not match the recorded plan')
    else:
        state = {'plan_hash': identity, 'elapsed_seconds': 0, 'decisions': [], 'trials': {}}
        write_json_atomic(state_path, state)
    metadata = copy.deepcopy(plan.metadata)
    candidates = {c.id: asdict(c) for c in plan.candidates}
    metadata['candidates'] = candidates
    limits = metadata['search']
    max_trials = int(limits['max_trials'])
    smoke_enabled = metadata.get('smoke', False) is True
    repetitions = int(limits['repetitions'])
    max_seconds = float(limits['max_seconds'])
    concurrency_max = min(int(limits['concurrency_max']), int(metadata['expected_request_count']))
    start_concurrency = _start_concurrency(limits, candidates, concurrency_max)
    promotion_tolerance = float(limits.get('promotion_tolerance', 0.05))
    explore_limit = max_trials - repetitions
    started = time.monotonic()
    previous_elapsed = float(state['elapsed_seconds'])
    decisions = []
    adapter = None
    if executor is None:
        from .adapters import ReplayAdapter, read_attempt
        adapter = ReplayAdapter(metadata, root / 'trials')

        def executor(task, attempt):
            attempt_dir = root / 'trials' / task.id / f'attempt-{attempt:03d}'
            exit_code = 0
            try:
                adapter(task, attempt)
            except RuntimeError:
                exit_code = 1
            result = read_attempt(attempt_dir, task, metadata, exit_code=exit_code)
            result['result_path'] = str(attempt_dir)
            return result

    def save():
        state['elapsed_seconds'] = previous_elapsed + time.monotonic() - started
        state['decisions'] = decisions
        write_json_atomic(state_path, state)

    def used():
        return sum(
            1
            for attempts in state['trials'].values()
            for entry in attempts
            if entry.get('fingerprint') is not None
        )

    def trial(candidate_id, concurrency, run_class='concurrency', suffix='', scale=None):
        mode = 'open-loop' if scale is not None else 'closed-loop'
        ident = f'{candidate_id}-{run_class}-c{concurrency}{suffix}'
        task = PlanTask(ident, 5 if run_class == 'final_repeat' else 3, candidate_id,
                        run_class=run_class, concurrency=concurrency, mode=mode, scale=scale,
                        candidate_hash=candidates[candidate_id].get('config_hash', ''))
        history = state['trials'].get(ident, [])
        if history:
            last = history[-1]
            path = root / last['manifest']
            try:
                cached = json.loads(path.read_text())
                cached_fingerprint = fingerprint(cached)
                if last.get('fingerprint') is None:
                    last['fingerprint'] = cached_fingerprint
                    save()
                elif cached_fingerprint != last.get('fingerprint'):
                    raise ValueError('trial evidence changed')
                # Invalid trials are terminal search observations. They stop this
                # branch; interrupted attempts have no finalized trial document.
                if adapter is not None and cached.get('status') == 'VALID':
                    from .adapters import read_attempt
                    checked = read_attempt(Path(cached['result_path']), task, metadata)
                    evidence_keys = ('effective_static_config', 'evidence_fingerprint',
                                     'summary_path', 'server_command_path')
                    if (_score(checked) != _score(cached) or
                            any(checked.get(key) != cached.get(key) for key in evidence_keys)):
                        raise ValueError('native result evidence no longer validates')
                return cached
            except (OSError, ValueError, KeyError):
                decisions.append({'task': ident, 'reason': 'cached_evidence_invalid; retry'})
        cap = max_trials if run_class in {'final_repeat', 'open_loop'} else explore_limit
        if used() >= cap or previous_elapsed + time.monotonic() - started >= max_seconds:
            decisions.append({'task': ident, 'reason': 'budget_exhausted'})
            return None
        attempt = len(history) + 1
        path = root / 'trials' / ident / f'attempt-{attempt:03d}' / 'trial.json'
        # Persist allocation before launch. It counts against the budget only
        # after a normalized trial document is written and fingerprinted.
        entry = {'manifest': str(path.relative_to(root)), 'task': asdict(task), 'fingerprint': None}
        state['trials'].setdefault(ident, []).append(entry)
        save()
        print(json.dumps({'event': 'trial', 'task': ident, 'attempt': attempt,
                          'trial': used() + 1, 'max_trials': max_trials}), flush=True)
        try:
            result = dict(executor(task, attempt))
        except Exception as exc:
            result = {'status': 'FAILED', 'reasons': [str(exc)]}
        if result.get('status') == 'VALID' and _score(result) is None:
            result.update(status='INCONCLUSIVE', output_tokens_per_second=None,
                          reasons=['invalid_throughput_measurement'])
        result.update(task_id=ident, candidate_id=candidate_id, concurrency=concurrency,
                      mode=mode, scale=scale, run_class=run_class, attempt=attempt,
                      candidate_hash=task.candidate_hash)
        write_json_atomic(path, result)
        entry['fingerprint'] = fingerprint(result)
        save()
        _write_progress_indexes(root)
        return result

    measured = {}
    active = []
    # Leave room for at least one additional concurrency point per screened
    # candidate, plus repeats.
    screen_count = min(len(plan.candidates), max(1, explore_limit // 4))
    screening_trials = 0
    for candidate in plan.candidates:
        if active and screening_trials >= max(2, explore_limit // 2):
            break
        if len(active) >= screen_count:
            break
        if smoke_enabled:
            smoke = trial(candidate.id, 1, 'smoke')
            if smoke is None:
                break
            screening_trials += smoke['attempt']
            if _score(smoke) is None:
                decisions.append({'candidate': candidate.id, 'reason': 'smoke_failed'})
                continue
        baseline = trial(candidate.id, start_concurrency)
        if baseline is not None:
            screening_trials += baseline['attempt']
        if baseline is not None and _score(baseline) is not None:
            measured[candidate.id] = {start_concurrency: baseline}
            active.append(candidate.id)
    concurrency = _next_concurrency(start_concurrency, concurrency_max)
    while active and start_concurrency < concurrency <= concurrency_max:
        next_active = []
        for ident in active:
            result = trial(ident, concurrency)
            if result is None:
                continue
            if _score(result) is None:
                decisions.append({'candidate': ident, 'concurrency': concurrency, 'reason': 'failed_load_stop'})
                continue
            earlier = max(_score(r) for r in measured[ident].values())
            measured[ident][concurrency] = result
            if _score(result) > earlier * 1.01:
                next_active.append(ident)
            else:
                decisions.append({'candidate': ident, 'concurrency': concurrency, 'reason': 'throughput_plateau'})
        active = next_active
        if concurrency == concurrency_max:
            break
        concurrency = _next_concurrency(concurrency, concurrency_max)

    def winner():
        valid = [r for points in measured.values() for r in points.values() if _score(r) is not None]
        return max(valid, key=_score) if valid else None

    provisional = winner()

    # Small tuning neighborhood; each variation has its own candidate identity.
    if provisional:
        base = candidates[provisional['candidate_id']]
        options = metadata.get('environment', {}).get('container', {}).get('capabilities', {}).get('options', [])
        changes = []
        if '--mem-fraction-static' in options:
            changes += [('mem_fraction_static', .90), ('mem_fraction_static', .80)]
        if '--chunked-prefill-size' in options:
            changes += [('chunked_prefill_size', 4096), ('chunked_prefill_size', 16384)]
        for key, value in changes:
            tuned = copy.deepcopy(base)
            tuned['static_config'][key] = value
            tuned['config_hash'] = fingerprint(tuned['static_config'])
            tuned['id'] = base['id'] + '-tune-' + tuned['config_hash'][:8]
            candidates[tuned['id']] = tuned
            result = trial(tuned['id'], provisional['concurrency'], 'tuning')
            if result is not None and _score(result) is not None:
                measured[tuned['id']] = {provisional['concurrency']: result}
        provisional = winner()

    best = None
    repeat_groups = []
    promoted = _promoted_results(measured, promotion_tolerance)
    if promoted:
        for rank, promoted_result in enumerate(promoted, 1):
            candidate_id = promoted_result['candidate_id']
            concurrency = promoted_result['concurrency']
            repeats = []
            for rep in range(repetitions):
                result = trial(candidate_id, concurrency, 'final_repeat',
                               suffix=f'-p{rank}-r{rep + 1}')
                if result is not None:
                    repeats.append(result)
            group = {
                'candidate_id': candidate_id,
                'concurrency': concurrency,
                'promoted_score': _score(promoted_result),
                'repeats': repeats,
            }
            repeat_groups.append(group)
            if len(repeats) == repetitions and all(_score(r) is not None for r in repeats):
                values = [_score(r) for r in repeats]
                planned = candidates[candidate_id]
                configuration = copy.deepcopy(planned)
                effective = repeats[0].get('effective_static_config')
                if effective is not None:
                    configuration['static_config'] = effective
                    configuration['config_hash'] = fingerprint(effective)
                if all(r.get('effective_static_config') == effective for r in repeats):
                    candidate_best = dict(candidate_id=candidate_id, concurrency=concurrency,
                                output_tokens_per_second=statistics.median(values), repetition_scores=values,
                                repetitions=repetitions, configuration=configuration,
                                planned_configuration=planned,
                                promoted_score=_score(promoted_result),
                                server_command_paths=[r['server_command_path'] for r in repeats
                                                      if r.get('server_command_path')],
                                measured_task_ids=[r['task_id'] for r in repeats])
                    if best is None or candidate_best['output_tokens_per_second'] > best['output_tokens_per_second']:
                        best = candidate_best
                else:
                    decisions.append({'candidate': candidate_id, 'concurrency': concurrency,
                                      'reason': 'effective_configuration_changed_between_repetitions'})
    open_results = []
    if best:
        for scale in limits.get('open_loop_scales') or ():
            result = trial(best['candidate_id'], best['concurrency'], 'open_loop',
                           suffix='-s' + fingerprint(float(scale))[:16], scale=scale)
            if result is not None:
                open_results.append(result)
    save()
    status = 'PASS' if best else 'INCONCLUSIVE'
    requested_scales = limits.get('open_loop_scales') or ()
    if best and (len(open_results) != len(requested_scales) or any(_score(r) is None for r in open_results)):
        status = 'PARTIAL'
    report = dict(version=2, status=status, best=best,
                  provisional_best=provisional, promoted_finalists=[
                      {
                          'candidate_id': item['candidate_id'],
                          'concurrency': item['concurrency'],
                          'promoted_score': item['promoted_score'],
                          'repeat_scores': [_score(r) for r in item['repeats']],
                      }
                      for item in repeat_groups
                  ], trials=used(), elapsed_seconds=state['elapsed_seconds'],
                  screened_candidates=len(measured), planned_candidates=len(plan.candidates),
                  stop_reasons=decisions, open_loop_results=open_results,
                  objective='best repeated full-workload closed-loop output tokens/second after limited exploration within this search budget')
    write_json_atomic(root / 'best.json', report)
    _write_progress_indexes(root)
    return report
