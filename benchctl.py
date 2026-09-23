#!/usr/bin/env python3
"""Portable one-configuration throughput LLMPerf (Python 3.11+)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit('benchctl requires Python 3.11 or newer')

# The local package works when LLMPerf is copied without its parent tree.
if __package__:
    from .llmperf.errors import ConfigError
    from .llmperf.search import collect_results
    from .llmperf.workflow import run_saved_plan, run_workflow
else:
    from llmperf.errors import ConfigError
    from llmperf.search import collect_results
    from llmperf.workflow import run_saved_plan, run_workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description='Docker-only automatic throughput search from one JSON configuration')
    sub = parser.add_subparsers(dest='command', required=True)
    for name, help_text in (
        ('auto', 'discover, prepare, search, verify and collect in one invocation'),
        ('doctor', 'check the configuration, inputs, GPU host and pinned image'),
        ('prepare', 'check environment and build the immutable request index'),
        ('plan', 'prepare inputs and save automatic candidates without benchmarking')):
        command = sub.add_parser(name, help=help_text)
        command.add_argument('--config', required=True, help='single experiment JSON')
    run = sub.add_parser('run', help='execute or resume a saved plan')
    run.add_argument('--plan', required=True)
    run.add_argument('--resume', action='store_true')
    collect = sub.add_parser('collect', help='rebuild results-index.json from recorded attempts')
    collect.add_argument('--run', required=True, help='experiment result directory')
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            result = run_saved_plan(args.plan, resume=args.resume)
        elif args.command == 'collect':
            if not Path(args.run).is_dir():
                raise ConfigError(f'result directory does not exist: {args.run}')
            collected = collect_results(args.run)
            result = {'status': 'PASS', 'rows': len(collected['rows']), 'result_dir': args.run}
        else:
            result = run_workflow(args.config, stop_after=args.command)
        # Detailed measurements stay in best.json; stdout keeps the usable result.
        output = {k: result[k] for k in ('status', 'result_dir', 'plan', 'candidates', 'requests', 'rows', 'trials') if k in result}
        if result.get('best'):
            best = result['best']
            output['best'] = {k: best[k] for k in ('candidate_id', 'concurrency', 'output_tokens_per_second', 'repetitions')}
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
        return 0 if result.get('status') in {'PASS', 'WARN'} else 1
    except ConfigError as exc:
        print(f'benchctl: configuration error: {exc}', file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f'benchctl: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('benchctl: interrupted; resume with run --plan <result-dir>/plan.json --resume', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
