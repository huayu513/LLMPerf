"""Contract checks for image-derived MoE backend planning."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llmperf.capabilities import EnvironmentSnapshot, GPUInfo, ImageSnapshot
from llmperf.backend_resolution import (
    _read_resolution_output, apply_backend_resolution, expand_auto_backends,
    resolve_in_image,
)
from llmperf.planner import fingerprint
from llmperf.types import CandidateConfig, DockerConfig, Plan
from llmperf.workflow import run_workflow


ROOT = Path(__file__).resolve().parents[1]


class BackendResolutionTests(unittest.TestCase):
    def plan(self):
        common = {'tp': 2, 'dp': 1, 'pp': 1, 'gpu_indexes': [0, 1],
                  'comparison_group': 'tp2'}
        auto = CandidateConfig('auto', backend=None, gpu_indexes=(0, 1),
                               static_config={**common, 'backend': None})
        flash = CandidateConfig('flash', backend='flashinfer_mxfp4', gpu_indexes=(0, 1),
                                static_config={**common, 'backend': 'flashinfer_mxfp4'})
        return Plan('test', candidates=(auto, flash), metadata={
            'candidates': {'auto': asdict(auto), 'flash': asdict(flash)},
            'comparison_groups': {'tp2': ['auto', 'flash']},
        })

    def test_deduplicates_only_matching_effective_configuration(self):
        plan = self.plan()
        resolved = apply_backend_resolution(plan, {
            'auto': {'status': 'resolved', 'effective_backend': 'flashinfer_mxfp4',
                     'effective_config_sha256': 'a' * 64},
            'flash': {'status': 'resolved', 'effective_backend': 'flashinfer_mxfp4',
                      'effective_config_sha256': 'a' * 64},
        }, source='test image')
        self.assertEqual([candidate.id for candidate in resolved.candidates], ['flash'])
        self.assertEqual(resolved.metadata['comparison_groups']['tp2'], ['flash'])
        self.assertEqual(resolved.metadata['backend_resolution']['removed_auto_duplicates'],
                         {'auto': 'flash'})
        different = apply_backend_resolution(plan, {
            'auto': {'status': 'resolved', 'effective_backend': 'flashinfer_mxfp4',
                     'effective_config_sha256': 'a' * 64},
            'flash': {'status': 'resolved', 'effective_backend': 'flashinfer_mxfp4',
                      'effective_config_sha256': 'b' * 64},
        }, source='test image')
        self.assertEqual(len(different.candidates), 2)

    def test_auto_resolution_adds_missing_explicit_backend_then_deduplicates(self):
        original = self.plan()
        auto = original.candidates[0]
        plan = Plan(original.id, candidates=(auto,), metadata={
            'search': {'backends': ()},
            'candidates': {'auto': asdict(auto)},
            'comparison_groups': {'tp2': ['auto']},
        })
        auto_result = {'status': 'resolved', 'effective_backend': 'flashinfer_mxfp4',
                       'effective_config_sha256': 'a' * 64}
        expanded, added = expand_auto_backends(plan, {'auto': auto_result})
        self.assertEqual(len(added), 1)
        concrete = expanded.candidates[1]
        self.assertEqual(concrete.backend, 'flashinfer_mxfp4')
        self.assertEqual(concrete.static_config['backend'], 'flashinfer_mxfp4')
        self.assertIn(concrete.id, expanded.metadata['comparison_groups']['tp2'])
        self.assertEqual(concrete.config_hash, fingerprint(concrete.static_config))
        final = apply_backend_resolution(expanded, {
            'auto': auto_result, concrete.id: auto_result,
        }, source='test image', added=added)
        self.assertEqual([candidate.id for candidate in final.candidates], [concrete.id])
        self.assertEqual(final.metadata['backend_resolution']['added_from_auto'], [concrete.id])

    def test_failed_explicit_probe_keeps_auto_and_records_failure(self):
        original = self.plan()
        auto = original.candidates[0]
        plan = Plan(original.id, candidates=(auto,), metadata={
            'search': {'backends': ()},
            'candidates': {'auto': asdict(auto)},
            'comparison_groups': {'tp2': ['auto']},
        })
        auto_result = {'status': 'resolved', 'effective_backend': 'flashinfer_mxfp4',
                       'effective_config_sha256': 'a' * 64}
        expanded, added = expand_auto_backends(plan, {'auto': auto_result})
        final = apply_backend_resolution(expanded, {
            'auto': auto_result, added[0]: {'status': 'error', 'reason': 'unsupported'},
        }, source='test image', added=added)
        self.assertEqual([candidate.id for candidate in final.candidates], ['auto'])
        self.assertIn(added[0], final.metadata['backend_resolution']['rejected_auto_expansions'])

    def test_explicit_backend_restriction_does_not_expand(self):
        plan = self.plan()
        plan.metadata['search'] = {'backends': ('triton',)}
        unchanged, added = expand_auto_backends(plan, {
            'auto': {'status': 'resolved', 'effective_backend': 'flashinfer_mxfp4'},
        })
        self.assertEqual(added, ())
        self.assertIs(unchanged, plan)

    def test_resolver_uses_benchmark_launcher_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'sglang' / 'srt'
            package.mkdir(parents=True)
            (root / 'sglang' / '__init__.py').write_text('')
            (package / '__init__.py').write_text('')
            (package / 'plugins.py').write_text('def load_plugins(): pass\n')
            (package / 'server_args.py').write_text('''
def prepare_server_args(argv):
    class Args:
        def resolve_once(self): pass
        def resolved_dict(self):
            backend = (argv[argv.index('--moe-runner-backend') + 1]
                       if '--moe-runner-backend' in argv else 'flashinfer_mxfp4')
            return {'moe_runner_backend': backend,
                    'tp': argv[argv.index('--tp-size') + 1]}
    return Args()
''')
            process = subprocess.run(
                [sys.executable, str(ROOT / 'benchmarks/server/resolve_config.py'), '--one'],
                input=json.dumps({'env': {'MODEL_PATH': '/model', 'TP_SIZE': '2',
                                          'CUDA_VISIBLE_DEVICES': '0,1'}}),
                capture_output=True, text=True,
                env={**os.environ, 'PYTHONPATH': directory},
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            line = next(line for line in process.stdout.splitlines()
                        if line.startswith('S1SLOW_RESOLVE_JSON='))
            result = json.loads(line.split('=', 1)[1])
            self.assertEqual(result['status'], 'resolved')
            self.assertEqual(result['effective_backend'], 'flashinfer_mxfp4')

    def test_partial_probe_output_keeps_completed_results(self):
        output = ('S1SLOW_RESOLVE_CASE_JSON=' + json.dumps({
            'key': 'finished',
            'result': {'status': 'resolved', 'effective_backend': 'triton',
                       'effective_config_sha256': 'a' * 64},
        }) + '\n')
        self.assertEqual(_read_resolution_output(output.encode())['finished']['status'],
                         'resolved')

    def test_host_batches_candidate_launches_in_pinned_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            model.mkdir()
            source = root / 'input.jsonl'
            source.write_text('{}\n')
            index = root / 'index.json'
            index.write_text('{}\n')
            image = 'repo/engine@sha256:' + 'a' * 64
            plan = self.plan()
            plan.metadata.update({
                'image': image, 'model_host': str(model), 'jsonl_host': str(source),
                'index_host': str(index), 'benchmark_dir': str(ROOT / 'benchmarks'),
                'results_host': str(root), 'served_model_name': 'model',
                'model_snapshot': {'raw': {'is_moe': True}},
            })
            config = SimpleNamespace(
                image=image, model_path=model, docker=DockerConfig(),
                search=SimpleNamespace(concurrency_max=16),
            )
            commands = []

            def run(command, **kwargs):
                commands.append(command)
                mounts = [command[index + 1] for index, value in enumerate(command[:-1])
                          if value == '--mount']
                payload = next(item for item in mounts
                               if 'dst=/run/resolution/cases.json' in item)
                path = Path(payload.split('src=', 1)[1].split(',dst=', 1)[0])
                cases = json.loads(path.read_text())
                results = {}
                for key, case in cases.items():
                    backend = case['env'].get('MOE_RUNNER_BACKEND') or 'flashinfer_mxfp4'
                    results[key] = {
                        'status': 'resolved', 'effective_backend': backend,
                        'effective_config_sha256': 'a' * 64,
                    }
                return SimpleNamespace(stdout='S1SLOW_RESOLVE_BATCH_JSON='
                                       + json.dumps(results) + '\n', stderr='', returncode=0)

            with patch('llmperf.backend_resolution.subprocess.run', side_effect=run):
                results = resolve_in_image(plan, config, root)
                only_added = resolve_in_image(plan, config, root,
                                              candidate_ids={'flash'})
            self.assertEqual(len(commands), 2)
            self.assertIn(image, commands[0])
            self.assertEqual(results['auto']['effective_backend'], 'flashinfer_mxfp4')
            self.assertEqual(results['flash']['effective_backend'], 'flashinfer_mxfp4')
            self.assertEqual(set(only_added), {'flash'})

    def test_plan_stage_uses_image_auto_result_when_metadata_omits_mxfp4(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            model.mkdir()
            (model / 'config.json').write_text(json.dumps({
                'model_type': 'deepseek_v4',
                'architectures': ['DeepseekV4ForCausalLM'],
                'num_attention_heads': 2, 'num_hidden_layers': 2,
                'n_routed_experts': 16,
                'quantization_config': {'quant_method': 'fp8'},
            }))
            source = root / 'input.jsonl'
            source.write_text(json.dumps({
                'api': 'openai_chat_completions', 'request_id': '1',
                'source_message_id': '1', 'captured_at': '2026-09-09T00:00:00Z',
                'request': {'model': 'input-alias',
                            'messages': [{'role': 'user', 'content': 'test'}],
                            'max_tokens': 5},
            }) + '\n')
            image = 'repo/engine@sha256:' + 'a' * 64
            config_path = root / 'experiment.json'
            config_path.write_text(json.dumps({
                'model_path': str(model), 'input_path': str(source),
                'image': image, 'output_dir': str(root / 'results'),
                'search': {'max_trials': 20, 'concurrency_max': 4},
            }))
            options = [
                '--trust-remote-code', '--model-path', '--served-model-name',
                '--enable-metrics', '--enable-cache-report', '--tp-size', '--dp',
                '--pp-size', '--mem-fraction-static', '--max-running-requests',
                '--host', '--port', '--chunked-prefill-size',
                '--default-chat-template-kwargs', '--moe-runner-backend',
                '--moe-a2a-backend', '--tool-call-parser', '--reasoning-parser',
            ]
            snapshot = EnvironmentSnapshot(
                'PASS', image=ImageSnapshot(image, 'PASS', image_id='sha256:test'),
                gpus=(GPUInfo(0, 'H100', 80000, '9.0'), GPUInfo(1, 'H100', 80000, '9.0')),
                container={'capabilities': {
                    'options': options,
                    'runner_backends': ['auto', 'triton', 'deep_gemm',
                                        'flashinfer_trtllm', 'flashinfer_mxfp4'],
                    'tool_parsers': ['deepseekv4'],
                    'reasoning_parsers': ['deepseek-v4'],
                }},
            )

            def resolver(plan, _config, _root):
                results = {}
                for candidate in plan.candidates:
                    backend = candidate.backend or 'flashinfer_mxfp4'
                    context = {key: value for key, value in candidate.static_config.items()
                               if key != 'backend'}
                    results[candidate.id] = {
                        'status': 'resolved', 'effective_backend': backend,
                        'effective_config_sha256': fingerprint({
                            'context': context, 'backend': backend,
                        }),
                    }
                return results

            outcome = run_workflow(config_path, stop_after='plan',
                                   probe=lambda _: snapshot,
                                   backend_resolver=resolver)
            plan = json.loads((Path(outcome['result_dir']) / 'plan.json').read_text())
            self.assertIn('flashinfer_mxfp4',
                          {candidate['backend'] for candidate in plan['candidates']})
            self.assertNotIn(None,
                             {candidate['backend'] for candidate in plan['candidates']})
            self.assertTrue(plan['metadata']['backend_resolution']['added_from_auto'])


if __name__ == '__main__':
    unittest.main()
