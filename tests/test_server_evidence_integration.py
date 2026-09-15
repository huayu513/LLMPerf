"""Exercise the shell's actual evidence writer through the host result reader."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from s1slow.Automation.automation.adapters import read_attempt
from s1slow.Automation.automation.types import PlanTask


ROOT = Path(__file__).resolve().parents[1]


class ServerEvidenceIntegrationTests(unittest.TestCase):
    def run_native_evidence(self, *, moe=False, dpa=False, kwargs=None, runner=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        attempt = Path(directory.name)
        output = attempt / 'formal' / 'AUTO' / 'run_001'
        output.mkdir(parents=True)
        kwargs = kwargs or {}
        static = {
            'tp': 2 if dpa else 1, 'dp': 2 if dpa else 1, 'pp': 1,
            'dp_attention': dpa, 'dp_lm_head': dpa, 'dspark': False,
            'backend': runner, 'moe_a2a_backend': 'none' if moe else None,
            'mem_fraction_static': .85, 'max_running_requests': 64,
            'chunked_prefill_size': 8192,
        }
        requested = {k: v for k, v in static.items() if v is not None}
        requested.update(model_path='/model', served_model_name='captured-alias',
                         max_running_requests=4, chat_template_kwargs=kwargs)
        if moe:
            requested['backend'] = runner or 'auto'
        server_args = {
            'cuda_graph_config': {'prefill': {'backend': 'tc_piecewise'}},
            'model_path': '/model', 'served_model_name': 'captured-alias',
            'tp_size': static['tp'], 'dp_size': static['dp'], 'pp_size': 1,
            'enable_dp_attention': dpa, 'enable_dp_lm_head': dpa,
            'speculative_algorithm': None, 'moe_runner_backend': 'triton' if moe else 'auto',
            'moe_a2a_backend': 'none', 'mem_fraction_static': .85,
            'max_running_requests': 4,
            'chunked_prefill_size': 4096 if dpa else 8192,
            'default_chat_template_kwargs': dict(reversed(list(kwargs.items()))),
        }
        (output / 'server.requested.json').write_text(json.dumps(requested))
        (output / 'server.info.json').write_text(json.dumps({'server_args': server_args}))
        controller = (ROOT / 'benchmarks' / 'run_point.sh').read_text()
        source = controller.split('evidence_resolved=', 1)[1].split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        process = subprocess.run([
            sys.executable, '-B', '-', str(output / 'server.requested.json'),
            str(output / 'server.info.json'), str(output / 'server.evidence.json'), '1',
        ], input=source, text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        evidence = json.loads((output / 'server.evidence.json').read_text())
        self.assertTrue(evidence['resolved'], evidence)
        (output / 'formal.summary.json').write_text(json.dumps({
            'jsonl_sha256': 'a' * 64, 'request_count': 4,
            'successes': 4, 'failures': 0, 'server_usage_available': 4,
            'server_usage_missing': 0, 'completion_tokens': 40,
            'measured_seconds': 2.0, 'output_tokens_per_second': 20.0,
            'base_url': 'http://127.0.0.1:25080',
        }))
        (output / 'run_manifest.json').write_text(json.dumps({'state': 'complete', 'exit_code': 0}))
        (output / 'server.command.sh').write_text('sglang serve --model-path /model\n')
        (attempt / 'container-exit-code').write_text('0\n')
        metadata = {
            'model_host': '/host/checkpoint', 'served_model_name': 'captured-alias',
            'chat_template_kwargs': kwargs, 'model_snapshot': {'raw': {'is_moe': moe}},
            'expected_source_sha256': 'a' * 64, 'expected_request_count': 4,
            'candidates': {'candidate': {'static_config': static}},
        }
        task = PlanTask('test', 3, 'candidate', run_class='formal', concurrency=4)
        result = read_attempt(attempt, task, metadata)
        self.assertEqual(result['status'], 'VALID', result['reasons'])
        self.assertEqual(result['effective_static_config']['max_running_requests'], 4)
        self.assertTrue(result['evidence_fingerprint'])
        self.assertEqual(result['server_command_path'], str(output / 'server.command.sh'))
        self.native_fixture = (attempt, output, task, metadata)
        return result

    def test_dense_defaults_accept_empty_template_and_disabled_speculation(self):
        self.run_native_evidence()

    def test_template_object_comparison_is_independent_of_key_order(self):
        self.run_native_evidence(kwargs={'thinking': False, 'reasoning_effort': 'low'})

    def test_moe_auto_backend_resolves_to_concrete_runner(self):
        result = self.run_native_evidence(moe=True)
        self.assertEqual(result['effective_static_config']['backend'], 'triton')

    def test_explicit_moe_runner_is_distinct_from_cuda_graph_backend(self):
        result = self.run_native_evidence(moe=True, runner='triton')
        self.assertEqual(result['effective_static_config']['backend'], 'triton')

    def test_dp_attention_preserves_effective_per_rank_prefill(self):
        result = self.run_native_evidence(moe=True, dpa=True)
        self.assertEqual(result['effective_static_config']['chunked_prefill_size'], 4096)


    def test_incomplete_or_wrong_source_measurements_are_rejected(self):
        self.run_native_evidence()
        attempt, output, task, metadata = self.native_fixture
        path = output / 'formal.summary.json'
        original = json.loads(path.read_text())
        changes = (
            {'jsonl_sha256': 'changed'}, {'successes': 3}, {'successes': 'four'},
            {'server_usage_available': 3, 'server_usage_missing': 1},
            {'completion_tokens': float('nan')}, {'output_tokens_per_second': 99},
            {'base_url': 'http://127.0.0.1:25081'},
            {'base_url': 'http://localhost:25080@example.invalid'},
        )
        for change in changes:
            with self.subTest(change=change):
                path.write_text(json.dumps({**original, **change}))
                self.assertNotEqual(read_attempt(attempt, task, metadata)['status'], 'VALID')
        path.write_text(json.dumps(original))
        (output / 'run_manifest.json').write_text('{"state":"complete"}')
        self.assertNotEqual(read_attempt(attempt, task, metadata)['status'], 'VALID')

    def test_native_backend_change_changes_effective_config_and_fingerprint(self):
        initial = self.run_native_evidence(moe=True)
        attempt, output, task, metadata = self.native_fixture
        path = output / 'server.info.json'
        changed = json.loads(path.read_text())
        changed['server_args']['moe_runner_backend'] = 'flashinfer'
        path.write_text(json.dumps(changed))
        result = read_attempt(attempt, task, metadata)
        self.assertNotEqual(result['evidence_fingerprint'], initial['evidence_fingerprint'])
        self.assertEqual(result['effective_static_config']['backend'], 'flashinfer')



if __name__ == '__main__':
    unittest.main()
