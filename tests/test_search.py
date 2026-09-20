import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from s1slow.Automation.automation.types import CandidateConfig, Plan
from s1slow.Automation.automation.search import execute_search
from s1slow.Automation.automation.planner import fingerprint, plan_to_dict


def plan(max_trials=40, repetitions=3, candidates=1, smoke=False,
         start_concurrency=1, gpu_indexes=(0,), concurrency_max=16,
         expected_request_count=None, backend_values=None):
    if expected_request_count is None:
        expected_request_count = max(20, concurrency_max)
    backend_values = backend_values or [None] * candidates
    items = tuple(CandidateConfig('c' + str(i), gpu_indexes=gpu_indexes,
                  backend=backend_values[i],
                  static_config={'tp': 1, 'dp': 1, 'pp': 1,
                                 'backend': backend_values[i]},
                  config_hash='hash' + str(i))
                  for i in range(candidates))
    return Plan('test', candidates=items, metadata={
        'expected_source_sha256': 'abc', 'expected_request_count': expected_request_count,
        'smoke': smoke,
        'search': {'concurrency_max': concurrency_max, 'max_trials': max_trials,
                   'start_concurrency': start_concurrency, 'explore_request_limit': 256,
                   'promotion_tolerance': 0.05, 'max_seconds': 10000,
                   'repetitions': repetitions, 'open_loop_scales': []},
        'candidates': {c.id: {'id': c.id, 'static_config': c.static_config} for c in items}})


class SearchTests(unittest.TestCase):
    def execute(self, p, executor, root=None, **kwargs):
        return execute_search(p, root or Path(self.directory.name), executor=executor, **kwargs)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.calls = []

    def executor(self, task, attempt):
        self.calls.append(task)
        score = {1: 10, 2: 18, 3: 24, 4: 25, 5: 23, 6: 21, 8: 19, 16: 18}.get(task.concurrency, 20)
        return {'status': 'VALID', 'output_tokens_per_second': score,
                'candidate_id': task.candidate_id, 'concurrency': task.concurrency, 'reasons': []}

    def test_searches_past_one_and_repeats_measured_winner(self):
        p = plan(start_concurrency=16, concurrency_max=64, expected_request_count=64)

        def execute(task, attempt):
            self.calls.append(task)
            score = {16: 10, 32: 18, 48: 25, 64: 23}[task.concurrency]
            return {'status': 'VALID', 'output_tokens_per_second': score,
                    'candidate_id': task.candidate_id, 'concurrency': task.concurrency,
                    'reasons': []}

        result = self.execute(p, execute)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['best']['concurrency'], 48)
        self.assertEqual(result['best']['output_tokens_per_second'], 25)
        explored = [t.concurrency for t in self.calls if t.run_class == 'concurrency']
        self.assertEqual(explored, [16, 32, 48, 64])
        repeats = [t for t in self.calls if t.run_class == 'final_repeat']
        self.assertEqual(len(repeats), 3)
        self.assertEqual(sum(1 for t in repeats if t.concurrency == 48), 3)
        self.assertFalse(any(t.concurrency in {47, 49} for t in self.calls))
        self.assertTrue((Path(self.directory.name) / 'best.json').is_file())

    def test_auto_start_concurrency_uses_selected_gpu_count(self):
        self.execute(plan(start_concurrency=None, gpu_indexes=(0, 1)), self.executor)
        first_load = [t for t in self.calls if t.run_class == 'concurrency'][0]
        self.assertEqual(first_load.concurrency, 16)

    def test_winner_exports_effective_config_and_repeat_commands(self):
        def execute(task, attempt):
            result = self.executor(task, attempt)
            result['effective_static_config'] = {
                'tp': 1, 'dp': 1, 'pp': 1,
                'max_running_requests': task.concurrency,
                'backend': 'resolved-runner',
            }
            result['server_command_path'] = f'/results/{task.id}/server.command.txt'
            return result
        result = self.execute(plan(), execute)
        best = result['best']
        self.assertEqual(best['configuration']['static_config']['max_running_requests'], 16)
        self.assertEqual(best['configuration']['static_config']['backend'], 'resolved-runner')
        self.assertEqual(len(best['server_command_paths']), 3)
        self.assertTrue(all('final_repeat' in path for path in best['server_command_paths']))

    def test_changed_effective_config_cannot_publish_repeat_median(self):
        def execute(task, attempt):
            result = self.executor(task, attempt)
            result['effective_static_config'] = {'backend': task.id}
            return result
        result = self.execute(plan(), execute)
        self.assertIsNone(result['best'])
        self.assertEqual(result['status'], 'INCONCLUSIVE')
        self.assertIn('effective_configuration_changed_between_repetitions',
                      [d['reason'] for d in result['stop_reasons']])

    def test_oom_stops_increasing_concurrency(self):
        p = plan(start_concurrency=16, concurrency_max=64, expected_request_count=64)

        def execute(task, attempt):
            if task.concurrency >= 48:
                self.calls.append(task)
                return {'status': 'FAILED', 'reasons': ['OOM']}
            return self.executor(task, attempt)
        result = self.execute(p, execute)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['best']['concurrency'], 32)
        self.assertTrue(any(t.concurrency == 32 for t in self.calls))
        self.assertFalse(any(t.concurrency > 48 for t in self.calls))

    def test_budget_reserves_repeats_and_screens_multiple_candidates(self):
        result = self.execute(plan(max_trials=12, candidates=2, smoke=True), self.executor)
        self.assertLessEqual(len(self.calls), 12)
        self.assertEqual({t.candidate_id for t in self.calls if t.run_class == 'smoke'}, {'c0', 'c1'})
        self.assertEqual(result['status'], 'PASS')
        self.assertGreaterEqual(len([t for t in self.calls if t.run_class == 'final_repeat']), 3)

    def test_smoke_is_skipped_by_default(self):
        result = self.execute(plan(candidates=2), self.executor)
        self.assertEqual(result['status'], 'PASS')
        self.assertFalse(any(t.run_class == 'smoke' for t in self.calls))
        self.assertTrue(any(t.run_class == 'concurrency' and t.concurrency == 1 for t in self.calls))

    def test_inconclusive_or_nonfinite_results_cannot_win(self):
        for score in (float('nan'), float('inf'), -1):
            with tempfile.TemporaryDirectory() as directory:
                result = self.execute(plan(), lambda t, a: {'status': 'VALID',
                    'output_tokens_per_second': score}, Path(directory))
                self.assertIsNone(result['best'])
                self.assertEqual(result['status'], 'INCONCLUSIVE')

    def test_failed_repetition_does_not_publish_verified_winner(self):
        def execute(task, attempt):
            if task.run_class == 'final_repeat':
                return {'status': 'INCONCLUSIVE', 'reasons': ['usage_missing']}
            return self.executor(task, attempt)
        result = self.execute(plan(), execute)
        self.assertIsNone(result['best'])
        self.assertEqual(result['status'], 'INCONCLUSIVE')
        self.assertIsNotNone(result['provisional_best'])

    def test_failed_smoke_blocks_only_its_candidate(self):
        def execute(task, attempt):
            self.calls.append(task)
            if task.candidate_id == 'c0':
                return {'status': 'FAILED', 'reasons': ['startup_failed']}
            return {'status': 'VALID', 'output_tokens_per_second': task.concurrency * 10}
        result = self.execute(plan(candidates=2, smoke=True), execute)
        self.assertEqual(result['best']['candidate_id'], 'c1')
        self.assertEqual([t.run_class for t in self.calls if t.candidate_id == 'c0'], ['smoke'])

    def test_startup_failure_blocks_later_candidates_with_same_backend(self):
        p = plan(candidates=3, backend_values=['triton', 'triton', 'flashinfer'])

        def execute(task, attempt):
            self.calls.append(task)
            if task.candidate_id == 'c0':
                return {
                    'status': 'FAILED',
                    'failure_phase': 'startup',
                    'reasons': ['server_info_missing'],
                }
            return {
                'status': 'VALID',
                'output_tokens_per_second': task.concurrency * 10,
            }

        result = self.execute(p, execute)

        self.assertEqual(result['backend_blocks']['triton']['candidate_id'], 'c0')
        self.assertEqual(result['candidate_states']['c0']['status'], 'FAILED')
        self.assertEqual(result['candidate_states']['c1']['status'], 'SKIPPED')
        self.assertEqual(result['best']['candidate_id'], 'c2')
        self.assertFalse(any(t.candidate_id == 'c1' for t in self.calls))
        self.assertTrue(any(t.candidate_id == 'c2' for t in self.calls))
        skipped_rows = [
            row for row in json.loads(
                (Path(self.directory.name) / 'results-index.json').read_text()
            )['rows']
            if row.get('candidate_id') == 'c1'
        ]
        self.assertTrue(skipped_rows)
        self.assertEqual(skipped_rows[0]['status'], 'SKIPPED')

    def test_replay_failure_does_not_block_backend(self):
        p = plan(candidates=2, backend_values=['triton', 'triton'])

        def execute(task, attempt):
            self.calls.append(task)
            if task.candidate_id == 'c0':
                return {'status': 'FAILED', 'reasons': ['OOM']}
            return {'status': 'VALID', 'output_tokens_per_second': 10}

        result = self.execute(p, execute)

        self.assertEqual(result['backend_blocks'], {})
        self.assertEqual(result['candidate_states']['c1']['status'], 'VALID')
        self.assertTrue(any(t.candidate_id == 'c1' for t in self.calls))

    def test_backend_failure_does_not_skip_already_valid_candidate_repeats(self):
        p = plan(candidates=2, backend_values=['triton', 'triton'])

        def execute(task, attempt):
            self.calls.append(task)
            if task.candidate_id == 'c1':
                return {
                    'status': 'FAILED',
                    'failure_phase': 'startup',
                    'reasons': ['readiness_missing'],
                }
            return {'status': 'VALID', 'output_tokens_per_second': 10}

        result = self.execute(p, execute)

        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['best']['candidate_id'], 'c0')
        c0_repeats = [t for t in self.calls if t.candidate_id == 'c0'
                      and t.run_class == 'final_repeat']
        self.assertGreaterEqual(len(c0_repeats), 3)
        self.assertEqual(result['candidate_states']['c1']['status'], 'FAILED')
        self.assertEqual(result['backend_blocks']['triton']['candidate_id'], 'c1')

    def test_requested_open_loop_failure_is_reported_separately(self):
        p = plan()
        p.metadata['search']['open_loop_scales'] = [1, 2]
        def execute(task, attempt):
            if task.run_class == 'open_loop':
                return {'status': 'INCONCLUSIVE', 'reasons': ['requests_failed']}
            return self.executor(task, attempt)
        result = self.execute(p, execute)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertIsNotNone(result['best'])
        self.assertEqual(len(result['open_loop_results']), 2)

    def test_failed_screens_are_backfilled_while_budget_remains(self):
        def execute(task, attempt):
            self.calls.append(task)
            if task.candidate_id != 'c15':
                return {'status': 'FAILED', 'reasons': ['incompatible']}
            return {'status': 'VALID', 'output_tokens_per_second': task.concurrency * 10}
        result = self.execute(plan(max_trials=64, candidates=16), execute)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['best']['candidate_id'], 'c15')

    def test_screen_replacements_preserve_concurrency_search_budget(self):
        def execute(task, attempt):
            self.calls.append(task)
            if task.candidate_id != 'c0':
                return {'status': 'FAILED', 'reasons': ['incompatible']}
            return {'status': 'VALID', 'output_tokens_per_second': task.concurrency * 10}
        p = plan(max_trials=64, candidates=64)
        result = self.execute(p, execute)
        self.assertEqual(result['status'], 'PASS')
        self.assertGreater(result['best']['concurrency'], 1)
        self.calls.clear()
        resumed = self.execute(p, execute, resume=True)
        self.assertEqual(resumed['best'], result['best'])
        self.assertEqual(self.calls, [])

    def test_close_open_loop_scales_are_distinct_trials(self):
        p = plan()
        p.metadata['search']['open_loop_scales'] = [1.0, 1.0000001]
        self.execute(p, self.executor)
        self.assertEqual([t.scale for t in self.calls if t.run_class == 'open_loop'], [1.0, 1.0000001])

    def test_resume_rechecks_effective_native_configuration(self):
        backend = ['triton']
        launches = []
        root = Path(self.directory.name)
        def launch(task, attempt):
            launches.append(task.id)
        def read(attempt_dir, task, metadata, exit_code=0):
            return {'status': 'VALID', 'output_tokens_per_second': 10.0,
                    'result_path': str(attempt_dir),
                    'effective_static_config': {'backend': backend[0]}}
        with patch('s1slow.Automation.automation.adapters.ReplayAdapter') as adapter, \
             patch('s1slow.Automation.automation.adapters.read_attempt', side_effect=read):
            adapter.return_value.side_effect = launch
            execute_search(plan(), root)
            launches.clear()
            backend[0] = 'flashinfer'
            result = execute_search(plan(), root, resume=True)
        self.assertTrue(launches, 'changed native parameters must invalidate cached trials')
        self.assertEqual(result['best']['configuration']['static_config']['backend'], 'flashinfer')

    def test_resume_reuses_completed_evidence_and_checks_plan_identity(self):
        self.execute(plan(), self.executor)
        self.calls.clear()
        self.execute(plan(), self.executor, resume=True)
        self.assertEqual(self.calls, [])
        changed = plan()
        changed.metadata['expected_source_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'plan'):
            self.execute(changed, self.executor, resume=True)

    def test_resume_does_not_count_interrupted_allocation_against_budget(self):
        p = plan(max_trials=4, repetitions=3)
        root = Path(self.directory.name)
        state = {
            'plan_hash': fingerprint(plan_to_dict(p)),
            'elapsed_seconds': 0,
            'decisions': [],
            'trials': {
                'c0-concurrency-c1': [{
                    'manifest': 'trials/c0-concurrency-c1/attempt-001/trial.json',
                    'task': {},
                    'fingerprint': None,
                }],
            },
        }
        (root / 'search-state.json').write_text(json.dumps(state))

        result = self.execute(p, self.executor, root, resume=True)

        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(self.calls[0].id, 'c0-concurrency-c1')
        self.assertEqual(result['trials'], 4)


if __name__ == '__main__':
    unittest.main()
