import json
import tempfile
import unittest
from pathlib import Path

from s1slow.Automation.automation.errors import ConfigError
from s1slow.Automation.automation.workflow import run_workflow, run_saved_plan
from s1slow.Automation.tests.test_auto_planner import environment


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / 'renamed'
        self.model.mkdir()
        (self.model / 'config.json').write_text(json.dumps({'model_type': 'qwen3',
            'architectures': ['Qwen3ForCausalLM'], 'num_attention_heads': 8, 'num_hidden_layers': 8}))
        (self.model / 'tokenizer_config.json').write_text('{"chat_template":"<tool_call> {{ tools }} </tool_call>"}')
        self.source = self.root / 'input.jsonl'
        self.source.write_text(''.join(json.dumps({'api': 'openai_chat_completions', 'request_id': str(i), 'source_message_id': str(i),
            'captured_at': '2026-09-09T00:00:00Z', 'request': {'model': 'input-alias',
            'messages': [{'role': 'user', 'content': 'test'}], 'max_tokens': 5}}) + '\n' for i in range(4)))
        self.config = self.root / 'experiment.json'
        self.config.write_text(json.dumps({'model_path': 'renamed', 'input_path': 'input.jsonl',
            'image': 'repo/engine@sha256:' + 'a' * 64,
            'output_dir': 'outputs', 'search': {'max_trials': 20, 'concurrency_max': 4}}))
        self.events = []

    def probe(self, config):
        self.events.append('probe')
        return environment(1)

    def execute(self, task, attempt):
        self.events.append(task.run_class)
        return {'status': 'VALID', 'output_tokens_per_second': 10 * task.concurrency, 'reasons': []}

    def test_single_config_runs_all_stages_and_preserves_input(self):
        original = self.source.read_bytes()
        result = run_workflow(self.config, probe=self.probe, executor=self.execute)
        self.assertEqual(result['status'], 'PASS')
        root = Path(result['result_dir'])
        for filename in ('resolved-config.json', 'environment.json', 'data/replay_index.json',
                         'plan.json', 'results-index.json', 'best.json'):
            self.assertTrue((root / filename).is_file(), filename)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertIn('final_repeat', self.events)
        resolved = json.loads((root / 'resolved-config.json').read_text())
        self.assertEqual(resolved['input']['models'], {'input-alias': 4})

    def test_each_invocation_gets_a_fresh_result_directory(self):
        first = run_workflow(self.config, stop_after='prepare', probe=self.probe)
        second = run_workflow(self.config, stop_after='prepare', probe=self.probe)
        self.assertNotEqual(first['result_dir'], second['result_dir'])

    def test_invalid_input_fails_before_environment_probe(self):
        self.source.write_text('{}\n')
        with self.assertRaises(ConfigError):
            run_workflow(self.config, probe=self.probe)
        self.assertEqual(self.events, [])

    def test_saved_plan_refuses_changed_input_before_execution(self):
        result = run_workflow(self.config, stop_after='plan', probe=self.probe)
        self.source.write_text(self.source.read_text() + '\n')
        self.events.clear()
        with self.assertRaisesRegex(ConfigError, 'input|source'):
            run_saved_plan(Path(result['result_dir']) / 'plan.json', executor=self.execute, probe=self.probe)
        self.assertEqual(self.events, [])

    def test_saved_plan_refuses_changed_driver_before_trials(self):
        result = run_workflow(self.config, stop_after='plan', probe=self.probe)
        self.events.clear()
        def changed_probe(config):
            env = environment(1)
            env.docker['facts'] = {'driver': 'new-driver'}
            return env
        with self.assertRaisesRegex(ConfigError, 'environment|driver'):
            run_saved_plan(Path(result['result_dir']) / 'plan.json', executor=self.execute, probe=changed_probe)
        self.assertEqual(self.events, [])

    def test_saved_plan_refuses_corrupted_index_before_probe(self):
        result = run_workflow(self.config, stop_after='plan', probe=self.probe)
        root = Path(result['result_dir'])
        path = root / 'data' / 'replay_index.json'
        data = json.loads(path.read_text())
        data['records'][0]['offset'] = 12345
        path.write_text(json.dumps(data))
        self.events.clear()
        with self.assertRaisesRegex(ConfigError, 'index'):
            run_saved_plan(root / 'plan.json', executor=self.execute, probe=self.probe)
        self.assertEqual(self.events, [])


if __name__ == '__main__':
    unittest.main()
