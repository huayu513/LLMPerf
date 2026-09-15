import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from s1slow.Automation.automation.capabilities import EnvironmentSnapshot, GPUInfo
from s1slow.Automation.automation.errors import ConfigError
from s1slow.Automation.automation import planner
from s1slow.Automation.automation.types import DockerConfig, ModelManifest, WorkloadManifest


def configuration(root, smoke=False, **search):
    limits = dict(concurrency_max=16, repetitions=3, max_trials=64, max_seconds=14400,
                  start_concurrency=None, explore_request_limit=256, promotion_tolerance=0.05,
                  backends=(), open_loop_scales=())
    limits.update(search)
    return SimpleNamespace(model_path=root / 'model', input_path=root / 'input.jsonl',
                           image='repo/engine@sha256:' + 'a' * 64, gpu_indexes=None, smoke=smoke,
                           search=SimpleNamespace(**limits), docker=DockerConfig(),
                           warmup=1, request_timeout=600, ready_timeout=3600)


def environment(count=2):
    return EnvironmentSnapshot('PASS', gpus=tuple(GPUInfo(i, 'H100', 80000, '9.0') for i in range(count)),
        container={'capabilities': {'options': ['--tp-size', '--dp', '--pp-size', '--enable-dp-attention',
        '--enable-dp-lm-head', '--moe-runner-backend', '--moe-a2a-backend',
        '--trust-remote-code', '--model-path', '--served-model-name', '--enable-metrics',
        '--enable-cache-report', '--mem-fraction-static', '--max-running-requests',
        '--host', '--port', '--chunked-prefill-size', '--default-chat-template-kwargs',
        '--tool-call-parser', '--reasoning-parser', '--speculative-algorithm'],
        'runner_backends': ['auto', 'triton', 'flashinfer_mxfp4'], 'a2a_backends': ['none', 'deepep'],
        'tool_parsers': ['qwen', 'qwen25'], 'reasoning_parsers': ['qwen3']}})


class AutoPlannerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('/tmp/planner-unit')
        self.config = configuration(self.root)
        self.model = ModelManifest('renamed-model', '/model', reasoning_parser='qwen3',
                                   raw={'model_type': 'qwen3', 'num_attention_heads': 8,
                                        'num_hidden_layers': 8, 'is_moe': False})
        self.workload = WorkloadManifest('input', '/input', sha256='abc',
                                         raw={'count': 20, 'models': {'captured-model': 20}})

    def plan(self, env=None):
        return planner.create_search_plan(self.config, self.model, self.workload,
                                          env or environment(), self.root / 'results')

    def test_builds_hardware_candidates_and_uses_captured_alias(self):
        plan = self.plan()
        self.assertTrue(any(c.tp == 2 for c in plan.candidates))
        self.assertTrue(any(c.dp == 2 and not c.dp_attention for c in plan.candidates))
        self.assertEqual(plan.metadata['served_model_name'], 'captured-model')
        self.assertEqual(plan.metadata['expected_request_count'], 20)
        self.assertEqual(plan.metadata['profile'], 'AUTO')
        self.assertFalse(plan.metadata['smoke'])
        self.assertTrue(all(c.backend is None and not c.dspark for c in plan.candidates))

    def test_deployment_topology_is_a_search_dimension(self):
        self.config.gpu_indexes = (1, 3)
        plan = self.plan(environment(4))
        deployments = {c.static_config['deployment']['label']: c.static_config['deployment']
                       for c in plan.candidates}
        self.assertIn('2卡1实例', deployments)
        self.assertIn('2卡2实例', deployments)
        self.assertEqual(deployments['2卡1实例']['instance_count'], 1)
        self.assertEqual(deployments['2卡1实例']['gpus_per_instance'], 2)
        self.assertEqual(deployments['2卡2实例']['instance_count'], 2)
        self.assertEqual(deployments['2卡2实例']['gpus_per_instance'], 1)
        self.assertTrue(all(c.gpu_indexes == (1, 3) for c in plan.candidates))

    def test_attention_heads_and_gpu_masks_bound_parallelism(self):
        self.config.gpu_indexes = (1, 3)
        self.model = ModelManifest('model', '/m', raw={'num_attention_heads': 3,
                              'num_hidden_layers': 1, 'is_moe': False})
        plan = self.plan(environment(4))
        self.assertTrue(all(c.tp == 1 for c in plan.candidates))
        self.assertTrue(all(set(c.gpu_indexes) <= {1, 3} for c in plan.candidates))
        self.assertTrue(all(c.pp == 1 for c in plan.candidates))

    def test_missing_parser_support_fails_before_launch(self):
        env = environment()
        env.container['capabilities']['reasoning_parsers'] = []
        with self.assertRaisesRegex(ConfigError, 'qwen3'):
            self.plan(env)

    def test_missing_mandatory_launch_option_fails_before_smoke(self):
        for option in ('--pp-size', '--default-chat-template-kwargs', '--enable-cache-report'):
            with self.subTest(option=option):
                env = environment()
                env.container['capabilities']['options'].remove(option)
                with self.assertRaisesRegex(ConfigError, option):
                    self.plan(env)

    def test_unknown_explicit_backend_is_an_error(self):
        self.config.search.backends = ('invented',)
        with self.assertRaisesRegex(ConfigError, 'backend'):
            self.plan()

    def test_invalid_gpu_selection_and_mig_are_rejected(self):
        self.config.gpu_indexes = (9,)
        with self.assertRaises(ConfigError):
            self.plan()
        self.config.gpu_indexes = None
        env = EnvironmentSnapshot('PASS', gpus=(GPUInfo(0, 'H100', mig_mode='Enabled'),),
                                  container=environment().container)
        with self.assertRaisesRegex(ConfigError, 'MIG'):
            self.plan(env)

    def test_early_candidates_cover_strategy_families(self):
        self.model = ModelManifest('moe', '/m', raw={'model_type': 'deepseek_v4',
            'is_moe': True, 'num_attention_heads': 64, 'num_hidden_layers': 80})
        env = environment(8)
        env.container['capabilities']['options'].append('--speculative-dspark-block-size')
        first = self.plan(env).candidates[:15]
        self.assertTrue(any(c.dp_attention for c in first))
        self.assertTrue(any(c.pp > 1 for c in first))
        self.assertTrue(any(c.backend is not None for c in first))
        self.assertTrue(any(c.dspark for c in first))

    def test_plan_roundtrip_and_immutable_write(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            planner.write_plan(path, plan)
            self.assertEqual(planner.plan_to_dict(planner.load_plan(path)), planner.plan_to_dict(plan))
            with self.assertRaises(FileExistsError):
                planner.write_plan(path, plan)


if __name__ == '__main__':
    unittest.main()
