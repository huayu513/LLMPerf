import json
import math
import tempfile
import unittest
from pathlib import Path

from s1slow.Automation.automation.configuration import load_config
from s1slow.Automation.automation.errors import ConfigError
from s1slow.Automation.automation.types import DockerConfig


class ConfigurationTests(unittest.TestCase):
    IMAGE = "registry.example/sglang@sha256:" + "a" * 64

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "weights").mkdir()
        (self.root / "requests.jsonl").write_text("{}\n", encoding="utf-8")

    def write_config(self, **updates):
        data = {
            "model_path": "weights",
            "input_path": "requests.jsonl",
            "image": self.IMAGE,
        }
        data.update(updates)
        path = self.root / "experiment.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_minimal_config_resolves_paths_and_uses_bounded_defaults(self):
        config = load_config(self.write_config())

        self.assertEqual(config.model_path, (self.root / "weights").resolve())
        self.assertEqual(config.input_path, (self.root / "requests.jsonl").resolve())
        self.assertIsNone(config.output_dir)
        self.assertIsNone(config.gpu_indexes)
        self.assertEqual(config.search.concurrency_max, 64)
        self.assertIsNone(config.search.start_concurrency)
        self.assertEqual(config.search.explore_request_limit, 256)
        self.assertEqual(config.search.promotion_tolerance, 0.05)
        self.assertFalse(config.smoke)
        self.assertEqual(config.search.repetitions, 3)
        self.assertEqual(config.search.max_trials, 64)
        self.assertEqual(config.search.max_seconds, 14400)
        self.assertIsNone(config.search.backends)
        self.assertIsNone(config.search.open_loop_scales)
        self.assertEqual(config.warmup, 0)
        self.assertEqual(config.request_timeout, 3600.0)
        self.assertEqual(config.ready_timeout, 3600)
        self.assertEqual(config.model_overrides, {})
        self.assertEqual(config.docker, DockerConfig())
        self.assertEqual(config.source_path, (self.root / "experiment.json").resolve())

    def test_optional_values_are_typed_and_relative_output_is_resolved(self):
        config = load_config(self.write_config(
            output_dir="run output",
            gpu_indexes=[3, 1],
            warmup=2,
            smoke=True,
            request_timeout=12.5,
            ready_timeout=90,
            model_overrides={"chat_template_kwargs": {"thinking": False}},
            search={
                "concurrency_max": 8,
                "start_concurrency": 4,
                "explore_request_limit": 12,
                "promotion_tolerance": 0.1,
                "repetitions": 2,
                "max_trials": 4,
                "max_seconds": 30,
                "backends": ["builtin", "triton"],
                "open_loop_scales": [0.5, 2],
            },
            docker={
                "name_prefix": "bench",
                "network_mode": "bridge",
                "service_port": 12345,
                "shm_size": "8g",
                "ipc": "private",
            },
        ))

        self.assertEqual(config.output_dir, (self.root / "run output").resolve())
        self.assertEqual(config.gpu_indexes, (3, 1))
        self.assertEqual(config.search.start_concurrency, 4)
        self.assertEqual(config.search.explore_request_limit, 12)
        self.assertEqual(config.search.promotion_tolerance, 0.1)
        self.assertEqual(config.search.backends, ("builtin", "triton"))
        self.assertTrue(config.smoke)
        self.assertEqual(config.search.open_loop_scales, (0.5, 2.0))
        self.assertEqual(config.docker.service_port, 12345)

    def test_unknown_keys_are_rejected_at_each_strict_object_boundary(self):
        cases = [
            {"typo": 1},
            {"search": {"mystery": 1}},
            {"docker": {"privileged": True}},
        ]
        for updates in cases:
            with self.subTest(updates=updates), self.assertRaisesRegex(ConfigError, "unknown"):
                load_config(self.write_config(**updates))

    def test_required_fields_and_immutable_image_are_validated(self):
        for key in ("model_path", "input_path", "image"):
            with self.subTest(key=key):
                path = self.write_config()
                data = json.loads(path.read_text())
                del data[key]
                path.write_text(json.dumps(data))
                with self.assertRaises(ConfigError):
                    load_config(path)
        for image in ("sglang:latest", "repo@sha256:abc", "repo:tag@sha256:" + "a" * 64):
            with self.subTest(image=image), self.assertRaisesRegex(ConfigError, "immutable"):
                load_config(self.write_config(image=image))

    def test_boolean_nonfinite_duplicate_and_budget_values_are_rejected(self):
        bad_updates = [
            {"warmup": True},
            {"smoke": "false"},
            {"request_timeout": math.inf},
            {"ready_timeout": False},
            {"gpu_indexes": [0, 0]},
            {"gpu_indexes": []},
            {"gpu_indexes": [True]},
            {"search": {"open_loop_scales": [float("nan")]}},
            {"search": {"open_loop_scales": [1, 1.0]}},
            {"search": {"start_concurrency": 0}},
            {"search": {"explore_request_limit": -1}},
            {"search": {"promotion_tolerance": 1.0}},
            {"docker": {"name_prefix": "x" * 115}},
            {"docker": {"service_port": 54322}},
            {"search": {"repetitions": 3, "max_trials": 4}},
        ]
        for updates in bad_updates:
            with self.subTest(updates=updates), self.assertRaises(ConfigError):
                load_config(self.write_config(**updates))


if __name__ == "__main__":
    unittest.main()
