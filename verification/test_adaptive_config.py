import unittest
from dataclasses import asdict

from llmperf.configuration import _search, validate_advisor
from llmperf.errors import ConfigError


class AdaptiveConfigurationTests(unittest.TestCase):
    def test_defaults_preserve_strict(self):
        config = _search({}, "search")
        self.assertEqual(config.strategy, "strict")
        self.assertEqual(asdict(config)["adaptive"]["batch_size"], 3)

    def test_adaptive_reserves_finalists_and_baseline(self):
        with self.assertRaisesRegex(ConfigError, "reserve"):
            _search({"strategy": "adaptive", "max_trials": 10}, "search")
        config = _search({"strategy": "adaptive", "max_trials": 11}, "search")
        self.assertEqual(config.adaptive.final_top_k, 2)

    def test_unknown_and_invalid_values_are_rejected(self):
        for data in (
            {"strategy": "agent"},
            {"adaptive": {"batch_size": True}},
            {"adaptive": {"batch_size": 9}},
            {"adaptive": {"candidate_pool_limit": 1}},
            {"adaptive": {"max_advisor_calls": -1}},
            {"adaptive": {"final_top_k": 6}},
            {"adaptive": {"commands": []}},
        ):
            with self.subTest(data=data), self.assertRaises(ConfigError):
                _search(data, "search")

    def test_rule_only_adaptive_is_supported(self):
        config = _search({"strategy": "adaptive", "adaptive": {"max_advisor_calls": 0}}, "search")
        self.assertEqual(config.adaptive.max_advisor_calls, 0)

    def test_task_token_budgets_are_validated(self):
        settings = {"base_url": "http://localhost:9000/v1", "model": "local"}
        config = validate_advisor({**settings, "task_max_tokens": {"next_experiments": 1024}})
        self.assertEqual(config["task_max_tokens"], {"next_experiments": 1024})
        for budgets in ({"next_experiments": 0}, {"shell": 64}, {"scene": True}):
            with self.subTest(budgets=budgets), self.assertRaises(ConfigError):
                validate_advisor({**settings, "task_max_tokens": budgets})

    def test_thinking_is_optional_and_requires_a_boolean(self):
        settings = {"base_url": "http://localhost:9000/v1", "model": "local"}
        self.assertIs(validate_advisor(settings)["thinking"], False)
        self.assertIs(validate_advisor({**settings, "thinking": True})["thinking"], True)
        for thinking in ("true", 1, None):
            with self.subTest(thinking=thinking), self.assertRaises(ConfigError):
                validate_advisor({**settings, "thinking": thinking})


if __name__ == "__main__":
    unittest.main()
