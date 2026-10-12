import unittest
import json
import tempfile
from pathlib import Path

from adaptive_fixtures import EmptyPlanningAdvisor, environment, write_config
from llmperf.adaptive import build_candidate_pool, launch_identity
from llmperf.tuning import plan_tuning
from llmperf.types import CandidateConfig


class AdaptivePoolTests(unittest.TestCase):
    def base(self, ident="baseline", instances=1):
        static = {"tp": 1, "dp": 1, "pp": 1, "max_running_requests": 32,
                  "mem_fraction_static": 0.85, "chunked_prefill_size": 8192,
                  "deployment": {"instance_count": instances, "label": ident}}
        return CandidateConfig(ident, static_config=static)

    def test_runtime_equivalents_are_deduplicated(self):
        pool = build_candidate_pool([self.base(instances=4)],
                                    {"max_running_requests": [16, 32]}, 32, 256)
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0].id, "baseline")
        self.assertEqual(pool[0].static_config["max_running_requests"], 8)

    def test_small_pool_covers_knob_families(self):
        pool = build_candidate_pool([self.base()],
                                    {"mem_fraction_static": [0.75, 0.8],
                                     "chunked_prefill_size": [2048, 4096],
                                     "max_running_requests": [16, 32]}, 32, 4)
        changed = {key for row in pool for key in
                   ("mem_fraction_static", "chunked_prefill_size", "max_running_requests")
                   if row.static_config[key] != self.base().static_config[key]}
        self.assertEqual(len(pool), 4)
        self.assertEqual(len(changed), 3)

    def test_bases_are_interleaved_and_identities_unique(self):
        first = self.base("first")
        second = self.base("second", instances=2)
        pool = build_candidate_pool([first, second], {"mem_fraction_static": [0.75, 0.9]}, 32, 6)
        self.assertEqual([row.id for row in pool[:2]], ["first", "second"])
        self.assertEqual(len({launch_identity(row.static_config) for row in pool}), len(pool))

    def test_plan_freezes_pool_independently_of_trial_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            config = write_config(Path(directory))
            result = plan_tuning(config, probe=lambda _: environment(),
                                 advisor=EmptyPlanningAdvisor(), calibrate=False)
            plan = json.loads(Path(result["plan"]).read_text())
            metadata = plan["metadata"]
            self.assertGreater(len(plan["candidates"]), metadata["planning"]["exploration_capacity"])
            self.assertEqual(metadata["planning"]["reserved_final_trials"], 3)
            self.assertEqual(metadata["adaptive_policy"]["version"], 1)
            self.assertNotIn("test-token", json.dumps(plan))
            self.assertEqual(metadata["search"]["adaptive"]["batch_size"], 3)


if __name__ == "__main__":
    unittest.main()
