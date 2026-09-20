import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_POINT = ROOT / "benchmarks" / "run_point.sh"


class RuntimeShellContractTests(unittest.TestCase):
    def test_actual_controller_dry_run_preserves_candidate_environment(self):
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            env = dict(os.environ)
            env.update({
                "MODEL_PATH": str(temp),
                "SGLANG_BIN": "/bin/echo",
                "S1_PYTHON_BIN": "python3",
                "S1_RESULTS_ROOT": str(temp / "results"),
                "TP_SIZE": "4", "DP_SIZE": "2", "PP_SIZE": "3",
                "ENABLE_DP_ATTENTION": "1", "ENABLE_DP_LM_HEAD": "1",
                "MEM_FRACTION_STATIC": "0.72", "MAX_RUNNING_REQUESTS": "77",
                "CHUNKED_PREFILL_SIZE": "2048", "MOE_RUNNER_BACKEND": "flashinfer",
                "MOE_A2A_BACKEND": "deepep", "DEFAULT_CHAT_TEMPLATE_KWARGS": '{"thinking":false}',
            })
            result = subprocess.run(
                ["bash", str(RUN_POINT), "smoke", "--profile", "AUTO", "--mode", "closed-loop",
                 "--concurrency", "4", "--warmup", "2", "--request-timeout", "12.5",
                 "--ready-timeout", "33", "--limit", "1", "--dry-run"],
                cwd=ROOT, env=env, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = result.stdout
            for expected in (
                "--tp-size 4", "--dp 2", "--pp-size 3", "--mem-fraction-static 0.72",
                "--max-running-requests 77", "--chunked-prefill-size 2048",
                "--enable-dp-attention", "--enable-dp-lm-head",
                "--moe-runner-backend flashinfer", "--moe-a2a-backend deepep",
                "--default-chat-template-kwargs \\{\\\"thinking\\\":false\\}",
                "--mode closed-loop", "--warmup 2", "--request-timeout 12.5",
            ):
                self.assertIn(expected, output)
            replay_line = [line for line in output.splitlines() if line.startswith("exec ")][-1]
            self.assertNotEqual(replay_line.split()[1], "python3")
            self.assertTrue(Path(replay_line.split()[1]).is_absolute(), replay_line)

    def test_auto_profile_defaults_to_high_thinking_json_template(self):
        with tempfile.TemporaryDirectory() as td:
            env = dict(os.environ)
            env.update({"MODEL_PATH": td, "SGLANG_BIN": "/bin/echo"})
            env.pop("DEFAULT_CHAT_TEMPLATE_KWARGS", None)
            result = subprocess.run(
                ["bash", str(ROOT / "benchmarks/server/launch_server.sh"), "dry-run", "AUTO"],
                cwd=ROOT, env=env, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                '--default-chat-template-kwargs \\{\\"enable_thinking\\":true\\,'
                '\\"reasoning_effort\\":\\"high\\"\\,'
                '\\"thinking\\":true\\}',
                result.stdout,
            )


if __name__ == "__main__":
    unittest.main()
