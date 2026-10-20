import os
import json
import subprocess
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

from benchmarks import run_deployment_point as controller


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

    def test_auto_profile_defaults_to_max_thinking_json_template(self):
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
                '\\"reasoning_effort\\":\\"max\\"\\,'
                '\\"thinking\\":true\\}',
                result.stdout,
            )

    def test_every_profile_defaults_to_max_and_preserves_explicit_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            for profile in sorted((ROOT / "benchmarks/server/profiles").glob("*.sh")):
                for override in (None, {"thinking": False}, {"reasoning_effort": "high", "thinking": True}):
                    with self.subTest(profile=profile.stem, override=override):
                        env = dict(os.environ)
                        env.update({"MODEL_PATH": directory, "SGLANG_BIN": "/bin/echo", "S1_ENGINE": "sglang"})
                        env.pop("DEFAULT_CHAT_TEMPLATE_KWARGS", None)
                        if override is not None:
                            env["DEFAULT_CHAT_TEMPLATE_KWARGS"] = json.dumps(override)
                        result = subprocess.run(
                            ["bash", str(ROOT / "benchmarks/server/launch_server.sh"), "argv-json", profile.stem],
                            cwd=ROOT, env=env, text=True, capture_output=True,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        command = json.loads(result.stdout)["argv"]
                        actual = json.loads(command[command.index("--default-chat-template-kwargs") + 1])
                        self.assertEqual(actual, override if override is not None else {
                            "enable_thinking": True, "reasoning_effort": "max", "thinking": True})


    def test_single_instance_completion_runs_after_replay_before_cleanup(self):
        source = RUN_POINT.read_text()
        replay_end = source.index('"$REPLAY" "${replay_args[@]}" --output-dir "$output_dir"')
        completion_source = source[replay_end:].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        for healthy in (True, False):
            with self.subTest(healthy=healthy), tempfile.TemporaryDirectory() as directory:
                report = {"instances": [{"id": "i0", "process_alive": True, "healthy": healthy}]}
                with patch.dict(sys.modules, {"run_deployment_point": controller}), \
                        patch.object(controller, "capture_server_completion", return_value=report) as capture, \
                        patch.object(sys, "path", list(sys.path)), \
                        patch.object(sys, "argv", ["-", directory, "http://localhost:25080", str(ROOT / "benchmarks")]):
                    with self.assertRaises(SystemExit) as outcome:
                        exec(compile(completion_source, str(RUN_POINT), "exec"), {})
                self.assertEqual(outcome.exception.code, 0 if healthy else 1)
                self.assertEqual(json.loads((Path(directory) / "server.completion.json").read_text()), report)
                self.assertEqual(capture.call_args.args[0][0][0], {"id": "i0"})
                self.assertEqual(capture.call_args.args[1], ["http://localhost:25080"])


if __name__ == "__main__":
    unittest.main()
