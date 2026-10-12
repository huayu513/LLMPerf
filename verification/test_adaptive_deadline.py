import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from llmperf.docker_runtime import DockerRuntime, DockerTaskSpec
from llmperf.adapters import read_attempt
from llmperf.types import PlanTask


class DeadlineTests(unittest.TestCase):
    def test_deadline_terminates_a_blocked_stream_without_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            started = time.monotonic()
            code = DockerRuntime()._stream_run(
                [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(30)"],
                Path(directory) / "run.log", timeout_seconds=0.1)
            self.assertEqual(code, 124)
            self.assertLess(time.monotonic() - started, 5)

    def test_invalid_deadline_fails_before_starting_container(self):
        spec = DockerTaskSpec("repo/engine@sha256:" + "a" * 64, "deadline-test")
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                DockerRuntime().run(spec, Path("unused.log"), timeout_seconds=timeout)

    def test_deadline_removes_only_the_named_container(self):
        calls = []
        def runner(command, **kwargs):
            calls.append(command)
            if command[:3] == ["docker", "rm", "-f"]:
                return subprocess.CompletedProcess(command, 0)
            return subprocess.Popen(command, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            runtime = DockerRuntime(runner)
            code = runtime._stream_run([sys.executable, "-c", "import time; time.sleep(30)"],
                                      Path(directory) / "run.log", 0.1, "owned-trial")
            self.assertEqual(code, 124)
            deadline = time.monotonic() + 2
            while ["docker", "rm", "-f", "owned-trial"] not in calls and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIn(["docker", "rm", "-f", "owned-trial"], calls)

    def test_deadline_marker_cannot_be_scored_as_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "container-timeout").write_text("1\n")
            result = read_attempt(root, PlanTask("trial", 3, "c0", concurrency=4),
                                  {"candidates": {"c0": {"static_config": {}}}})
            self.assertNotEqual(result["status"], "VALID")
            self.assertEqual(result["failure_kind"], "timeout")
            self.assertIn("trial_deadline_exceeded", result["reasons"])


if __name__ == "__main__":
    unittest.main()
