import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from s1slow.Automation.automation.adapters import ReplayAdapter
from s1slow.Automation.automation.types import PlanTask


AUTOMATION_ROOT = Path(__file__).resolve().parents[1]


class SelfContainedAutomationTests(unittest.TestCase):
    def test_runtime_defaults_to_bundled_benchmarks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"
            model.mkdir()
            jsonl = root / "input.jsonl"
            jsonl.write_text("{}\n", encoding="utf-8")
            index = root / "index.json"
            index.write_text("{}\n", encoding="utf-8")
            adapter = ReplayAdapter(
                {
                    "image": "registry/image@sha256:" + "a" * 64,
                    "model_host": str(model),
                    "jsonl_host": str(jsonl),
                    "index_host": str(index),
                },
                root / "results",
            )

            spec = adapter.build_spec(PlanTask("task", 0, "candidate"), 1)

            benchmark_mount = next(
                mount for mount in spec.mounts
                if mount.dst == "/opt/s1slow/benchmarks"
            )
            self.assertEqual(
                benchmark_mount.src,
                (AUTOMATION_ROOT / "benchmarks").resolve(),
            )

    def test_bundled_runtime_files_are_complete_and_shell_valid(self):
        required = (
            "run_point.sh",
            "run_deployment_point.py",
            "config/portable.env",
            "server/common.sh",
            "server/launch_server.sh",
            "server/profile_utils.sh",
            "server/profiles/AUTO.sh",
            "replay/run_replay.sh",
            "replay/replay_jsonl_sglang.py",
            "replay/prepare_jsonl_replay.py",
        )
        for relative in required:
            self.assertTrue(
                (AUTOMATION_ROOT / "benchmarks" / relative).is_file(),
                relative,
            )
        for relative in ("run_point.sh", "server/common.sh", "replay/run_replay.sh"):
            content = (AUTOMATION_ROOT / "benchmarks" / relative).read_text(encoding="utf-8")
            self.assertIn("config/portable.env", content, relative)
        shell_files = list((AUTOMATION_ROOT / "benchmarks").rglob("*.sh"))
        self.assertTrue(shell_files)
        for shell_file in shell_files:
            completed = subprocess.run(
                ["bash", "-n", str(shell_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cli_runs_after_automation_directory_is_copied_alone(self):
        with tempfile.TemporaryDirectory() as td:
            copied = Path(td) / "Automation"
            shutil.copytree(
                AUTOMATION_ROOT,
                copied,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            completed = subprocess.run(
                [sys.executable, str(copied / "benchctl.py"), "--help"],
                cwd=td,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("auto", completed.stdout)
            self.assertIn("one JSON configuration", completed.stdout)


if __name__ == "__main__":
    unittest.main()
