import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from s1slow.Automation.automation.docker_runtime import DockerRuntime, DockerTaskSpec, Mount


class DockerRuntimeTests(unittest.TestCase):
    IMAGE = "registry.example/s1slow@sha256:" + "a" * 64

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "model dir").mkdir()
        (self.root / "input data.jsonl").write_text("{}\n")
        self.spec = DockerTaskSpec(self.IMAGE, "run-123", (0, 1), (
            Mount(self.root / "model dir", "/mnt/model", True),
            Mount(self.root / "input data.jsonl", "/mnt/input.jsonl", True),
            Mount(self.root / "results", "/mnt/results", False),
        ), {"API_TOKEN": "secret", "VISIBLE": "yes"}, ("python", "-c", "print('hi there')"), 54322, 15432)

    def test_build_run_command_uses_argv_and_secure_mounts(self):
        command = DockerRuntime().build_run_command(self.spec)
        self.assertEqual(command[:3], ["docker", "run", "--rm"])
        self.assertIn("--gpus", command)
        self.assertEqual(next(csv.reader([command[command.index("--gpus") + 1]])), ["device=0,1"])
        self.assertIn("--ipc=host", command)
        self.assertIn("--shm-size", command)
        self.assertIn("127.0.0.1:15432:54322", command)
        self.assertNotIn("host", command[command.index("--publish") + 1:])
        self.assertEqual(command[-3:], ["python", "-c", "print('hi there')"])
        mounts = [command[i + 1] for i, x in enumerate(command) if x == "--mount"]
        self.assertIn(f"type=bind,src={self.root / 'model dir'},dst=/mnt/model,readonly", mounts)

    def test_custom_runtime_options_are_applied_and_host_network_rejected(self):
        spec = DockerTaskSpec(self.IMAGE, "n", network_mode="bench-net", shm_size="2g", ipc="private")
        command = DockerRuntime().build_run_command(spec)
        self.assertEqual(command[command.index("--network") + 1], "bench-net")
        self.assertEqual(command[command.index("--shm-size") + 1], "2g")
        self.assertIn("--ipc=private", command)
        with self.assertRaises(ValueError):
            DockerRuntime().build_run_command(DockerTaskSpec(self.IMAGE, "n", network_mode="host"))

    def test_rejects_mutable_or_short_digest_images(self):
        for image in ("registry.example/s1slow:latest", "registry.example/s1slow@sha256:abc"):
            with self.assertRaises(ValueError):
                DockerRuntime().build_run_command(DockerTaskSpec(image, "n"))

    def test_rejects_invalid_destination_and_missing_source(self):
        with self.assertRaises(ValueError):
            DockerRuntime().build_run_command(DockerTaskSpec(self.IMAGE, "n", (), (Mount(self.root / "input data.jsonl", "relative", True),), {}, (), 1, None))
        with self.assertRaises(FileNotFoundError):
            DockerRuntime().build_run_command(DockerTaskSpec(self.IMAGE, "n", (), (Mount(self.root / "missing", "/x", True),), {}, (), 1, None))

    def test_run_records_lifecycle_and_redacts_inspect(self):
        runner = Mock()
        runner.side_effect = [Mock(returncode=1, stdout="", stderr="not found"), Mock(returncode=0, stdout="output\n", stderr=""), Mock(returncode=0, stdout=json.dumps({"Name": "run-123", "Config": {"Env": ["API_TOKEN=secret", "VISIBLE=yes"]}}), stderr=""), Mock(returncode=0, stdout="", stderr="")]
        log = self.root / "run.log"
        result = DockerRuntime(runner=runner).run(self.spec, log)
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.started_at and result.finished_at)
        self.assertIn("output", log.read_text())
        inspect_call = runner.call_args_list[2][0][0]
        self.assertEqual(inspect_call[:3], ["docker", "inspect", "run-123"])
        self.assertEqual(runner.call_args_list[-1][0][0][:2], ["docker", "rm"])


if __name__ == "__main__":
    unittest.main()
