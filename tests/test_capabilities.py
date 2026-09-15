import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from s1slow.Automation.automation.capabilities import (
    CommandResult,
    EnvironmentSnapshot,
    _CONTAINER_PROBE,
    check_image,
    probe_environment,
    subprocess_runner,
    write_environment,
)
from s1slow.Automation.automation.types import DockerConfig, PathConfig


SGLANG_HELP = """usage: sglang serve [-h] [--tp-size TP_SIZE]
                    [--tensor-parallel-size TP_SIZE] [--pp-size PP_SIZE]
                    [--moe-runner-backend {auto,deep_gemm,flashinfer_trtllm}]
                    [--moe-a2a-backend {none,deepep,megamoe}]
                    [--tool-call-parser {auto,qwen25,deepseekv3}]
                    [--reasoning-parser {auto,deepseek-r1,qwen3}]

options:
  -h, --help            show this help message and exit
  --tp-size TP_SIZE, --tensor-parallel-size TP_SIZE
  --pp-size PP_SIZE
  --moe-runner-backend {auto,deep_gemm,flashinfer_trtllm}
  --moe-a2a-backend {none,deepep,megamoe}
  --tool-call-parser {auto,qwen25,deepseekv3}
  --reasoning-parser {auto,deepseek-r1,qwen3}
"""


def probe_output(*, cuda_available=True, cuda_device_count=1, help_text=SGLANG_HELP,
                 kernel_packages=None):
    payload = {
        "python": "3.12.8",
        "torch": "2.8.0",
        "cuda": "13.0",
        "nccl": "2.27.3",
        "sglang": "0.5.2",
        "kernel": "6.8.0",
        "kernel_packages": (
            {"sgl_kernel": "0.3.12"} if kernel_packages is None else kernel_packages
        ),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_visible_devices": "0",
        "shm_bytes": 17179869184,
        "nvidia_devices": 5,
    }
    output = "S1SLOW_PROBE_JSON=" + json.dumps(payload, sort_keys=True) + "\n"
    if help_text is None:
        return output
    return (
        output + "S1SLOW_SGLANG_HELP_BEGIN\n"
        + help_text + "S1SLOW_SGLANG_HELP_END\n"
    )


class FakeRunner:
    def __init__(self, config, *, gpu_text=None, runtime='{"nvidia": {}}',
                 topology="GPU0\n", probe=None, compute_query_supported=True,
                 fallback_gpu_text=None):
        self.calls = []
        self.timeouts = []
        self.config = config
        self.gpu_text = gpu_text or "0, NVIDIA B300, 180000, 10.0, GPU-0"
        self.fallback_gpu_text = (
            fallback_gpu_text or "0, NVIDIA B300, 180000, GPU-0"
        )
        self.runtime = runtime
        self.topology = topology
        self.probe = probe if probe is not None else probe_output()
        self.compute_query_supported = compute_query_supported

    def __call__(self, args, **kwargs):
        call = tuple(args)
        self.calls.append(list(args))
        self.timeouts.append((call, kwargs.get("timeout")))
        fixed = {
            ("docker", "version", "--format", "{{json .}}"): (0, "{}", ""),
            ("docker", "info", "--format", "{{json .Runtimes}}"): (0, self.runtime, ""),
            ("docker", "image", "inspect", self.config.image): (
                0,
                '[{"Id":"sha256:x","RepoDigests":["repo/image@sha256:x"]}]',
                "",
            ),
            ("nvidia-smi", "--query-gpu=index,mig.mode.current", "--format=csv,noheader"): (
                0, "0, Disabled\n1, Disabled\n2, Disabled", ""
            ),
            ("nvidia-smi", "topo", "-m"): (0, self.topology, ""),
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"): (
                0, "580.1", ""
            ),
            ("printenv", "CUDA_VISIBLE_DEVICES"): (0, "0", ""),
            ("numactl", "--hardware"): (0, "available: 1 nodes", ""),
            ("ls", "-l", "/dev/nvidiactl", "/dev/nvidia-uvm"): (0, "devices", ""),
            ("df", "-Pk", str(self.config.paths.results_host)): (0, "Avail 1000", ""),
        }
        if call == (
            "nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,uuid",
            "--format=csv,noheader,nounits",
        ):
            if self.compute_query_supported:
                return CommandResult(call, 0, self.gpu_text, "")
            return CommandResult(call, 2, "", "Field compute_cap is not supported")
        if call == (
            "nvidia-smi", "--query-gpu=index,name,memory.total,uuid",
            "--format=csv,noheader,nounits",
        ):
            return CommandResult(call, 0, self.fallback_gpu_text, "")
        if call in fixed:
            return CommandResult(call, *fixed[call])
        if call[:3] == ("docker", "run", "--rm"):
            return CommandResult(call, 0, self.probe, "")
        return CommandResult(call, 127, "", "unexpected fake command")


class CapabilityTests(unittest.TestCase):
    def config(self, td, image="repo/image", *, gpu_indexes=None, docker=None):
        root = Path(td)
        (root / "model").mkdir()
        (root / "in.jsonl").write_text("{}\n")
        (root / "results").mkdir()
        paths = PathConfig(root / "model", root / "in.jsonl", root / "results")
        return SimpleNamespace(image=image, paths=paths, docker=docker or DockerConfig(),
                               gpu_indexes=gpu_indexes)

    def test_discovers_only_cli_options_and_choices_rendered_by_pinned_image(self):
        with tempfile.TemporaryDirectory() as td:
            config = self.config(td)
            snapshot = probe_environment(config, FakeRunner(config))

        capabilities = snapshot.container["capabilities"]
        self.assertEqual(capabilities["runner_backends"],
                         ["auto", "deep_gemm", "flashinfer_trtllm"])
        self.assertEqual(capabilities["a2a_backends"], ["none", "deepep", "megamoe"])
        self.assertEqual(capabilities["tool_parsers"], ["auto", "qwen25", "deepseekv3"])
        self.assertEqual(capabilities["reasoning_parsers"], ["auto", "deepseek-r1", "qwen3"])
        self.assertTrue({"--tp-size", "--tensor-parallel-size", "--pp-size"}
                        <= set(capabilities["options"]))
        self.assertNotIn("--made-up-option", capabilities["options"])

    def test_uses_documented_nvidia_compute_cap_field(self):
        with tempfile.TemporaryDirectory() as td:
            config = self.config(td)
            runner = FakeRunner(config)
            probe_environment(config, runner)

        self.assertIn([
            "nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,uuid",
            "--format=csv,noheader,nounits",
        ], runner.calls)
        self.assertFalse(any(
            "compute_capability" in argument
            for call in runner.calls for argument in call
        ))

    def test_gpu_query_fallback_preserves_uuid_and_warns_on_unknown_compute_capability(self):
        with tempfile.TemporaryDirectory() as td:
            config = self.config(td)
            runner = FakeRunner(
                config,
                compute_query_supported=False,
                fallback_gpu_text="0, NVIDIA B300, 180000, GPU-fallback",
            )
            snapshot = probe_environment(config, runner)

        self.assertEqual(snapshot.status, "WARN")
        self.assertEqual(snapshot.gpus[0].uuid, "GPU-fallback")
        self.assertIsNone(snapshot.gpus[0].compute_capability)
        self.assertTrue(any(
            check["name"] == "compute_capability" and check["status"] == "WARN"
            for check in snapshot.checks
        ))

    def test_only_container_probe_receives_extended_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            config = self.config(td)
            runner = FakeRunner(config)
            probe_environment(config, runner)

        container_timeout = next(
            timeout for call, timeout in runner.timeouts
            if call[:3] == ("docker", "run", "--rm")
        )
        self.assertEqual(container_timeout, 180)
        self.assertTrue(all(
            timeout == 30 for call, timeout in runner.timeouts
            if call[:3] != ("docker", "run", "--rm")
        ))

    def test_absent_capability_choices_are_recorded_as_empty(self):
        help_text = "usage: sglang serve [-h] [--tp-size TP_SIZE]\n\noptions:\n  --tp-size TP_SIZE\n"
        with tempfile.TemporaryDirectory() as td:
            config = self.config(td)
            snapshot = probe_environment(config, FakeRunner(config, probe=probe_output(help_text=help_text)))

        self.assertEqual(snapshot.container["capabilities"], {
            "options": ["--tp-size"],
            "runner_backends": [],
            "a2a_backends": [],
            "tool_parsers": [],
            "reasoning_parsers": [],
        })

    def test_container_probe_requires_working_sglang_serve_command(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "torch.py").write_text(
                '__version__="2.8.0"\n'
                'class version: cuda="13.0"\n'
                'class _Nccl:\n'
                ' @staticmethod\n'
                ' def version(): return (2, 27, 3)\n'
                'class cuda:\n'
                ' nccl=_Nccl()\n'
                ' @staticmethod\n'
                ' def is_available(): return True\n'
                ' @staticmethod\n'
                ' def device_count(): return 1\n'
            )
            package = root / "sglang"
            package.mkdir()
            (package / "__init__.py").write_text('__version__="0.5.2"\n')
            (package / "launch_server.py").write_text(
                'print("usage: python -m sglang.launch_server [--tp-size TP_SIZE]")\n'
            )
            bin_dir = root / "bin"
            bin_dir.mkdir()
            executable = bin_dir / "sglang"
            executable.write_text("#!/bin/sh\necho serve unavailable >&2\nexit 2\n")
            executable.chmod(0o755)
            env = dict(os.environ)
            env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
            env["PYTHONPATH"] = str(root)

            completed = subprocess.run(
                [sys.executable, "-c", _CONTAINER_PROBE],
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertNotIn("S1SLOW_SGLANG_HELP_BEGIN", completed.stdout)
        self.assertIn("sglang serve", completed.stdout)

    def test_missing_sglang_serve_help_has_actionable_snapshot_reason(self):
        with tempfile.TemporaryDirectory() as td:
            config = self.config(td)
            snapshot = probe_environment(
                config, FakeRunner(config, probe=probe_output(help_text=None))
            )

        self.assertEqual(snapshot.container["status"], "FAIL")
        self.assertIn("SGLang serve capabilities unavailable", snapshot.reasons)

    def test_false_cuda_availability_fails_container_probe(self):
        with tempfile.TemporaryDirectory() as td:
            config = self.config(td)
            snapshot = probe_environment(
                config, FakeRunner(config, probe=probe_output(cuda_available=False, cuda_device_count=0))
            )

        self.assertEqual(snapshot.status, "FAIL")
        self.assertEqual(snapshot.container["status"], "FAIL")
        self.assertIn("container CUDA runtime unavailable", snapshot.reasons)

    def test_selected_gpu_indexes_limit_only_the_container_probe(self):
        gpu_text = "\n".join(
            f"{i}, NVIDIA H100, 81559, 9.0, GPU-{i}" for i in range(3)
        )
        with tempfile.TemporaryDirectory() as td:
            config = self.config(td, gpu_indexes=(0, 2))
            runner = FakeRunner(config, gpu_text=gpu_text,
                                probe=probe_output(cuda_device_count=2))
            snapshot = probe_environment(config, runner)

        run = next(call for call in runner.calls if call[:2] == ["docker", "run"])
        self.assertEqual(run[run.index("--gpus") + 1], '"device=0,2"')
        self.assertEqual([gpu.index for gpu in snapshot.gpus], [0, 1, 2])

    def test_container_probe_does_not_publish_host_ports(self):
        docker = DockerConfig(network_mode="host", service_port=15432, shm_size="2g", ipc="private")
        with tempfile.TemporaryDirectory() as td:
            config = self.config(td, docker=docker)
            runner = FakeRunner(config)
            probe_environment(config, runner)

        run = next(call for call in runner.calls if call[:2] == ["docker", "run"])
        self.assertNotIn("--publish", run)
        self.assertNotIn("-p", run)
        self.assertEqual(run[run.index("--ipc=private")], "--ipc=private")
        self.assertEqual(run[run.index("--shm-size") + 1], "2g")
        self.assertEqual(run[run.index("--network") + 1], "host")

    def test_sgl_kernel_package_name_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            config = self.config(td)
            snapshot = probe_environment(config, FakeRunner(config))

        self.assertEqual(snapshot.status, "PASS")
        self.assertEqual(snapshot.container["versions"], {
            "python": "3.12.8", "torch": "2.8.0", "cuda": "13.0",
            "nccl": "2.27.3", "sglang": "0.5.2",
            "kernel_packages": {"sgl_kernel": "0.3.12"},
        })

    def test_kernel_package_is_not_required(self):
        with tempfile.TemporaryDirectory() as td:
            config = self.config(td)
            snapshot = probe_environment(
                config, FakeRunner(config, probe=probe_output(kernel_packages={}))
            )

        self.assertEqual(snapshot.status, "PASS")
        self.assertEqual(snapshot.container["versions"]["kernel_packages"], {})

    def test_missing_nvidia_container_runtime_fails_environment(self):
        with tempfile.TemporaryDirectory() as td:
            config = self.config(td)
            snapshot = probe_environment(config, FakeRunner(config, runtime="{}"))

        self.assertEqual(snapshot.status, "FAIL")
        self.assertIn("NVIDIA container runtime unavailable", snapshot.reasons)

    def test_full_host_gpu_fixture_and_structured_checks(self):
        gpu_text = "\n".join(
            f"{i}, NVIDIA H100, 81559, 9.0, GPU-{i}" for i in range(8)
        )
        with tempfile.TemporaryDirectory() as td:
            config = self.config(td)
            snapshot = probe_environment(config, FakeRunner(config, gpu_text=gpu_text,
                                                              probe=probe_output(cuda_device_count=8)))

        self.assertEqual([gpu.index for gpu in snapshot.gpus], list(range(8)))
        self.assertTrue({"mig", "driver", "cuda_visible_devices", "numa",
                         "nvidia_devices", "disk"} <= {check["name"] for check in snapshot.checks})

    def test_malformed_image_inspection_preserves_raw_output(self):
        fake = lambda args, **kwargs: CommandResult(tuple(args), 0, "not-json", "stderr")
        image = check_image("x", fake)
        self.assertEqual(image.status, "FAIL")
        self.assertEqual(image.to_dict()["raw"]["stdout"], "not-json")

    def test_subprocess_runner_is_shell_free(self):
        result = subprocess_runner(["/definitely/not/a/command"])
        self.assertNotEqual(result.returncode, 0)

    def test_image_raw_metadata_round_trips(self):
        def fake(args, **kwargs):
            return CommandResult(tuple(args), 0,
                                 '[{"Id":"i","Config":{"Labels":{"x":"y"}}}]', "")
        self.assertEqual(check_image("x", fake).to_dict()["raw"]["Config"]["Labels"]["x"], "y")

    def test_environment_write_is_atomic_and_json_serializable(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "environment.json"
            write_environment(path, EnvironmentSnapshot("WARN"))
            self.assertEqual(json.loads(path.read_text())["status"], "WARN")


if __name__ == "__main__":
    unittest.main()
