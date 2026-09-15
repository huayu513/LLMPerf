import tempfile
import unittest
from pathlib import Path

from s1slow.Automation.automation.docker_runtime import DockerRuntime, DockerTaskSpec, Mount


class RegressionTests(unittest.TestCase):
    def test_writable_mount_symlink_parent_to_root_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            link = root / "link"
            link.symlink_to(Path("/"), target_is_directory=True)
            escaped = link / "new"
            spec = DockerTaskSpec(
                "registry.example/image@sha256:" + "a" * 64, "safe",
                mounts=(Mount(escaped, "/out", False),),
            )
            with self.assertRaises(ValueError):
                DockerRuntime().build_run_command(spec)


if __name__ == '__main__':
    unittest.main()
