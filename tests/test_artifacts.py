import json
import tempfile
import unittest
from pathlib import Path

from s1slow.Automation.automation.artifacts import sha256_file, write_json_atomic


class ArtifactTests(unittest.TestCase):
    def test_atomic_write_replaces_json_without_leaving_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'value.json'
            write_json_atomic(path, {'version': 1})
            write_json_atomic(path, {'version': 2})
            self.assertEqual(json.loads(path.read_text()), {'version': 2})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_bytes(b"hello")
            self.assertEqual(sha256_file(path), "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")


if __name__ == '__main__':
    unittest.main()
