import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmperf.advisor import LocalAdvisor, load_runtime_advisor, runtime_advisor_spec
from llmperf.evidence import redact, summarize_trial
from llmperf.errors import ConfigError


class AdvisorEvidenceTests(unittest.TestCase):
    def test_nonfinite_diagnostics_are_not_sent_as_invalid_json(self):
        self.assertEqual(redact({"ttft": float("inf"), "tpot": float("nan")}),
                         {"ttft": None, "tpot": None})

    def test_feedback_has_separate_token_budget(self):
        captured = {}
        class Opener:
            def open(self, request, timeout):
                captured.update(json.loads(request.data))
                return io.BytesIO(b'{"choices":[{"message":{"content":"{}"}}]}')
        with patch("llmperf.advisor.urllib.request.build_opener", return_value=Opener()):
            LocalAdvisor({"base_url": "http://localhost/v1", "model": "local",
                          "timeout_seconds": 5}).ask("next_experiments", {})
        self.assertEqual(captured["max_tokens"], 1024)

    def test_credentials_are_not_saved_and_only_auth_is_reloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"advisor": {"authorization": "Bearer private-token"},
                                        "search": {"max_trials": "invalid"}}))
            settings = {"base_url": "http://localhost/v1", "model": "frozen",
                        "timeout_seconds": 60, "authorization": "Bearer private-token"}
            spec = runtime_advisor_spec(settings, path)
            self.assertNotIn("private-token", json.dumps(spec))
            with patch.dict("os.environ", {"LLMPERF_ADVISOR_KEY": ""}):
                advisor = load_runtime_advisor(spec, 3)
            self.assertEqual(advisor.settings["model"], "frozen")
            self.assertEqual(advisor.settings["timeout_seconds"], 3)
            self.assertEqual(advisor.settings["authorization"], "Bearer private-token")
            with patch.dict("os.environ", {"LLMPERF_ADVISOR_KEY": "env-token"}):
                advisor = load_runtime_advisor(spec, 3)
            self.assertNotIn("authorization", advisor.settings)

    def test_missing_credentials_fail_closed(self):
        spec = {"base_url": "http://localhost/v1", "model": "local", "timeout_seconds": 5,
                "auth_required": True, "api_key_env": "MISSING_TEST_KEY"}
        with patch.dict("os.environ", {"MISSING_TEST_KEY": ""}), self.assertRaises(ConfigError):
            load_runtime_advisor(spec, 3)

    def test_redaction_and_log_size_bounds(self):
        text = 'Authorization: Bearer hidden-token api_key="another-token" password=secret-value'
        cleaned = redact({"tail": text, "authorization": "secret"})
        for secret in ("hidden-token", "another-token", "secret-value"):
            self.assertNotIn(secret, json.dumps(cleaned))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "trials" / "trial" / "attempt-001"
            attempt.mkdir(parents=True)
            (attempt / "server.log").write_text("x" * 10000 + text)
            candidate = {"id": "test", "config_hash": "hash", "static_config": {}}
            summary = summarize_trial({"status": "FAILED", "result_path": str(attempt)}, candidate, root)
            self.assertLessEqual(len(summary["log_excerpts"][0]["tail"]), 2048)
            self.assertNotIn("hidden-token", json.dumps(summary))


if __name__ == "__main__":
    unittest.main()
