import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmperf.advisor import LocalAdvisor, extract_and_save_scenario
from llmperf.configuration import load_config
from llmperf.errors import ConfigError
from llmperf.scene_workload import create_scene_workload
from llmperf.tuning import _result_length_evidence, run_tuning
from test_auto_planner import environment


SCENE = {"max_input_tokens": 8192, "max_output_tokens": 2048,
         "target_concurrency": 4, "priority": "latency"}


class AdvisorStub:
    def __init__(self):
        self.calls = []

    def ask(self, task, payload):
        self.calls.append((task, payload))
        if task == "scene":
            return {"scenario": SCENE}
        if task == "candidates":
            base = payload["catalog"]["base_candidates"][0]["base_id"]
            return {"candidates": [{"base_id": base,
                                    "params": {"mem_fraction_static": 0.9},
                                    "reason": "try a larger KV cache"}]}
        return {"summary": "实测时延下降。"}


class TuningTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({
            "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
            "num_attention_heads": 8, "num_hidden_layers": 8,
        }))
        (model / "tokenizer_config.json").write_text(
            '{"chat_template":"<tool_call> {{ tools }} </tool_call>"}')
        self.config = self.root / "experiment.json"
        self.config.write_text(json.dumps({
            "model_path": "model", "image": "repo/engine@sha256:" + "a" * 64,
            "output_dir": "results",
            "advisor": {"base_url": "http://127.0.0.1:9000/v1", "model": "local"},
            "search": {"max_trials": 10, "repetitions": 2},
        }))

    def test_scene_is_written_only_after_validation(self):
        advisor = AdvisorStub()
        result = extract_and_save_scenario(self.config,
            "输入 8K、输出 2K、并发 4、时延优先", advisor=advisor)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(load_config(self.config, tuning=True).scenario, SCENE)
        self.assertEqual(advisor.calls[0][0], "scene")

    def test_openai_compatible_client_sends_json_to_local_endpoint(self):
        captured = {}
        class Opener:
            def open(self, request, timeout):
                captured["url"] = request.full_url
                captured["body"] = json.loads(request.data)
                captured["timeout"] = timeout
                return io.BytesIO(json.dumps({"choices": [{"message": {"content":
                    json.dumps({"scenario": SCENE})}}]}).encode())
        with patch("llmperf.advisor.urllib.request.build_opener", return_value=Opener()):
            reply = LocalAdvisor({"base_url": "http://127.0.0.1:9000/v1",
                                  "model": "local", "timeout_seconds": 5}).ask(
                                      "scene", {"description": "输入 8K"})
        self.assertEqual(reply["scenario"], SCENE)
        self.assertEqual(captured["url"], "http://127.0.0.1:9000/v1/chat/completions")
        self.assertEqual(captured["body"]["model"], "local")
        self.assertEqual(captured["timeout"], 5)

    def test_question_does_not_change_config(self):
        class Question:
            def ask(self, *_):
                return {"question": "业务并发是多少？"}
        before = self.config.read_bytes()
        result = extract_and_save_scenario(self.config, "输入 8K", advisor=Question())
        self.assertEqual(result["status"], "NEEDS_INPUT")
        self.assertEqual(self.config.read_bytes(), before)

    def test_generated_workload_is_fixed_and_valid(self):
        path = self.root / "input.jsonl"
        info = create_scene_workload(path, SCENE)
        self.assertEqual(info["requests"], 36)
        requests = [json.loads(line)["request"] for line in path.read_text().splitlines()]
        self.assertEqual(max(row["max_tokens"] for row in requests), 2048)
        self.assertTrue(all(row["stream_options"]["include_usage"] for row in requests))

    def test_missing_prompt_usage_cannot_win_scene_tuning(self):
        request_file = self.root / "trial.requests.jsonl"
        request_file.write_text(json.dumps({"ok": True, "prompt_tokens": None,
                                            "completion_tokens": 100}) + "\n")
        result = {"status": "VALID", "reasons": [], "result_path": str(self.root)}
        _result_length_evidence(result, SCENE)
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertIn("scene_request_usage_missing", result["reasons"])

    def test_dynamic_tuning_uses_target_concurrency_and_repeats(self):
        extract_and_save_scenario(self.config, "输入 8K 输出 2K 并发 4 时延优先",
                                  advisor=AdvisorStub())
        advisor = AdvisorStub()
        tasks = []

        def execute(task, attempt):
            tasks.append(task)
            p95 = 0.4 if task.candidate_id.startswith("scene-") else 0.8
            return {"status": "VALID", "output_tokens_per_second": 20,
                    "latency_seconds": {"p95": p95},
                    "effective_static_config": {"candidate": task.candidate_id},
                    "reasons": []}

        report = run_tuning(self.config, probe=lambda _: environment(1),
                            advisor=advisor, executor=execute)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["best"]["priority"], "latency")
        self.assertTrue(report["best"]["candidate_id"].startswith("scene-"))
        self.assertTrue(all(task.concurrency == 4 for task in tasks))
        self.assertEqual(sum(task.run_class == "final_repeat" for task in tasks), 4)
        self.assertLessEqual(len(tasks), 10)
        run = Path(report["result_dir"])
        self.assertTrue((run / "catalog.json").is_file())
        self.assertTrue((run / "results-index.json").is_file())
        self.assertEqual(report["comparison"]["metric"], "latency_seconds.p95")
        task_count = len(tasks)
        resumed = run_tuning(self.config, resume_run=run,
                             probe=lambda _: environment(1), advisor=advisor, executor=execute)
        self.assertEqual(resumed["best"]["candidate_id"], report["best"]["candidate_id"])
        self.assertEqual(len(tasks), task_count)

    def test_invalid_scene_is_rejected(self):
        class Bad:
            def ask(self, *_):
                return {"scenario": {**SCENE, "target_concurrency": -3}}
        with self.assertRaises(ConfigError):
            extract_and_save_scenario(self.config, "abc", advisor=Bad())

    def test_invalid_advisor_candidates_fall_back_to_rule_search(self):
        extract_and_save_scenario(self.config, "输入 8K 输出 2K 并发 4 时延优先",
                                  advisor=AdvisorStub())
        class InvalidAdvisor:
            def ask(self, task, payload):
                if task == "candidates":
                    return {"candidates": [{"base_id": "invented", "params": {}, "reason": "bad"}]}
                return {"summary": "fallback"}
        def execute(task, attempt):
            return {"status": "VALID", "output_tokens_per_second": 10,
                    "latency_seconds": {"p95": 1.0},
                    "effective_static_config": {"candidate": task.candidate_id},
                    "reasons": []}
        report = run_tuning(self.config, probe=lambda _: environment(1),
                            advisor=InvalidAdvisor(), executor=execute)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(any(row.get("reason") == "advisor_candidate_rejected"
                            for row in report["stop_reasons"]))

    def test_unstable_top_candidate_falls_back_to_verified_baseline(self):
        extract_and_save_scenario(self.config, "输入 8K 输出 2K 并发 4 时延优先",
                                  advisor=AdvisorStub())
        def execute(task, attempt):
            tuned = task.candidate_id.startswith("scene-")
            if tuned and task.run_class == "final_repeat":
                return {"status": "FAILED", "reasons": ["startup_failed"]}
            return {"status": "VALID", "output_tokens_per_second": 20,
                    "latency_seconds": {"p95": 0.4 if tuned else 0.8},
                    "effective_static_config": {"candidate": task.candidate_id},
                    "reasons": []}
        report = run_tuning(self.config, probe=lambda _: environment(1),
                            advisor=AdvisorStub(), executor=execute)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["best"]["candidate_id"].startswith("scene-"))

    def test_final_repetitions_can_promote_baseline_over_exploration_winner(self):
        extract_and_save_scenario(self.config, "输入 8K 输出 2K 并发 4 时延优先",
                                  advisor=AdvisorStub())
        def execute(task, attempt):
            tuned = task.candidate_id.startswith("scene-")
            if task.run_class == "final_repeat":
                p95 = 0.9 if tuned else 0.7
            else:
                p95 = 0.4 if tuned else 0.8
            return {"status": "VALID", "output_tokens_per_second": 20,
                    "latency_seconds": {"p95": p95},
                    "effective_static_config": {"candidate": task.candidate_id},
                    "reasons": []}
        report = run_tuning(self.config, probe=lambda _: environment(1),
                            advisor=AdvisorStub(), executor=execute)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["best"]["candidate_id"].startswith("scene-"))
        self.assertEqual(report["best"]["metric_value"], 0.7)
