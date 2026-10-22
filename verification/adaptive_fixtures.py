import json

from llmperf.capabilities import EnvironmentSnapshot, GPUInfo
from llmperf.scene_workload import create_scene_workload


SCENE = {"max_input_tokens": 8192, "max_output_tokens": 2048,
         "target_concurrency": 4, "priority": "latency"}


def environment():
    return EnvironmentSnapshot("PASS", gpus=tuple(GPUInfo(index, "H100", 80000, "9.0")
                                                 for index in range(2)),
        container={"capabilities": {"options": [
            "--tp-size", "--dp", "--pp-size", "--trust-remote-code", "--model-path",
            "--served-model-name", "--enable-metrics", "--enable-cache-report",
            "--mem-fraction-static", "--max-running-requests", "--host", "--port",
            "--chunked-prefill-size", "--default-chat-template-kwargs", "--tool-call-parser",
            "--reasoning-parser"], "tool_parsers": ["qwen", "qwen25"], "reasoning_parsers": ["qwen3"]}})


def write_config(root, **adaptive):
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"], "num_attention_heads": 8, "num_hidden_layers": 8}))
    (model / "tokenizer_config.json").write_text(
        '{"chat_template":"<tool_call> {{ tools }} </tool_call>"}')
    path = root / "experiment.json"
    create_scene_workload(root / "input.jsonl", SCENE)
    path.write_text(json.dumps({
        "model_path": "model", "image": "repo/engine@sha256:" + "a" * 64,
        "output_dir": "results", "scenario": SCENE, "input_path": "input.jsonl",
        "advisor": {"base_url": "http://127.0.0.1:9000/v1", "model": "local",
                    "authorization": "Bearer test-token"},
        "search": {"strategy": "adaptive", "max_trials": 14, "repetitions": 1,
                   "adaptive": {"candidate_pool_limit": 32, "max_advisor_calls": 3, **adaptive}},
    }))
    return path


class EmptyPlanningAdvisor:
    def ask(self, task, payload):
        return {"candidates": []}
