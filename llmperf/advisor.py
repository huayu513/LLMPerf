"""Small, bounded OpenAI-compatible advisor for an isolated deployment."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .artifacts import sha256_file, write_json_atomic
from .configuration import load_config, validate_scenario
from .errors import ConfigError


def _json_object(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
    if not isinstance(content, str):
        raise ValueError("advisor returned non-text content")
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("advisor response must be a JSON object")
    return value


class LocalAdvisor:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings

    def ask(self, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        base = self.settings["base_url"]
        url = base if base.endswith("/chat/completions") else base + (
            "/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        instructions = {
            "scene": (
                "Extract only facts explicitly supplied by the Chinese user. Return JSON only. "
                "If any of max_input_tokens, max_output_tokens, target_concurrency, priority "
                "is missing or ambiguous, return {\"question\":\"one concise Chinese clarification\"}. "
                "Otherwise return {\"scenario\":{\"max_input_tokens\":integer,"
                "\"max_output_tokens\":integer,\"target_concurrency\":integer,"
                "\"priority\":\"latency or throughput\"}}. K means 1024. "
                "Do not invent numeric limits or a latency SLO."
            ),
            "candidates": (
                "You are a SGLang configuration advisor. Return JSON only: "
                "{\"candidates\":[{\"base_id\":\"catalog id\",\"params\":{...},"
                "\"reason\":\"short Chinese reason\"}]}. Select only base_id values and "
                "parameter values allowed by the supplied catalog. Return exactly one candidate "
                "with a reason of at most 8 Chinese characters. Prefer a new experiment. "
                "Do not issue commands, alter the workload, or claim a winner."
            ),
            "explain": (
                "Explain the supplied measured comparison in concise Chinese. "
                "Return JSON only: {\"summary\":\"...\"}. Do not change the computed winner."
            ),
        }
        if task not in instructions:
            raise ValueError(f"unknown advisor task: {task}")
        body = json.dumps({
            "model": self.settings["model"],
            "messages": [
                {"role": "user", "content": instructions[task] + "\n\n" +
                 json.dumps(payload, ensure_ascii=False)},
            ],
            "reasoning_effort": "none",
            "chat_template_kwargs": {"thinking": False},
            "max_tokens": 64,
            "temperature": 0,
            "stream": False,
        }, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.settings.get("model_pool"):
            headers["X-Model-Pool"] = self.settings["model_pool"]
        authorization = self.settings.get("authorization")
        if authorization:
            headers["Authorization"] = authorization
        else:
            key_env = self.settings.get("api_key_env")
            if key_env:
                key = os.environ.get(key_env)
                if not key:
                    raise ConfigError(f"advisor API key environment variable is missing: {key_env}")
                headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(url, body, headers=headers, method="POST")
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=float(self.settings["timeout_seconds"])) as response:
                document = json.load(response)
            return _json_object(document["choices"][0]["message"]["content"])
        except (OSError, urllib.error.URLError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise ConfigError(f"local advisor {task} call failed: {exc}") from exc


def extract_and_save_scenario(config_path: str | Path, description: str,
                              advisor: LocalAdvisor | None = None) -> dict[str, Any]:
    if not isinstance(description, str) or not description.strip():
        raise ConfigError("scene description must be non-empty")
    path = Path(config_path).expanduser().resolve()
    config = load_config(path, tuning=True)
    if config.advisor is None:
        raise ConfigError("advisor settings are required to parse a natural-language scene")
    before = sha256_file(path)
    reply = (advisor or LocalAdvisor(config.advisor)).ask("scene", {"description": description.strip()})
    if "question" in reply:
        question = reply["question"]
        if not isinstance(question, str) or not question.strip():
            raise ConfigError("advisor clarification question is invalid")
        return {"status": "NEEDS_INPUT", "question": question.strip()}
    scene = validate_scenario(reply.get("scenario"))
    if sha256_file(path) != before:
        raise ConfigError("experiment.json changed while parsing the scene; retry")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["scenario"] = scene
    write_json_atomic(path, document)
    return {"status": "PASS", "scenario": scene, "config_path": str(path)}
