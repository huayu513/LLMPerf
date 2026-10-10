"""Deterministic synthetic requests derived from scene maxima."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .artifacts import write_json_atomic
from .discovery import inspect_workload


def create_scene_workload(path: Path, scenario: dict, model_alias: str = "llmperf-scene",
                          repeat_counts: list[int] | None = None) -> dict:
    """Write a fixed workload. Lengths are targets until server usage verifies them.

    Chat templates and tokenizers differ by checkpoint, so character counts
    cannot establish exact input-token counts. The manifest deliberately calls
    these *requested* tiers and never presents them as measured token lengths.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    max_input = scenario["max_input_tokens"]
    max_output = scenario["max_output_tokens"]
    concurrency = scenario["target_concurrency"]
    count = math.ceil(max(36, concurrency * 2) / 9) * 9
    tiers = (0.25, 0.5, 1.0)
    if repeat_counts is not None and (len(repeat_counts) != 3 or
                                      any(type(n) is not int or n < 0 for n in repeat_counts)):
        raise ValueError("repeat_counts must contain three non-negative integers")
    rows = []
    layout = []
    for index in range(count):
        input_tier = (index // 3) % len(tiers)
        output_tier = index % len(tiers)
        input_target = max(1, round(max_input * tiers[input_tier]))
        output_limit = max(1, round(max_output * tiers[output_tier]))
        # ASCII repetitions are deliberately conservative. True prompt-token
        # counts are observed from the server and reported separately.
        repeat = (repeat_counts[input_tier] if repeat_counts is not None
                  else max(0, int((input_target - 48) * 0.8)))
        content = "请连续输出编号，尽量用满允许的输出长度。" + (" x" * repeat)
        rows.append({
            "api": "openai_chat_completions",
            "request_id": f"scene-{index:06d}",
            "source_message_id": f"scene-{index:06d}",
            "captured_at": "2026-01-01T00:00:00Z",
            "request": {
                "model": model_alias,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": output_limit,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        })
        layout.append({"request_id": rows[-1]["request_id"],
                       "requested_input_tier": input_target,
                       "requested_output_limit": output_limit})
    with path.open("x", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = inspect_workload(path)
    info = {"kind": "synthetic_scene", "requests": count,
            "max_input_tokens": max_input, "max_output_tokens": max_output,
            "input_length_note": "requested tiers are approximations; actual prompt tokens require server usage",
            "output_length_note": "max_tokens is an upper bound; actual completion tokens require server usage",
            "repeat_counts": repeat_counts, "tiers": layout, "sha256": manifest.sha256}
    write_json_atomic(path.with_suffix(".manifest.json"), info)
    return info
