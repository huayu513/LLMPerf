#!/usr/bin/env python3
"""Calibrate synthetic scene prompts with the target checkpoint tokenizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFIX = "请连续输出编号，尽量用满允许的输出长度。"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    spec = json.loads(Path(args.input).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False, use_fast=True)
    kwargs = spec.get("chat_template_kwargs") or {}
    has_template = bool(getattr(tokenizer, "chat_template", None))

    def count(repetitions: int) -> int:
        content = PREFIX + " x" * repetitions
        if has_template:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=True,
                add_generation_prompt=True, **kwargs)
            return len(encoded)
        # Some supported checkpoints use an engine-specific chat formatter.
        # Their content tokens can still be measured, with a conservative
        # allowance for the formatter. Server usage remains the final check.
        return len(tokenizer.encode(content)) + 128

    maximum = int(spec["max_input_tokens"])
    tiers = [max(1, round(maximum * fraction)) for fraction in (0.25, 0.5, 1.0)]
    repeats, actual = [], []
    for target in tiers:
        # Retain a small margin for engine-side prompt adjustments.
        ceiling = max(1, target - max(8, target // 100))
        if count(0) > ceiling:
            raise RuntimeError(f"chat template overhead exceeds requested input tier {target}")
        low, high = 0, max(1, target * 2)
        while low < high:
            middle = (low + high + 1) // 2
            if count(middle) <= ceiling:
                low = middle
            else:
                high = middle - 1
        repeats.append(low)
        actual.append(count(low))
    Path(args.output).write_text(json.dumps({
        "status": "PASS", "repeat_counts": repeats,
        "tokenizer_input_tokens": actual, "requested_input_tiers": tiers,
        "method": ("target checkpoint tokenizer.apply_chat_template" if has_template
                   else "target checkpoint tokenizer.encode plus 128-token template allowance"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
