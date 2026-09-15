#!/usr/bin/env python3
"""Canonical indexer for captured OpenAI Chat Completions JSONL requests.

Each JSONL row is expected to contain capture metadata and a complete
``request`` object.  The request body is never rewritten.  The resulting
index stores byte offsets and metadata so replay workers can use ``os.pread``
without loading or copying the complete corpus.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_API = "openai_chat_completions"
INDEX_FORMAT = "s1_jsonl_chat_requests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional tokenizer path. Formal indexes should provide it.",
    )
    parser.add_argument(
        "--chat-template-kwargs",
        default="{}",
        help="JSON object passed to tokenizer.apply_chat_template().",
    )
    parser.add_argument(
        "--dsv4-encoder-path",
        default=None,
        help="Path to SGLang encoding_dsv4.py for models without an HF chat template.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-stream-usage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require stream=true and stream_options.include_usage=true.",
    )
    return parser.parse_args()


def percentile(values: list[int | float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return float(ordered[low] * (1.0 - fraction) + ordered[high] * fraction)


def distribution(values: list[int | float]) -> dict[str, int | float]:
    if not values:
        return {
            "count": 0,
            "total": 0,
            "mean": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }
    return {
        "count": len(values),
        "total": sum(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def parse_timestamp(value: Any) -> float:
    if not isinstance(value, str) or not value:
        raise ValueError("captured_at must be a non-empty ISO-8601 string")
    if not value.endswith("Z"):
        raise ValueError("captured_at must be a UTC timestamp ending in Z")
    date_part, separator, fractional = value[:-1].partition(".")
    parsed = datetime.strptime(date_part, "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    if not separator:
        return parsed.timestamp()
    if not fractional.isdigit() or len(fractional) > 9:
        raise ValueError("captured_at has an invalid fractional second")
    nanoseconds = int(fractional.ljust(9, "0"))
    return parsed.timestamp() + nanoseconds / 1_000_000_000


def canonical_request_bytes(request: dict[str, Any]) -> bytes:
    return json.dumps(
        request,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("utf-8")


def content_char_count(messages: list[Any]) -> int:
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += len(json.dumps(content, ensure_ascii=False, separators=(",", ":")))
    return total


def load_tokenizer(args: argparse.Namespace) -> Any | None:
    if not args.model_path:
        return None
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(f"failed to import transformers: {exc}") from exc
    print(f"loading tokenizer from {args.model_path}", flush=True)
    return AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )


def load_dsv4_encoder(path: str | None) -> Any | None:
    if not path:
        return None
    encoder_path = Path(path)
    if not encoder_path.is_file():
        raise RuntimeError(f"DeepSeek-V4 encoder does not exist: {encoder_path}")
    spec = importlib.util.spec_from_file_location("s1slow_encoding_dsv4", encoder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load DeepSeek-V4 encoder: {encoder_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "encode_messages", None)):
        raise RuntimeError(f"encoder has no encode_messages(): {encoder_path}")
    return module


def count_chat_tokens(
    tokenizer: Any,
    request: dict[str, Any],
    chat_template_kwargs: dict[str, Any],
    dsv4_encoder: Any | None,
) -> int:
    if dsv4_encoder is not None:
        messages = copy.deepcopy(request["messages"])
        for message in messages:
            if message.get("content") is None:
                message["content"] = ""
        if messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": ""})
        if request.get("tools"):
            messages[0]["tools"] = copy.deepcopy(request["tools"])
        thinking_requested = chat_template_kwargs.get(
            "thinking", chat_template_kwargs.get("enable_thinking", False)
        )
        thinking_mode = "thinking" if thinking_requested else "chat"
        reasoning_effort = chat_template_kwargs.get("reasoning_effort")
        if reasoning_effort not in (None, "high", "max"):
            reasoning_effort = None
        rendered = dsv4_encoder.encode_messages(
            messages,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )
        return len(tokenizer.encode(rendered))
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "tokenizer has no HF chat_template; pass --dsv4-encoder-path "
            "from the target SGLang installation"
        )
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        **chat_template_kwargs,
    }
    tools = request.get("tools")
    if tools is not None:
        kwargs["tools"] = tools
    token_ids = tokenizer.apply_chat_template(request["messages"], **kwargs)
    if hasattr(token_ids, "shape"):
        shape = token_ids.shape
        return int(shape[-1])
    return len(token_ids)


def validate_request(
    payload: Any,
    line_number: int,
    *,
    require_stream_usage: bool = False,
) -> tuple[dict[str, Any], float]:
    if not isinstance(payload, dict):
        raise ValueError(f"line {line_number}: top-level JSON must be an object")
    if payload.get("api") != EXPECTED_API:
        raise ValueError(
            f"line {line_number}: api must be {EXPECTED_API!r}, got {payload.get('api')!r}"
        )
    request_id = payload.get("request_id")
    source_message_id = payload.get("source_message_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError(f"line {line_number}: request_id must be a non-empty string")
    if not isinstance(source_message_id, (str, int)) or str(source_message_id) == "":
        raise ValueError(f"line {line_number}: source_message_id is missing")
    request = payload.get("request")
    if not isinstance(request, dict):
        raise ValueError(f"line {line_number}: request must be an object")
    if not isinstance(request.get("model"), str) or not request["model"]:
        raise ValueError(f"line {line_number}: request.model must be a non-empty string")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"line {line_number}: request.messages must be a non-empty array")
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"line {line_number}: message {message_index} must be an object"
            )
        if not isinstance(message.get("role"), str) or not message["role"]:
            raise ValueError(
                f"line {line_number}: message {message_index} has no valid role"
            )
    if require_stream_usage:
        if request.get("stream") is not True:
            raise ValueError(f"line {line_number}: request.stream must be true")
        stream_options = request.get("stream_options")
        if not isinstance(stream_options, dict) or stream_options.get("include_usage") is not True:
            raise ValueError(
                f"line {line_number}: stream_options.include_usage must be true"
            )
    return request, parse_timestamp(payload.get("captured_at"))


def main() -> int:
    args = parse_args()
    source_path = Path(args.jsonl)
    if not source_path.is_file():
        print(f"JSONL does not exist: {source_path}", file=sys.stderr)
        return 2
    try:
        chat_template_kwargs = json.loads(args.chat_template_kwargs)
        if not isinstance(chat_template_kwargs, dict):
            raise ValueError("must be a JSON object")
    except Exception as exc:
        print(f"invalid --chat-template-kwargs: {exc}", file=sys.stderr)
        return 2
    try:
        tokenizer = load_tokenizer(args)
        dsv4_encoder = load_dsv4_encoder(args.dsv4_encoder_path)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    started = time.time()
    file_digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    source_message_ids: set[str] = set()
    input_chars: list[int] = []
    input_tokens: list[int] = []
    line_lengths: list[int] = []
    models: Counter[str] = Counter()
    max_tokens_values: Counter[str] = Counter()
    temperatures: Counter[str] = Counter()
    role_sequences: Counter[str] = Counter()
    request_key_shapes: Counter[str] = Counter()
    captured_reverse_count = 0
    previous_timestamp: float | None = None

    try:
        with source_path.open("rb") as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                line_number = len(records) + 1
                file_digest.update(line)
                if not line.strip():
                    raise ValueError(f"line {line_number}: empty line")
                try:
                    payload = json.loads(line)
                except Exception as exc:
                    raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
                request, captured_at_unix = validate_request(
                    payload,
                    line_number,
                    require_stream_usage=args.require_stream_usage,
                )
                request_id = payload["request_id"]
                source_message_id = str(payload["source_message_id"])
                if request_id in request_ids:
                    raise ValueError(f"line {line_number}: duplicate request_id {request_id}")
                if source_message_id in source_message_ids:
                    raise ValueError(
                        f"line {line_number}: duplicate source_message_id {source_message_id}"
                    )
                request_ids.add(request_id)
                source_message_ids.add(source_message_id)
                if previous_timestamp is not None and captured_at_unix < previous_timestamp:
                    captured_reverse_count += 1
                previous_timestamp = captured_at_unix

                messages = request["messages"]
                char_count = content_char_count(messages)
                token_count: int | None = None
                if tokenizer is not None:
                    try:
                        token_count = count_chat_tokens(
                            tokenizer, request, chat_template_kwargs, dsv4_encoder
                        )
                    except Exception as exc:
                        raise ValueError(
                            f"line {line_number}: chat template tokenization failed: {exc}"
                        ) from exc

                request_bytes = canonical_request_bytes(request)
                record = {
                    "index": len(records),
                    "offset": offset,
                    "length": len(line),
                    "request_id": request_id,
                    "source_message_id": source_message_id,
                    "captured_at": payload["captured_at"],
                    "captured_at_unix": captured_at_unix,
                    "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                    "input_chars": char_count,
                    "input_tokens": token_count,
                }
                records.append(record)
                input_chars.append(char_count)
                line_lengths.append(len(line))
                if token_count is not None:
                    input_tokens.append(token_count)
                models[str(request.get("model"))] += 1
                max_tokens_values[json.dumps(request.get("max_tokens"))] += 1
                temperatures[json.dumps(request.get("temperature"))] += 1
                role_sequences[",".join(str(x.get("role")) for x in messages)] += 1
                request_key_shapes[",".join(sorted(request))] += 1
                if len(records) % 100 == 0:
                    print(f"validated {len(records)} records", flush=True)
    except Exception as exc:
        print(f"indexing failed: {exc}", file=sys.stderr)
        return 2

    if not records:
        print("indexing failed: JSONL has no records", file=sys.stderr)
        return 2

    captured_start = records[0]["captured_at_unix"]
    captured_end = records[-1]["captured_at_unix"]
    result = {
        "version": 1,
        "format": INDEX_FORMAT,
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "jsonl_path": str(source_path),
        "jsonl_size_bytes": source_path.stat().st_size,
        "jsonl_sha256": file_digest.hexdigest(),
        "model_path": args.model_path,
        "tokenizer_class": tokenizer.__class__.__name__ if tokenizer is not None else None,
        "dsv4_encoder_path": args.dsv4_encoder_path,
        "chat_template_kwargs": chat_template_kwargs,
        "count": len(records),
        "unique_request_ids": len(request_ids),
        "unique_source_message_ids": len(source_message_ids),
        "captured_at": {
            "first": records[0]["captured_at"],
            "last": records[-1]["captured_at"],
            "span_seconds": captured_end - captured_start,
            "reverse_count": captured_reverse_count,
        },
        "models": dict(models),
        "max_tokens": dict(max_tokens_values),
        "temperatures": dict(temperatures),
        "role_sequences": dict(role_sequences),
        "request_key_shapes": dict(request_key_shapes),
        "line_bytes": distribution(line_lengths),
        "input_chars": distribution(input_chars),
        "input_tokens": distribution(input_tokens) if tokenizer is not None else None,
        "records": records,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as target:
            json.dump(result, target, ensure_ascii=False, separators=(",", ":"))
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, output_path)
    except Exception as exc:
        print(f"failed to write index: {exc}", file=sys.stderr)
        return 2

    printable = {key: value for key, value in result.items() if key != "records"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
