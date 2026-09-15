#!/usr/bin/env python3
"""Canonical full-request Chat Completions replay client for SGLang.

The JSON object stored under each JSONL row's ``request`` key is sent without
field additions, removals, or overrides.  The runner supports both closed-loop
fixed concurrency and open-loop replay based on captured timestamps.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import aiohttp


INDEX_FORMAT = "s1_jsonl_chat_requests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:25080")
    parser.add_argument(
        "--base-urls",
        default="",
        help="Comma-separated endpoint base URLs for aggregate multi-instance replay.",
    )
    parser.add_argument(
        "--endpoint-ids",
        default="",
        help="Optional comma-separated stable endpoint IDs matching --base-urls.",
    )
    parser.add_argument(
        "--mode",
        choices=("closed-loop", "open-loop"),
        default="closed-loop",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Worker count for closed-loop mode.",
    )
    parser.add_argument(
        "--arrival-rate-scale",
        type=float,
        default=1.0,
        help="Open-loop arrival multiplier: 2 means twice the captured rate.",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help="Optional open-loop in-flight cap; 0 means no client-side cap.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Exact captured requests to send before measurement.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument(
        "--verify-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify JSONL size and SHA-256 against the index before replay.",
    )
    return parser.parse_args()


def percentile(values: list[float], p: float) -> float | None:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return None
    rank = (len(ordered) - 1) * p / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def distribution(values: list[int | float]) -> dict[str, float | int | None]:
    numeric = [float(value) for value in values if math.isfinite(float(value))]
    if not numeric:
        return {
            "count": 0,
            "total": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(numeric),
        "total": sum(numeric),
        "mean": statistics.fmean(numeric),
        "p50": percentile(numeric, 50),
        "p90": percentile(numeric, 90),
        "p95": percentile(numeric, 95),
        "p99": percentile(numeric, 99),
        "max": max(numeric),
    }


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def build_endpoints(args: argparse.Namespace) -> list[dict[str, str]]:
    urls = parse_csv(args.base_urls) or [args.base_url]
    ids = parse_csv(args.endpoint_ids)
    if ids and len(ids) != len(urls):
        raise ValueError("--endpoint-ids must have the same item count as --base-urls")
    endpoints = []
    for index, url in enumerate(urls):
        endpoint_id = ids[index] if ids else f"endpoint-{index}"
        endpoints.append({
            "id": endpoint_id,
            "base_url": url.rstrip("/"),
            "url": url.rstrip("/") + "/v1/chat/completions",
            "ordinal": index,
        })
    if len({endpoint["id"] for endpoint in endpoints}) != len(endpoints):
        raise ValueError("endpoint IDs must be unique")
    return endpoints


def load_index(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        data = json.load(source)
    if data.get("format") != INDEX_FORMAT:
        raise ValueError(
            f"unsupported index format {data.get('format')!r}; expected {INDEX_FORMAT!r}"
        )
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"index has no records: {path}")
    return data, records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(path: Path, index_meta: dict[str, Any]) -> None:
    expected_size = index_meta.get("jsonl_size_bytes")
    if isinstance(expected_size, int) and path.stat().st_size != expected_size:
        raise ValueError(
            f"JSONL size changed: expected {expected_size}, got {path.stat().st_size}"
        )
    expected_digest = index_meta.get("jsonl_sha256")
    if isinstance(expected_digest, str) and sha256_file(path) != expected_digest:
        raise ValueError("JSONL SHA-256 does not match the replay index")


def canonical_request_bytes(request: dict[str, Any]) -> bytes:
    return json.dumps(
        request,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("utf-8")


def read_request(fd: int, record: dict[str, Any]) -> dict[str, Any]:
    line = os.pread(fd, int(record["length"]), int(record["offset"]))
    try:
        payload = json.loads(line)
        request = payload["request"]
        if not isinstance(request, dict):
            raise TypeError("request is not an object")
        if payload.get("request_id") != record.get("request_id"):
            raise ValueError("request_id does not match index")
        if str(payload.get("source_message_id")) != record.get("source_message_id"):
            raise ValueError("source_message_id does not match index")
        digest = hashlib.sha256(canonical_request_bytes(request)).hexdigest()
        if digest != record.get("request_sha256"):
            raise ValueError("request SHA-256 does not match index")
    except Exception as exc:
        raise ValueError(
            f"cannot read request index={record.get('index')} "
            f"request_id={record.get('request_id')}: {exc}"
        ) from exc
    return request


def parse_sse_event(event: bytes) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for raw_line in event.splitlines():
        if not raw_line.startswith(b"data:"):
            continue
        data = raw_line[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            value = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed.append(value)
    return parsed


def sse_event_is_done(event: bytes) -> bool:
    """Return whether an SSE event contains the OpenAI stream terminator."""
    return any(
        raw_line.startswith(b"data:")
        and raw_line[5:].strip() == b"[DONE]"
        for raw_line in event.splitlines()
    )


def split_sse_events(buffer: bytes) -> tuple[list[bytes], bytes]:
    normalized = buffer.replace(b"\r\n", b"\n")
    events: list[bytes] = []
    while b"\n\n" in normalized:
        event, normalized = normalized.split(b"\n\n", 1)
        events.append(event)
    return events, normalized


def numeric_token(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def observe_response_item(
    item: dict[str, Any],
    now: float,
    state: dict[str, Any],
) -> None:
    if isinstance(item.get("usage"), dict):
        state["usage"] = {**(state.get("usage") or {}), **item["usage"]}
    choices = item.get("choices") or []
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        if choice.get("finish_reason") is not None:
            state["finish_reason"] = choice["finish_reason"]
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            delta = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        has_activity = False
        reasoning = delta.get("reasoning_content")
        content = delta.get("content")
        tool_calls = delta.get("tool_calls")
        if isinstance(reasoning, str) and reasoning:
            state["reasoning_chars"] += len(reasoning)
            has_activity = True
        if isinstance(content, str) and content:
            state["content_chars"] += len(content)
            has_activity = True
        if tool_calls:
            state["tool_call_chunks"] += 1
            state["tool_call_chars"] += len(
                json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":"))
            )
            has_activity = True
        if has_activity:
            if state["first_activity"] is None:
                state["first_activity"] = now
            state["last_activity"] = now


async def request_one(
    session: aiohttp.ClientSession,
    fd: int,
    record: dict[str, Any],
    *,
    url: str,
    timeout: float,
    headers: dict[str, str],
    scheduled_at: float,
    worker_id: int | None,
    endpoint_id: str,
    endpoint_url: str,
) -> dict[str, Any]:
    payload = read_request(fd, record)
    started = time.perf_counter()
    started_unix = time.time()
    state: dict[str, Any] = {
        "usage": None,
        "finish_reason": None,
        "first_activity": None,
        "last_activity": None,
        "content_chars": 0,
        "reasoning_chars": 0,
        "tool_call_chunks": 0,
        "tool_call_chars": 0,
    }
    status = 0
    error: str | None = None
    response_content_type: str | None = None
    try:
        request_timeout = aiohttp.ClientTimeout(total=timeout)
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=request_timeout,
        ) as response:
            status = response.status
            response_content_type = response.headers.get("Content-Type")
            if response.status != 200:
                error_body = await response.text(errors="replace")
                error = f"HTTP {response.status}: {error_body[:1000]}"
            elif payload.get("stream") is True:
                buffer = b""
                stream_done = False
                async for chunk in response.content.iter_any():
                    buffer += chunk
                    events, buffer = split_sse_events(buffer)
                    for event in events:
                        if sse_event_is_done(event):
                            stream_done = True
                            break
                        now = time.perf_counter()
                        for item in parse_sse_event(event):
                            observe_response_item(item, now, state)
                    if stream_done:
                        break
                if not stream_done and buffer.strip():
                    now = time.perf_counter()
                    for item in parse_sse_event(buffer):
                        observe_response_item(item, now, state)
            else:
                item = await response.json(content_type=None)
                if not isinstance(item, dict):
                    raise ValueError("non-stream response is not a JSON object")
                observe_response_item(item, time.perf_counter(), state)
    except Exception as exc:  # noqa: BLE001 - retain per-request failure
        error = f"{type(exc).__name__}: {exc}"

    ended = time.perf_counter()
    usage = state["usage"]
    prompt_tokens = numeric_token(usage.get("prompt_tokens")) if usage else None
    completion_tokens = numeric_token(usage.get("completion_tokens")) if usage else None
    total_tokens = numeric_token(usage.get("total_tokens")) if usage else None
    first_activity = state["first_activity"]
    last_activity = state["last_activity"]
    return {
        "index": int(record["index"]),
        "request_id": record.get("request_id"),
        "source_message_id": record.get("source_message_id"),
        "captured_at": record.get("captured_at"),
        "worker_id": worker_id,
        "endpoint_id": endpoint_id,
        "endpoint_url": endpoint_url,
        "request_model": payload.get("model"),
        "request_max_tokens": payload.get("max_tokens"),
        "request_sha256": record.get("request_sha256"),
        "scheduled_time_unix": started_unix - (started - scheduled_at),
        "dispatch_lag_seconds": max(started - scheduled_at, 0.0),
        "start_time_unix": started_unix,
        "status": status,
        "ok": error is None and status == 200,
        "usage_available": completion_tokens is not None,
        "error": error,
        "response_content_type": response_content_type,
        "finish_reason": state["finish_reason"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": ended - started,
        "ttft_seconds": first_activity - started if first_activity is not None else None,
        "last_activity_seconds": (
            last_activity - started if last_activity is not None else None
        ),
        "tpot_seconds": (
            (last_activity - first_activity) / max(completion_tokens - 1, 1)
            if first_activity is not None
            and last_activity is not None
            and completion_tokens is not None
            and completion_tokens > 0
            else None
        ),
        "content_chars": state["content_chars"],
        "reasoning_chars": state["reasoning_chars"],
        "tool_call_chunks": state["tool_call_chunks"],
        "tool_call_chars": state["tool_call_chars"],
        "usage": usage,
    }


async def result_writer(
    queue: asyncio.Queue[dict[str, Any] | None],
    path: Path,
    flush_every: int,
) -> None:
    written = 0
    with path.open("x", encoding="utf-8") as target:
        while True:
            row = await queue.get()
            if row is None:
                break
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            if written % max(flush_every, 1) == 0:
                target.flush()
        target.flush()
        os.fsync(target.fileno())


def summarize_result_rows(
    results: list[dict[str, Any]],
    measured_seconds: float,
) -> dict[str, Any]:
    successful = [row for row in results if row["ok"]]
    failed = [row for row in results if not row["ok"]]
    with_usage = [row for row in successful if row["completion_tokens"] is not None]
    prompt_tokens = [
        int(row["prompt_tokens"])
        for row in successful
        if row["prompt_tokens"] is not None
    ]
    completion_tokens = [int(row["completion_tokens"]) for row in with_usage]
    total_tokens = [
        int(row["total_tokens"])
        for row in successful
        if row["total_tokens"] is not None
    ]
    latencies = [float(row["latency_seconds"]) for row in successful]
    ttfts = [
        float(row["ttft_seconds"])
        for row in successful
        if row["ttft_seconds"] is not None
    ]
    tpots = [
        float(row["tpot_seconds"])
        for row in successful
        if row["tpot_seconds"] is not None
    ]
    dispatch_lags = [float(row["dispatch_lag_seconds"]) for row in results]
    safe_seconds = max(measured_seconds, 1e-9)
    input_total = sum(prompt_tokens)
    output_total = sum(completion_tokens)
    total_total = sum(total_tokens)
    return {
        "successes": len(successful),
        "failures": len(failed),
        "error_rate": len(failed) / len(results) if results else 1.0,
        "server_usage_available": len(with_usage),
        "server_usage_missing": len(successful) - len(with_usage),
        "prompt_tokens": input_total,
        "completion_tokens": output_total,
        "total_tokens": total_total,
        "input_tokens_per_second": input_total / safe_seconds,
        "output_tokens_per_second": output_total / safe_seconds,
        "total_tokens_per_second": total_total / safe_seconds,
        "successful_requests_per_second": len(successful) / safe_seconds,
        "completion_tokens_per_request": distribution(completion_tokens),
        "latency_seconds": distribution(latencies),
        "ttft_seconds": distribution(ttfts),
        "tpot_seconds": distribution(tpots),
        "dispatch_lag_seconds": distribution(dispatch_lags),
        "finish_reasons": dict(
            Counter(str(row.get("finish_reason")) for row in successful)
        ),
        "http_statuses": dict(Counter(str(row.get("status")) for row in results)),
        "failed_request_ids": [row.get("request_id") for row in failed],
        "missing_usage_request_ids": [
            row.get("request_id")
            for row in successful
            if row["completion_tokens"] is None
        ],
    }


def summarize(
    args: argparse.Namespace,
    index_meta: dict[str, Any],
    records: list[dict[str, Any]],
    results: list[dict[str, Any]],
    measured_seconds: float,
    warmup_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    warmup_results = warmup_results or []
    warmup_successes = sum(1 for row in warmup_results if row.get("ok"))
    warmup_failures = len(warmup_results) - warmup_successes
    endpoints = build_endpoints(args)
    metrics = summarize_result_rows(results, measured_seconds)
    endpoint_summaries = []
    for endpoint in endpoints:
        endpoint_rows = [row for row in results if row.get("endpoint_id") == endpoint["id"]]
        endpoint_summary = {
            "endpoint_id": endpoint["id"],
            "base_url": endpoint["base_url"],
            "request_count": len(endpoint_rows),
            **summarize_result_rows(endpoint_rows, measured_seconds),
        }
        endpoint_summaries.append(endpoint_summary)
    return {
        "run_name": args.run_name,
        "created_at_unix": time.time(),
        "mode": args.mode,
        "base_url": endpoints[0]["base_url"],
        "base_urls": [endpoint["base_url"] for endpoint in endpoints],
        "endpoints": [
            {"id": endpoint["id"], "base_url": endpoint["base_url"], "ordinal": endpoint["ordinal"]}
            for endpoint in endpoints
        ],
        "jsonl": args.jsonl,
        "index": args.index,
        "jsonl_sha256": index_meta.get("jsonl_sha256"),
        "index_count": index_meta.get("count"),
        "request_count": len(records),
        "concurrency": args.concurrency if args.mode == "closed-loop" else None,
        "arrival_rate_scale": (
            args.arrival_rate_scale if args.mode == "open-loop" else None
        ),
        "max_in_flight": args.max_in_flight if args.mode == "open-loop" else None,
        "warmup": min(args.warmup, len(records)),
        "warmup_successes": warmup_successes,
        "warmup_failures": warmup_failures,
        "measured_seconds": measured_seconds,
        **metrics,
        "endpoint_summaries": endpoint_summaries,
    }


async def run_replay(
    args: argparse.Namespace,
    result_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    jsonl_path = Path(args.jsonl)
    index_meta, all_records = load_index(Path(args.index))
    if args.verify_source:
        verify_source(jsonl_path, index_meta)
    records = all_records[: args.limit] if args.limit else all_records
    if not records:
        raise ValueError("no records selected")
    if args.warmup < 0:
        raise ValueError("warmup cannot be negative")
    if args.mode == "closed-loop" and args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if args.mode == "open-loop" and args.arrival_rate_scale <= 0:
        raise ValueError("arrival-rate-scale must be positive")
    if args.max_in_flight < 0:
        raise ValueError("max-in-flight cannot be negative")

    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
    api_key = os.environ.get("SGLANG_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    endpoints = build_endpoints(args)
    in_flight = [0 for _ in endpoints]
    connector_limit = (
        max(args.concurrency * 2, 32)
        if args.mode == "closed-loop"
        else (args.max_in_flight if args.max_in_flight > 0 else 0)
    )
    connector = aiohttp.TCPConnector(limit=connector_limit, limit_per_host=0)
    fd = os.open(jsonl_path, os.O_RDONLY)
    results: list[dict[str, Any]] = []
    warmup_results: list[dict[str, Any]] = []
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    writer_task = asyncio.create_task(
        result_writer(queue, result_path, args.flush_every)
    )

    async def record_result(result: dict[str, Any]) -> None:
        results.append(result)
        await queue.put(result)
        completed = len(results)
        if completed % max(args.progress_every, 1) == 0 or completed == len(records):
            print(f"completed {completed}/{len(records)}", flush=True)

    def select_endpoint_index() -> int:
        return min(range(len(endpoints)), key=lambda index: (in_flight[index], index))

    async def request_selected(
        session: aiohttp.ClientSession,
        record: dict[str, Any],
        scheduled_at: float,
        worker_id: int | None,
    ) -> dict[str, Any]:
        endpoint_index = select_endpoint_index()
        endpoint = endpoints[endpoint_index]
        in_flight[endpoint_index] += 1
        try:
            return await request_one(
                session,
                fd,
                record,
                url=endpoint["url"],
                timeout=args.request_timeout,
                headers=headers,
                scheduled_at=scheduled_at,
                worker_id=worker_id,
                endpoint_id=endpoint["id"],
                endpoint_url=endpoint["base_url"],
            )
        finally:
            in_flight[endpoint_index] -= 1

    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            for warmup_index in range(min(args.warmup, len(records))):
                warmup_result = await request_selected(
                    session,
                    records[warmup_index],
                    time.perf_counter(),
                    -1,
                )
                warmup_results.append(warmup_result)
                if not warmup_result["ok"]:
                    raise RuntimeError(
                        f"warmup request failed: request_id={warmup_result['request_id']} "
                        f"error={warmup_result['error']}"
                    )

            measured_start = time.perf_counter()
            if args.mode == "closed-loop":
                cursor = 0

                async def worker(worker_id: int) -> None:
                    nonlocal cursor
                    while cursor < len(records):
                        record = records[cursor]
                        cursor += 1
                        result = await request_selected(
                            session,
                            record,
                            time.perf_counter(),
                            worker_id,
                        )
                        await record_result(result)

                await asyncio.gather(*(worker(i) for i in range(args.concurrency)))
            else:
                first_capture = float(records[0]["captured_at_unix"])
                semaphore = (
                    asyncio.Semaphore(args.max_in_flight)
                    if args.max_in_flight > 0
                    else None
                )

                async def scheduled_request(record: dict[str, Any]) -> None:
                    relative = (
                        float(record["captured_at_unix"]) - first_capture
                    ) / args.arrival_rate_scale
                    scheduled = measured_start + max(relative, 0.0)
                    delay = scheduled - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    if semaphore is None:
                        result = await request_selected(
                            session,
                            record,
                            scheduled,
                            None,
                        )
                    else:
                        async with semaphore:
                            result = await request_selected(
                                session,
                                record,
                                scheduled,
                                None,
                            )
                    await record_result(result)

                await asyncio.gather(*(scheduled_request(record) for record in records))
            measured_end = time.perf_counter()
    finally:
        os.close(fd)
        await queue.put(None)
        await writer_task

    measured_seconds = max(measured_end - measured_start, 1e-9)
    summary = summarize(args, index_meta, records, results, measured_seconds, warmup_results)
    return results, summary


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.run_name}.requests.jsonl"
    summary_path = output_dir / f"{args.run_name}.summary.json"
    if result_path.exists() or summary_path.exists():
        print(
            f"refusing to overwrite existing result for run {args.run_name!r}",
            file=sys.stderr,
        )
        return 2
    try:
        _, summary = asyncio.run(run_replay(args, result_path))
    except Exception as exc:  # pragma: no cover - CLI diagnostic
        print(f"replay failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    with summary_path.open("x", encoding="utf-8") as target:
        json.dump(summary, target, ensure_ascii=False, indent=2)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")
    return 0 if summary["failures"] == 0 and summary["server_usage_missing"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
