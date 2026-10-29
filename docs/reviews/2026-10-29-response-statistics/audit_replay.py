"""Review harness: synthetic responses only; does not modify LLMPerf sources.

Usage: python audit_replay.py /path/to/LLMPerf /new/output/directory
Requires aiohttp. All network requests use a temporary loopback HTTP server.
Server evidence used to exercise read_attempt is explicitly synthetic.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import web


def event(value):
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


def delta(content=None, finish=None, **extra):
    message = {**extra}
    if content is not None:
        message["content"] = content
    return {"choices": [{"index": 0, "delta": message, "finish_reason": finish}]}


def usage(completion=3, prompt=5, total=None):
    return {"choices": [], "usage": {"prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion if total is None else total}}


DONE = b"data: [DONE]\n\n"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


async def main(repo, out):
    sys.path.insert(0, str(repo))
    from llmperf.discovery import prepare_workload
    from llmperf.adapters import read_attempt
    from llmperf.types import PlanTask
    from benchmarks.server.protocol_check import deployment_report, validate_protocol

    spec = importlib.util.spec_from_file_location("audit_replay", repo / "benchmarks/replay/replay_jsonl_sglang.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    out.mkdir(parents=True, exist_ok=False)
    good = [event(delta("你")), event(delta("好")), event(delta("!")),
            event(delta(finish="stop")), event(usage()), DONE]
    cases = {
        "normal_stream": {"chunks": good},
        "fragmented_utf8_stream": {"chunks": [bytes([b]) for b in b"".join(good)], "delay": 0},
        "coalesced_stream": {"chunks": [b"".join(good)]},
        "missing_done": {"chunks": good[:-1]},
        "missing_finish_and_done": {"chunks": [event(delta("你")), event(usage())]},
        "usage_without_output": {"chunks": [event(usage()), DONE]},
        "malformed_event_with_usage": {"chunks": [b'data: {"choices": BROKEN}\n\n', event(usage()), DONE]},
        "api_error_with_usage": {"chunks": [event({"error": {"message": "generation failed"}}), event(usage()), DONE]},
        "missing_usage": {"chunks": [event(delta("你好!", finish="stop")), DONE]},
        "http_500": {"status": 500, "chunks": [b"synthetic server failure"]},
        "fractional_usage": {"chunks": [event(delta("你好!", finish="stop")), event(usage(3.9)), DONE]},
        "inconsistent_total_usage": {"chunks": [event(delta("你好!", finish="stop")), event(usage(total=999)), DONE]},
        "truncated_tool_arguments": {"chunks": [event(delta(finish="length", tool_calls=[{
            "index": 0, "type": "function", "function": {"name": "calculator", "arguments": '{"value":'}
        }])), event(usage()), DONE]},
        "normal_nonstream": {"stream": False, "body": {"choices": [{"index": 0,
            "message": {"role": "assistant", "content": "你好!"}, "finish_reason": "stop"}], "usage": usage()["usage"]}},
        "wrong_semantic_answer": {"chunks": [event(delta("8", finish="stop")), event(usage(1)), DONE]},
    }
    selected = None
    seen_payloads = []

    async def handler(request):
        seen_payloads.append(await request.json())
        case = cases[selected]
        await asyncio.sleep(0.01)
        if "body" in case:
            return web.json_response(case["body"])
        response = web.StreamResponse(status=case.get("status", 200),
                                      headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in case["chunks"]:
            if case.get("delay", 0.005):
                await asyncio.sleep(case.get("delay", 0.005))
            await response.write(chunk)
        with contextlib.suppress(ConnectionResetError):
            await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    observations = []
    batch_observations = []
    try:
        for name, case in cases.items():
            selected = name
            case_dir = out / name
            case_dir.mkdir()
            source = case_dir / "input.jsonl"
            payload = {"model": "audit-model", "messages": [{"role": "user", "content": "What is 3 + 4?"}],
                       "stream": case.get("stream", True), "max_tokens": 16}
            if payload["stream"]:
                payload["stream_options"] = {"include_usage": True}
            capture = {"api": "openai_chat_completions", "request_id": name,
                       "source_message_id": name, "captured_at": "2026-10-29T00:00:00Z", "request": payload}
            source.write_text(json.dumps(capture, ensure_ascii=False) + "\n")
            index = prepare_workload(source, case_dir / "index.json")
            write_json(case_dir / "synthetic-response.json", {
                "description": "Controlled audit fixture, not a model response",
                "chunks_utf8": [b.decode("utf-8", errors="backslashreplace") for b in case.get("chunks", [])],
                "body": case.get("body")})
            cmd = [sys.executable, str(repo / "benchmarks/replay/replay_jsonl_sglang.py"),
                   "--jsonl", str(source), "--index", str(case_dir / "index.json"),
                   "--base-url", f"http://127.0.0.1:{port}", "--output-dir", str(case_dir),
                   "--run-name", "audit", "--request-timeout", "5", "--flush-every", "1"]
            process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await process.communicate()
            (case_dir / "client.stdout.log").write_bytes(stdout)
            (case_dir / "client.stderr.log").write_bytes(stderr)
            assert seen_payloads[-1] == payload, name
            rows = [json.loads(line) for line in (case_dir / "audit.requests.jsonl").read_text().splitlines()]
            assert len(rows) == 1, name
            row = rows[0]
            summary = json.loads((case_dir / "audit.summary.json").read_text())
            assert summary["successes"] + summary["failures"] == 1
            assert summary["completion_tokens"] == (row["completion_tokens"] or 0 if row["ok"] else 0)
            assert math.isclose(summary["output_tokens_per_second"], summary["completion_tokens"] / summary["measured_seconds"])

            # Downstream validator: all unrelated server proof is a known-good synthetic fixture.
            requested = {"model_path": "/model", "max_running_requests": 1}
            for filename, data in {
                "server.requested.json": requested,
                "server.info.json": {"server_args": requested},
                "server.evidence.json": {"readiness": True, "server_info_captured": True, "resolved": True,
                    "requested_server_parameters": requested, "unsupported_parameters": [], "mismatched_parameters": []},
                "run_manifest.json": {"state": "complete", "exit_code": process.returncode},
            }.items():
                write_json(case_dir / filename, data)
            def probe_sender(base, request, timeout):
                return {"items": [{"choices": [{"delta" if request["stream"] else "message": {"content": "7"},
                                              "finish_reason": "stop"}], "usage": usage(1)["usage"]}], "completed": True}
            protocol_spec = {"version": 1, "streaming": True}
            protocol = deployment_report([{"id": "i0", **validate_protocol("http://fixture", "audit-model", protocol_spec, probe_sender)}])
            write_json(case_dir / "server.protocol.json", protocol)
            readback = read_attempt(case_dir, PlanTask("audit", 1, "candidate", run_class="formal", concurrency=1),
                {"expected_source_sha256": index["jsonl_sha256"], "expected_request_count": 1,
                 "service_port": port, "protocol_validation": protocol_spec}, exit_code=process.returncode)
            write_json(case_dir / "downstream-validation.json", readback)
            observation = {"case": name, "cli_exit_code": process.returncode,
                **{key: row[key] for key in ("ok", "error", "finish_reason", "content_chars", "reasoning_chars",
                    "completion_tokens", "total_tokens", "ttft_seconds", "tpot_seconds", "latency_seconds")},
                "downstream_status": readback["status"], "downstream_reasons": readback["reasons"]}
            observations.append(observation)
            print(json.dumps(observation, ensure_ascii=False), flush=True)
        # Exercise warmup exclusion, parallel dispatch, request identity, and both load modes.
        selected = "normal_stream"
        for mode in ("closed-loop", "open-loop"):
            case_dir = out / ("batch-" + mode)
            case_dir.mkdir()
            source = case_dir / "input.jsonl"
            captures = [{"api": "openai_chat_completions", "request_id": f"batch-{i}",
                "source_message_id": f"batch-{i}", "captured_at": f"2026-10-29T00:00:00.{i * 20:03d}Z",
                "request": {"model": "audit-model", "messages": [{"role": "user", "content": str(i)}],
                            "stream": True, "stream_options": {"include_usage": True}}} for i in range(5)]
            source.write_text("".join(json.dumps(row) + "\n" for row in captures))
            prepare_workload(source, case_dir / "index.json")
            before = len(seen_payloads)
            cmd = [sys.executable, str(repo / "benchmarks/replay/replay_jsonl_sglang.py"),
                   "--jsonl", str(source), "--index", str(case_dir / "index.json"),
                   "--base-url", f"http://127.0.0.1:{port}", "--output-dir", str(case_dir), "--run-name", "audit",
                   "--mode", mode, "--concurrency", "3", "--warmup", "2", "--max-in-flight", "2",
                   "--arrival-rate-scale", "2", "--request-timeout", "5"]
            process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await process.communicate()
            (case_dir / "client.stdout.log").write_bytes(stdout)
            (case_dir / "client.stderr.log").write_bytes(stderr)
            assert process.returncode == 0, stderr
            rows = [json.loads(line) for line in (case_dir / "audit.requests.jsonl").read_text().splitlines()]
            summary = json.loads((case_dir / "audit.summary.json").read_text())
            assert len(seen_payloads) - before == 7
            assert len(rows) == 5 and {row["request_id"] for row in rows} == {row["request_id"] for row in captures}
            assert summary["request_count"] == summary["successes"] == 5
            assert summary["warmup_successes"] == 2 and summary["warmup_failures"] == summary["failures"] == 0
            assert summary["completion_tokens"] == 15 and summary["prompt_tokens"] == 25 and summary["total_tokens"] == 40
            assert math.isclose(summary["output_tokens_per_second"], 15 / summary["measured_seconds"])
            assert summary["endpoint_summaries"][0]["completion_tokens"] == 15
            item = {"mode": mode, "passed": True, "requests_received": 7, "measured_requests": 5,
                    "warmup_requests": 2, "completion_tokens": 15, "output_tokens_per_second": summary["output_tokens_per_second"]}
            batch_observations.append(item)
            print(json.dumps(item), flush=True)
    finally:
        await runner.cleanup()

    # Verify the metric formulas against fixed, known clock values.
    normal_dir = out / "normal_stream"
    record = json.loads((normal_dir / "index.json").read_text())["records"][0]
    class Content:
        async def iter_any(self):
            for chunk in [event(delta(role="assistant")), event(delta(reasoning_content="算")),
                          event(delta("你")), event(delta("好")), event(delta(finish="stop")), event(usage()), DONE]:
                yield chunk
    class Response:
        status = 200
        headers = {"Content-Type": "text/event-stream"}
        content = Content()
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
    class Session:
        def post(self, *args, **kwargs): return Response()
    ticks = iter([100, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.8])
    fd = os.open(normal_dir / "input.jsonl", os.O_RDONLY)
    try:
        with patch.object(replay, "time", SimpleNamespace(perf_counter=lambda: next(ticks), time=lambda: 1_000_000)):
            timed = await replay.request_one(Session(), fd, record, url="http://fixture", timeout=1,
                headers={}, scheduled_at=99.9, worker_id=0, endpoint_id="i0", endpoint_url="http://fixture")
    finally:
        os.close(fd)
    expected = {"latency_seconds": 0.8, "ttft_seconds": 0.2, "tpot_seconds": 0.1,
                "dispatch_lag_seconds": 0.1, "content_chars": 2, "reasoning_chars": 1, "completion_tokens": 3}
    for field, value in expected.items():
        assert math.isclose(timed[field], value, abs_tol=1e-9), (field, timed[field], value)
    write_json(out / "deterministic-metric-check.json", {"expected": expected, "actual": timed, "passed": True})

    # A well-formed but wrong answer passes the startup protocol probe too.
    def wrong_sender(base, request, timeout):
        return {"items": [{"choices": [{"message": {"content": "8"}, "finish_reason": "stop"}],
                           "usage": usage(1)["usage"]}], "completed": True}
    wrong_probe = validate_protocol("http://fixture", "audit-model", {}, wrong_sender)
    write_json(out / "wrong-answer-protocol-check.json", wrong_probe)
    assert wrong_probe["status"] == "PASS"
    source_digest = replay.sha256_file(normal_dir / "input.jsonl")
    try:
        replay.verify_source(normal_dir / "input.jsonl", {"jsonl_sha256": "0" * 64})
    except ValueError:
        source_mismatch_rejected = True
    else:
        source_mismatch_rejected = False
    assert source_mismatch_rejected
    write_json(out / "audit-results.json", {
        "repository": str(repo), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "python": sys.version, "aiohttp": __import__("aiohttp").__version__,
        "scope": "Synthetic local HTTP responses; no live model was tested. Downstream server evidence is a synthetic fixture.",
        "source_hash_mismatch_rejected": source_mismatch_rejected, "source_sha256": source_digest,
        "normal_metric_formula_check_passed": True, "wrong_answer_probe_status": wrong_probe["status"],
        "cases": observations, "batch_checks": batch_observations})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    asyncio.run(main(args.repository.resolve(), args.output.resolve()))
