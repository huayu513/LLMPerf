"""Small, non-scored OpenAI protocol checks against a freshly started server."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_RAW_MARKERS = ("<｜DSML｜", "<tool_call>", "<function=", "<think>", "</think>")


def exchange(base_url, payload, timeout):
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    deadline = time.monotonic() + timeout
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not payload.get("stream"):
            body = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(body) > _MAX_RESPONSE_BYTES:
                raise ValueError("protocol response exceeds size limit")
            return {"items": [json.loads(body)], "completed": True}
        items = []
        event_data = []
        total_bytes = 0
        completed = False
        while time.monotonic() < deadline:
            line = response.readline(65537)
            if not line:
                break
            total_bytes += len(line)
            if len(line) > 65536 or total_bytes > _MAX_RESPONSE_BYTES:
                raise ValueError("protocol stream exceeds size limit")
            stripped = line.decode("utf-8").rstrip("\r\n")
            if stripped.startswith("data:"):
                event_data.append(stripped[5:].lstrip(" "))
            elif not stripped and event_data:
                data = "\n".join(event_data)
                event_data = []
                if data == "[DONE]":
                    completed = True
                    break
                items.append(json.loads(data))
        return {"items": items, "completed": completed}


def inspect_response(response, *, stream=False, tools=False, reasoning=False):
    errors = []
    unknowns = []
    content = []
    reasoning_chars = 0
    tool_calls = {}
    completion_tokens = None
    finishes = []
    items = response.get("items", [])
    if not isinstance(items, list) or not items:
        return {"status": "UNKNOWN", "errors": [], "unknowns": ["no complete response item observed"]}
    for item in items:
        if not isinstance(item, dict) or item.get("error"):
            errors.append("invalid response object or API error")
            continue
        usage = item.get("usage")
        if isinstance(usage, dict) and type(usage.get("completion_tokens")) is int:
            completion_tokens = usage["completion_tokens"]
        choices = item.get("choices")
        if not isinstance(choices, list):
            errors.append("choices must be an array")
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                errors.append("choice must be an object")
                continue
            if choice.get("finish_reason"):
                finishes.append(choice["finish_reason"])
            message = choice.get("delta" if stream else "message")
            if not isinstance(message, dict):
                errors.append("missing message or delta object")
                continue
            text = message.get("content")
            if isinstance(text, str):
                content.append(text)
            reason = message.get("reasoning_content") or message.get("reasoning")
            if isinstance(reason, str):
                reasoning_chars += len(reason)
            calls = message.get("tool_calls", [])
            if calls is None:
                calls = []
            if not isinstance(calls, list):
                errors.append("tool_calls must be an array")
                continue
            for offset, call in enumerate(calls):
                if not isinstance(call, dict):
                    errors.append("tool call must be an object")
                    continue
                index = call.get("index", offset)
                function = call.get("function")
                if type(index) is not int or index < 0 or not isinstance(function, dict):
                    errors.append("invalid tool call index or function")
                    continue
                collected = tool_calls.setdefault(index, {"name": "", "arguments": ""})
                for key in ("name", "arguments"):
                    value = function.get(key, "")
                    if not isinstance(value, str):
                        errors.append("tool function " + key + " must be a string")
                    else:
                        collected[key] += value
    text = "".join(content)
    if any(marker in text for marker in _RAW_MARKERS):
        errors.append("raw reasoning/tool protocol markers leaked into content")
    if not text and not reasoning_chars and not tool_calls:
        unknowns.append("no generated content observed")
    if completion_tokens is None or completion_tokens < 1:
        unknowns.append("positive completion token usage not observed")
    if stream and not response.get("completed"):
        unknowns.append("stream completion marker not observed")
    if not finishes:
        unknowns.append("generation finish reason not observed")
    if reasoning and not reasoning_chars:
        unknowns.append("reasoning channel not observed; thinking semantics remain unverified")
    if tools and not tool_calls:
        unknowns.append("tool call not observed; tool semantics remain unverified")
    if "length" in finishes:
        unknowns.append("probe reached output limit; complete encoding semantics remain unverified")
    if tools:
        for call in tool_calls.values():
            try:
                arguments = json.loads(call["arguments"])
                if call["name"] != "llmperf_probe" or not isinstance(arguments, dict) or type(arguments.get("value")) is not int:
                    raise ValueError("tool schema mismatch")
            except (ValueError, TypeError):
                if "length" in finishes:
                    unknowns.append("tool call truncated by probe output limit")
                else:
                    errors.append("tool name or JSON arguments do not match the probe schema")
    return {"status": "FAIL" if errors else "UNKNOWN" if unknowns else "PASS",
            "errors": sorted(set(errors)), "unknowns": sorted(set(unknowns)),
            "content_chars": len(text), "reasoning_chars": reasoning_chars,
            "tool_call_count": len(tool_calls), "completion_tokens": completion_tokens,
            "finish_reasons": finishes}


def validate_protocol(base_url, model, specification, sender=exchange):
    deadline = time.monotonic() + float(specification.get("timeout_seconds", 60))
    payload = {"model": model, "messages": [{"role": "user", "content": "What is 3 + 4? Give a short answer."}],
               "stream": False, "max_tokens": 256, "temperature": 0}
    cases = [("chat", payload)]
    if specification.get("streaming"):
        cases.append(("streaming", {**payload, "stream": True, "stream_options": {"include_usage": True}}))
    if specification.get("tools"):
        cases.append(("tools", {**payload, "messages": [{"role": "user", "content":
            "Call llmperf_probe with the integer value 7. Do not execute anything else."}],
            "tools": [{"type": "function", "function": {"name": "llmperf_probe", "description": "Return a number.",
                "parameters": {"type": "object", "properties": {"value": {"type": "integer"}},
                               "required": ["value"]}}}], "tool_choice": "auto"}))
        if specification.get("streaming"):
            cases.append(("tools_streaming", {**cases[-1][1], "stream": True,
                                               "stream_options": {"include_usage": True}}))
    results = []
    for name, request in cases:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            results.append({"case": name, "status": "UNKNOWN", "unknowns": ["protocol check budget exhausted"]})
            continue
        try:
            response = sender(base_url, request, remaining)
            row = inspect_response(response, stream=request["stream"], tools=name.startswith("tools"),
                                   reasoning=bool(specification.get("reasoning")) and not name.startswith("tools"))
        except urllib.error.HTTPError as exc:
            row = {"status": "FAIL", "errors": ["protocol request rejected: HTTP " + str(exc.code)]}
        except (TimeoutError, OSError) as exc:
            row = {"status": "UNKNOWN", "unknowns": ["protocol transport: " + type(exc).__name__]}
        except (ValueError, TypeError) as exc:
            row = {"status": "FAIL", "errors": ["malformed protocol response: " + type(exc).__name__]}
        results.append({"case": name, **row})
    statuses = {row["status"] for row in results}
    status = "FAIL" if "FAIL" in statuses else "UNKNOWN" if "UNKNOWN" in statuses else "PASS"
    if specification.get("multimodal") and status == "PASS":
        status = "UNKNOWN"
    return {"version": 1, "status": status, "runtime_verified": status == "PASS", "cases": results,
            "scope": "text protocol probes only; not model quality or multimodal correctness",
            "multimodal_verified": False, "scored": False, "tools_executed": False}


def deployment_report(instances):
    statuses = {row["status"] for row in instances}
    status = "FAIL" if "FAIL" in statuses else "UNKNOWN" if not instances or "UNKNOWN" in statuses else "PASS"
    return {"version": 1, "status": status, "runtime_verified": status == "PASS", "instances": instances}


def report_is_valid(report, specification, expected_ids):
    if not isinstance(report, dict) or type(report.get("version")) is not int or report["version"] != 1:
        return False
    instances = report.get("instances")
    if not isinstance(instances, list) or not instances or any(not isinstance(item, dict) for item in instances):
        return False
    identifiers = [item.get("id") for item in instances]
    if any(not isinstance(ident, str) for ident in identifiers) or sorted(identifiers) != sorted(expected_ids):
        return False
    required = {"chat"} | {key for key in ("streaming", "tools") if specification.get(key)}
    if specification.get("streaming") and specification.get("tools"):
        required.add("tools_streaming")
    for instance in instances:
        cases = instance.get("cases")
        if not isinstance(cases, list) or any(not isinstance(row, dict) for row in cases):
            return False
        names = [row.get("case") for row in cases]
        if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names) or not required <= set(names):
            return False
        if any(not isinstance(row.get("status"), str) for row in cases):
            return False
        statuses = {row["status"] for row in cases}
        if not statuses <= {"PASS", "UNKNOWN", "FAIL"}:
            return False
        expected = "FAIL" if "FAIL" in statuses else "UNKNOWN" if "UNKNOWN" in statuses or specification.get("multimodal") else "PASS"
        if instance.get("status") != expected or instance.get("runtime_verified") is not (expected == "PASS"):
            return False
    expected_report = deployment_report(instances)
    return (report.get("status") == expected_report["status"] and
            report.get("runtime_verified") is expected_report["runtime_verified"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    specification = json.loads(os.environ["S1_PROTOCOL_VALIDATION"])
    report = deployment_report([{"id": "i0", **validate_protocol(args.base_url, args.model, specification)}])
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    return 2 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
