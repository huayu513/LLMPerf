"""Read installed engine argument definitions without loading model weights."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib
import inspect
import io
import json
import typing


def plain(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        return None


def build_parser(engine):
    if engine == "sglang":
        module = importlib.import_module("sglang.srt.server_args")
        parser = argparse.ArgumentParser(exit_on_error=False)
        module.ServerArgs.add_cli_args(parser)
        return parser, module
    if engine != "vllm":
        raise ValueError("unknown engine")
    module = importlib.import_module("vllm.engine.arg_utils")
    try:
        parser_type = importlib.import_module("vllm.utils.argparse_utils").FlexibleArgumentParser
    except ImportError:
        parser_type = argparse.ArgumentParser
    parser = parser_type()
    module.EngineArgs.add_cli_args(parser)
    return parser, module


def annotation_info(annotation):
    origin = typing.get_origin(annotation)
    arguments = typing.get_args(annotation)
    result = {"annotation": str(annotation)[:1000]}
    literals = []
    if origin is typing.Literal:
        literals = list(arguments)
    else:
        for argument in arguments:
            if typing.get_origin(argument) is typing.Literal:
                literals.extend(typing.get_args(argument))
    if literals:
        result["choices"] = plain(literals)
    for datatype, name in ((bool, "boolean"), (int, "integer"), (float, "number"), (str, "string")):
        if annotation is datatype or datatype in arguments:
            result["type"] = name
            break
    return result


def source_document(module, keywords):
    try:
        source = inspect.getsource(module)
        lines = source.splitlines()
        selected = set()
        for index, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in keywords):
                selected.update(range(max(0, index - 2), min(len(lines), index + 9)))
        excerpt = "\n".join(f"{index + 1}: {lines[index]}" for index in sorted(selected))[:24000]
        return {"module": module.__name__, "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "excerpt": excerpt, "truncated": len(excerpt) >= 24000}
    except (OSError, TypeError):
        return {"module": module.__name__, "error": "source unavailable"}


def collect_schema(engine):
    result = {"version": 1, "engine": engine, "status": "unknown", "parameters": {},
              "nested": {}, "documents": {}, "errors": []}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            parser, module = build_parser(engine)
        keywords = ("speculat", "dspark", "expert", "attn_dp", "dp_attention", "all2all", "moe")
        result["documents"][module.__name__] = source_document(module, keywords)
        for action in parser._actions:
            flags = [flag for flag in action.option_strings if flag.startswith("--")]
            if not flags:
                continue
            datatype = "unknown"
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction,
                                   argparse.BooleanOptionalAction)):
                datatype = "boolean"
            elif action.type in (int, float, str):
                datatype = {int: "integer", float: "number", str: "string"}[action.type]
            result["parameters"][action.dest] = {
                "flags": flags, "type": datatype, "default": plain(action.default),
                "choices": plain(list(action.choices)) if action.choices is not None else None,
                "help": str(action.help or "")[:3000], "nargs": plain(action.nargs),
                "source": module.__name__ + ":argparse", "action": type(action).__name__,
                "status": "discovered"}
        if engine == "vllm":
            try:
                spec_module = importlib.import_module("vllm.config.speculative")
                config_type = spec_module.SpeculativeConfig
                hints = typing.get_type_hints(config_type)
                fields = {field.name: field for field in dataclasses.fields(config_type)}
                result["nested"]["speculative_config"] = {
                    name: {**annotation_info(annotation),
                           "default": plain(fields[name].default) if name in fields else None,
                           "source": spec_module.__name__ + ":" + name}
                    for name, annotation in hints.items() if not name.startswith("_")}
                result["documents"][spec_module.__name__] = source_document(spec_module, keywords)
            except Exception as exc:
                result["errors"].append("speculative schema: " + str(exc)[:500])
        result["status"] = "discovered" if result["parameters"] else "unknown"
    except Exception as exc:
        result["errors"].append(type(exc).__name__ + ": " + str(exc)[:500])
    return result


def preflight(engine, cases):
    results = []
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            parser, _ = build_parser(engine)
    except Exception as exc:
        return [{"id": case["id"], "status": "unknown", "scope": "native_parser",
                 "reason": type(exc).__name__ + ": " + str(exc)[:500]} for case in cases]
    for case in cases:
        diagnostics = io.StringIO()
        try:
            with contextlib.redirect_stdout(diagnostics), contextlib.redirect_stderr(diagnostics):
                parser.parse_args(case["argv"])
            result = {"status": "preflight_passed", "scope": "native_parser",
                      "runtime_verified": False}
        except (Exception, SystemExit) as exc:
            result = {"status": "parser_rejected", "scope": "native_parser",
                      "reason": type(exc).__name__ + ": " + str(exc)[:500] + " " + diagnostics.getvalue()[-1500:]}
        results.append({"id": case["id"], **result})
    return results
