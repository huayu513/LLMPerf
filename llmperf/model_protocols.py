"""Checkpoint protocol profiles, independent of engine CLI spellings."""
from __future__ import annotations

import copy


_LEGACY_THINKING = {"enable_thinking": True, "reasoning_effort": "max", "thinking": True}
_PROFILES = (
    ("deepseek-v41", {"deepseek_v41", "deepseek_v41_text"},
     {"DeepseekV41ForCausalLM"}, "deepseekv41", "deepseek-v41",
     {"thinking": True, "reasoning_effort": "max"}),
    ("qwen4-exp", {"qwen4_exp", "qwen4_exp_text"},
     {"Qwen4ExpForConditionalGeneration"}, None, "qwen3",
     {"enable_thinking": True, "reasoning_effort": "max"}),
    ("glm5-next", {"glm5_next", "glm5_next_text"},
     {"Glm5NextForConditionalGeneration"}, "glm47", "glm45",
     {"enable_thinking": True, "reasoning_effort": "max"}),
    ("deepseek-v4", {"deepseek_v4"}, {"DeepseekV4ForCausalLM"},
     "deepseekv4", "deepseek-v4", _LEGACY_THINKING),
    ("deepseek-v3", {"deepseek_v3"}, {"DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM"},
     None, "deepseek-v3", _LEGACY_THINKING),
    ("glm", {"glm4_moe", "glm4_moe_lite", "glm_moe_dsa"},
     {"Glm4MoeForCausalLM", "Glm4MoeLiteForCausalLM", "GlmMoeDsaForCausalLM"},
     "glm", "glm45", _LEGACY_THINKING),
    ("qwen3", {"qwen3", "qwen3_moe", "qwen3_next", "qwen3_5", "qwen3_5_text",
               "qwen3_5_moe", "qwen3_5_moe_text"},
     {"Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "Qwen3NextForCausalLM",
      "Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM"}, None, "qwen3", _LEGACY_THINKING),
    ("qwen2", {"qwen", "qwen2", "qwen2_moe"},
     {"QWenLMHeadModel", "Qwen2ForCausalLM", "Qwen2MoeForCausalLM"},
     "qwen", None, _LEGACY_THINKING),
)


def protocol_profile(model_type, architectures):
    for family, types, archs, tool_parser, reasoning_parser, template_kwargs in _PROFILES:
        if model_type in types or set(architectures) & archs:
            if "DeepseekV32ForCausalLM" in architectures and family == "deepseek-v3":
                tool_parser = "deepseekv32"
            return {"version": 1, "id": family, "known": True,
                    "tool_call_parser": tool_parser, "reasoning_parser": reasoning_parser,
                    "chat_template_kwargs": copy.deepcopy(template_kwargs),
                    "encoding": "engine_native" if family == "deepseek-v41" else "checkpoint_or_engine",
                    "runtime_verified": False}
    return {"version": 1, "id": "generic", "known": False, "tool_call_parser": None,
            "reasoning_parser": None, "chat_template_kwargs": {},
            "encoding": "checkpoint_or_engine", "runtime_verified": False}


def parser_candidates(parser, engine, kind):
    if engine != "vllm":
        return (parser,) if parser else ()
    mapping = {
        "deepseekv4": ("deepseek_v4",), "deepseek-v4": ("deepseek_v4",),
        "deepseekv41": ("deepseek_v41",), "deepseek-v41": ("deepseek_v41",),
        "deepseekv3": ("deepseek_v3",), "deepseek-v3": ("deepseek_v3",),
        "deepseekv31": ("deepseek_v31",), "deepseekv32": ("deepseek_v32",),
        "qwen": ("qwen3_json", "hermes"), "qwen25": ("qwen3_json", "hermes"),
    }
    if kind == "reasoning_parser" and parser == "glm45":
        return ("glm45", "glm47")
    return mapping.get(parser, (parser,)) if parser else ()
