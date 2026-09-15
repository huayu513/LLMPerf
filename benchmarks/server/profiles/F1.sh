#!/usr/bin/env bash
PROFILE_ID="F1"
PROFILE_DESCRIPTION="Converted FP8 checkpoint, two-GPU TP2/DP2 with DP Attention, no DSpark"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/var/lib/docker/models/DeepSeek-V4-Flash-0731-FP8-mt}"
MODEL_PATH="${MODEL_PATH:-/model}"
CHECKPOINT_KIND="converted_fp8"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-Local}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-deepseekv4}"
REASONING_PARSER="${REASONING_PARSER:-deepseek-v4}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
if [[ -z "$DEFAULT_CHAT_TEMPLATE_KWARGS" ]]; then
  DEFAULT_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"high","thinking":true,"enable_thinking":true}'
fi
TP_SIZE="2"
DP_SIZE="2"
ENABLE_DP_ATTENTION="1"
ENABLE_DP_LM_HEAD="1"
PROFILE_ENV+=("SGLANG_DSV4_FP4_EXPERTS=0")
