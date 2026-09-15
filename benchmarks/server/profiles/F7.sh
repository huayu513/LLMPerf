#!/usr/bin/env bash
PROFILE_ID="F7"
PROFILE_DESCRIPTION="Converted FP8 checkpoint, two-GPU TP2/DP2 DeepEP with DP Attention, without DSpark"
ENABLE_DSPARK="0"
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
MOE_A2A_BACKEND="deepep"
DEEPEP_CONFIG='{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'
PROFILE_ENV+=(
  "SGLANG_DSV4_FP4_EXPERTS=0"
  "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024"
)
