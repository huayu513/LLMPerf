#!/usr/bin/env bash
PROFILE_ID="O1"
PROFILE_DESCRIPTION="Official mixed MXFP4 checkpoint, auto-resolved B300 runner, no DSpark"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/var/lib/docker/models/DeepSeek-V4-Flash-0731}"
MODEL_PATH="${MODEL_PATH:-/model}"
CHECKPOINT_KIND="official_mixed"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-Local}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-deepseekv4}"
REASONING_PARSER="${REASONING_PARSER:-deepseek-v4}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
if [[ -z "$DEFAULT_CHAT_TEMPLATE_KWARGS" ]]; then
  DEFAULT_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"high","thinking":true,"enable_thinking":true}'
fi
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-flashinfer_mxfp4}"
MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-none}"
