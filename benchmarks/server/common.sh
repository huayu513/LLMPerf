#!/usr/bin/env bash
# Common defaults sourced after a candidate profile. Environment variables may
# override any scalar value for controlled parallel-strategy and tuning runs.

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "${COMMON_DIR}/.." && pwd)"
CONFIG_FILE="${S1_CONFIG_FILE:-${BENCHMARK_DIR}/config/portable.env}"
if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

SGLANG_BIN="${SGLANG_BIN:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-25080}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local-model}"

TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
ENABLE_DP_ATTENTION="${ENABLE_DP_ATTENTION:-0}"
ENABLE_DP_LM_HEAD="${ENABLE_DP_LM_HEAD:-${ENABLE_DP_ATTENTION}}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"

ENABLE_HIERARCHICAL_CACHE="${ENABLE_HIERARCHICAL_CACHE:-0}"
HICACHE_RATIO="${HICACHE_RATIO:-2.0}"
HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"

TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"
REASONING_PARSER="${REASONING_PARSER:-}"
QUANTIZATION="${QUANTIZATION:-}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
if [[ -z "$DEFAULT_CHAT_TEMPLATE_KWARGS" ]]; then
  DEFAULT_CHAT_TEMPLATE_KWARGS='{}'
fi

ENABLE_DSPARK="${ENABLE_DSPARK:-0}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-5}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND-}"
MOE_A2A_BACKEND="${MOE_A2A_BACKEND-}"
DEEPEP_CONFIG="${DEEPEP_CONFIG-}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO-}"

SGLANG_ENABLE_UNIFIED_RADIX_TREE="${SGLANG_ENABLE_UNIFIED_RADIX_TREE:-1}"
SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-static}"

STATE_DIR="${S1_SERVER_STATE_DIR:-${BENCHMARK_DIR}/run/server}"

s1_effective_config_hash() {
  local profile_env_item
  {
    printf 'profile=%s\n' "$PROFILE_ID"
    printf 'model=%s\n' "$MODEL_PATH"
    printf 'served_model=%s\n' "$SERVED_MODEL_NAME"
    printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
    printf 'tp=%s\ndp=%s\npp=%s\ndpa=%s\ndplm=%s\n'       "$TP_SIZE" "$DP_SIZE" "$PP_SIZE" "$ENABLE_DP_ATTENTION" "$ENABLE_DP_LM_HEAD"
    printf 'mem=%s\nmax_running=%s\nchunked_prefill=%s\n'       "$MEM_FRACTION_STATIC" "$MAX_RUNNING_REQUESTS" "$CHUNKED_PREFILL_SIZE"
    printf 'hicache=%s\nhicache_ratio=%s\n'       "$ENABLE_HIERARCHICAL_CACHE" "$HICACHE_RATIO"
    printf 'moe_runner=%s\nmoe_a2a=%s\ndeepep_config=%s\n'       "$MOE_RUNNER_BACKEND" "$MOE_A2A_BACKEND" "$DEEPEP_CONFIG"
    printf 'dspark=%s\ndspark_block=%s\nswa_ratio=%s\n'       "$ENABLE_DSPARK" "$DSPARK_BLOCK_SIZE" "$SWA_FULL_TOKENS_RATIO"
    printf 'chat_template_kwargs=%s\n' "$DEFAULT_CHAT_TEMPLATE_KWARGS"
    printf 'quantization=%s\n' "$QUANTIZATION"
    for profile_env_item in "${PROFILE_ENV[@]:-}"; do
      printf 'profile_env=%s\n' "$profile_env_item"
    done
  } | sha256sum | awk '{print $1}'
}
