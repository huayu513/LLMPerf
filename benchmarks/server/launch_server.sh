#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_FILE="${SCRIPT_DIR}/common.sh"
PROFILE_DIR="${S1_PROFILE_DIR:-${SCRIPT_DIR}/profiles}"
S1_PROFILE_DIR="$PROFILE_DIR"
source "${SCRIPT_DIR}/profile_utils.sh"
STATE_DIR="${S1_SERVER_STATE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)/run/server}"
SETSID_BIN="${S1_SETSID_BIN:-$(command -v setsid || true)}"

usage() {
  local supported_profiles
  supported_profiles="$(profile_names | paste -sd ' ' -)"
  cat <<USAGE
Usage:
  launch_server.sh dry-run PROFILE
  launch_server.sh start PROFILE [--log-file PATH]
  launch_server.sh status
  launch_server.sh stop [--force]

Supported profiles: ${supported_profiles}

Scalar settings such as TP_SIZE, DP_SIZE, CHUNKED_PREFILL_SIZE and
MEM_FRACTION_STATIC may be overridden through environment variables.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 2
}

require_bool() {
  local name="$1"
  local value="$2"
  [[ "$value" == "0" || "$value" == "1" ]] || die "${name} must be 0 or 1"
}

require_positive_int() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer"
}

load_profile() {
  local requested="$1"
  profile_exists "$requested" ||
    die "unknown profile: ${requested}"
  PROFILE_FILE="${PROFILE_DIR}/${requested}.sh"
  [[ -f "$PROFILE_FILE" ]] || die "profile does not exist: ${PROFILE_FILE}"

  PROFILE_ENV=()
  # shellcheck disable=SC1090
  source "$PROFILE_FILE"
  # shellcheck disable=SC1090
  source "$COMMON_FILE"

  [[ "${PROFILE_ID:-}" == "$requested" ]] ||
    die "profile identity mismatch: expected ${requested}, got ${PROFILE_ID}"
  [[ -n "${MODEL_PATH:-}" ]] || die "MODEL_PATH is empty"
  require_positive_int TP_SIZE "$TP_SIZE"
  require_positive_int DP_SIZE "$DP_SIZE"
  require_positive_int PP_SIZE "$PP_SIZE"
  require_positive_int MAX_RUNNING_REQUESTS "$MAX_RUNNING_REQUESTS"
  require_positive_int CHUNKED_PREFILL_SIZE "$CHUNKED_PREFILL_SIZE"
  require_bool ENABLE_DP_ATTENTION "$ENABLE_DP_ATTENTION"
  require_bool ENABLE_DP_LM_HEAD "$ENABLE_DP_LM_HEAD"
  require_bool ENABLE_HIERARCHICAL_CACHE "$ENABLE_HIERARCHICAL_CACHE"
  require_bool ENABLE_DSPARK "$ENABLE_DSPARK"

  if [[ "$ENABLE_DP_LM_HEAD" == "1" && "$ENABLE_DP_ATTENTION" != "1" ]]; then
    die "DP LM Head requires DP Attention in the unified profiles"
  fi
  if [[ "$ENABLE_DSPARK" == "1" && "$ENABLE_DP_ATTENTION" == "1" &&
        "$MOE_A2A_BACKEND" != "none" ]]; then
    die "DSpark with DP Attention requires MOE_A2A_BACKEND=none"
  fi
  if [[ "${CHECKPOINT_KIND:-}" == "converted_fp8" &&
        "$MOE_RUNNER_BACKEND" == "marlin" ]]; then
    die "converted FP8 profiles must not use the incompatible Marlin runner"
  fi
}

build_command() {
  SERVER_ENV=(
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    "SGLANG_ENABLE_UNIFIED_RADIX_TREE=${SGLANG_ENABLE_UNIFIED_RADIX_TREE}"
    "SGLANG_RAGGED_VERIFY_MODE=${SGLANG_RAGGED_VERIFY_MODE}"
  )
  local profile_env_item
  for profile_env_item in "${PROFILE_ENV[@]:-}"; do
    [[ -n "$profile_env_item" ]] && SERVER_ENV+=("$profile_env_item")
  done

  SERVER_COMMAND=(
    "$SGLANG_BIN" serve
    --trust-remote-code
    --model-path "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --enable-metrics
    --enable-cache-report
    --tp-size "$TP_SIZE"
    --dp "$DP_SIZE"
    --pp-size "$PP_SIZE"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --host "$SERVER_HOST"
    --port "$SERVER_PORT"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS"
  )

  [[ -n "$TOOL_CALL_PARSER" ]] &&
    SERVER_COMMAND+=(--tool-call-parser "$TOOL_CALL_PARSER")
  [[ -n "$REASONING_PARSER" ]] &&
    SERVER_COMMAND+=(--reasoning-parser "$REASONING_PARSER")
  [[ -n "$QUANTIZATION" ]] &&
    SERVER_COMMAND+=(--quantization "$QUANTIZATION")

  if [[ "$ENABLE_HIERARCHICAL_CACHE" == "1" ]]; then
    SERVER_COMMAND+=(
      --enable-hierarchical-cache
      --hicache-ratio "$HICACHE_RATIO"
      --hicache-write-policy "$HICACHE_WRITE_POLICY"
      --hicache-io-backend "$HICACHE_IO_BACKEND"
      --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
    )
  fi
  [[ "$ENABLE_DP_ATTENTION" == "1" ]] &&
    SERVER_COMMAND+=(--enable-dp-attention)
  [[ "$ENABLE_DP_LM_HEAD" == "1" ]] &&
    SERVER_COMMAND+=(--enable-dp-lm-head)
  [[ -n "$MOE_RUNNER_BACKEND" ]] &&
    SERVER_COMMAND+=(--moe-runner-backend "$MOE_RUNNER_BACKEND")
  [[ -n "$MOE_A2A_BACKEND" ]] &&
    SERVER_COMMAND+=(--moe-a2a-backend "$MOE_A2A_BACKEND")
  [[ -n "$DEEPEP_CONFIG" ]] &&
    SERVER_COMMAND+=(--deepep-config "$DEEPEP_CONFIG")
  if [[ "$ENABLE_DSPARK" == "1" ]]; then
    SERVER_COMMAND+=(
      --speculative-algorithm DSPARK
      --speculative-dspark-block-size "$DSPARK_BLOCK_SIZE"
    )
  fi
  [[ -n "$SWA_FULL_TOKENS_RATIO" ]] &&
    SERVER_COMMAND+=(--swa-full-tokens-ratio "$SWA_FULL_TOKENS_RATIO")
  return 0
}

print_shell_command() {
  printf '%q env' "${SETSID_BIN:-setsid}"
  printf ' %q' "${SERVER_ENV[@]}"
  printf ' %q' "${SERVER_COMMAND[@]}"
  printf '\n'
}

read_pid() {
  local pid_file="${STATE_DIR}/server.pid"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(<"$pid_file")"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s\n' "$pid"
}

pid_state() {
  local pid="$1"
  [[ -r "/proc/${pid}/stat" ]] || return 1
  local stat_line
  stat_line="$(<"/proc/${pid}/stat")"
  # The command name is enclosed in parentheses and may contain spaces. Strip
  # through the final ") " so the first remaining field is the process state.
  stat_line="${stat_line##*) }"
  printf '%s\n' "${stat_line%% *}"
}

pid_is_zombie() {
  local state
  state="$(pid_state "$1" 2>/dev/null || true)"
  [[ "$state" == Z* ]]
}

pid_is_running() {
  local pid="$1"
  local state
  state="$(pid_state "$pid" 2>/dev/null || true)"
  [[ -n "$state" && "$state" != Z* ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

pid_is_managed_sglang() {
  local pid="$1"
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  local command_line
  command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  [[ "$command_line" == *sglang* && "$command_line" == *serve* ]]
}

process_group_for_pid() {
  local pid="$1"
  local pgid
  pgid="$(command ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
  [[ "$pgid" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s\n' "$pgid"
}

process_group_has_live_members() {
  local pgid="$1"
  [[ "$pgid" =~ ^[1-9][0-9]*$ ]] || return 1
  command ps -eo pgid=,stat= 2>/dev/null |
    awk -v target="$pgid" '$1 == target && $2 !~ /^Z/ { found = 1 }
      END { exit(found ? 0 : 1) }'
}

read_pgid() {
  local pgid_file="${STATE_DIR}/server.pgid"
  [[ -f "$pgid_file" ]] || return 1
  local pgid
  pgid="$(<"$pgid_file")"
  [[ "$pgid" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s\n' "$pgid"
}

remove_server_state() {
  rm -f "${STATE_DIR}/server.pid" "${STATE_DIR}/server.pgid"
}

show_status() {
  local pid
  if ! pid="$(read_pid)"; then
    echo "server state: stopped (no valid PID file)"
    return 1
  fi
  if ! pid_is_running "$pid"; then
    if pid_is_zombie "$pid"; then
      echo "server state: stopped (zombie PID ${pid})"
    else
      echo "server state: stopped (stale PID ${pid})"
    fi
    return 1
  fi
  echo "server state: running"
  echo "pid: ${pid}"
  local pgid
  if pgid="$(read_pgid 2>/dev/null)"; then
    echo "pgid: ${pgid}"
  fi
  [[ -f "${STATE_DIR}/server.meta" ]] && sed -n '1,20p' "${STATE_DIR}/server.meta"
  ps -p "$pid" -o pid=,etimes=,cmd=
}

stop_server() {
  local force="$1"
  local pid
  if ! pid="$(read_pid)"; then
    echo "server is not managed by this launcher"
    return 0
  fi

  local pgid=""
  pgid="$(read_pgid 2>/dev/null || true)"
  if [[ -z "$pgid" ]] && pid_is_running "$pid"; then
    pgid="$(process_group_for_pid "$pid" 2>/dev/null || true)"
  fi

  # A terminated process can remain as a zombie until its parent reaps it.
  # Treat that as exited, but still clean up any live members of our dedicated
  # process group before removing the state files.
  if ! pid_is_running "$pid"; then
    if [[ -n "$pgid" && "$pgid" == "$pid" ]] &&
       process_group_has_live_members "$pgid"; then
      echo "server parent PID ${pid} has exited; stopping remaining process-group members (PGID ${pgid})"
    else
      if pid_is_zombie "$pid"; then
        echo "server PID ${pid} is already exited (zombie); cleaning state"
      else
        echo "removing stale server PID ${pid}"
      fi
      remove_server_state
      return 0
    fi
  else
    pid_is_managed_sglang "$pid" ||
      die "PID ${pid} is not an SGLang serve process; refusing to signal it"
    echo "stopping SGLang PID ${pid}"
    # Let SGLang perform its own graceful drain first. If it exits while a
    # worker remains alive, the dedicated process-group cleanup below catches
    # that worker.
    kill -TERM "$pid" 2>/dev/null || true
  fi

  local use_group=0
  if [[ "$pgid" =~ ^[1-9][0-9]*$ && "$pgid" == "$pid" ]]; then
    use_group=1
  fi
  local group_term_sent=0
  local waited
  for waited in {1..60}; do
    if [[ "$use_group" == "1" ]]; then
      process_group_has_live_members "$pgid" || break
      # If the SGLang parent is gone but a worker survived, ask the remaining
      # members to terminate before resorting to KILL at the deadline.
      if [[ "$group_term_sent" == "0" ]] && ! pid_is_running "$pid"; then
        kill -TERM -- "-${pgid}" 2>/dev/null || true
        group_term_sent=1
      fi
    else
      pid_is_running "$pid" || break
    fi
    sleep 1
  done

  local still_running=0
  if [[ "$use_group" == "1" ]]; then
    process_group_has_live_members "$pgid" && still_running=1
  else
    pid_is_running "$pid" && still_running=1
  fi
  if [[ "$still_running" == "1" ]]; then
    if [[ "$force" == "1" ]]; then
      echo "SGLang did not exit after 60s; sending KILL"
      if [[ "$use_group" == "1" ]]; then
        kill -KILL -- "-${pgid}" 2>/dev/null || true
      else
        kill -KILL "$pid" 2>/dev/null || true
      fi
      for waited in {1..10}; do
        if [[ "$use_group" == "1" ]]; then
          process_group_has_live_members "$pgid" || break
        else
          pid_is_running "$pid" || break
        fi
        sleep 1
      done
      if [[ "$use_group" == "1" ]]; then
        process_group_has_live_members "$pgid" &&
          die "SGLang process group ${pgid} is still running after KILL"
      else
        pid_is_running "$pid" &&
          die "SGLang PID ${pid} is still running after KILL"
      fi
    else
      die "SGLang did not exit after 60s; rerun stop --force if appropriate"
    fi
  fi
  remove_server_state
  echo "server stopped"
}

action="${1:-}"
case "$action" in
  dry-run)
    [[ "$#" -eq 2 ]] || { usage; exit 2; }
    load_profile "$2"
    build_command
    echo "profile: ${PROFILE_ID} - ${PROFILE_DESCRIPTION:-custom profile}"
    echo "checkpoint: ${MODEL_PATH}"
    echo "parallelism: tp=${TP_SIZE} dp=${DP_SIZE} dp_attention=${ENABLE_DP_ATTENTION}"
    print_shell_command
    ;;
  start)
    [[ "$#" -ge 2 ]] || { usage; exit 2; }
    profile="$2"
    shift 2
    log_file=""
    while [[ "$#" -gt 0 ]]; do
      case "$1" in
        --log-file)
          [[ "$#" -ge 2 ]] || die "--log-file requires a path"
          log_file="$2"
          shift 2
          ;;
        *)
          die "unknown start argument: $1"
          ;;
      esac
    done
    load_profile "$profile"
    build_command
    if [[ "$SGLANG_BIN" != */* ]]; then
      SGLANG_BIN="$(command -v "$SGLANG_BIN" 2>/dev/null || true)"
      SERVER_COMMAND[0]="$SGLANG_BIN"
    fi
    [[ -x "$SGLANG_BIN" ]] || die "SGLang executable is missing: ${SGLANG_BIN}"
    [[ -x "$SETSID_BIN" ]] || die "setsid executable is missing; cannot create a dedicated SGLang process group"
    [[ -d "$MODEL_PATH" ]] || die "model directory is missing: ${MODEL_PATH}"
    mkdir -p "$STATE_DIR"
    if existing_pid="$(read_pid 2>/dev/null)"; then
      if pid_is_running "$existing_pid"; then
        die "managed server is already running with PID ${existing_pid}"
      fi
      echo "removing stale server state for PID ${existing_pid}"
    fi
    remove_server_state
    log_file="${log_file:-${STATE_DIR}/${PROFILE_ID}.server.log}"
    mkdir -p "$(dirname "$log_file")"
    manifest_python="${S1_MANIFEST_PYTHON:-python3}"
    if [[ "$manifest_python" != */* ]]; then
      manifest_python="$(command -v "$manifest_python" 2>/dev/null || true)"
    fi
    [[ -x "$manifest_python" ]] || die "manifest Python executable is missing"
    "$manifest_python" - "${STATE_DIR}/server.requested.json" \
      "$TP_SIZE" "$DP_SIZE" "$PP_SIZE" "$ENABLE_DP_ATTENTION" \
      "$ENABLE_DP_LM_HEAD" "${MOE_RUNNER_BACKEND-}" \
      "${MOE_A2A_BACKEND-}" "$ENABLE_DSPARK" \
      "$MEM_FRACTION_STATIC" "$MAX_RUNNING_REQUESTS" "$CHUNKED_PREFILL_SIZE" \
      "$MODEL_PATH" "$SERVED_MODEL_NAME" "$TOOL_CALL_PARSER" \
      "$REASONING_PARSER" "$DEFAULT_CHAT_TEMPLATE_KWARGS" \
      "$QUANTIZATION" "${MODEL_IS_MOE:-0}" <<'PY'
import json
import sys
from pathlib import Path

(
    path, tp, dp, pp, dpa, dplm, runner, a2a, dspark, memory, running,
    chunked, model_path, served_name, tool_parser, reasoning_parser,
    template_kwargs, quantization, model_is_moe,
) = sys.argv[1:]
data = {
    "model_path": model_path,
    "served_model_name": served_name,
    "tp": int(tp), "dp": int(dp), "pp": int(pp),
    "dp_attention": dpa == "1", "dp_lm_head": dplm == "1",
    "dspark": dspark == "1",
    "mem_fraction_static": float(memory),
    "max_running_requests": int(running), "chunked_prefill_size": int(chunked),
    "chat_template_kwargs": json.loads(template_kwargs),
}
if tool_parser:
    data["tool_call_parser"] = tool_parser
if reasoning_parser:
    data["reasoning_parser"] = reasoning_parser
if quantization:
    data["quantization"] = quantization
if model_is_moe == "1" or runner:
    data["backend"] = runner or "auto"
if model_is_moe == "1" or a2a:
    data["moe_a2a_backend"] = a2a or "auto"
Path(path).write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
    print_shell_command > "${STATE_DIR}/server.command.sh"
    {
      printf 'profile=%q\n' "$PROFILE_ID"
      printf 'description=%q\n' "${PROFILE_DESCRIPTION:-custom profile}"
      printf 'model_path=%q\n' "$MODEL_PATH"
      printf 'tp=%q\n' "$TP_SIZE"
      printf 'dp=%q\n' "$DP_SIZE"
      printf 'pp=%q\n' "$PP_SIZE"
      printf 'dp_attention=%q\n' "$ENABLE_DP_ATTENTION"
      printf 'dp_lm_head=%q\n' "$ENABLE_DP_LM_HEAD"
      printf 'moe_runner_backend=%q\n' "$MOE_RUNNER_BACKEND"
      printf 'moe_a2a_backend=%q\n' "$MOE_A2A_BACKEND"
      printf 'dspark=%q\n' "$ENABLE_DSPARK"
      printf 'mem_fraction_static=%q\n' "$MEM_FRACTION_STATIC"
      printf 'max_running_requests=%q\n' "$MAX_RUNNING_REQUESTS"
      printf 'chunked_prefill_size=%q\n' "$CHUNKED_PREFILL_SIZE"
      printf 'log_file=%q\n' "$log_file"
      printf 'started_at=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'effective_config_sha256=%q\n' "$(s1_effective_config_hash)"
      printf 'source_config_sha256=%q\n' "$(
        sha256sum "$COMMON_FILE" "$PROFILE_FILE" "${BASH_SOURCE[0]}" |
          sha256sum | awk '{print $1}'
      )"
    } > "${STATE_DIR}/server.meta"

    nohup "$SETSID_BIN" env "${SERVER_ENV[@]}" "${SERVER_COMMAND[@]}"       >>"$log_file" 2>&1 </dev/null &
    pid="$!"
    printf '%s\n' "$pid" > "${STATE_DIR}/server.pid"
    sleep 2
    if ! pid_is_running "$pid"; then
      remove_server_state
      die "SGLang exited during startup; inspect ${log_file}"
    fi
    pgid="$(process_group_for_pid "$pid" 2>/dev/null || true)"
    [[ "$pgid" =~ ^[1-9][0-9]*$ ]] || {
      kill -TERM "$pid" 2>/dev/null || true
      remove_server_state
      die "unable to determine SGLang process group for PID ${pid}"
    }
    printf '%s\n' "$pgid" > "${STATE_DIR}/server.pgid"
    printf 'pid=%q\npgid=%q\n' "$pid" "$pgid" >> "${STATE_DIR}/server.meta"
    echo "started ${PROFILE_ID} with PID ${pid}"
    echo "process group: ${pgid}"
    echo "log: ${log_file}"
    ;;
  status)
    [[ "$#" -eq 1 ]] || { usage; exit 2; }
    show_status
    ;;
  stop)
    force=0
    shift
    while [[ "$#" -gt 0 ]]; do
      case "$1" in
        --force)
          force=1
          shift
          ;;
        *)
          die "unknown stop argument: $1"
          ;;
      esac
    done
    stop_server "$force"
    ;;
  *)
    usage
    exit 2
    ;;
esac
