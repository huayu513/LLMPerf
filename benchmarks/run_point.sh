#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/server/launch_server.sh"
REPLAY="${SCRIPT_DIR}/replay/run_replay.sh"
BENCHMARK_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
CONFIG_FILE="${S1_CONFIG_FILE:-${BENCHMARK_DIR}/config/portable.env}"
if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi
STATE_DIR="${S1_SERVER_STATE_DIR:-${BENCHMARK_DIR}/run/server}"

usage() {
  cat <<'USAGE'
Usage:
  run_point.sh CLASS --profile PROFILE [options]

CLASS is smoke, formal, or diagnostic. This controller starts one fresh server,
waits for readiness, runs a Pod-local replay, and stops the server afterward.

Options:
  --mode closed-loop|open-loop
  --concurrency N
  --arrival-rate-scale N
  --max-in-flight N
  --run N
  --limit N
  --warmup N
  --ready-timeout SECONDS
  --request-timeout SECONDS
  --base-url URL
  --jsonl PATH
  --index PATH
  --keep-server
  --no-gpu-monitor
  --dry-run
USAGE
}

die() {
  echo "error: $*" >&2
  exit 2
}

startup_log_has_failure() {
  local log_file="$1"
  [[ -f "$log_file" ]] || return 1
  LC_ALL=C grep -Eq \
    '(^Traceback \(most recent call last\):|^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*(Error|Exception): )' \
    "$log_file"
}

run_class="${1:-}"
case "$run_class" in
  smoke|formal|diagnostic) ;;
  *) usage; exit 2 ;;
esac
shift

profile=""
mode="closed-loop"
concurrency=1
arrival_rate_scale=1.0
max_in_flight=0
run_index=1
limit=""
warmup=0
ready_timeout=3600
request_timeout=3600
base_url="${S1_BASE_URL:-http://127.0.0.1:25080}"
jsonl_path="${S1_JSONL_PATH:-${BENCHMARK_DIR}/input.jsonl}"
index_path="${S1_INDEX_PATH:-${BENCHMARK_DIR}/replay_index.json}"
keep_server=0
enable_gpu_monitor="${S1_ENABLE_GPU_MONITOR:-1}"
verify_source=1
dry_run=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ "$#" -ge 2 ]] || die "--profile requires a value"
      profile="$2"
      shift 2
      ;;
    --mode)
      [[ "$#" -ge 2 ]] || die "--mode requires a value"
      mode="$2"
      shift 2
      ;;
    --concurrency)
      [[ "$#" -ge 2 ]] || die "--concurrency requires a value"
      concurrency="$2"
      shift 2
      ;;
    --arrival-rate-scale)
      [[ "$#" -ge 2 ]] || die "--arrival-rate-scale requires a value"
      arrival_rate_scale="$2"
      shift 2
      ;;
    --max-in-flight)
      [[ "$#" -ge 2 ]] || die "--max-in-flight requires a value"
      max_in_flight="$2"
      shift 2
      ;;
    --run)
      [[ "$#" -ge 2 ]] || die "--run requires a value"
      run_index="$2"
      shift 2
      ;;
    --limit)
      [[ "$#" -ge 2 ]] || die "--limit requires a value"
      limit="$2"
      shift 2
      ;;
    --warmup)
      [[ "$#" -ge 2 ]] || die "--warmup requires a value"
      warmup="$2"
      shift 2
      ;;
    --ready-timeout)
      [[ "$#" -ge 2 ]] || die "--ready-timeout requires a value"
      ready_timeout="$2"
      shift 2
      ;;
    --request-timeout)
      [[ "$#" -ge 2 ]] || die "--request-timeout requires a value"
      request_timeout="$2"
      shift 2
      ;;
    --base-url)
      [[ "$#" -ge 2 ]] || die "--base-url requires a value"
      base_url="$2"
      shift 2
      ;;
    --jsonl)
      [[ "$#" -ge 2 ]] || die "--jsonl requires a value"
      jsonl_path="$2"
      shift 2
      ;;
    --index)
      [[ "$#" -ge 2 ]] || die "--index requires a value"
      index_path="$2"
      shift 2
      ;;
    --keep-server)
      keep_server=1
      shift
      ;;
    --no-gpu-monitor)
      enable_gpu_monitor=0
      shift
      ;;
    --no-verify-source)
      verify_source=0
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$profile" ]] || die "--profile is required"
[[ "$ready_timeout" =~ ^[1-9][0-9]*$ ]] ||
  die "--ready-timeout must be a positive integer"

replay_args=(
  "$run_class"
  --profile "$profile"
  --mode "$mode"
  --concurrency "$concurrency"
  --arrival-rate-scale "$arrival_rate_scale"
  --max-in-flight "$max_in_flight"
  --run "$run_index"
  --warmup "$warmup"
  --request-timeout "$request_timeout"
  --base-url "$base_url"
  --jsonl "$jsonl_path"
  --index "$index_path"
)
[[ -n "$limit" ]] && replay_args+=(--limit "$limit")
[[ "$verify_source" == "0" ]] && replay_args+=(--no-verify-source)

output_dir="$("$REPLAY" "${replay_args[@]}" --print-output-dir)"

if [[ "$dry_run" == "1" ]]; then
  "$LAUNCHER" dry-run "$profile"
  echo
  "$REPLAY" "${replay_args[@]}" --dry-run
  exit 0
fi

if [[ -d "$output_dir" ]] &&
   [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  die "output directory is not empty: ${output_dir}"
fi
mkdir -p "$output_dir"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
manifest_python="${S1_MANIFEST_PYTHON:-python3}"
if [[ "$manifest_python" != */* ]]; then
  manifest_python="$(command -v "$manifest_python" 2>/dev/null || true)"
fi
[[ -x "$manifest_python" ]] || die "manifest Python executable is missing"

write_manifest() {
  local state="$1"
  local exit_code="$2"
  "$manifest_python" -     "${output_dir}/run_manifest.json" "$state" "$exit_code"     "$profile" "$run_class" "$mode" "$concurrency"     "$arrival_rate_scale" "$run_index" "$base_url" "$output_dir"     "$started_at" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    path,
    state,
    exit_code,
    profile,
    run_class,
    mode,
    concurrency,
    arrival_rate_scale,
    run_index,
    base_url,
    output_dir,
    started_at,
) = sys.argv[1:]
manifest_path = Path(path)
data = {}
if manifest_path.exists():
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
data.update(
    {
        "schema_version": 1,
        "profile": profile,
        "class": run_class,
        "mode": mode,
        "concurrency": int(concurrency),
        "arrival_rate_scale": float(arrival_rate_scale),
        "run": int(run_index),
        "base_url": base_url,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "7"),
        "gpu_mapping": "physical GPU mask; process-local device numbering starts at cuda:0",
        "request_sender_location": "target_pod",
        "output_dir": output_dir,
        "started_at": started_at,
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
)
if exit_code:
    data["exit_code"] = int(exit_code)
temporary = manifest_path.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(data, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, manifest_path)
PY
}

server_started=0
monitor_pid=""
cleanup() {
  local exit_code="$?"
  local final_exit_code="$exit_code"
  local stop_exit_code=0
  trap - EXIT
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill -TERM "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  if [[ "$server_started" == "1" && "$keep_server" != "1" ]]; then
    "$LAUNCHER" stop || stop_exit_code="$?"
  fi
  if [[ "$stop_exit_code" != "0" ]]; then
    echo "error: server stop failed with exit code ${stop_exit_code}" >&2
    if [[ "$final_exit_code" == "0" ]]; then
      final_exit_code="$stop_exit_code"
    fi
  fi
  if [[ "$final_exit_code" == "0" ]]; then
    write_manifest complete "$final_exit_code" || true
  else
    write_manifest failed "$final_exit_code" || true
  fi
  exit "$final_exit_code"
}
trap cleanup EXIT

write_manifest running ""

if command -v curl >/dev/null 2>&1 &&
   curl -fsS --max-time 2 "${base_url%/}/health" >/dev/null 2>&1; then
  die "a service is already responding at ${base_url}; refuse to mix server state"
fi

server_log="${output_dir}/server.log"
"$LAUNCHER" start "$profile" --log-file "$server_log"
server_started=1

deadline="$((SECONDS + ready_timeout))"
while ! curl -fsS --max-time 5 "${base_url%/}/health" >/dev/null 2>&1; do
  "$LAUNCHER" status >/dev/null ||
    die "SGLang exited before becoming ready; inspect ${server_log}"
  ! startup_log_has_failure "$server_log" ||
    die "SGLang logged a startup exception before becoming ready; inspect ${server_log}"
  (( SECONDS < deadline )) ||
    die "SGLang did not become ready within ${ready_timeout}s"
  sleep 2
done
echo "server ready: ${base_url}"

cp "${STATE_DIR}/server.command.sh" "${output_dir}/server.command.sh"
cp "${STATE_DIR}/server.meta" "${output_dir}/server.meta"
cp "${STATE_DIR}/server.requested.json" "${output_dir}/server.requested.json"
server_info_ok=1
curl -fsS --max-time 10 "${base_url%/}/get_server_info" \
  -o "${output_dir}/server.info.json" || server_info_ok=0
if [[ "$server_info_ok" == "0" ]]; then
  printf '{}\n' > "${output_dir}/server.info.json"
fi
evidence_resolved="$("$manifest_python" - \
  "${output_dir}/server.requested.json" \
  "${output_dir}/server.info.json" \
  "${output_dir}/server.evidence.json" "$server_info_ok" <<'PY'
import json
import math
import sys
from pathlib import Path

requested_path, info_path, evidence_path, captured = sys.argv[1:]
requested = json.loads(Path(requested_path).read_text(encoding="utf-8"))
try:
    server_info = json.loads(Path(info_path).read_text(encoding="utf-8"))
except (OSError, ValueError, TypeError):
    server_info = {}

def normalized(name):
    return str(name).strip().lower().replace("-", "_")

# Keep direct ServerArgs fields; nested CUDA graph backends are unrelated.
arguments = server_info.get("server_args", server_info) if isinstance(server_info, dict) else {}
leaves = {normalized(key): value for key, value in arguments.items()} if isinstance(arguments, dict) else {}

aliases = {
    "chat_template_kwargs": ("chat_template_kwargs", "default_chat_template_kwargs"),
    "served_model_name": ("served_model_name", "served_model"),
    "tp": ("tp", "tp_size", "tensor_parallel_size"),
    "dp": ("dp", "dp_size", "data_parallel_size"),
    "pp": ("pp", "pp_size", "pipeline_parallel_size"),
    "dp_attention": ("dp_attention", "enable_dp_attention"),
    "dp_lm_head": ("dp_lm_head", "enable_dp_lm_head"),
    "backend": ("moe_runner_backend", "moe_runner", "backend"),
    "moe_a2a_backend": ("moe_a2a_backend", "a2a_backend"),
    "dspark": ("dspark", "enable_dspark", "speculative_algorithm"),
    "mem_fraction_static": ("mem_fraction_static",),
    "max_running_requests": ("max_running_requests",),
    "chunked_prefill_size": ("chunked_prefill_size",),
}

def find_value(names):
    for name in names:
        if name in leaves:
            return True, leaves[name]
    return False, None

def as_bool(value, key):
    if key == "dspark" and isinstance(value, str):
        return value.strip().upper() == "DSPARK"
    if key == "dspark" and value is None:
        return False

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", "none", ""}:
            return False
    return value

checks = {}
resolved_values = {}
unsupported = []
mismatched = []
for key, expected in requested.items():
    supported, actual = find_value(aliases.get(key, (key,)))
    auto_resolved = False
    if not supported:
        matches = False
        unsupported.append(key)
    elif key in {"backend", "moe_a2a_backend"} and isinstance(expected, str) and expected.lower() == "auto":
        invalid_auto = {"", "auto"} if key == "moe_a2a_backend" else {"", "auto", "none"}
        auto_resolved = actual is not None and str(actual).strip().lower() not in invalid_auto
        matches = auto_resolved
        if not matches:
            mismatched.append(key)
    elif isinstance(expected, bool):
        matches = as_bool(actual, key) == expected
        if not matches:
            mismatched.append(key)
    elif isinstance(expected, float):
        try:
            matches = math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            mismatched.append(key)
    elif isinstance(expected, int):
        try:
            matches = int(actual) == expected
            if (
                not matches
                and key == "chunked_prefill_size"
                and requested.get("dp_attention") is True
                and isinstance(requested.get("dp"), int)
                and requested["dp"] > 0
            ):
                matches = int(actual) == expected // requested["dp"]
        except (TypeError, ValueError):
            matches = False
        if not matches:
            mismatched.append(key)
    elif isinstance(expected, dict):
        if isinstance(actual, str):
            try:
                actual = json.loads(actual)
            except (TypeError, ValueError):
                pass
        matches = actual == expected
        if not matches:
            mismatched.append(key)
    else:
        matches = str(actual) == str(expected)
        if not matches:
            mismatched.append(key)
    if supported:
        resolved_values[key] = actual
    checks[key] = {
        "requested": expected, "actual": actual,
        "supported": supported, "matches": matches,
        "auto_resolved": auto_resolved,
    }

resolved = captured == "1" and not unsupported and not mismatched
evidence = {
    "readiness": True,
    "server_info_captured": captured == "1",
    "resolved": resolved,
    "requested_server_parameters": requested,
    "resolved_server_parameters": resolved_values,
    "parameter_checks": checks,
    "unsupported_parameters": unsupported,
    "mismatched_parameters": mismatched,
}
Path(evidence_path).write_text(
    json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
print("true" if resolved else "false")
PY
)"
if [[ "$evidence_resolved" != "true" ]]; then
  echo "server parameter evidence did not resolve; skipping replay" >&2
  exit 0
fi
grep -Ei 'resolved|ready to roll|chunked|dp attention|moe|dspark'   "${output_dir}/server.log" > "${output_dir}/server.resolved.log" || true

if [[ "$enable_gpu_monitor" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  mkdir -p "${output_dir}/monitor"
  nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}"     --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu     --format=csv -l 1 -f "${output_dir}/monitor/gpu.csv"     >/dev/null 2>&1 &
  monitor_pid="$!"
fi

"$REPLAY" "${replay_args[@]}" --output-dir "$output_dir"
grep -Ei 'resolved|ready to roll|chunked|dp attention|moe|dspark|oom|retract'   "${output_dir}/server.log" > "${output_dir}/server.resolved.log" || true
