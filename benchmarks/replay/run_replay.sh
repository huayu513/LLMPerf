#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVER_DIR="${BENCHMARK_DIR}/server"
REPLAY_PY="${SCRIPT_DIR}/replay_jsonl_sglang.py"
S1_PROFILE_DIR="${S1_PROFILE_DIR:-${SERVER_DIR}/profiles}"
CONFIG_FILE="${S1_CONFIG_FILE:-${BENCHMARK_DIR}/config/portable.env}"
if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi
source "${SERVER_DIR}/profile_utils.sh"

usage() {
  cat <<'USAGE'
Usage:
  run_replay.sh CLASS --profile PROFILE [options]

CLASS is one of: smoke, formal, diagnostic

Core options:
  --profile PROFILE (any profile file under server/profiles)
  --mode closed-loop|open-loop
  --concurrency N
  --arrival-rate-scale N
  --max-in-flight N
  --run N
  --limit N
  --warmup N
  --output-dir PATH
  --dry-run
  --print-output-dir

Formal runs must use a loopback base URL and the complete indexed request set.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 2
}

require_nonnegative_int() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer"
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
request_timeout=3600
progress_every=50
flush_every=10
verify_source=1
dry_run=0
print_output_dir=0
output_dir=""
base_url="${S1_BASE_URL:-http://127.0.0.1:25080}"
base_urls=""
endpoint_ids=""
jsonl_path="${S1_JSONL_PATH:-${BENCHMARK_DIR}/input.jsonl}"
index_path="${S1_INDEX_PATH:-${BENCHMARK_DIR}/replay_index.json}"
results_root="${S1_RESULTS_ROOT:-${BENCHMARK_DIR}/results}"
python_bin="${S1_PYTHON_BIN:-$(command -v python3 2>/dev/null || true)}"
if [[ "$python_bin" != */* ]]; then
  python_bin="$(command -v "$python_bin" 2>/dev/null || true)"
fi

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
    --request-timeout)
      [[ "$#" -ge 2 ]] || die "--request-timeout requires a value"
      request_timeout="$2"
      shift 2
      ;;
    --progress-every)
      [[ "$#" -ge 2 ]] || die "--progress-every requires a value"
      progress_every="$2"
      shift 2
      ;;
    --flush-every)
      [[ "$#" -ge 2 ]] || die "--flush-every requires a value"
      flush_every="$2"
      shift 2
      ;;
    --base-url)
      [[ "$#" -ge 2 ]] || die "--base-url requires a value"
      base_url="$2"
      shift 2
      ;;
    --base-urls)
      [[ "$#" -ge 2 ]] || die "--base-urls requires a value"
      base_urls="$2"
      shift 2
      ;;
    --endpoint-ids)
      [[ "$#" -ge 2 ]] || die "--endpoint-ids requires a value"
      endpoint_ids="$2"
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
    --output-dir)
      [[ "$#" -ge 2 ]] || die "--output-dir requires a value"
      output_dir="$2"
      shift 2
      ;;
    --no-verify-source)
      verify_source=0
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --print-output-dir)
      print_output_dir=1
      shift
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

profile_exists "$profile" ||
  die "--profile must name a profile file under ${S1_PROFILE_DIR}"
[[ "$mode" == "closed-loop" || "$mode" == "open-loop" ]] ||
  die "--mode must be closed-loop or open-loop"
require_nonnegative_int concurrency "$concurrency"
require_nonnegative_int max-in-flight "$max_in_flight"
require_nonnegative_int run "$run_index"
require_nonnegative_int warmup "$warmup"
require_nonnegative_int progress-every "$progress_every"
require_nonnegative_int flush-every "$flush_every"
[[ "$run_index" != "0" ]] || die "--run must be positive"
if [[ -z "$limit" ]]; then
  [[ "$run_class" == "smoke" ]] && limit=8 || limit=0
fi
require_nonnegative_int limit "$limit"
if [[ "$mode" == "closed-loop" && "$concurrency" == "0" ]]; then
  die "closed-loop concurrency must be positive"
fi

if [[ "$run_class" == "formal" ]]; then
  [[ "$limit" == "0" ]] || die "formal runs must use the complete request set"
  urls_to_check="$base_url"
  [[ -n "$base_urls" ]] && urls_to_check="$base_urls"
  IFS=',' read -r -a formal_urls <<< "$urls_to_check"
  for formal_url in "${formal_urls[@]}"; do
    [[ "$formal_url" =~ ^http://(127\.0\.0\.1|localhost)(:[0-9]+)?/?$ ]] ||
      die "formal runs must target Pod-local loopback URLs"
  done
fi

PROFILE_ENV=()
# shellcheck disable=SC1090
source "$(profile_dir_path)/${profile}.sh"
# shellcheck disable=SC1090
source "${SERVER_DIR}/common.sh"
config_tag="tp${TP_SIZE}_dp${DP_SIZE}_dpa${ENABLE_DP_ATTENTION}"
config_hash_full="$(s1_effective_config_hash)"
config_hash="${config_hash_full:0:12}"
if [[ "$mode" == "closed-loop" ]]; then
  load_tag="c${concurrency}"
else
  scale_tag="${arrival_rate_scale//./p}"
  load_tag="trace_${scale_tag}x"
fi
run_decimal="$((10#${run_index}))"
printf -v run_pad '%03d' "$run_decimal"
run_name="${profile}_${config_tag}_cfg${config_hash}_${load_tag}_run${run_pad}"
if [[ -z "$output_dir" ]]; then
  output_dir="${results_root}/${run_class}/${profile}/${config_tag}/cfg_${config_hash}/${load_tag}/run_${run_pad}"
fi

if [[ "$print_output_dir" == "1" ]]; then
  printf '%s\n' "$output_dir"
  exit 0
fi

COMMAND=(
  "$python_bin" "$REPLAY_PY"
  --jsonl "$jsonl_path"
  --index "$index_path"
  --base-url "$base_url"
  --mode "$mode"
  --concurrency "$concurrency"
  --arrival-rate-scale "$arrival_rate_scale"
  --max-in-flight "$max_in_flight"
  --output-dir "$output_dir"
  --run-name "$run_name"
  --warmup "$warmup"
  --limit "$limit"
  --request-timeout "$request_timeout"
  --progress-every "$progress_every"
  --flush-every "$flush_every"
)
[[ -n "$base_urls" ]] && COMMAND+=(--base-urls "$base_urls")
[[ -n "$endpoint_ids" ]] && COMMAND+=(--endpoint-ids "$endpoint_ids")
[[ "$verify_source" == "1" ]] && COMMAND+=(--verify-source) ||
  COMMAND+=(--no-verify-source)

print_command() {
  printf 'exec'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
}

if [[ "$dry_run" == "1" ]]; then
  echo "output_dir: ${output_dir}"
  echo "request_path: Pod-local ${base_url}"
  print_command
  exit 0
fi

[[ -f "$REPLAY_PY" ]] || die "replay client is missing: ${REPLAY_PY}"
[[ -x "$python_bin" ]] || die "Python executable is missing: ${python_bin}"
[[ -f "$jsonl_path" ]] || die "JSONL file is missing: ${jsonl_path}"
[[ -f "$index_path" ]] || die "replay index is missing: ${index_path}"
mkdir -p "$output_dir"
result_path="${output_dir}/${run_name}.requests.jsonl"
summary_path="${output_dir}/${run_name}.summary.json"
command_path="${output_dir}/replay.command.sh"
[[ ! -e "$result_path" && ! -e "$summary_path" && ! -e "$command_path" ]] ||
  die "run artifacts already exist in ${output_dir}; refusing to overwrite"

{
  echo '#!/usr/bin/env bash'
  print_command
} > "$command_path"
chmod 0755 "$command_path"
cp "${SERVER_DIR}/common.sh" "${output_dir}/server.common.snapshot.sh"
cp "$(profile_dir_path)/${profile}.sh"   "${output_dir}/server.profile.snapshot.sh"
{
  printf 'class=%q\n' "$run_class"
  printf 'profile=%q\n' "$profile"
  printf 'mode=%q\n' "$mode"
  printf 'config_sha256=%q\n' "$config_hash_full"
  printf 'concurrency=%q\n' "$concurrency"
  printf 'arrival_rate_scale=%q\n' "$arrival_rate_scale"
  printf 'run=%q\n' "$run_decimal"
  printf 'base_url=%q\n' "$base_url"
  printf 'base_urls=%q\n' "$base_urls"
  printf 'endpoint_ids=%q\n' "$endpoint_ids"
  printf 'jsonl=%q\n' "$jsonl_path"
  printf 'index=%q\n' "$index_path"
} > "${output_dir}/replay.meta"

exec "${COMMAND[@]}"
