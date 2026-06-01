#!/usr/bin/env bash
# Exhaustive CLI smoke matrix for InvarLock.
#
# The matrix is split into three lanes so fast command-surface coverage does not
# get conflated with negative-path expectations or slower realistic runs.
#
# Default lanes:
# - fast: command surface + positive-path tiny-model flows
# - negative: malformed/policy_fail/fail-closed categories
# - realistic: slower GPT-2-sized user path

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib/smoke_common.sh"
ts() { smoke_ts; }
WORK_ROOT="${1:-$(mktemp -d -t invarlock_cli_smoke_matrix.XXXXXX.dir)}"
LOG_FILE="${INVARLOCK_SMOKE_LOG_FILE:-$(mktemp -t invarlock_cli_smoke_matrix.XXXXXX.log)}"
LANES_RAW="${INVARLOCK_SMOKE_LANES:-fast,negative,realistic}"

mkdir -p "$WORK_ROOT"

echo "[info] $(ts) work_root=$WORK_ROOT" | tee -a "$LOG_FILE"
echo "[info] $(ts) log_file=$LOG_FILE" | tee -a "$LOG_FILE"
echo "[info] $(ts) lanes=$LANES_RAW" | tee -a "$LOG_FILE"

TOTAL_LANES=0
FAILED_LANES=0

run_realistic_lane() {
  local lane_work_root="$1"
  local mode="${INVARLOCK_REALISTIC_SMOKE_MODE:-local}"
  local journeys="${INVARLOCK_REALISTIC_SMOKE_JOURNEYS:-noop,negative}"

  echo "[info] $(ts) realistic smoke mode=$mode journeys=$journeys work_root=$lane_work_root" | tee -a "$LOG_FILE"
  set +e
  INVARLOCK_SMOKE_MODE="$mode" INVARLOCK_SMOKE_JOURNEYS="$journeys" bash "$REPO_ROOT/scripts/smoke/run_gpt2_user_journey_smoke.sh" "$lane_work_root" >>"$LOG_FILE" 2>&1
  local rc=$?
  set -e
  echo "[summary] $(ts) lane=realistic exit_code=$rc" | tee -a "$LOG_FILE"
  return "$rc"
}

run_lane() {
  local lane="$1"
  local script_path=""
  case "$lane" in
    fast)
      script_path="$REPO_ROOT/scripts/smoke/cli_smoke_fast.sh"
      ;;
    negative)
      script_path="$REPO_ROOT/scripts/smoke/cli_smoke_negative.sh"
      ;;
    realistic)
      script_path="$REPO_ROOT/scripts/smoke/run_gpt2_user_journey_smoke.sh"
      ;;
    *)
      echo "[error] unknown smoke lane: $lane" | tee -a "$LOG_FILE"
      return 2
      ;;
  esac

  TOTAL_LANES=$((TOTAL_LANES + 1))
  echo "==== BEGIN lane:$lane ====" | tee -a "$LOG_FILE"
  echo "[script] $script_path" | tee -a "$LOG_FILE"
  local rc=0
  set +e
  if [[ "$lane" == "realistic" ]]; then
    run_realistic_lane "$WORK_ROOT/$lane"
  else
    bash "$script_path" "$WORK_ROOT/$lane" >>"$LOG_FILE" 2>&1
  fi
  rc=$?
  set -e
  echo "[exit_code] $rc" | tee -a "$LOG_FILE"
  echo "==== END lane:$lane ====" | tee -a "$LOG_FILE"
  if [[ "$rc" -ne 0 ]]; then
    FAILED_LANES=$((FAILED_LANES + 1))
  fi
}

IFS=',' read -r -a LANES <<<"$LANES_RAW"
for lane in "${LANES[@]}"; do
  lane="${lane//[[:space:]]/}"
  [[ -n "$lane" ]] || continue
  run_lane "$lane"
done

echo "[summary] $(ts) lanes=${TOTAL_LANES} failed=${FAILED_LANES}" | tee -a "$LOG_FILE"
echo "[done] $(ts) Log captured to: $LOG_FILE"
echo "$LOG_FILE"

if [[ "$FAILED_LANES" -ne 0 ]]; then
  exit 1
fi
