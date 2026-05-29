#!/usr/bin/env bash
# Realistic CLI smoke lane for InvarLock.
#
# Uses the GPT-2 user-journey smoke so at least one slower lane runs on a model
# large enough to surface issues tiny smoke models can hide.

set -euo pipefail

ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORK_ROOT="${1:-$(mktemp -d -t invarlock_cli_realistic_smoke.XXXXXX.dir)}"
MODE="${INVARLOCK_REALISTIC_SMOKE_MODE:-local}"
JOURNEYS="${INVARLOCK_REALISTIC_SMOKE_JOURNEYS:-noop,negative}"
LOG_FILE="${INVARLOCK_SMOKE_LOG_FILE:-$(mktemp -t invarlock_cli_realistic_smoke.XXXXXX.log)}"

echo "[info] $(ts) realistic smoke mode=$MODE journeys=$JOURNEYS work_root=$WORK_ROOT" | tee -a "$LOG_FILE"
echo "[info] $(ts) Log file: $LOG_FILE" | tee -a "$LOG_FILE"

set +e
INVARLOCK_SMOKE_MODE="$MODE" INVARLOCK_SMOKE_JOURNEYS="$JOURNEYS" bash "$REPO_ROOT/scripts/smoke/run_gpt2_user_journey_smoke.sh" "$WORK_ROOT" >>"$LOG_FILE" 2>&1
RC=$?
set -e

echo "[summary] $(ts) lane=realistic exit_code=$RC" | tee -a "$LOG_FILE"
echo "[done] $(ts) Log captured to: $LOG_FILE"
echo "$LOG_FILE"
exit "$RC"
