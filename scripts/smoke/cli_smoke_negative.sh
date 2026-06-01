#!/usr/bin/env bash
# Negative-path CLI smoke lane for InvarLock.
#
# This lane exercises canonical user-visible failure categories:
# - malformed verify input
# - policy_fail verify results for guard and PM categories
# - fail-closed report generation from a failed subject run report
# - command-line validation errors for bad paths

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib/smoke_common.sh"
ts() { smoke_ts; }

cd "$ROOT"

PYTHON_BIN="$(smoke_select_python "$ROOT" "${INVARLOCK_PYTHON:-}")"
smoke_setup_pythonpath "$ROOT"
printf -v CLI '%q ' "$PYTHON_BIN" -m invarlock
CLI="${CLI% }"

LOG_FILE="${INVARLOCK_SMOKE_LOG_FILE:-$(mktemp -t invarlock_cli_negative_smoke.XXXXXX.log)}"
TOTAL_COMMANDS=0
UNEXPECTED_FAILURES=0

echo "[info] $(ts) CLI runner: $CLI" | tee -a "$LOG_FILE"
echo "[info] $(ts) Log file: $LOG_FILE"

expected_exit_match() {
  smoke_expected_exit_match "$@"
}

record_result() {
  local label="$1"
  local ec="$2"
  local expected="${3:-0}"
  local status="pass"
  TOTAL_COMMANDS=$((TOTAL_COMMANDS + 1))
  if ! expected_exit_match "$ec" "$expected"; then
    status="fail"
    UNEXPECTED_FAILURES=$((UNEXPECTED_FAILURES + 1))
  fi
  {
    echo "[expected_exit_codes] $expected"
    echo "[status] $status"
  } >>"$LOG_FILE"
}

run_capture() {
  local label="$1"
  local outfile="$2"
  local cmd="$3"
  local expected="${4:-0}"
  {
    echo "\n==== BEGIN $label ===="
    echo "[cmd] $cmd"
    echo "[outfile] $outfile"
    echo "[ts] $(ts)"
  } >>"$LOG_FILE"
  set +e
  bash -lc "$cmd" >"$outfile" 2>&1
  local ec=$?
  set -e
  cat "$outfile" >>"$LOG_FILE"
  {
    echo "[exit_code] $ec"
    record_result "$label" "$ec" "$expected"
    echo "==== END $label ====\n"
  } >>"$LOG_FILE"
}

assert_verify_reason() {
  local outfile="$1"
  local expected_reason="$2"
  VERIFY_OUTPUT="$outfile" VERIFY_REASON="$expected_reason" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["VERIFY_OUTPUT"])
reason = os.environ["VERIFY_REASON"]
payload = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[-1])
actual = payload.get("summary", {}).get("reason")
if actual != reason:
    raise SystemExit(f"expected summary.reason={reason!r}, found {actual!r}")
PY
}

assert_contains() {
  local outfile="$1"
  local expected_fragment="$2"
  OUTPUT_FILE="$outfile" OUTPUT_FRAGMENT="$expected_fragment" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

text = Path(os.environ["OUTPUT_FILE"]).read_text(encoding="utf-8")
fragment = os.environ["OUTPUT_FRAGMENT"]
if fragment not in text:
    raise SystemExit(f"expected output to contain {fragment!r}")
PY
}

WORK_ROOT="${1:-$(mktemp -d -t invarlock_cli_negative_smoke.XXXXXX.dir)}"
mkdir -p "$WORK_ROOT"

GOLDEN_REPORT="$ROOT/tests/artifacts/golden_runs/gpt2/evaluation.report.json"
FIXTURE_ROOT="$WORK_ROOT/fixtures"
VERIFY_OUT="$WORK_ROOT/verify"
mkdir -p "$FIXTURE_ROOT" "$VERIFY_OUT"

GOLDEN_REPORT="$GOLDEN_REPORT" FIXTURE_ROOT="$FIXTURE_ROOT" "$PYTHON_BIN" - <<'PY'
import json
import math
import os
from copy import deepcopy
from pathlib import Path

golden = json.loads(Path(os.environ["GOLDEN_REPORT"]).read_text(encoding="utf-8"))
root = Path(os.environ["FIXTURE_ROOT"])

def write(name: str, payload: dict) -> None:
    (root / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

malformed = {"schema_version": "v1", "primary_metric": {}}
write("malformed.json", malformed)

pm_fail = deepcopy(golden)
pm_fail.setdefault("validation", {})["primary_metric_acceptable"] = False
pm_fail.setdefault("primary_metric", {})["ratio_vs_baseline"] = 1.25
write("pm_fail.json", pm_fail)

invariants_fail = deepcopy(golden)
invariants_fail.setdefault("validation", {})["invariants_pass"] = False
invariants_fail["invariants"] = {"status": "fail"}
write("invariants_fail.json", invariants_fail)

spectral_fail = deepcopy(golden)
spectral_fail.setdefault("validation", {})["spectral_stable"] = False
spectral_fail.setdefault("spectral", {})["caps_applied"] = 99
write("spectral_fail.json", spectral_fail)

rmt_fail = deepcopy(golden)
rmt_fail.setdefault("validation", {})["rmt_stable"] = False
rmt_fail.setdefault("rmt", {})["stable"] = False
write("rmt_fail.json", rmt_fail)

baseline_run = {
    "meta": {"model_id": "gpt2"},
    "edit": {"name": "baseline"},
    "metrics": {
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.0,
            "ratio_vs_baseline": 1.0,
        }
    },
    "data": {"seq_len": 8, "preview_n": 2, "final_n": 2},
}
failed_subject = {
    "meta": {"model_id": "gpt2"},
    "status": "failed",
    "edit": {"name": "quant_rtn"},
    "metrics": {
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.5,
            "ratio_vs_baseline": 1.05,
        }
    },
    "data": {"seq_len": 8, "preview_n": 2, "final_n": 2},
    "flags": {"guard_recovered": True, "rollback_reason": "guards_failed or metrics_unacceptable"},
}
write("baseline_run.json", baseline_run)
write("failed_subject_run.json", failed_subject)
PY

run_capture \
  "invarlock run (removed public command)" \
  "$VERIFY_OUT/run_removed.out" \
  "$CLI run" \
  "2"
assert_contains "$VERIFY_OUT/run_removed.out" "No such command 'run'"

run_capture \
  "invarlock verify --json (malformed fixture)" \
  "$VERIFY_OUT/malformed.out" \
  "$CLI verify --json \"$FIXTURE_ROOT/malformed.json\"" \
  "2"
assert_verify_reason "$VERIFY_OUT/malformed.out" "malformed"
assert_contains "$VERIFY_OUT/malformed.out" "\"code\": \"E601\""

run_capture \
  "invarlock verify --json (primary metric policy fail)" \
  "$VERIFY_OUT/pm_fail.out" \
  "$CLI verify --json --profile release \"$FIXTURE_ROOT/pm_fail.json\"" \
  "3"
assert_verify_reason "$VERIFY_OUT/pm_fail.out" "policy_fail"

run_capture \
  "invarlock verify --json (invariants policy fail)" \
  "$VERIFY_OUT/invariants_fail.out" \
  "$CLI verify --json --profile release \"$FIXTURE_ROOT/invariants_fail.json\"" \
  "3"
assert_verify_reason "$VERIFY_OUT/invariants_fail.out" "policy_fail"

run_capture \
  "invarlock verify --json (spectral policy fail)" \
  "$VERIFY_OUT/spectral_fail.out" \
  "$CLI verify --json --profile release \"$FIXTURE_ROOT/spectral_fail.json\"" \
  "3"
assert_verify_reason "$VERIFY_OUT/spectral_fail.out" "policy_fail"

run_capture \
  "invarlock verify --json (rmt policy fail)" \
  "$VERIFY_OUT/rmt_fail.out" \
  "$CLI verify --json --profile release \"$FIXTURE_ROOT/rmt_fail.json\"" \
  "3"
assert_verify_reason "$VERIFY_OUT/rmt_fail.out" "policy_fail"

run_capture \
  "invarlock report generate (failed subject run report)" \
  "$VERIFY_OUT/report_generate_failed_subject.out" \
  "$CLI report generate --run \"$FIXTURE_ROOT/failed_subject_run.json\" --baseline-run-report \"$FIXTURE_ROOT/baseline_run.json\" --format report -o \"$WORK_ROOT/generated_failed_subject\"" \
  "2"
assert_contains "$VERIFY_OUT/report_generate_failed_subject.out" "subject run report with status"

run_capture \
  "invarlock advanced calibrate null-sweep (missing config)" \
  "$VERIFY_OUT/calibrate_missing_config.out" \
  "$CLI advanced calibrate null-sweep --config \"$WORK_ROOT/does-not-exist.yaml\"" \
  "2"
assert_contains "$VERIFY_OUT/calibrate_missing_config.out" "Invalid value for '--config'"
assert_contains "$VERIFY_OUT/calibrate_missing_config.out" "does-not-exist.yaml"

echo "[summary] $(ts) total=${TOTAL_COMMANDS} unexpected_failures=${UNEXPECTED_FAILURES}" | tee -a "$LOG_FILE"
echo "[done] $(ts) Log captured to: $LOG_FILE"
echo "$LOG_FILE"

if [[ "$UNEXPECTED_FAILURES" -ne 0 ]]; then
  exit 1
fi
