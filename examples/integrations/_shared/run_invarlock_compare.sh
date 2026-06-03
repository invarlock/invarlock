#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_invarlock_compare.sh --baseline MODEL --subject MODEL_OR_PATH [options]

Required:
  --baseline VALUE             Baseline model ID or local path.
  --subject VALUE              Subject model ID or local path.

Options:
  --report-out DIR             Output directory for evaluation artifacts.
                               Default: reports/integration/<artifact-lane>
  --baseline-adapter NAME      Baseline adapter. Default: auto
  --subject-adapter NAME       Subject adapter. Default: auto
  --profile NAME               InvarLock profile. Default: ci
  --tier NAME                  InvarLock tier. Default: balanced
  --lane MODE                  Standard lane shortcut: host or cuda.
                               host => host execution, off assurance, host provenance.
                               cuda => cuda-container-strict.
  --execution-mode MODE        container or host. Default: container
  --assurance MODE             strict or off. Default: strict
  --runtime-provenance MODE    container or host for verify. Defaults to
                               execution mode.
  --device VALUE               Optional device override.
  --preset PATH                Optional InvarLock preset path.
  --edit-label VALUE           Optional edit label for BYOE subjects.
  --allow-network              Allow model/dataset downloads for evaluate.
  --require-backend-inventory  Fail if backend_inventory.json is missing.
  --no-html                    Skip HTML rendering.
  -h, --help                   Show this help.

The default path is strict/container-backed. For host-mode exploratory runs,
pass --lane host.
USAGE
}

baseline=""
subject=""
report_out="reports/integration"
report_out_was_default=1
baseline_adapter="auto"
subject_adapter="auto"
profile="ci"
tier="balanced"
lane=""
execution_mode="container"
assurance="strict"
runtime_provenance=""
device=""
preset=""
edit_label=""
allow_network=0
render_html=1
require_backend_inventory=0
original_args=("$@")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

# shellcheck source=preflight.sh
source "$SCRIPT_DIR/preflight.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      baseline="${2:-}"
      shift 2
      ;;
    --subject)
      subject="${2:-}"
      shift 2
      ;;
    --report-out)
      report_out="${2:-}"
      report_out_was_default=0
      shift 2
      ;;
    --baseline-adapter)
      baseline_adapter="${2:-}"
      shift 2
      ;;
    --subject-adapter)
      subject_adapter="${2:-}"
      shift 2
      ;;
    --profile)
      profile="${2:-}"
      shift 2
      ;;
    --tier)
      tier="${2:-}"
      shift 2
      ;;
    --lane)
      lane="${2:-}"
      shift 2
      ;;
    --execution-mode)
      execution_mode="${2:-}"
      shift 2
      ;;
    --assurance)
      assurance="${2:-}"
      shift 2
      ;;
    --runtime-provenance)
      runtime_provenance="${2:-}"
      shift 2
      ;;
    --device)
      device="${2:-}"
      shift 2
      ;;
    --preset)
      preset="${2:-}"
      shift 2
      ;;
    --edit-label)
      edit_label="${2:-}"
      shift 2
      ;;
    --allow-network)
      allow_network=1
      shift
      ;;
    --require-backend-inventory)
      require_backend_inventory=1
      shift
      ;;
    --no-html)
      render_html=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$baseline" || -z "$subject" ]]; then
  echo "Missing required --baseline or --subject." >&2
  usage >&2
  exit 2
fi

if [[ -n "$lane" ]]; then
  case "$lane" in
    host)
      execution_mode="host"
      assurance="off"
      runtime_provenance="host"
      ;;
    cuda)
      execution_mode="container"
      assurance="strict"
      runtime_provenance="container"
      device="cuda"
      ;;
    *)
      echo "Unknown lane: $lane" >&2
      usage >&2
      exit 2
      ;;
  esac
fi

if [[ "$execution_mode" == "host" && "$assurance" == "strict" ]]; then
  echo "Host execution requires --assurance off for this shared wrapper." >&2
  exit 2
fi

if [[ -z "$runtime_provenance" ]]; then
  runtime_provenance="$execution_mode"
fi

device="$(integration_default_host_device "$execution_mode" "$device")"
lane_artifact_label="$(
  integration_lane_artifact_label "$execution_mode" "$assurance" "$device"
)"
report_out="$(
  integration_lane_report_out "$report_out" "$report_out_was_default" "$lane_artifact_label"
)"
integration_log_header "InvarLock integration compare"
integration_log_kv "lane" "$lane_artifact_label"
integration_log_kv "execution_mode" "$execution_mode"
integration_log_kv "assurance" "$assurance"
integration_log_kv "runtime_provenance" "$runtime_provenance"
integration_log_kv "device" "$device"
integration_log_kv "report_out" "$report_out"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CLI=("$PYTHON_BIN" -m invarlock)

mkdir -p "$report_out"
report_json="$report_out/evaluation.report.json"
verify_json="$report_out/verify.json"
html_out="$report_out/evaluation.html"
backend_inventory_json="$report_out/backend_inventory.json"
lane_artifact_json="$report_out/lane_artifact.json"
run_command_txt="$report_out/run_command.txt"
run_summary_txt="$report_out/run_summary.txt"
rm -f "$report_json" "$verify_json" "$backend_inventory_json"
rm -f "$lane_artifact_json" "$run_summary_txt"
if [[ "$render_html" -eq 1 ]]; then
  rm -f "$html_out"
fi

run_complete=0

emit_verify_summary_fields() {
  local output_mode="$1"

  if [[ ! -s "$verify_json" ]]; then
    if [[ "$output_mode" == "terminal" ]]; then
      printf '  verify status: not available\n'
    else
      printf 'verify_status: not_available\n'
    fi
    return
  fi

  "$PYTHON_BIN" - "$verify_json" "$output_mode" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
output_mode = sys.argv[2]

try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:  # pragma: no cover - defensive shell UX path
    if output_mode == "terminal":
        print(f"  verify status: unreadable ({exc})")
    else:
        print("verify_status: unreadable")
        print(f"verify_error: {exc}")
    raise SystemExit(0)

summary = payload.get("summary") if isinstance(payload, dict) else {}
if not isinstance(summary, dict):
    summary = {}

ok = summary.get("ok")
if ok is True:
    verify_status = "ok"
elif ok is False:
    verify_status = "failed"
else:
    verify_status = "unknown"
verify_reason = summary.get("reason")

runtime = {}
for result in payload.get("results", []):
    if not isinstance(result, dict):
        continue
    verification = result.get("verification")
    if not isinstance(verification, dict):
        continue
    candidate = verification.get("runtime_provenance")
    if isinstance(candidate, dict):
        runtime = candidate
        break

declared = runtime.get("declared_mode")
runtime_status = runtime.get("status")
verified = runtime.get("verified")
issues = runtime.get("issues")

if output_mode == "terminal":
    if verify_reason and verify_reason != "ok":
        print(f"  verify status: {verify_status} ({verify_reason})")
    else:
        print(f"  verify status: {verify_status}")
    runtime_bits = []
    if runtime_status:
        runtime_bits.append(str(runtime_status))
    if declared:
        runtime_bits.append(f"declared={declared}")
    if verified is not None:
        runtime_bits.append(f"verified={str(bool(verified)).lower()}")
    if runtime_bits:
        print("  runtime provenance: " + ", ".join(runtime_bits))
else:
    print(f"verify_status: {verify_status}")
    if verify_reason:
        print(f"verify_reason: {verify_reason}")
    if declared:
        print(f"verify_runtime_provenance_declared: {declared}")
    if runtime_status:
        print(f"verify_runtime_provenance_status: {runtime_status}")
    if verified is not None:
        print(f"verify_runtime_provenance_verified: {str(bool(verified)).lower()}")
    if issues:
        print("verify_runtime_provenance_issues: " + json.dumps(issues, sort_keys=True))
PY
}

write_run_summary() {
  local status="$1"
  {
    printf 'status: %s\n' "$status"
    printf 'lane_artifact_label: %s\n' "$lane_artifact_label"
    printf 'execution_mode: %s\n' "$execution_mode"
    printf 'assurance: %s\n' "$assurance"
    printf 'runtime_provenance: %s\n' "$runtime_provenance"
    printf 'device: %s\n' "$device"
    printf 'report: %s\n' "$report_json"
    printf 'verify: %s\n' "$verify_json"
    emit_verify_summary_fields "file"
    if [[ "$render_html" -eq 1 ]]; then
      printf 'html: %s\n' "$html_out"
    fi
    printf 'lane_artifact: %s\n' "$lane_artifact_json"
    printf 'run_command: %s\n' "$run_command_txt"
  } > "$run_summary_txt"
}

print_success_summary() {
  cat <<MSG

InvarLock integration run complete
  status: success
  lane: $lane_artifact_label
  report: $report_json
  verify: $verify_json
MSG
  emit_verify_summary_fields "terminal"
  if [[ "$render_html" -eq 1 ]]; then
    printf '  html: %s\n' "$html_out"
  fi
  cat <<MSG
  lane artifact: $lane_artifact_json
  summary: $run_summary_txt
MSG
}

on_exit() {
  local rc=$?
  if [[ "$run_complete" -eq 0 && "$rc" -ne 0 ]]; then
    write_run_summary "failed" || true
    cat >&2 <<MSG

InvarLock integration run failed
  lane: $lane_artifact_label
  report out: $report_out
  command log: $run_command_txt
  summary: $run_summary_txt

Check the prerequisite message above first. If the failure happened during
evaluation or verification, inspect the concrete command recorded in
$run_command_txt.
MSG
  fi
}
trap on_exit EXIT

evaluate_cmd=(
  "${CLI[@]}" evaluate
  --baseline "$baseline"
  --subject "$subject"
  --baseline-adapter "$baseline_adapter"
  --subject-adapter "$subject_adapter"
  --profile "$profile"
  --tier "$tier"
  --report-out "$report_out"
  --execution-mode "$execution_mode"
  --assurance "$assurance"
)

if [[ "$allow_network" -eq 1 ]]; then
  evaluate_cmd+=(--allow-network)
fi
if [[ -n "$device" ]]; then
  evaluate_cmd+=(--device "$device")
fi
if [[ -n "$preset" ]]; then
  evaluate_cmd+=(--preset "$preset")
fi
if [[ -n "$edit_label" ]]; then
  evaluate_cmd+=(--edit-label "$edit_label")
fi

{
  printf 'lane_artifact_label: %s\n' "$lane_artifact_label"
  printf 'execution_mode: %s\n' "$execution_mode"
  printf 'assurance: %s\n' "$assurance"
  printf 'runtime_provenance: %s\n' "$runtime_provenance"
  printf 'device: %s\n' "$device"
  printf 'wrapper: '
  printf '%q ' "$0" "${original_args[@]}"
  printf '\n'
  printf 'evaluate: '
  printf '%q ' "${evaluate_cmd[@]}"
  printf '\n'
} > "$run_command_txt"

"$PYTHON_BIN" - "$lane_artifact_json" "$lane_artifact_label" "$lane" "$execution_mode" "$assurance" "$runtime_provenance" "$device" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "lane_artifact_label": sys.argv[2],
    "lane": sys.argv[3] or None,
    "execution_mode": sys.argv[4],
    "assurance": sys.argv[5],
    "runtime_provenance": sys.argv[6],
    "device": sys.argv[7],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

integration_log_step "evaluate baseline and subject"
"${evaluate_cmd[@]}"

if [[ ! -s "$report_json" ]]; then
  cat >&2 <<MSG
Evaluate completed but did not write the expected report:
  $report_json

Check --report-out path mapping and the evaluate command recorded in:
  $run_command_txt
MSG
  exit 1
fi

if [[ "$require_backend_inventory" -eq 1 && ! -s "$backend_inventory_json" ]]; then
  cat >&2 <<MSG
Evaluate completed but did not write the required backend inventory:
  $backend_inventory_json

This run requested --require-backend-inventory because the example documents
adapter provenance sidecars for this strict quantized lane. Check the selected
subject adapter and report persistence output.
MSG
  exit 1
fi

verify_cmd=(
  "${CLI[@]}" verify
  --json \
  --profile "$profile" \
  --assurance "$assurance" \
  --runtime-provenance "$runtime_provenance" \
  "$report_json"
)

printf 'verify: ' >> "$run_command_txt"
printf '%q ' "${verify_cmd[@]}" >> "$run_command_txt"
printf '> %q\n' "$verify_json" >> "$run_command_txt"

integration_log_step "verify evaluation report"
"${verify_cmd[@]}" > "$verify_json"

if [[ "$render_html" -eq 1 ]]; then
  html_cmd=(
    "${CLI[@]}" report html
    -i "$report_json"
    -o "$html_out"
    --force
  )
  printf 'html: ' >> "$run_command_txt"
  printf '%q ' "${html_cmd[@]}" >> "$run_command_txt"
  printf '\n' >> "$run_command_txt"
  integration_log_step "render HTML report"
  "${html_cmd[@]}"
else
  integration_log_step "skip HTML render (--no-html)"
fi

write_run_summary "success"
run_complete=1
print_success_summary
