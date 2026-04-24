#!/usr/bin/env bash
# run_mini_pack_gate.sh - smaller confirmation gate before full evidence-pack reruns.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK_MINI_ORIG_CLEAN_EDIT_RUNS_SET="${CLEAN_EDIT_RUNS+x}"
PACK_MINI_ORIG_CLEAN_EDIT_RUNS="${CLEAN_EDIT_RUNS-}"
PACK_MINI_ORIG_STRESS_EDIT_RUNS_SET="${STRESS_EDIT_RUNS+x}"
PACK_MINI_ORIG_STRESS_EDIT_RUNS="${STRESS_EDIT_RUNS-}"
PACK_MINI_ORIG_RUN_ERROR_INJECTION_SET="${RUN_ERROR_INJECTION+x}"
PACK_MINI_ORIG_RUN_ERROR_INJECTION="${RUN_ERROR_INJECTION-}"
PACK_MINI_ORIG_DRIFT_CALIBRATION_RUNS_SET="${DRIFT_CALIBRATION_RUNS+x}"
PACK_MINI_ORIG_DRIFT_CALIBRATION_RUNS="${DRIFT_CALIBRATION_RUNS-}"
PACK_MINI_ORIG_PACK_USE_BATCH_EDITS_SET="${PACK_USE_BATCH_EDITS+x}"
PACK_MINI_ORIG_PACK_USE_BATCH_EDITS="${PACK_USE_BATCH_EDITS-}"

# shellcheck source=run_suite.sh
source "${SCRIPT_DIR}/run_suite.sh"

pack_mini_pack_usage() {
    cat <<'EOF'
InvarLock Evidence Pack Mini-Pack Gate
Usage: scripts/evidence_packs/run_mini_pack_gate.sh [options]

Runs a cheaper confirmation gate before a full reduced-pack rerun.

Options:
  --models CSV          Comma-separated model IDs to run (required)
  --gate NAME           Gate recipe (closure|targeted). Default: closure
  --scenario-ids IDS    Extra comma-separated scenario IDs to append
  --failed-verdict FILE Append failed scenario IDs from a prior final_verdict.json
  --manifest FILE       Scenario manifest to read (default: scripts/evidence_packs/scenarios.json)
  --net 1|0             Enable network access for preflight/downloads (default: 0)
  --out DIR             Output directory (default: ./evidence_pack_runs/mini_<gate>_<timestamp>)
  --determinism MODE    Determinism mode (strict|throughput)
  --repeats N           Determinism repeat count metadata (default: 0)
  --dry-run             Print selected scenario IDs and effective env, then exit
  --help                Show this help message

Gate recipes:
  closure   All clean controls, catastrophic-required stress lanes, and
            primary-guard-required error scenarios.
  targeted  Only the scenarios supplied via --scenario-ids and/or --failed-verdict.

Default mini-pack env overrides, unless already set by the caller:
  CLEAN_EDIT_RUNS=1
  STRESS_EDIT_RUNS=1
  RUN_ERROR_INJECTION=true
  DRIFT_CALIBRATION_RUNS=1
  PACK_USE_BATCH_EDITS=false
EOF
}

pack_mini_pack_select_scenarios() {
    local manifest_file="$1"
    local gate="$2"
    local extra_ids_csv="${3:-}"
    local failed_verdict_file="${4:-}"

    python3 - "${manifest_file}" "${gate}" "${extra_ids_csv}" "${failed_verdict_file}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
gate = sys.argv[2]
extra_ids_csv = sys.argv[3]
failed_verdict_path = Path(sys.argv[4]) if sys.argv[4] else None

if not manifest_path.is_file():
    raise SystemExit(f"manifest not found: {manifest_path}")

manifest = json.loads(manifest_path.read_text())
scenarios = manifest.get("scenarios") or []

selected = []
seen = set()

def add(sid: str) -> None:
    sid = (sid or "").strip()
    if not sid or sid in seen:
        return
    seen.add(sid)
    selected.append(sid)

if gate == "closure":
    for scenario in scenarios:
        req = scenario.get("requirements") or {}
        if (
            scenario.get("category") == "clean"
            or req.get("catastrophic_required") is True
            or req.get("primary_guard_required") is True
        ):
            add(scenario.get("id", ""))
elif gate == "targeted":
    pass
else:
    raise SystemExit(f"unsupported gate: {gate}")

if failed_verdict_path:
    if not failed_verdict_path.is_file():
        raise SystemExit(f"failed verdict not found: {failed_verdict_path}")
    verdict = json.loads(failed_verdict_path.read_text())
    for failed in verdict.get("failed_requirements") or []:
        add(failed.get("scenario", ""))

for raw in extra_ids_csv.split(","):
    add(raw)

print(",".join(selected))
PY
}

pack_mini_pack_apply_defaults() {
    if [[ -n "${PACK_MINI_ORIG_CLEAN_EDIT_RUNS_SET}" ]]; then
        CLEAN_EDIT_RUNS="${PACK_MINI_ORIG_CLEAN_EDIT_RUNS}"
    else
        CLEAN_EDIT_RUNS="1"
    fi
    if [[ -n "${PACK_MINI_ORIG_STRESS_EDIT_RUNS_SET}" ]]; then
        STRESS_EDIT_RUNS="${PACK_MINI_ORIG_STRESS_EDIT_RUNS}"
    else
        STRESS_EDIT_RUNS="1"
    fi
    if [[ -n "${PACK_MINI_ORIG_RUN_ERROR_INJECTION_SET}" ]]; then
        RUN_ERROR_INJECTION="${PACK_MINI_ORIG_RUN_ERROR_INJECTION}"
    else
        RUN_ERROR_INJECTION="true"
    fi
    if [[ -n "${PACK_MINI_ORIG_DRIFT_CALIBRATION_RUNS_SET}" ]]; then
        DRIFT_CALIBRATION_RUNS="${PACK_MINI_ORIG_DRIFT_CALIBRATION_RUNS}"
    else
        DRIFT_CALIBRATION_RUNS="1"
    fi
    if [[ -n "${PACK_MINI_ORIG_PACK_USE_BATCH_EDITS_SET}" ]]; then
        PACK_USE_BATCH_EDITS="${PACK_MINI_ORIG_PACK_USE_BATCH_EDITS}"
    else
        PACK_USE_BATCH_EDITS="false"
    fi
    export CLEAN_EDIT_RUNS STRESS_EDIT_RUNS RUN_ERROR_INJECTION DRIFT_CALIBRATION_RUNS PACK_USE_BATCH_EDITS
}

pack_mini_pack_entrypoint() {
    set -euo pipefail

    local gate="closure"
    local models_csv="${PACK_MODELS_CSV:-${PACK_MODELS:-}}"
    local manifest_file="${PACK_SCENARIOS_MANIFEST_FILE:-${SCRIPT_DIR}/scenarios.json}"
    local extra_ids_csv="${PACK_SCENARIO_IDS:-}"
    local failed_verdict_file="${PACK_FAILED_VERDICT_FILE:-}"
    local net="${PACK_NET:-0}"
    local out="${PACK_OUTPUT_DIR:-${OUTPUT_DIR:-}}"
    local determinism="${PACK_DETERMINISM:-throughput}"
    local repeats="${PACK_REPEATS:-0}"
    local dry_run="false"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                pack_mini_pack_usage
                return 0
                ;;
            --models)
                models_csv="${2:-}"
                [[ -n "${models_csv}" ]] || { echo "ERROR: --models requires a value" >&2; return 2; }
                shift 2
                ;;
            --gate)
                gate="${2:-}"
                [[ -n "${gate}" ]] || { echo "ERROR: --gate requires a value" >&2; return 2; }
                shift 2
                ;;
            --scenario-ids)
                extra_ids_csv="${2:-}"
                [[ -n "${extra_ids_csv}" ]] || { echo "ERROR: --scenario-ids requires a value" >&2; return 2; }
                shift 2
                ;;
            --failed-verdict)
                failed_verdict_file="${2:-}"
                [[ -n "${failed_verdict_file}" ]] || { echo "ERROR: --failed-verdict requires a value" >&2; return 2; }
                shift 2
                ;;
            --manifest)
                manifest_file="${2:-}"
                [[ -n "${manifest_file}" ]] || { echo "ERROR: --manifest requires a value" >&2; return 2; }
                shift 2
                ;;
            --net)
                net="${2:-}"
                [[ -n "${net}" ]] || { echo "ERROR: --net requires 1 or 0" >&2; return 2; }
                shift 2
                ;;
            --out)
                out="${2:-}"
                [[ -n "${out}" ]] || { echo "ERROR: --out requires a value" >&2; return 2; }
                shift 2
                ;;
            --determinism)
                determinism="${2:-}"
                [[ -n "${determinism}" ]] || { echo "ERROR: --determinism requires a value" >&2; return 2; }
                shift 2
                ;;
            --repeats)
                repeats="${2:-}"
                [[ -n "${repeats}" && "${repeats}" =~ ^[0-9]+$ ]] || { echo "ERROR: --repeats requires an integer" >&2; return 2; }
                shift 2
                ;;
            --dry-run)
                dry_run="true"
                shift
                ;;
            *)
                echo "Unknown arg: $1" >&2
                pack_mini_pack_usage >&2
                return 2
                ;;
        esac
    done

    [[ -n "${models_csv}" ]] || { echo "ERROR: --models is required" >&2; return 2; }

    if [[ -z "${out}" ]]; then
        local stamp
        stamp="$(date -u +%Y%m%d_%H%M%S)"
        out="./evidence_pack_runs/mini_${gate}_${stamp}"
    fi

    local scenario_ids
    scenario_ids="$(pack_mini_pack_select_scenarios "${manifest_file}" "${gate}" "${extra_ids_csv}" "${failed_verdict_file}")"
    [[ -n "${scenario_ids}" ]] || { echo "ERROR: mini-pack resolved no scenarios" >&2; return 2; }

    pack_mini_pack_apply_defaults

    if [[ "${dry_run}" == "true" ]]; then
        printf 'gate=%s\n' "${gate}"
        printf 'models=%s\n' "${models_csv}"
        printf 'scenario_ids=%s\n' "${scenario_ids}"
        printf 'env=CLEAN_EDIT_RUNS=%s STRESS_EDIT_RUNS=%s RUN_ERROR_INJECTION=%s DRIFT_CALIBRATION_RUNS=%s PACK_USE_BATCH_EDITS=%s\n' \
            "${CLEAN_EDIT_RUNS}" "${STRESS_EDIT_RUNS}" "${RUN_ERROR_INJECTION}" "${DRIFT_CALIBRATION_RUNS}" "${PACK_USE_BATCH_EDITS}"
        return 0
    fi

    export PACK_SCENARIOS_MANIFEST_FILE="${manifest_file}"

    pack_entrypoint \
        --suite subset \
        --models "${models_csv}" \
        --net "${net}" \
        --out "${out}" \
        --determinism "${determinism}" \
        --repeats "${repeats}" \
        --scenario-ids "${scenario_ids}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    pack_mini_pack_entrypoint "$@"
fi
