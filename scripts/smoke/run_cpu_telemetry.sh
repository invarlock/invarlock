#!/usr/bin/env bash

# Lightweight CPU-only telemetry sweep for CI profile edits.
# Produces container-backed reports under reports/telemetry/cpu-ci with latency/memory metrics.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib/smoke_common.sh"
cd "$ROOT"

PYTHON_BIN="$(smoke_select_python "$ROOT" "${INVARLOCK_PYTHON:-}")"
smoke_setup_pythonpath "$ROOT"
CLI=("$PYTHON_BIN" -m invarlock)

OUT_ROOT="${ROOT}/reports/telemetry/cpu-ci"
mkdir -p "${OUT_ROOT}"

echo "=== CPU telemetry sweep (quant8 attention) ==="

# Defaults (override via env if desired)
MODEL_ID="${MODEL_ID:-sshleifer/tiny-gpt2}"
PROFILE="${PROFILE:-ci_cpu}"
TIER="${TIER:-balanced}"
PRESET="${PRESET:-configs/presets/causal_lm/wikitext2_512.yaml}"
EDIT_CFG="${EDIT_CFG:-configs/overlays/edits/quant_rtn/8bit_attn.yaml}"

RUN_ROOT="${ROOT}/runs/telemetry_cpu/quant8"
REPORT_ROOT="${OUT_ROOT}/quant8"

smoke_seed_local_runtime_image "cpu"
smoke_ensure_current_runtime_image "container" "cpu"

set +e
INVARLOCK_ALLOW_NETWORK=1 "${CLI[@]}" evaluate \
  --baseline "${MODEL_ID}" \
  --subject "${MODEL_ID}" \
  --baseline-adapter auto --subject-adapter auto \
  --profile "${PROFILE}" \
  --assurance off \
  --tier "${TIER}" \
  --device cpu \
  --preset "${PRESET}" \
  --edit-config "${EDIT_CFG}" \
  --out "${RUN_ROOT}" \
  --report-out "${REPORT_ROOT}" >/dev/null
EVAL_RC=$?
set -e

if [[ "${EVAL_RC}" != "0" ]]; then
  exit "${EVAL_RC}"
fi

"${CLI[@]}" report validate "${REPORT_ROOT}/evaluation.report.json" >/dev/null

echo "Telemetry reports written to ${REPORT_ROOT}"
