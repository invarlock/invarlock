#!/usr/bin/env bash

# Lightweight CPU-only telemetry sweep for CI profile edits.
# Produces attested reports under reports/telemetry/cpu-ci with latency/memory metrics.

set -euo pipefail

if ! command -v invarlock >/dev/null 2>&1; then
  echo "ERROR: invarlock CLI not found in PATH. Activate the invarlock environment first." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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

INVARLOCK_ALLOW_NETWORK=1 invarlock evaluate \
  --baseline "${MODEL_ID}" \
  --subject "${MODEL_ID}" \
  --adapter auto \
  --profile "${PROFILE}" \
  --tier "${TIER}" \
  --preset "${PRESET}" \
  --edit-config "${EDIT_CFG}" \
  --out "${RUN_ROOT}" \
  --report-out "${REPORT_ROOT}" >/dev/null

invarlock verify "${REPORT_ROOT}/evaluation.report.json" >/dev/null

echo "Telemetry reports written to ${REPORT_ROOT}"
