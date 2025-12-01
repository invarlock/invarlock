#!/usr/bin/env bash

# Lightweight CPU-only telemetry sweep for CI profile edits.
# Produces certs under reports/telemetry/cpu-ci with latency/memory metrics.

set -euo pipefail

if ! command -v invarlock >/dev/null 2>&1; then
  echo "ERROR: invarlock CLI not found in PATH. Activate the invarlock environment first." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${ROOT}/reports/telemetry/cpu-ci"
mkdir -p "${OUT_ROOT}"

echo "=== CPU telemetry sweep (GPT-2 small quant8) ==="

# Baseline (CPU)
BASE_RUN="${ROOT}/runs/telemetry_cpu/baseline"
invarlock run -c configs/tasks/causal_lm/ci_cpu.yaml --profile ci_cpu --out "${BASE_RUN}"
BASE_REPORT="$(ls -t "${BASE_RUN}"/*/report.json | head -n1)"

# Quant8 (CPU)
QUANT_RUN="${ROOT}/runs/telemetry_cpu/quant8"
invarlock run -c configs/edits/quant_rtn/8bit_attn.yaml --profile ci_cpu --baseline "${BASE_REPORT}" --out "${QUANT_RUN}"
QUANT_LATEST="$(ls -t "${QUANT_RUN}"/*/report.json | head -n1)"
invarlock report --run "$(dirname "${QUANT_LATEST}")" --baseline "${BASE_REPORT}" --format cert --output "${OUT_ROOT}/quant8" >/dev/null
invarlock verify "${OUT_ROOT}/quant8/evaluation.cert.json" >/dev/null

echo "Telemetry certs written to ${OUT_ROOT}"
