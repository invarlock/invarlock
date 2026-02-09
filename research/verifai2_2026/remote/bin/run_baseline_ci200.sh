#!/usr/bin/env bash
set -euo pipefail

# Remote runner: generate a clean per-model baseline_ci200 with stored windows.
#
# Usage:
#   run_baseline_ci200.sh GPU_ID SLUG MODEL_ID
#
# Produces:
# - /root/verifai2/f4/$SLUG/baseline_ci200/runs/{source,edited}/.../report.json
# - /root/verifai2/f4/$SLUG/baseline_ci200/reports/evaluation.report.json
# - /root/verifai2/f4/$SLUG/baseline_ci200/reports/verify.ci.json
#
# Critical invariant:
# - baseline_ref.model_id MUST equal MODEL_ID and ratio_vs_baseline MUST be 1.0.
#
# This prevents accidental cross-model baseline reuse, which can silently corrupt
# all downstream evaluation reports.

GPU_ID="$1"
SLUG="$2"
MODEL_ID="$3"

PRESET="/root/verifai2/f4/presets/code_canary_256.yaml"
ROOT="/root/verifai2/f4/${SLUG}"
OUT="${ROOT}/baseline_ci200"

mkdir -p "${OUT}/runs" "${OUT}/reports" "${OUT}/logs"

cd /root/invarlock-public
source .venv/bin/activate

echo "GPU_ID=${GPU_ID}"
echo "SLUG=${SLUG}"
echo "MODEL_ID=${MODEL_ID}"
echo "OUT=${OUT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  INVARLOCK_ALLOW_NETWORK=1 \
  HF_HUB_DISABLE_PROGRESS_BARS=1 \
  TOKENIZERS_PARALLELISM=false \
  PYTHONPATH=src \
  python -m invarlock evaluate \
    --source "${MODEL_ID}" \
    --edited "${MODEL_ID}" \
    --adapter hf_causal \
    --profile ci --tier balanced \
    --preset "${PRESET}" \
    --out "${OUT}/runs" \
    --report-out "${OUT}/reports" \
    --no-color --quiet \
    >"${OUT}/logs/evaluate_baseline.log" 2>&1

(PYTHONPATH=src python -m invarlock verify --profile ci --json "${OUT}/reports/evaluation.report.json" \
  >"${OUT}/reports/verify.ci.json") || true

# Enforce baseline sanity checks for this pipeline.
PYTHONPATH=src MODEL_ID="${MODEL_ID}" \
  python -c "import json, os; from pathlib import Path; p=Path(\"${OUT}/reports/evaluation.report.json\"); d=json.loads(p.read_text()); meta=(d.get(\"meta\") or {}).get(\"model_id\"); bref=(d.get(\"baseline_ref\") or {}).get(\"model_id\"); ratio=(d.get(\"primary_metric\") or {}).get(\"ratio_vs_baseline\"); want=os.environ[\"MODEL_ID\"]; assert meta == want, f\"meta.model_id mismatch: {meta} != {want}\"; assert bref == want, f\"baseline_ref.model_id mismatch: {bref} != {want}\"; assert float(ratio) == 1.0, f\"baseline ratio != 1.0: {ratio}\""

echo ALL_TASKS_COMPLETE
