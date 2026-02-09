#!/usr/bin/env bash
set -euo pipefail

# Remote runner: execute the locked INT8 quant_rtn sweep for one model.
#
# Usage:
#   run_quant_matrix.sh GPU_ID SLUG MODEL_ID
#
# This script assumes:
# - repo at /root/invarlock-public with an activated venv at .venv/
# - F4 canary preset at /root/verifai2/f4/presets/code_canary_256.yaml
# - per-edit overlays at /root/verifai2/f4/edits/f4_quant_rtn/*.yaml

GPU_ID="$1"
SLUG="$2"
MODEL_ID="$3"

PRESET="/root/verifai2/f4/presets/code_canary_256.yaml"
EDITS_DIR="/root/verifai2/f4/edits/f4_quant_rtn"
ROOT="/root/verifai2/f4/${SLUG}"
BASE_DIR="${ROOT}/baseline_ci200"

BASELINE_REPORT=$(ls -1 "${BASE_DIR}/runs/source"/*/report.json | sort | tail -n 1)
if [[ -z "${BASELINE_REPORT}" ]]; then
  echo "Missing baseline report under ${BASE_DIR}/runs/source" >&2
  exit 2
fi

echo "GPU_ID=${GPU_ID}"
echo "SLUG=${SLUG}"
echo "MODEL_ID=${MODEL_ID}"
echo "BASELINE_REPORT=${BASELINE_REPORT}"

action_edit() {
  local edit="$1"
  local overlay="${EDITS_DIR}/${edit}.yaml"
  local out="${ROOT}/${edit}"

  mkdir -p "${out}/runs" "${out}/reports" "${out}/logs"
  if [[ -s "${out}/reports/evaluation.report.json" ]]; then
    echo "[SKIP] ${SLUG} ${edit} (evaluation.report.json exists)"
    return 0
  fi

  echo "[EVAL] ${SLUG} ${edit}"

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
      --baseline-report "${BASELINE_REPORT}" \
      --edit-config "${overlay}" \
      --out "${out}/runs" \
      --report-out "${out}/reports" \
      --no-color --quiet \
      >"${out}/logs/evaluate.log" 2>&1

  # verify returns non-zero on policy_fail; capture JSON regardless.
  (PYTHONPATH=src python -m invarlock verify --profile ci --json "${out}/reports/evaluation.report.json" \
    >"${out}/reports/verify.ci.json") || true

  echo "[DONE] ${SLUG} ${edit}"
}

cd /root/invarlock-public
source .venv/bin/activate

# Sanity check: the baseline windows we are reusing must come from the same
# model family as the run we are about to execute.
#
# This prevents accidental cross-model baseline reuse (which can silently
# produce huge, meaningless ratios).
PYTHONPATH=src BASELINE_REPORT="${BASELINE_REPORT}" MODEL_ID="${MODEL_ID}" \
  python -c "import json, os; from pathlib import Path; d=json.loads(Path(os.environ[\"BASELINE_REPORT\"]).read_text()); mid=(d.get(\"meta\") or {}).get(\"model_id\"); want=os.environ[\"MODEL_ID\"]; assert mid == want, f\"baseline-report mismatch: {mid} != {want}\""

# Quant variants (locked in research/verifai2_2026/f4_matrix_final.md)
action_edit quant_rtn_int8_all_clamp0
action_edit quant_rtn_int8_all_clamp0p1
action_edit quant_rtn_int8_all_clamp0p25
action_edit quant_rtn_int8_all_clamp0p5
action_edit quant_rtn_int8_ffn_clamp0p25
action_edit quant_rtn_int8_attn_clamp0p25

echo ALL_TASKS_COMPLETE
