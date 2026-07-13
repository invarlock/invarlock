#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib/smoke_common.sh"

WORK_ROOT="${1:-$(mktemp -d -t invarlock_tiny_container_smoke.XXXXXX)}"
MODEL_ID="${INVARLOCK_TINY_SMOKE_MODEL_ID:-sshleifer/tiny-gpt2}"
MODEL_CACHE_NAME="models--${MODEL_ID//\//--}"
MODE="${INVARLOCK_SMOKE_MODE:-container}"
PROFILE="${INVARLOCK_SMOKE_PROFILE:-dev}"
SMOKE_DEVICE="${INVARLOCK_SMOKE_DEVICE:-auto}"

PYTHON_BIN="$(smoke_select_python "$REPO_ROOT" "${INVARLOCK_PYTHON:-}")"
smoke_setup_pythonpath "$REPO_ROOT"
CLI=("$PYTHON_BIN" -m invarlock)

export INVARLOCK_ALLOW_NETWORK="${INVARLOCK_ALLOW_NETWORK:-1}"
export INVARLOCK_DEDUP_TEXTS="${INVARLOCK_DEDUP_TEXTS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${INVARLOCK_SMOKE_PLAN:-0}" == "1" ]]; then
  SMOKE_PLAN_WORK_ROOT="$WORK_ROOT" \
  SMOKE_PLAN_MODEL_ID="$MODEL_ID" \
  SMOKE_PLAN_MODEL_CACHE_NAME="$MODEL_CACHE_NAME" \
  SMOKE_PLAN_MODE="$MODE" \
  SMOKE_PLAN_PROFILE="$PROFILE" \
  SMOKE_PLAN_DEVICE="$SMOKE_DEVICE" \
  SMOKE_PLAN_COMMANDS="$(smoke_plan_markers command "${BASH_SOURCE[0]}")" \
  "$PYTHON_BIN" - <<'PY'
import json
import os

mode = os.environ["SMOKE_PLAN_MODE"]
commands = [
    command.strip()
    for command in os.environ["SMOKE_PLAN_COMMANDS"].splitlines()
    if command.strip()
]
plan = {
    "script": "run_tiny_container_smoke",
    "work_root": os.environ["SMOKE_PLAN_WORK_ROOT"],
    "model_id": os.environ["SMOKE_PLAN_MODEL_ID"],
    "model_cache_name": os.environ["SMOKE_PLAN_MODEL_CACHE_NAME"],
    "mode": mode,
    "profile": os.environ["SMOKE_PLAN_PROFILE"],
    "device": os.environ["SMOKE_PLAN_DEVICE"],
    "runtime_provenance": "host" if mode == "local" else "container",
    "dataset": {
        "provider": {"kind": "local_jsonl"},
        "seq_len": 16,
        "preview_n": 2,
        "final_n": 2,
        "tiny_relax": True,
    },
    "commands": commands,
    "runtime_image": {
        "seed_local_image": mode == "container",
        "seed_digest": mode == "container",
    },
}
print(json.dumps(plan, sort_keys=True))
PY
  exit 0
fi

if [[ "$MODE" == "container" && -z "${INVARLOCK_RUNTIME_IMAGE:-}" ]]; then
  smoke_seed_local_runtime_image "$SMOKE_DEVICE"
fi

mkdir -p "$WORK_ROOT"

SMOKE_RUN_DIR="$WORK_ROOT/runs"
SMOKE_REPORT_DIR="$WORK_ROOT/reports/eval"
SMOKE_EXPORT_DIR="$WORK_ROOT/exports"
SMOKE_CACHE_ROOT="$WORK_ROOT/.hf"
HOST_HF_CACHE_ROOT="${INVARLOCK_SMOKE_HOST_HF_CACHE_ROOT:-${HF_HOME:-${HOME}/.cache/huggingface}}"
DATA_FILE="$WORK_ROOT/smoke.jsonl"
PRESET_PATH="$WORK_ROOT/tiny_smoke_preset.yaml"

seed_hf_cache_from_host() {
  smoke_copy_cached_tree_if_present \
    "$HOST_HF_CACHE_ROOT/hub/$MODEL_CACHE_NAME" \
    "$SMOKE_CACHE_ROOT/hub/$MODEL_CACHE_NAME"
  if [[ -d "$SMOKE_CACHE_ROOT/hub/$MODEL_CACHE_NAME" ]]; then
    export INVARLOCK_SMOKE_CACHE_COMPLETE=1
    return 0
  fi
  return 1
}

prefetch_tiny_model_on_host() {
  if [[ "${INVARLOCK_ALLOW_NETWORK:-0}" != "1" ]]; then
    return 1
  fi
  mkdir -p "$HOST_HF_CACHE_ROOT" "$HOST_HF_CACHE_ROOT/hub"
  echo "[smoke] prefetching $MODEL_ID into host HF cache"
  MODEL_ID="$MODEL_ID" \
    HF_HOME="$HOST_HF_CACHE_ROOT" \
    HF_HUB_CACHE="$HOST_HF_CACHE_ROOT/hub" \
    "$PYTHON_BIN" - <<'PY'
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ["MODEL_ID"]

AutoTokenizer.from_pretrained(MODEL_ID)
AutoModelForCausalLM.from_pretrained(MODEL_ID)
PY
}

seed_runtime_image_digest() {
  if [[ "$MODE" != "container" ]]; then
    return 0
  fi
  if [[ -n "${INVARLOCK_RUNTIME_IMAGE_DIGEST:-}" ]]; then
    return 0
  fi
  if [[ -z "${INVARLOCK_RUNTIME_IMAGE:-}" ]]; then
    return 0
  fi
  local engine
  engine="$(smoke_resolve_container_engine || true)"
  if [[ -z "$engine" ]]; then
    return 0
  fi
  local digest
  digest="$("$engine" image inspect "$INVARLOCK_RUNTIME_IMAGE" --format '{{.Id}}' 2>/dev/null || true)"
  if [[ "$digest" == sha256:* ]]; then
    export INVARLOCK_RUNTIME_IMAGE_DIGEST="$digest"
    echo "[smoke] runtime_image_digest=$INVARLOCK_RUNTIME_IMAGE_DIGEST"
  fi
}

debug_verify_failure() {
  local manifest_path="$SMOKE_REPORT_DIR/runtime.manifest.json"
  if [[ -f "$manifest_path" ]]; then
    echo "[smoke] runtime_verify_diagnostics"
    "$PYTHON_BIN" -m invarlock advanced runtime-verify \
      --report "$EVAL_REPORT" \
      --manifest "$manifest_path" \
      --json || true
  fi
  echo "[smoke] verify_plain_diagnostics"
  "${CLI[@]}" verify "$EVAL_REPORT" "${VERIFY_ARGS[@]}" --profile "$PROFILE" || true
}

assert_semantic_pass() {
  local report_path="$1"
  REPORT_PATH="$report_path" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from invarlock.reporting.report_policy import resolve_tiny_relax_from_report

report = json.loads(Path(os.environ["REPORT_PATH"]).read_text(encoding="utf-8"))
if not resolve_tiny_relax_from_report(report):
    raise SystemExit("tiny smoke report is missing tiny_relax provenance")

validation = report.get("validation") or {}
required_true = (
    "primary_metric_acceptable",
    "preview_final_drift_acceptable",
    "invariants_pass",
    "spectral_stable",
    "rmt_stable",
)
for key in required_true:
    if validation.get(key) is not True:
        raise SystemExit(f"expected validation.{key} to be true, found {validation.get(key)!r}")
PY
}

if seed_hf_cache_from_host; then
  export INVARLOCK_SMOKE_CACHE_SEEDED=1
elif prefetch_tiny_model_on_host && seed_hf_cache_from_host; then
  export INVARLOCK_SMOKE_CACHE_SEEDED=1
fi

smoke_ensure_writable_hf_cache "$SMOKE_CACHE_ROOT"
smoke_ensure_current_runtime_image "$MODE" "$SMOKE_DEVICE"
seed_runtime_image_digest

if [[ "${INVARLOCK_SMOKE_CACHE_COMPLETE:-0}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

cat >"$DATA_FILE" <<'EOF'
{"text":"tiny container smoke sample one"}
{"text":"tiny container smoke sample two"}
{"text":"tiny container smoke sample three"}
{"text":"tiny container smoke sample four"}
EOF

cat >"$PRESET_PATH" <<EOF
dataset:
  provider:
    kind: local_jsonl
  file: $DATA_FILE
  split: validation
  seq_len: 16
  stride: 16
  preview_n: 2
  final_n: 2
  seed: 42
guards:
  order: []
eval:
  metric: {kind: ppl_causal}
  loss: {type: auto}
context:
  run:
    tiny_relax: true
  eval:
    tiny_relax: true
EOF

echo "[smoke] work_root=$WORK_ROOT"
echo "[smoke] model_id=$MODEL_ID"
echo "[smoke] preset=$PRESET_PATH"
echo "[smoke] mode=$MODE profile=$PROFILE"
echo "[smoke] device=$SMOKE_DEVICE"
echo "[smoke] hf_home=$HF_HOME"
echo "[smoke] hf_hub_cache=$HF_HUB_CACHE"

EXECUTION_MODE="container"
RUNTIME_PROVENANCE="container"
if [[ "$MODE" == "local" ]]; then
  EXECUTION_MODE="host"
  RUNTIME_PROVENANCE="host"
fi

# smoke-plan-command: evaluate
"${CLI[@]}" evaluate \
  --baseline "$MODEL_ID" \
  --subject "$MODEL_ID" \
  --baseline-adapter hf_causal --subject-adapter hf_causal \
  --profile "$PROFILE" \
  --assurance off \
  --preset "$PRESET_PATH" \
  --execution-mode "$EXECUTION_MODE" \
  --device "$SMOKE_DEVICE" \
  --out "$SMOKE_RUN_DIR" \
  --report-out "$SMOKE_REPORT_DIR" \
  --timing

BASELINE_REPORT="$(find "$SMOKE_RUN_DIR/source" -name report.json -print | sort | tail -n1)"
EDITED_REPORT="$(find "$SMOKE_RUN_DIR/edited" -name report.json -print | sort | tail -n1)"
EVAL_REPORT="$SMOKE_REPORT_DIR/evaluation.report.json"

if [[ -z "$BASELINE_REPORT" || -z "$EDITED_REPORT" ]]; then
  echo "[error] could not locate run reports under $SMOKE_RUN_DIR" >&2
  exit 1
fi

echo "[smoke] baseline_report=$BASELINE_REPORT"
echo "[smoke] edited_report=$EDITED_REPORT"
echo "[smoke] evaluation_report=$EVAL_REPORT"

VERIFY_ARGS=(--runtime-provenance "$RUNTIME_PROVENANCE")

VERIFY_RC=0
# smoke-plan-command: verify
"${CLI[@]}" verify "$EVAL_REPORT" "${VERIFY_ARGS[@]}" --profile "$PROFILE" --assurance off --json || VERIFY_RC=$?
VERIFY_RC="${VERIFY_RC:-0}"
echo "[smoke] verify_rc=$VERIFY_RC"
if [[ "$VERIFY_RC" != "0" ]]; then
  debug_verify_failure
  echo "[error] evaluation report verification failed" >&2
  exit "$VERIFY_RC"
fi
assert_semantic_pass "$EVAL_REPORT"
# smoke-plan-command: report validate
"${CLI[@]}" report validate "$EVAL_REPORT"
mkdir -p "$SMOKE_EXPORT_DIR"
# smoke-plan-command: report html
"${CLI[@]}" report html -i "$EVAL_REPORT" -o "$SMOKE_EXPORT_DIR/evaluation.html"
# smoke-plan-command: report explain
"${CLI[@]}" report explain --subject-report "$EDITED_REPORT" --baseline-report "$BASELINE_REPORT"

echo "[smoke] complete"
