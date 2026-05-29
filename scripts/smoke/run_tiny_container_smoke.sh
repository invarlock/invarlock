#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

WORK_ROOT="${1:-$(mktemp -d -t invarlock_tiny_container_smoke.XXXXXX)}"
MODEL_ID="${INVARLOCK_TINY_SMOKE_MODEL_ID:-sshleifer/tiny-gpt2}"
MODE="${INVARLOCK_SMOKE_MODE:-container}"
PROFILE="${INVARLOCK_SMOKE_PROFILE:-dev}"
SMOKE_DEVICE="${INVARLOCK_SMOKE_DEVICE:-auto}"

PYTHON_BIN="${INVARLOCK_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(bash "$REPO_ROOT/scripts/select_workspace_python.sh")"
fi
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CLI=("$PYTHON_BIN" -m invarlock)

export INVARLOCK_ALLOW_NETWORK="${INVARLOCK_ALLOW_NETWORK:-1}"
export INVARLOCK_DEDUP_TEXTS="${INVARLOCK_DEDUP_TEXTS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

host_gpu_visible() {
  [[ -e /dev/nvidiactl ]] || command -v nvidia-smi >/dev/null 2>&1
}

seed_local_runtime_image() {
  if host_gpu_visible && docker image inspect invarlock-runtime:cuda-local >/dev/null 2>&1; then
    export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:cuda-local"
    return 0
  fi
  if docker image inspect invarlock-runtime:local >/dev/null 2>&1; then
    export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:local"
  fi
}

if [[ "$MODE" == "container" && -z "${INVARLOCK_RUNTIME_IMAGE:-}" ]]; then
  seed_local_runtime_image
fi

mkdir -p "$WORK_ROOT"

SMOKE_RUN_DIR="$WORK_ROOT/runs"
SMOKE_REPORT_DIR="$WORK_ROOT/reports/eval"
SMOKE_EXPORT_DIR="$WORK_ROOT/exports"
EVIDENCE_PACK_DIR="$WORK_ROOT/evidence_pack"
SMOKE_CACHE_ROOT="$WORK_ROOT/.hf"
HOST_HF_CACHE_ROOT="${INVARLOCK_SMOKE_HOST_HF_CACHE_ROOT:-${HF_HOME:-${HOME}/.cache/huggingface}}"
DATA_FILE="$WORK_ROOT/smoke.jsonl"
PRESET_PATH="$WORK_ROOT/tiny_smoke_preset.yaml"

copy_cached_tree_if_present() {
  local source_dir="$1"
  local target_dir="$2"
  if [[ -d "$source_dir" && ! -e "$target_dir" ]]; then
    mkdir -p "$(dirname -- "$target_dir")"
    cp -a "$source_dir" "$target_dir"
  fi
}

seed_hf_cache_from_host() {
  copy_cached_tree_if_present \
    "$HOST_HF_CACHE_ROOT/hub/models--sshleifer--tiny-gpt2" \
    "$SMOKE_CACHE_ROOT/hub/models--sshleifer--tiny-gpt2"
  if [[ -d "$SMOKE_CACHE_ROOT/hub/models--sshleifer--tiny-gpt2" ]]; then
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
  echo "[smoke] prefetching tiny GPT-2 into host HF cache"
  HF_HOME="$HOST_HF_CACHE_ROOT" \
    HF_HUB_CACHE="$HOST_HF_CACHE_ROOT/hub" \
    "$PYTHON_BIN" - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "sshleifer/tiny-gpt2"

AutoTokenizer.from_pretrained(MODEL_ID)
AutoModelForCausalLM.from_pretrained(MODEL_ID)
PY
}

resolve_container_engine() {
  if command -v docker >/dev/null 2>&1; then
    echo "docker"
    return 0
  fi
  if command -v podman >/dev/null 2>&1; then
    echo "podman"
    return 0
  fi
  return 1
}

ensure_current_runtime_image() {
  if [[ "$MODE" != "container" ]]; then
    return 0
  fi
  if [[ -n "${INVARLOCK_RUNTIME_IMAGE:-}" \
    && "${INVARLOCK_RUNTIME_IMAGE}" != "invarlock-runtime:local" \
    && "${INVARLOCK_RUNTIME_IMAGE}" != "invarlock-runtime:cuda-local" ]]; then
    return 0
  fi
  if ! command -v docker >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
    return 0
  fi
  if host_gpu_visible; then
    echo "[smoke] refreshing local CUDA container runtime image"
    make runtime-image-cuda
    export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:cuda-local"
    return 0
  fi
  echo "[smoke] refreshing local container runtime image"
  make runtime-image
  export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:local"
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
  engine="$(resolve_container_engine || true)"
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
    "$PYTHON_BIN" -m invarlock.cli.runtime_verify \
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

ensure_writable_hf_cache() {
  local candidate_home="${HF_HOME:-$SMOKE_CACHE_ROOT}"
  local candidate_datasets="${HF_DATASETS_CACHE:-${candidate_home}/datasets}"
  local candidate_hub="${HF_HUB_CACHE:-${candidate_home}/hub}"

  if mkdir -p "$candidate_home" "$candidate_datasets" "$candidate_hub" >/dev/null 2>&1 \
    && touch "$candidate_datasets/.ivl_smoke_probe" >/dev/null 2>&1; then
    rm -f "$candidate_datasets/.ivl_smoke_probe" >/dev/null 2>&1 || true
    export HF_HOME="$candidate_home"
    export HF_HUB_CACHE="$candidate_hub"
    export HF_DATASETS_CACHE="$candidate_datasets"
    unset TRANSFORMERS_CACHE
    return 0
  fi

  export HF_HOME="$SMOKE_CACHE_ROOT"
  export HF_HUB_CACHE="$SMOKE_CACHE_ROOT/hub"
  export HF_DATASETS_CACHE="$SMOKE_CACHE_ROOT/datasets"
  unset TRANSFORMERS_CACHE
  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
  echo "[smoke] falling back to writable HF cache under $HF_HOME"
}

ensure_writable_hf_cache
ensure_current_runtime_image
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

"${CLI[@]}" verify "$EVAL_REPORT" "${VERIFY_ARGS[@]}" --profile "$PROFILE" --assurance off --json || VERIFY_RC=$?
VERIFY_RC="${VERIFY_RC:-0}"
echo "[smoke] verify_rc=$VERIFY_RC"
if [[ "$VERIFY_RC" != "0" ]]; then
  debug_verify_failure
  echo "[error] evaluation report verification failed" >&2
  exit "$VERIFY_RC"
fi
assert_semantic_pass "$EVAL_REPORT"
"${CLI[@]}" report validate "$EVAL_REPORT"
mkdir -p "$SMOKE_EXPORT_DIR"
"${CLI[@]}" report html -i "$EVAL_REPORT" -o "$SMOKE_EXPORT_DIR/evaluation.html"
"${CLI[@]}" report explain --subject-report "$EDITED_REPORT" --baseline-report "$BASELINE_REPORT"

printf '%s\n' '{"verdict":"PASS","note":"tiny container smoke campaign"}' > "$WORK_ROOT/final_verdict.json"
EVIDENCE_PACK_SIGNING_KEY="$WORK_ROOT/evidence_pack_signing_key.pem"
EVIDENCE_PACK_PUBLIC_KEY="$WORK_ROOT/evidence_pack_signing_key.pub.pem"

if [[ "$MODE" == "local" ]]; then
  echo "[smoke] skipping evidence-pack build/verify in local mode; emitted artifacts are host-bypass."
  echo "[smoke] complete"
  exit 0
fi

"${CLI[@]}" advanced evidence-pack keygen "$EVIDENCE_PACK_SIGNING_KEY" \
  --public-key-out "$EVIDENCE_PACK_PUBLIC_KEY" \
  --json
"${CLI[@]}" advanced evidence-pack build "$EVIDENCE_PACK_DIR" \
  --final-verdict "$WORK_ROOT/final_verdict.json" \
  --report "$EVAL_REPORT" \
  --signing-key "$EVIDENCE_PACK_SIGNING_KEY" \
  --profile "$PROFILE" \
  --json
"${CLI[@]}" advanced evidence-pack inspect "$EVIDENCE_PACK_DIR" --json
"${CLI[@]}" advanced evidence-pack verify "$EVIDENCE_PACK_DIR" --json || EVIDENCE_PACK_VERIFY_RC=$?
EVIDENCE_PACK_VERIFY_RC="${EVIDENCE_PACK_VERIFY_RC:-0}"
echo "[smoke] evidence_pack_verify_rc=$EVIDENCE_PACK_VERIFY_RC"
if [[ "$EVIDENCE_PACK_VERIFY_RC" != "0" ]]; then
  echo "[error] evidence-pack verification failed" >&2
  exit "$EVIDENCE_PACK_VERIFY_RC"
fi

echo "[smoke] complete"
