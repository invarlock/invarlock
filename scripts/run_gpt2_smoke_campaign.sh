#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

WORK_ROOT="${1:-$(mktemp -d -t invarlock_gpt2_smoke.XXXXXX)}"
PRESET="${INVARLOCK_GPT2_SMOKE_PRESET:-$REPO_ROOT/configs/presets/causal_lm/gpt2_smoke_128.yaml}"
MODE="${INVARLOCK_SMOKE_MODE:-local}"
PROFILE="${INVARLOCK_SMOKE_PROFILE:-dev}"

if [[ ! -f "$PRESET" ]]; then
  echo "[error] GPT-2 smoke preset not found: $PRESET" >&2
  exit 2
fi

PYTHON_BIN="${INVARLOCK_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(bash "$REPO_ROOT/scripts/select_python.sh")"
fi
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CLI=("$PYTHON_BIN" -m invarlock)

export INVARLOCK_ALLOW_NETWORK="${INVARLOCK_ALLOW_NETWORK:-1}"
export INVARLOCK_DEDUP_TEXTS="${INVARLOCK_DEDUP_TEXTS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "$MODE" == "attested" && -z "${INVARLOCK_RUNTIME_IMAGE:-}" ]]; then
  if docker image inspect invarlock-runtime:local >/dev/null 2>&1; then
    export INVARLOCK_RUNTIME_IMAGE="invarlock-runtime:local"
  fi
fi

mkdir -p "$WORK_ROOT"

SMOKE_RUN_DIR="$WORK_ROOT/runs"
SMOKE_REPORT_DIR="$WORK_ROOT/reports/eval"
PROOF_PACK_DIR="$WORK_ROOT/proof_pack"
SMOKE_CACHE_ROOT="$WORK_ROOT/.hf"
HOST_HF_CACHE_ROOT="${INVARLOCK_SMOKE_HOST_HF_CACHE_ROOT:-${HF_HOME:-${HOME}/.cache/huggingface}}"

copy_cached_tree_if_present() {
  local source_dir="$1"
  local target_dir="$2"
  if [[ -d "$source_dir" && ! -e "$target_dir" ]]; then
    mkdir -p "$(dirname -- "$target_dir")"
    cp -a "$source_dir" "$target_dir"
  fi
}

seed_hf_cache_from_host() {
  local seeded=0
  copy_cached_tree_if_present "$HOST_HF_CACHE_ROOT/hub/models--gpt2" "$SMOKE_CACHE_ROOT/hub/models--gpt2"
  copy_cached_tree_if_present "$HOST_HF_CACHE_ROOT/hub/datasets--wikitext" "$SMOKE_CACHE_ROOT/hub/datasets--wikitext"
  copy_cached_tree_if_present \
    "$HOST_HF_CACHE_ROOT/datasets/wikitext/wikitext-2-raw-v1" \
    "$SMOKE_CACHE_ROOT/datasets/wikitext/wikitext-2-raw-v1"
  if [[ -d "$SMOKE_CACHE_ROOT/hub/models--gpt2" || -d "$SMOKE_CACHE_ROOT/hub/datasets--wikitext" || -d "$SMOKE_CACHE_ROOT/datasets/wikitext/wikitext-2-raw-v1" ]]; then
    seeded=1
  fi
  if [[ "$seeded" != "1" ]]; then
    return 1
  fi
  if [[ -d "$SMOKE_CACHE_ROOT/hub/models--gpt2" ]] && \
    [[ -d "$SMOKE_CACHE_ROOT/hub/datasets--wikitext" || -d "$SMOKE_CACHE_ROOT/datasets/wikitext/wikitext-2-raw-v1" ]]; then
    export INVARLOCK_SMOKE_CACHE_COMPLETE=1
  fi
  return 0
}

if seed_hf_cache_from_host; then
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

if [[ "${INVARLOCK_SMOKE_CACHE_COMPLETE:-0}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

echo "[smoke] work_root=$WORK_ROOT"
echo "[smoke] preset=$PRESET"
echo "[smoke] mode=$MODE profile=$PROFILE"
echo "[smoke] hf_home=$HF_HOME"
echo "[smoke] hf_datasets_cache=$HF_DATASETS_CACHE"

"${CLI[@]}" evaluate \
  --baseline gpt2 \
  --subject gpt2 \
  --adapter auto \
  --profile "$PROFILE" \
  --preset "$PRESET" \
  --mode "$MODE" \
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

VERIFY_ARGS=()
if [[ "$MODE" == "local" ]]; then
  VERIFY_ARGS+=(--allow-unattested-artifacts)
fi

"${CLI[@]}" verify "$EVAL_REPORT" "${VERIFY_ARGS[@]}" --json || VERIFY_RC=$?
VERIFY_RC="${VERIFY_RC:-0}"
echo "[smoke] verify_rc=$VERIFY_RC"
if [[ "$VERIFY_RC" != "0" ]]; then
  echo "[error] evaluation report verification failed" >&2
  exit "$VERIFY_RC"
fi
"${CLI[@]}" report validate "$EVAL_REPORT"
"${CLI[@]}" report html -i "$EVAL_REPORT" -o "$SMOKE_REPORT_DIR/evaluation.html"
"${CLI[@]}" report explain --report "$EDITED_REPORT" --baseline "$BASELINE_REPORT"

printf '%s\n' '{"verdict":"PASS","note":"gpt2 smoke campaign"}' > "$WORK_ROOT/final_verdict.json"

if [[ "$MODE" == "local" ]]; then
  echo "[smoke] skipping proof-pack build/verify in local mode; emitted artifacts are host-bypass."
  echo "[smoke] complete"
  exit 0
fi

"${CLI[@]}" advanced proof-pack build "$PROOF_PACK_DIR" \
  --final-verdict "$WORK_ROOT/final_verdict.json" \
  --report "$EVAL_REPORT" \
  --profile "$PROFILE" \
  --json
"${CLI[@]}" advanced proof-pack inspect "$PROOF_PACK_DIR" --json
"${CLI[@]}" advanced proof-pack verify "$PROOF_PACK_DIR" --json || PROOF_PACK_VERIFY_RC=$?
PROOF_PACK_VERIFY_RC="${PROOF_PACK_VERIFY_RC:-0}"
echo "[smoke] proof_pack_verify_rc=$PROOF_PACK_VERIFY_RC"
if [[ "$PROOF_PACK_VERIFY_RC" != "0" ]]; then
  echo "[error] proof-pack verification failed" >&2
  exit "$PROOF_PACK_VERIFY_RC"
fi

echo "[smoke] complete"
