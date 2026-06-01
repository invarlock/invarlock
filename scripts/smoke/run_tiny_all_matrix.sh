#!/usr/bin/env bash
set -euo pipefail

# Tiny Models Matrix (GPT-2 causal, BERT MLM)
# ----------------------------------------------------------------------
# Generates a consolidated checklist of valid invarlock evaluate command permutations
# across two compact smoke models and optional quantization for GPT-2.
#
# Usage:
#   bash scripts/smoke/run_tiny_all_matrix.sh               # dry-run (print + write checklist)
#   RUN=1 bash scripts/smoke/run_tiny_all_matrix.sh         # execute commands
#   RUN=1 NET=1 bash scripts/smoke/run_tiny_all_matrix.sh   # allow network
#

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/lib/smoke_common.sh"

RUN="${RUN:-0}"
NET="${NET:-0}"
TORCH_CPU_INDEX_URL="${TORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
export TORCH_CPU_INDEX_URL

PYTHON_BIN="$(smoke_select_python "$REPO_ROOT" "${PYTHON_BIN:-}")"
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1; then
  echo "ERROR: scripts/smoke/run_tiny_all_matrix.sh requires Python 3.12+." >&2
  echo "Set PYTHON_BIN to a supported interpreter or activate a Python 3.12+ environment." >&2
  exit 2
fi

smoke_setup_pythonpath "$REPO_ROOT"
CLI=("$PYTHON_BIN" -m invarlock.cli)

render_cmd() {
  printf '%q ' "$@"
}

run_cmd() {
  if ! "$@"; then
    MATRIX_FAILURES=$((MATRIX_FAILURES + 1))
  fi
}
MATRIX_FAILURES=0

# Profile selection
# - If caller set PROFILE, respect it.
# - Otherwise default to 'ci', but auto-switch to 'dev' when tiny relax is on.
if [ -z "${PROFILE+x}" ]; then
  PROFILE="ci"
  case "${INVARLOCK_TINY_RELAX:-}" in
    1|true|TRUE|yes|on) PROFILE="dev" ;;
  esac
fi

STAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_TMP_DIR=0
if [ -z "${TMP_DIR:-}" ]; then
  TMP_DIR="tmp/tiny_all_$STAMP"
  DEFAULT_TMP_DIR=1
fi
mkdir -p "$TMP_DIR"
if [ "$DEFAULT_TMP_DIR" = "1" ]; then
  LATEST_TARGET=${TMP_DIR#tmp/}
  ln -sfn "$LATEST_TARGET" "tmp/tiny_all_latest"
fi

HF_HOME="${HF_HOME:-$TMP_DIR/.hf}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

# Env knobs for speed and determinism
export INVARLOCK_DEDUP_TEXTS=1
export INVARLOCK_CAPACITY_FAST=1
export TOKENIZERS_PARALLELISM=false

# Respect NET for networked downloads vs offline cache
if [ "$NET" = "1" ]; then
  export INVARLOCK_ALLOW_NETWORK=1
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export HF_DATASETS_OFFLINE=0
  # Avoid torchvision dependency path in transformers
  export TRANSFORMERS_NO_TORCHVISION=1
  # Reduce CI windows for speed to avoid capacity/dedupe floors
  export INVARLOCK_CI_PREVIEW=${INVARLOCK_CI_PREVIEW:-64}
  export INVARLOCK_CI_FINAL=${INVARLOCK_CI_FINAL:-64}
else
  export HF_HUB_ENABLE_HF_TRANSFER=0
  export HF_DATASETS_OFFLINE=1
fi

smoke_seed_local_runtime_image "auto"
if [ "$RUN" = "1" ] && [ "$NET" = "1" ]; then
  smoke_ensure_current_runtime_image "container" "auto"
fi

# Ensure required Python deps are present when NET=1
if [ "$NET" = "1" ]; then
  "$PYTHON_BIN" - << 'PY'
try:
    import google.protobuf  # noqa: F401
    import sentencepiece  # noqa: F401
    import tiktoken  # noqa: F401
    import torch, transformers, datasets  # noqa: F401
    print("deps: torch/transformers/datasets/protobuf/sentencepiece/tiktoken present")
except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as e:
    print("deps: missing core HF stack; attempting install via pip...", e)
    import os, sys, subprocess
    cpu_torch_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--index-url",
        os.environ["TORCH_CPU_INDEX_URL"],
        "torch",
    ]
    hf_cmd = [sys.executable, "-m", "pip", "install", "-q", ".[hf]"]
    subprocess.check_call(cpu_torch_cmd)
    subprocess.check_call(hf_cmd)
    print("deps: installed .[hf]")
PY
fi

echo "# Tiny Models Evaluation Matrix ($STAMP)" > "$TMP_DIR/checklist.md"
CHECKLIST_ENV="Env: INVARLOCK_DEDUP_TEXTS=1, INVARLOCK_CAPACITY_FAST=1, HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}"
if [ "$NET" = "1" ]; then
  CHECKLIST_ENV="${CHECKLIST_ENV}, INVARLOCK_ALLOW_NETWORK=1"
fi
CHECKLIST_ENV="${CHECKLIST_ENV}, HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}, HF_HOME=${HF_HOME}"
echo "$CHECKLIST_ENV" >> "$TMP_DIR/checklist.md"
echo >> "$TMP_DIR/checklist.md"

render_runtime_prefix() {
  if [ "$NET" = "1" ]; then
    printf 'INVARLOCK_ALLOW_NETWORK=1 '
  else
    printf ''
  fi
}

append() {
  printf -- "- [ ] %s\n      \`%s%s\`\n" "$1" "$(render_runtime_prefix)" "$2" >> "$TMP_DIR/checklist.md"
}

# 1) GPT-2: causal LM (Compare & evaluate + quant demo edit)
GPT2_ID=${GPT2_ID:-"sshleifer/tiny-gpt2"}
QUANT_PROFILE="${QUANT_PROFILE:-dev}"
echo "## GPT-2 (causal LM)" >> "$TMP_DIR/checklist.md"
for PRESET in \
  configs/presets/causal_lm/wikitext2_512.yaml \
  omit
do
  tag="gpt2_eval_${PRESET##*/}"
  [ "$PRESET" = "omit" ] && tag="gpt2_eval_auto"
  cmd=("${CLI[@]}" evaluate --baseline "$GPT2_ID" --subject "$GPT2_ID" --baseline-adapter hf_causal --subject-adapter hf_causal --profile "$PROFILE" --tier balanced --device cpu)
  [ "$PRESET" != "omit" ] && cmd+=(--preset "$PRESET")
  append "$tag" "$(render_cmd "${cmd[@]}")"
  if [ "$RUN" = "1" ]; then run_cmd "${cmd[@]}"; fi
done

echo >> "$TMP_DIR/checklist.md"
echo "### GPT-2 Quant (demo edit)" >> "$TMP_DIR/checklist.md"
QCFG="configs/overlays/edits/quant_rtn/tiny_demo.yaml"
# Keep the quant demo on a smoke-friendly profile so strict CI parity checks do
# not turn an example edit into a false red path.
cmd=("${CLI[@]}" evaluate --baseline "$GPT2_ID" --subject "$GPT2_ID" --baseline-adapter hf_causal --subject-adapter hf_causal --profile "$QUANT_PROFILE" --tier balanced --device cpu --preset configs/presets/causal_lm/wikitext2_512.yaml --edit-config "$QCFG" --assurance off)
append "gpt2_eval_quant8_${QUANT_PROFILE}" "$(render_cmd "${cmd[@]}")"
[ "$RUN" = "1" ] && run_cmd "${cmd[@]}"

echo >> "$TMP_DIR/checklist.md"

# 2) Tiny encoder MLM
BERT_ID=${BERT_ID:-"sshleifer/tiny-distilroberta-base"}
echo "## Encoder MLM" >> "$TMP_DIR/checklist.md"
cmd=("${CLI[@]}" evaluate --baseline "$BERT_ID" --subject "$BERT_ID" --baseline-adapter hf_mlm --subject-adapter hf_mlm --profile "$PROFILE" --tier balanced --device cpu --preset configs/presets/masked_lm/wikitext2_128.yaml)
append "bert_mlm_eval" "$(render_cmd "${cmd[@]}")"
[ "$RUN" = "1" ] && run_cmd "${cmd[@]}"

echo >> "$TMP_DIR/checklist.md"

echo
echo "Checklist written to: $TMP_DIR/checklist.md"
echo "Using profile: ${PROFILE} (INVARLOCK_TINY_RELAX=${INVARLOCK_TINY_RELAX:-0})"
if [ "$RUN" = "1" ] && [ "$MATRIX_FAILURES" -gt 0 ]; then
  echo "ERROR: ${MATRIX_FAILURES} matrix command(s) failed." >&2
  exit 1
fi
