#!/usr/bin/env bash
set -euo pipefail

# Tiny Models Matrix (GPT-2 causal, BERT MLM, DistilBERT classification)
# ----------------------------------------------------------------------
# Generates a consolidated checklist of valid invarlock certify command permutations
# across three tiny models and optional quantization for GPT-2.
#
# Usage:
#   bash scripts/run_tiny_all_matrix.sh               # dry-run (print + write checklist)
#   RUN=1 bash scripts/run_tiny_all_matrix.sh         # execute commands
#   RUN=1 NET=1 bash scripts/run_tiny_all_matrix.sh   # allow network
#

RUN="${RUN:-0}"
NET="${NET:-0}"

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
TMP_DIR=${TMP_DIR:-"tmp/tiny_all_$STAMP"}
mkdir -p "$TMP_DIR"

# Env knobs for speed and determinism
export INVARLOCK_DEDUP_TEXTS=1
export INVARLOCK_CAPACITY_FAST=1

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

# Ensure required Python deps are present when NET=1
if [ "$NET" = "1" ]; then
  python - << 'PY' || true
try:
    import torch, transformers, datasets  # noqa: F401
    print("deps: torch/transformers/datasets present")
except Exception as e:
    print("deps: missing core HF stack; attempting install via pip...", e)
    import sys, subprocess
    cmd = [sys.executable, "-m", "pip", "install", "-q", "invarlock[hf]"]
    subprocess.check_call(cmd)
    print("deps: installed invarlock[hf]")
PY
fi

echo "# Tiny Models Certification Matrix ($STAMP)" > "$TMP_DIR/checklist.md"
echo "Env: INVARLOCK_DEDUP_TEXTS=1, INVARLOCK_CAPACITY_FAST=1, HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}${NET:+, INVARLOCK_ALLOW_NETWORK=1, HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}}" >> "$TMP_DIR/checklist.md"
echo >> "$TMP_DIR/checklist.md"

append() { printf -- "- [ ] %s\n      \`%s\`\n" "$1" "$2" >> "$TMP_DIR/checklist.md"; }

# 1) GPT-2: causal LM (Compare & Certify + quant demo edit)
GPT2_ID=${GPT2_ID:-"sshleifer/tiny-gpt2"}
echo "## GPT-2 (causal LM)" >> "$TMP_DIR/checklist.md"
for PRESET in \
  configs/presets/causal_lm/wikitext2_512.yaml \
  omit
do
  tag="gpt2_cert_${PRESET##*/}"
  [ "$PRESET" = "omit" ] && tag="gpt2_cert_auto"
  cmd=(invarlock certify --baseline "$GPT2_ID" --subject "$GPT2_ID" --adapter hf_gpt2 --profile "$PROFILE" --tier balanced)
  [ "$PRESET" != "omit" ] && cmd+=(--preset "$PRESET")
  append "$tag" "${cmd[*]}"
  if [ "$RUN" = "1" ]; then eval "${cmd[*]}" || true; fi
done

echo >> "$TMP_DIR/checklist.md"
echo "### GPT-2 Quant (demo edit)" >> "$TMP_DIR/checklist.md"
QCFG="configs/overlays/edits/quant_rtn/tiny_demo.yaml"
cmd=(invarlock certify --baseline "$GPT2_ID" --subject "$GPT2_ID" --adapter hf_gpt2 --profile "$PROFILE" --tier balanced --preset configs/presets/causal_lm/wikitext2_512.yaml --edit-config "$QCFG")
append "gpt2_editcert_quant8" "${cmd[*]}"
[ "$RUN" = "1" ] && eval "${cmd[*]}" || true

echo >> "$TMP_DIR/checklist.md"

# 2) BERT-tiny: masked LM
BERT_ID=${BERT_ID:-"prajjwal1/bert-tiny"}
echo "## BERT (masked LM)" >> "$TMP_DIR/checklist.md"
cmd=(invarlock certify --baseline "$BERT_ID" --subject "$BERT_ID" --adapter hf_bert --profile "$PROFILE" --tier balanced --preset configs/presets/masked_lm/wikitext2_128.yaml)
append "bert_mlm_cert" "${cmd[*]}"
[ "$RUN" = "1" ] && eval "${cmd[*]}" || true

echo >> "$TMP_DIR/checklist.md"

# 3) DistilBERT SST-2: classification smoke
CLS_ID=${CLS_ID:-"distilbert-base-uncased-finetuned-sst-2-english"}
echo "## DistilBERT (classification)" >> "$TMP_DIR/checklist.md"
cmd=(invarlock certify --baseline "$CLS_ID" --subject "$CLS_ID" --adapter hf_bert --profile "$PROFILE" --tier balanced --preset configs/presets/masked_lm/wikitext2_128.yaml)
append "distilbert_cls_cert" "${cmd[*]}"
[ "$RUN" = "1" ] && eval "${cmd[*]}" || true

# Optional: measured accuracy (requires network) when INCLUDE_MEASURED_CLS=1
if [ "$NET" = "1" ] && [ "${INCLUDE_MEASURED_CLS:-0}" = "1" ]; then
  echo >> "$TMP_DIR/checklist.md"
  echo "## DistilBERT (classification, measured)" >> "$TMP_DIR/checklist.md"
  cmd=(invarlock certify --baseline "$CLS_ID" --subject "$CLS_ID" --adapter hf_bert --profile "$PROFILE" --tier balanced --preset configs/presets/masked_lm/wikitext2_128.yaml)
  append "distilbert_cls_measured" "${cmd[*]}"
  if [ "$RUN" = "1" ]; then eval "${cmd[*]}" || true; fi
fi

echo
echo "Checklist written to: $TMP_DIR/checklist.md"
echo "Using profile: ${PROFILE} (INVARLOCK_TINY_RELAX=${INVARLOCK_TINY_RELAX:-0})"
