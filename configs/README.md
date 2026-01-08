# Configs layout (task‑centric)

This repo ships example presets (not included in the wheel) organized by:

- tasks/ … what you evaluate (causal_lm, masked_lm)
  - seq2seq (synthetic provider for demos)
- edits/ … what you change (quant_rtn variants)
- models/ … optional overlays for adapter/id
- datasets/ … optional overlays for provider/seq len

Pick one from each axis when needed. Typical flows use a task preset + edit plan:

Examples

```bash
# Causal LM (CI) + 8‑bit attention edit
invarlock run -c configs/tasks/causal_lm/ci_cpu.yaml --profile ci \
  --tier balanced --out runs/baseline
invarlock run -c configs/edits/quant_rtn/8bit_attn.yaml --profile ci \
  --baseline runs/baseline/report.json --out runs/edited

# Compare & Certify (BYOE)
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock certify \
  --baseline gpt2 --subject gpt2 --adapter auto --profile ci \
  --preset configs/tasks/causal_lm/ci_cpu.yaml

# PASSing release certificates
# Causal LM (auto device; larger windows)
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock certify \
  --baseline gpt2 --subject gpt2 --adapter auto --profile release \
  --preset configs/tasks/causal_lm/release_auto.yaml \
  --edit-config configs/edits/quant_rtn/8bit_attn.yaml \
  --out runs/release_gpt2 --cert-out reports/cert/release_gpt2

# Masked LM (BERT) — release, auto device (no edit)
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock certify \
  --baseline bert-base-uncased --subject bert-base-uncased --adapter auto \
  --profile release --preset configs/tasks/masked_lm/release_auto.yaml \
  --out runs/release_bert --cert-out reports/cert/release_bert

# Seq2Seq (synthetic) — release, auto device (demo)
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock certify \
  --baseline gpt2 --subject gpt2 --adapter auto --profile release \
  --preset configs/tasks/seq2seq/release_auto.yaml \
  --out runs/release_s2s --cert-out reports/cert/release_s2s

# Seq2Seq (T5) — release, auto device (real HF adapter)
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock certify \
  --baseline t5-small --subject t5-small --adapter hf_t5 --profile release \
  --preset configs/tasks/seq2seq/release_auto_t5.yaml \
  --out runs/release_s2s_t5 --cert-out reports/cert/release_s2s_t5

Note: hf_seq2seq uses Hugging Face datasets. The project pins datasets==2.18.*
in extras to avoid API drift. If you override, ensure your datasets version can
load your chosen dataset. Alternatively, use a local JSONL pairs preset:

# Seq2Seq (T5) — release, auto device (local JSONL pairs)
# JSONL lines must include string keys: source and target
# A sample is provided under data/seq2seq_pairs/sample.jsonl
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock certify \
  --baseline t5-small --subject t5-small --adapter hf_t5 --profile release \
  --preset configs/tasks/seq2seq/release_auto_t5_local.yaml \
  --out runs/release_s2s_t5_local --cert-out reports/cert/release_s2s_t5_local
```

Runtime configs (canonical, in‑package) live under `src/invarlock/_data/runtime/`:
- `src/invarlock/_data/runtime/tiers.yaml` (published tier defaults)
- `src/invarlock/_data/runtime/profiles/` (profile overlays; e.g. `--profile release`)

The CLI loads runtime configs via `importlib.resources` (or `$INVARLOCK_CONFIG_ROOT/runtime/...`
if you override the runtime location). The `configs/` tree is repo-only examples.
