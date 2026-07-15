# Repo-only configs (examples)

This repository ships example YAML configs for repo checkouts (they are not
included in the wheel). Runtime policy is canonical and lives under
`src/invarlock/_data/runtime/`.

## Layout

- `configs/presets/` — complete, runnable presets for `invarlock evaluate --preset ...`
- `configs/overlays/` — partial overlays intended to be merged into presets
  - `overlays/edits/` (edit plans)
  - `overlays/models/` (model id/adapter overlays)
  - `overlays/datasets/` (dataset overlays)
- `configs/calibration/` — calibration harness configs (used by `invarlock advanced calibrate ...`)
- `configs/overrides/` — committed, copy-first examples for local guard overrides
- `configs/local/` — ignored by git; for your working presets/overrides

Preset families currently include:

- `configs/presets/causal_lm/` — decoder-only text models
- `configs/presets/masked_lm/` — BERT/RoBERTa-style masked language models
- `configs/presets/seq2seq/` — T5/BART-style encoder-decoder smoke presets
- `configs/presets/multimodal/` — multimodal model presets

The simple model and dataset overlays are:

- `configs/overlays/models/hf_causal.yaml`
- `configs/overlays/models/hf_mlm.yaml`
- `configs/overlays/datasets/wikitext2.yaml`

## Examples

These repo-only examples assume a host-side checkout, so they use
`invarlock evaluate --execution-mode host`. If you are running through the default
runtime-container path, drop `--execution-mode host`.

```bash
# Baseline vs subject with the repo preset
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --execution-mode host \
  --baseline sshleifer/tiny-gpt2 --subject sshleifer/tiny-gpt2 --baseline-adapter auto --subject-adapter auto \
  --profile ci --tier balanced \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --out runs/baseline --report-out reports/baseline

# Compare & Evaluate (preferred), using an edit overlay
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --execution-mode host \
  --baseline gpt2 --subject gpt2 --baseline-adapter auto --subject-adapter auto \
  --profile dev --tier balanced \
  --preset configs/presets/causal_lm/gpt2_smoke_128.yaml \
  --edit-config configs/overlays/edits/quant_rtn/8bit_attn.yaml

# First-class GPT-2 smoke preset used by the smoke campaign script/CI workflow
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --execution-mode host \
  --baseline gpt2 --subject gpt2 --baseline-adapter auto --subject-adapter auto \
  --profile dev \
  --preset configs/presets/causal_lm/gpt2_smoke_128.yaml

# Seq2seq smoke preset (synthetic pairs, T5-style adapter)
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --execution-mode host \
  --baseline t5-small --subject t5-small --baseline-adapter hf_seq2seq --subject-adapter hf_seq2seq \
  --profile dev \
  --preset configs/presets/seq2seq/synth_64.yaml
```

Runtime configs (canonical, in‑package) live under `src/invarlock/_data/runtime/`:

- `src/invarlock/_data/runtime/tiers.yaml` (maintained support-tier defaults)
- `src/invarlock/_data/runtime/profiles/` (profile overlays; e.g. `--profile release`)

The CLI loads runtime configs via `importlib.resources` (or `$INVARLOCK_CONFIG_ROOT/runtime/...`
if you override the runtime location). The `configs/` tree is repo-only examples.
