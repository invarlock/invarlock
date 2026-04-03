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

## Examples

These repo-only examples assume a trusted local checkout, so they use
`invarlock evaluate --assurance trusted-local`. If you are running through the secure-default
runtime-container path, drop `--assurance trusted-local`.

```bash
# Baseline vs subject with the repo preset
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --assurance trusted-local \
  --baseline sshleifer/tiny-gpt2 --subject sshleifer/tiny-gpt2 --adapter auto \
  --profile ci --tier balanced \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --out runs/baseline --report-out reports/baseline

# Compare & Evaluate (preferred), using an edit overlay
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --assurance trusted-local \
  --baseline sshleifer/tiny-gpt2 --subject sshleifer/tiny-gpt2 --adapter auto \
  --profile ci --tier balanced \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --edit-config configs/overlays/edits/quant_rtn/8bit_attn.yaml

# First-class GPT-2 smoke preset used by the smoke campaign script/CI workflow
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --assurance trusted-local \
  --baseline gpt2 --subject gpt2 --adapter auto \
  --profile dev \
  --preset configs/presets/causal_lm/gpt2_smoke_128.yaml
```

Runtime configs (canonical, in‑package) live under `src/invarlock/_data/runtime/`:
- `src/invarlock/_data/runtime/tiers.yaml` (published tier defaults)
- `src/invarlock/_data/runtime/profiles/` (profile overlays; e.g. `--profile release`)

The CLI loads runtime configs via `importlib.resources` (or `$INVARLOCK_CONFIG_ROOT/runtime/...`
if you override the runtime location). The `configs/` tree is repo-only examples.
