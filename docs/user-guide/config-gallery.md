# Configuration Gallery

Pointers to common presets in this repository you can start from. Presets are
repo assets (not shipped in wheels). Use flag‑only `invarlock certify` when
installing from PyPI, or clone this repo to reference these files.

Note: adapter‑based flows such as `invarlock certify` and `invarlock run` with
HF models require extras like `invarlock[hf]` or `invarlock[adapters]`. The
core install (`pip install invarlock`) remains torch‑free.

## Presets (runnable)

- Causal LM (WikiText‑2, 512): `configs/presets/causal_lm/wikitext2_512.yaml`
- Masked LM (WikiText‑2, 128): `configs/presets/masked_lm/wikitext2_128.yaml`
- Masked LM (synthetic, offline‑friendly): `configs/presets/masked_lm/synthetic_128.yaml`
- Seq2Seq (synthetic, quick): `configs/presets/seq2seq/synth_64.yaml`

## Edit Overlays (RTN quantization)

- 8‑bit attention‑only: `configs/overlays/edits/quant_rtn/8bit_attn.yaml`
- 8‑bit full‑model: `configs/overlays/edits/quant_rtn/8bit_full.yaml`
- Tiny demo: `configs/overlays/edits/quant_rtn/tiny_demo.yaml`

Notes

- Use `--profile ci|release|ci_cpu` to apply runtime profile window counts and bootstrapping defaults.
- Keep `seq_len = stride` for deterministic non‑overlapping windows.
