# Configuration Gallery

Pointers to common presets in this repository you can start from. Presets are
repo assets (not shipped in wheels). Use flag‑only `invarlock certify` when
installing from PyPI, or clone this repo to reference these files.

Note: adapter‑based flows such as `invarlock certify` and `invarlock run` with
HF models require extras like `invarlock[hf]` or `invarlock[adapters]`. The
core install (`pip install invarlock`) remains torch‑free.

## Task Presets (CPU‑friendly)

- Causal LM (CI): `configs/tasks/causal_lm/ci_cpu.yaml`
- Causal LM (Release): `configs/tasks/causal_lm/release_cpu.yaml`
- Masked LM (CI): `configs/tasks/masked_lm/ci_cpu.yaml`

## Edit Plans (RTN quantization)

- 8‑bit attention‑only: `configs/edits/quant_rtn/8bit_attn.yaml`
- 8‑bit full‑model: `configs/edits/quant_rtn/8bit_full.yaml`
- Tiny demo: `configs/edits/quant_rtn/tiny_demo.yaml`

Notes

- Adjust window counts (`preview_n`, `final_n`) for CI vs. release profiles.
- Keep `seq_len = stride` for deterministic non‑overlapping windows.
