# B200 GPU/CPU Backend Validation Harness

This folder contains a small, self-contained harness to **measure** how much CPU vs CUDA backends differ for SVD-based measurements used by InvarLock guards (notably RMT activation SVD and spectral `sigma_max`).

The goal is to produce an archived JSON report you can attach to a proposal that:

- Moves a strict/canonical measurement from CPU → GPU, or
- Introduces a new “GPU path” that would otherwise be treated as noncanonical.

It does **not** change InvarLock behavior by itself.

## What it measures

- **Equivalence (CPU vs CUDA):** outlier counts and key scalars on identical inputs
- **Determinism (per backend):** repeat runs and report min/max drift
- **Calibration impact estimate:** “required epsilon” implied by CPU vs CUDA outlier-count deltas

## Quickstart (single GPU)

Synthetic-only (fast, isolates backend math):

```bash
CUDA_VISIBLE_DEVICES=0 \
python3 scripts/b200_gpu_backend_validation/validate_svd_backend_equivalence.py \
  --synthetic \
  --strict-determinism \
  --out results/svd_backend_synth.json
```

Model-based (runs a forward pass and hooks a few linear modules):

```bash
CUDA_VISIBLE_DEVICES=0 INVARLOCK_ALLOW_NETWORK=1 \
python3 scripts/b200_gpu_backend_validation/validate_svd_backend_equivalence.py \
  --model NousResearch/Llama-2-13b-hf \
  --max-modules 4 \
  --seq-len 1024 \
  --strict-determinism \
  --out results/svd_backend_llama13b.json
```

## Multi-GPU runner (8× B200)

Run one model per GPU (recommended: keep **one GPU visible per process** so `device_map="auto"` does not shard across GPUs):

```bash
OUT_DIR=results/svd_backend_matrix_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_DIR"

bash scripts/b200_gpu_backend_validation/run_multi_gpu.sh \
  --out-dir "$OUT_DIR" \
  --strict-determinism \
  --model mistralai/Mistral-7B-v0.1 \
  --model NousResearch/Llama-2-13b-hf \
  --model Qwen/Qwen2.5-14B \
  --model Qwen/Qwen2.5-32B \
  --model 01-ai/Yi-34B \
  --model mistralai/Mixtral-8x7B-v0.1 \
  --model NousResearch/Llama-2-70b-hf \
  --model Qwen/Qwen1.5-72B
```

Each run writes `*.json` + `*.log` under `--out-dir`.

## Interpreting results (practical guidance)

- The strongest “strict-compatible” claim is: **synthetic mismatch rate = 0** and very small scalar drift under determinism settings.
- If CUDA differs in outlier counts, the JSON includes `required_eps` estimates: that’s a hint you’d need either:
  - a backend-specific calibration, or
  - to keep CPU as the canonical backend for strict claims.

If you want this harness to enforce a specific tolerance, use the `--fail-*` flags so CI can gate backend changes automatically.

