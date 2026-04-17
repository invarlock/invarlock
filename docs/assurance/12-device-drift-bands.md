# Cross‑Device Drift Bands (CPU ↔ MPS ↔ CUDA)

> **Plain language:** With deterministic settings, evaluation ratios across
> devices stay within small, documented bands. We publish the budgets and a
> reproducible check.

## Claim

With deterministic settings and identical evaluation schedules/policies,
cross‑device evaluation ratios remain within small, documented bands relative to
CPU (e.g., ≤ 0.5% MPS, ≤ 1.0% CUDA).

## Budgets (expected)

| Device | PM ratio vs CPU (Δ%) | Notes |
|--------|------------------------|-------|
| MPS    | within ±0.5%           | Apple Accelerate; deterministic seeds supported |
| CUDA   | within ±1.0%           | Deterministic algorithms; set `CUBLAS_WORKSPACE_CONFIG`, disable TF32 |

Bands were empirically derived on pilot models and are enforced in CI. Actual values may vary slightly by family/precision; verify on your setup.

## Determinism & Setup

- Enable framework determinism (PyTorch deterministic algorithms; disable TF32 where applicable).
- Record seed bundle and device in the container-backed report bundle: `meta.seeds.*`, `meta.device`.
- Use identical window plans (paired, non‑overlapping) and the same resolved policy/digest.

## Reproducible Check

```bash
# Baseline on CPU → report
INVARLOCK_ALLOW_HOST_EXECUTION=1 invarlock evaluate \
  --baseline gpt2 \
  --subject gpt2 \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --device cpu \
  --profile ci \
  --out runs/baseline_cpu \
  --report-out reports/baseline_cpu

# Same schedule on MPS → report
INVARLOCK_ALLOW_HOST_EXECUTION=1 invarlock evaluate \
  --baseline gpt2 \
  --subject gpt2 \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --device mps \
  --profile ci \
  --out runs/baseline_mps \
  --report-out reports/baseline_mps

# Lint cross-device drift (absolute ratio tolerance)
python scripts/check_device_drift.py \
  reports/baseline_cpu/evaluation.report.json \
  reports/baseline_mps/evaluation.report.json \
  --tolerance 0.005
```

## Runtime Contract (report)

- `primary_metric.ratio_vs_baseline` and `primary_metric.display_ci` report the ratio and CI when ppl‑like.
- `meta.device`, `meta.seeds` document the device context and seed bundle.

## Observability

- Archive a drift summary with release evidence; maintain pilot tables justifying chosen bands.

## Assumptions & Scope

- Deterministic flags must be enabled; TF32 must be disabled for CUDA.
- Window plans and seeds must match; schedule changes invalidate comparisons.
- Bands are empirical and may vary slightly by model family; verify locally and
  adjust tolerance for CI accordingly.
