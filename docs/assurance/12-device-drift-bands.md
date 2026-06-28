# Cross‑Device Drift Bands (CPU ↔ MPS ↔ CUDA)

> **Plain language:** With deterministic settings, evaluation ratios across
> devices are reviewed against small, documented pilot bands. We publish the
> budgets and a reproducible check.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Define the pilot review bands for comparing CPU, MPS, and CUDA evaluation ratios. |
| **Audience** | Maintainers, release approvers, and operators attaching cross-device evidence. |
| **Contract scope** | Empirical drift review for matching reports; PyTorch cross-platform reproducibility remains a separate platform concern. |
| **Source of truth** | `scripts/smoke/check_device_drift.py`, report `primary_metric.*`, and runtime metadata under `meta.*`. |

## Claim

With deterministic settings and identical evaluation schedules/policies,
cross-device evaluation ratios are expected to stay within small empirical
review bands relative to CPU (e.g., ≤ 0.5% MPS, ≤ 1.0% CUDA). These are pilot
budgets for InvarLock report comparison.

## Budgets (expected)

| Device | PM ratio vs CPU (Δ%) | Notes |
|--------|------------------------|-------|
| MPS    | within ±0.5%           | Apple Accelerate; deterministic seeds supported |
| CUDA   | within ±1.0%           | Deterministic algorithms; set `CUBLAS_WORKSPACE_CONFIG`, disable TF32 |

Bands were empirically derived on pilot models. The repo ships and tests
`scripts/smoke/check_device_drift.py`; CI enforces the checker behavior on fixtures,
while real CPU/MPS/CUDA drift enforcement requires CI or release evidence to
provide comparable reports from those devices. The checker compares absolute
drift in `primary_metric.ratio_vs_baseline`; report verification and provenance
review establish matching devices, seeds, policy digests, and window schedules.
Actual values may vary slightly by family/precision; verify on your setup.

## Determinism & Setup

- Enable framework determinism (PyTorch deterministic algorithms; disable TF32 where applicable).
- Record seed bundle and device in the report bundle: `meta.seeds.*`, `meta.device`.
- Use identical window plans (paired, non‑overlapping) and the same resolved policy/digest.

## Reproducible Check

```bash
# Calibration-only / non-assurance host-mode example.
# Do not accept host-mode output as strict assurance evidence.
# Baseline on CPU → report
invarlock evaluate --allow-network --execution-mode host \
  --assurance off \
  --baseline gpt2 \
  --subject gpt2 \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --device cpu \
  --profile ci \
  --out runs/baseline_cpu \
  --report-out reports/baseline_cpu

# Same schedule on MPS → report
invarlock evaluate --allow-network --execution-mode host \
  --assurance off \
  --baseline gpt2 \
  --subject gpt2 \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --device mps \
  --profile ci \
  --out runs/baseline_mps \
  --report-out reports/baseline_mps

# Lint cross-device drift (absolute ratio tolerance)
python scripts/smoke/check_device_drift.py \
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

## Related Documentation

- [Determinism Contracts](08-determinism-contracts.md)
- [Guard Contracts & Primer](04-guard-contracts.md)
- [Runtime Provenance Guide](../security/runtime-provenance-guide.md)

## References

- PyTorch. “Reproducibility.” <https://docs.pytorch.org/docs/2.12/notes/randomness.html>
