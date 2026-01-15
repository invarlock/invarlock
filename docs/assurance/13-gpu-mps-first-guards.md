# GPU/MPS-First Guards (Decision Memo)

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Record the guard design decisions enabling large-model execution. |
| **Audience** | Contributors and reviewers of guard measurement contracts. |
| **Scope** | Spectral + RMT guard contracts, accelerator-first design. |
| **Source of truth** | Guard implementations in `src/invarlock/guards/*.py`. |

## Quick Start

This is a decision memo; for implementation usage see [Guards](../reference/guards.md).

## Concepts

- **Accelerator-first**: guard math must run on CUDA/MPS without full SVD.
- **Approximation-only**: iterative estimators and deterministic sampling.
- **Measurement contracts**: estimator + sampling policy must be recorded in certificates.

## Reference

### Goals

- Device-resident guard computation for large models.
- Reproducible approximations with fixed budgets.
- Contract binding enforced at verification time.

### Decisions

1. **Single evidence mode**: one canonical contract for each guard.
2. **Spectral contract**: track `σ̂_max` and degeneracy proxies (stable-rank drift,
   row/col norm collapse).
3. **RMT contract**: activation edge-risk score normalized by MP edge.
4. **Verification gate**: certificates must record the measurement contract and hash.

### Non-goals

- Full-spectrum or exact SVD computations.
- Certificates missing measurement contracts.

## Troubleshooting

- See [Guards](../reference/guards.md) for operational guidance and guard configuration.

## Observability

- Contract hashes appear under `spectral.measurement_contract_hash` and
  `rmt.measurement_contract_hash` in certificates.

## Related Documentation

- [Guards](../reference/guards.md)
- [Guard Contracts & Primer](04-guard-contracts.md)
- [Spectral False-Positive Control](05-spectral-fpr-derivation.md)
- [RMT ε-Rule](06-rmt-epsilon-rule.md)
