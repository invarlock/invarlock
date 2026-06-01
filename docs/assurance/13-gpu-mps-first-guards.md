# GPU/MPS-First Guard Measurement Contracts

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Define the maintained measurement-contract expectations for accelerator-friendly spectral and RMT guards. |
| **Audience** | Contributors, maintainers, and reviewers of guard evidence. |
| **Scope** | Spectral + RMT guard contracts on CUDA/MPS-capable and CPU fallback paths. |
| **Source of truth** | Guard implementations in `src/invarlock/guards/*.py`. |

## Claim

Spectral and RMT guard evidence must remain reproducible without requiring full
matrix decompositions on large models. The runtime therefore records bounded
iterative estimator settings, sampling policy, and measurement-contract hashes
for the guard evidence that `invarlock verify` later checks.

For operational guard usage, see [Guards](../reference/guards.md).

## Concepts

- **Accelerator-first**: guard math runs on CUDA/MPS-capable paths without full SVD.
- **Bounded approximation**: iterative estimators and deterministic sampling replace
  exact decompositions for large tensors.
- **Measurement contracts**: estimator + sampling policy must be recorded in reports.

## Runtime Contract

Guard reports must preserve enough information for later verification:

- Spectral evidence records the estimator family, bounded iteration budget,
  degeneracy proxies, and measurement-contract hash.
- RMT evidence records activation edge-risk scoring, sampling policy, estimator
  budget, and measurement-contract hash.
- `invarlock verify` rejects missing measurement-contract hashes in CI/Release
  assurance paths.

## Contract Details

1. **Single evidence mode**: one canonical contract for each guard.
2. **Spectral contract**: track `σ̂_max` and degeneracy proxies (stable-rank drift,
   row/col norm collapse).
3. **RMT contract**: activation edge-risk score normalized by MP edge.
4. **Verification gate**: reports must record the measurement contract and hash.

## Non-goals

- Full-spectrum or exact SVD computations.
- Accepting reports missing measurement contracts in CI/Release assurance paths.

## Troubleshooting

- See [Guards](../reference/guards.md) for operational guidance and guard configuration.

## Observability

- Contract hashes appear under `spectral.measurement_contract_hash` and
  `rmt.measurement_contract_hash` in reports.

## Related Documentation

- [Guards](../reference/guards.md)
- [Guard Contracts & Primer](04-guard-contracts.md)
- [Spectral False-Positive Control](05-spectral-fpr-derivation.md)
- [RMT ε-Rule](06-rmt-epsilon-rule.md)
