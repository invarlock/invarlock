# GPU/MPS-First Guard Measurement Contracts

> **Plain language:** Accelerator-friendly guard measurements are acceptable
> only when their estimator settings, sampling policy, and measurement-contract
> hashes are recorded so later verification can check the declaration and its
> report-local consistency. The offline verifier does not replay guard
> measurements from tensors or independently compare those declarations with
> the independently supplied raw baseline report.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Define the maintained measurement-contract expectations for accelerator-friendly spectral and RMT guards. |
| **Audience** | Contributors, maintainers, and auditors of guard evidence. |
| **Contract scope** | Spectral + RMT guard contracts on CUDA/MPS-capable and CPU fallback paths. |
| **Source of truth** | Guard implementations in `src/invarlock/guards/*.py`. |

## Claim

Spectral and RMT guard evidence uses bounded estimators rather than full matrix
decompositions on large models. The runtime records the
iterative estimator settings, sampling policy, and measurement-contract hashes
for the guard evidence that `invarlock verify` later checks for internal
consistency.

The recorded hash binds the report to a declared procedure. It does not prove
that the procedure executed, that the sampled activations were genuine, or that
the reported measurements were recomputed by the verifier. Those claims require
an independent rerun, trusted execution evidence, or another external attestation.

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
- CI/Release verification rejects missing measurement-contract hashes for
  evaluated Spectral/RMT guard evidence, recomputes each hash from the reported
  contract, compares the report block with `resolved_policy`, and requires the
  report's `measurement_contract_match` flag to be true.

## Contract Details

1. **Single evidence mode**: one canonical contract for each guard.
2. **Spectral contract**: track $\hat{\sigma}_{\max}$ and degeneracy proxies
   (stable-rank drift, row/col norm collapse).
3. **RMT contract**: activation edge-risk score normalized by MP edge.
4. **Verification gate**: evaluated reports must record the measurement
   contract, its hash, and a true match flag. A baseline contract hash may also
   be emitted. Verification checks report-local presence and consistency, not
   numerical replay or an independent baseline-contract comparison.

## Non-goals

- Full-spectrum or exact SVD computations.
- Accepting reports missing measurement contracts in CI/Release assurance paths.

## Troubleshooting

- See [Guards](../reference/guards.md) for operational guidance and guard configuration.

## Observability

- Contract hashes appear under `spectral.measurement_contract_hash` and
  `rmt.measurement_contract_hash` in reports.
- When emitted, evaluation-time baseline comparison appears under
  `spectral.baseline_measurement_contract_hash`,
  `rmt.baseline_measurement_contract_hash`, and
  `*.measurement_contract_match`.
- Resolved-policy contracts appear under
  `resolved_policy.spectral.measurement_contract` and
  `resolved_policy.rmt.measurement_contract`.

## Related Documentation

- [Guards](../reference/guards.md)
- [Guard Contracts & Primer](04-guard-contracts.md)
- [Spectral False-Positive Control](05-spectral-fpr-derivation.md)
- [RMT ε-Rule](06-rmt-epsilon-rule.md)
