# RMT Epsilon-Band Rule And Evidence Limits

> **Plain language:** The RMT guard implements a deterministic,
> baseline-relative activation edge-risk comparison. The shipped one-percent
> epsilon is an operational policy constant in this repository snapshot; the
> public tree does not contain the null corpus needed to call it a demonstrated
> q95-q97 calibration.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Define the implemented activation edge-risk rule and its evidence boundary. |
| **Audience** | RMT maintainers, reviewers, and operators selecting local policy. |
| **Contract scope** | Global activation standardization, bounded power iteration, MP-edge normalization, and baseline-relative epsilon checks. |
| **Source of truth** | `src/invarlock/guards/rmt_activation_runtime.py`, `src/invarlock/guards/rmt_policy.py`, and the packaged `runtime/tiers.yaml` policy resource. |

## Implemented Rule

For family `f`, the report compares baseline and current edge-risk values:

$$
r_f^{cur} \leq (1 + \epsilon_f) r_f^{base}.
$$

If the current value exceeds that bound, the family is listed in
`rmt.epsilon_violations` and `validation.rmt_stable` becomes false.

## Edge-Risk Computation

For a token-by-hidden activation matrix, the implementation:

1. reshapes higher-rank activations to two dimensions;
2. subtracts one global scalar mean and divides by one global scalar standard
   deviation across all matrix entries;
3. estimates the leading singular value with a deterministic bounded power
   iteration (three iterations by default, initialized with all ones); and
4. divides by the shape-dependent Marchenko-Pastur upper edge.

The implementation performs global centering and scaling. It does **not**
perform feature-covariance whitening, so documentation and evidence should not
describe the matrix as whitened. The Marchenko-Pastur interpretation also relies
on assumptions that are not established for arbitrary correlated transformer
activations. Baseline-relative comparison may reduce some common bias, but does
not prove the estimator detects every structural change.

## Shipped Policy

Balanced and Conservative currently store
`{ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01}` in the packaged tier
resource (`runtime/tiers.yaml` as the logical package path).
Those are enforced constants, not a public statistical guarantee. The checkout
does not ship the raw null-run reports or a dedicated RMT epsilon summarizer
that re-derives them.

Use a local override when the target architecture, activation sampling plan,
precision, or device differs. Record that override in `resolved_policy`.

## Runtime Contract

Reports record:

- `rmt.{mode,edge_risk_by_family_base,edge_risk_by_family}`
- `rmt.{epsilon_default,epsilon_by_family,epsilon_violations,stable,status}`
- `rmt.families.*.{edge_base,edge_cur,epsilon,allowed,ratio,delta}`
- `rmt.measurement_contract` and its hash
- the matching resolved policy under `resolved_policy.rmt`

Verification checks presence, arithmetic consistency, policy matching, and the
declared acceptance inequality. It does not replay model activations from the
report alone.

The current runtime assigns a blocking decision to any epsilon violation, and
strict verification requires `rmt.stable=true` with an empty violation list.
There is no separate strict-mode organization-policy switch that accepts a
complete RMT measurement as diagnostic-only evidence. The estimator's
statistical interpretation remains experimental despite that blocking
behavior.

## Recalibration Procedure

For a local policy proposal:

1. specify model families, devices, precision, sampling, and null edits in advance;
2. run independent baseline/no-op pairs;
3. compute `delta(f) = r_cur(f) / r_base(f) - 1` when the baseline is positive;
4. report the full distribution, sample count, quantiles, and uncertainty;
5. validate the selected epsilon on held-out null and fault cases; and
6. archive raw reports, code revision, configuration, and hashes.

A locally selected quantile is not a shipped calibration unless its corpus and
derivation are attached and independently reviewable.

## Related Documentation

- [Guard Contracts](04-guard-contracts.md)
- [Tier Policy Values And Recalibration](09-tier-v1-calibration.md)
- [GPU/MPS-First Guard Measurement Contracts](13-gpu-mps-first-guards.md)
