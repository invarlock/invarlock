# RMT ε‑Band Acceptance (Edge Risk Score)

> **Plain language:** The RMT guard limits how much the activation **edge risk**
> can grow beyond its baseline, ensuring structural shifts trigger a failure
> while expected noise passes.

## Claim

The Random Matrix Theory (RMT) guard accepts an edit when the activation **edge
risk score** stays within the calibrated ε‑band for each family.

Let $r_f^{\text{base}}$ be the baseline edge risk score and
$r_f^{\text{cur}}$ the current score for family $f$.
The guard accepts if:

$$
r_f^{\text{cur}} \le (1+\epsilon_f)\, r_f^{\text{base}}
$$

with $\epsilon_f$ calibrated from **null** runs (e.g., 95th–99th percentile of
$r_f^{\text{cur}}/r_f^{\text{base}} - 1$).

### What is the edge risk score?

For a (token×hidden) activation matrix $A$, the guard forms a whitened matrix
$A'$ (centered and standardised), estimates its top singular value
$\hat{\sigma}_{\max}(A')$ via a deterministic matvec estimator, and normalizes by
the Marchenko–Pastur edge $\sigma_{\mathrm{MP}}(m,n)$ for the same shape:

$$
r = \frac{\hat{\sigma}_{\max}(A')}{\sigma_{\mathrm{MP}}(m,n)}
$$

The contract fixes the estimator budget and the activation sampling policy; those
knobs are recorded in the certificate.

## Derivation (sketch)

- Edge risk fluctuates under null due to finite‑sample deviations from the
  Marchenko–Pastur edge and estimator noise.
- The ε‑band permits expected null drift, flagging **structural** increases.
- Large edge risk indicates concentration of activation energy along a small
  number of directions beyond random‑matrix expectations.

## Assumptions & Scope

- Null calibration must cover each family `{ffn, attn, embed, other}`; default ε values are exposed whenever data is sparse.
- Baseline and current scores use identical activation sampling and **token‑weighted aggregation**.
- Evidence requires activation-based scoring; if activation batches are missing, the RMT guard fails closed.

## Calibration (pilot-derived)

- Balanced tier uses $\epsilon_f = \{0.10, 0.08, 0.12, 0.12\}$ for
  `{ffn, attn, embed, other}` respectively (q95–q97 of null deltas).
- Conservative tightens to $\epsilon_f = \{0.06, 0.05, 0.07, 0.07\}$
  (q97–q99 of null deltas).
  Values are recorded in the packaged `tiers.yaml`
  (`invarlock._data.runtime/tiers.yaml`) and surfaced in certificates. Provide
  overrides via `INVARLOCK_CONFIG_ROOT/runtime/tiers.yaml` when needed.

*Example:* with `r_base = 1.20` and ε = 0.10, the guard allows
`r_cur ≤ (1+0.10) × 1.20 = 1.32`.

## Calibration

Calibration values are derived from null-sweep runs and stored in the packaged
`tiers.yaml`. See the full calibration methodology in
[09-tier-v1-calibration.md](09-tier-v1-calibration.md).

To recalibrate, run null baselines (no edit) and compute per-family deltas
Δ(f) = r_cur(f)/r_base(f) − 1 (skip cases with missing baseline). Set ε(f) to the
q95–q99 quantile of Δ(f). For small families or tiny sample sizes, use a slightly
larger ε to avoid spurious failures.

## Runtime Contract (certificate)

- Certificate reports `rmt.{edge_risk_by_family_base,edge_risk_by_family,epsilon_default,epsilon_by_family,epsilon_violations,stable,status}`.
- Per-family details for rendering live under `rmt.families.*.{edge_base,edge_cur,epsilon,allowed,ratio,delta}`.
- Certificate lint verifies the inequality and marks violations; `validation.rmt_stable` reflects the ε‑band gate.

## Observability

- `rmt.edge_risk_by_family_base.*` and `rmt.edge_risk_by_family.*`.
- `rmt.epsilon_default` and `rmt.epsilon_by_family.*`.
- `rmt.status` / `rmt.stable` and `rmt.epsilon_violations` for pass/fail context.
- `resolved_policy.rmt.{margin,deadband,epsilon_by_family}` — resolved thresholds archived with the cert.

## Edge cases

- Small samples: estimator variance dominates; increase activation sample count or widen ε for tiny families.

## Background reading

- Pennington, J., & Worah, P. (2017). “Nonlinear Random Matrix Theory for Deep Learning.” *Advances in Neural Information Processing Systems (NeurIPS)*. <https://papers.nips.cc/paper/6857-nonlinear-random-matrix-theory-for-deep-learning>
- Martin, C. H., & Mahoney, M. W. (2021). “Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Learning.” *Journal of Machine Learning Research*, 22(165), 1–73. Preprint: <https://arxiv.org/abs/1810.01075>
