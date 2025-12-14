# RMT ε‑Rule Acceptance Band

> **Plain language:** The RMT guard limits how much activation outliers can
> grow beyond the Marchenko–Pastur edge, ensuring structural shifts trigger a
> WARN while expected noise passes.

## Claim

The Random Matrix Theory (RMT) guard accepts an edit when the guarded outlier
counts stay within the calibrated ε-band for each family.

Let $b_f$ be **bare** outliers and $g_f$ **guarded** outliers for family $f$.
The guard accepts if

$$
g_f \le \left\lceil (1+\epsilon_f) \, b_f \right\rceil
$$

with $\epsilon_f$ calibrated from **null** runs (e.g., 95th–99th percentile of $g_f/b_f - 1$).

## Derivation (sketch)

- Outlier counts fluctuate under null due to finite‑sample deviations from the
  Marchenko–Pastur edge.
- The ε‑band permits expected null growth, flagging **structural** increases.
- Outliers are the singular/eigen values that escape the Marchenko–Pastur
  bulk—large deviations suggest the edit introduced new structure the guard
  should flag.

## Assumptions & Scope

- Null calibration must cover each family `{ffn, attn, embed, other}`; default ε values are exposed whenever data is sparse.
- Bare and guarded counts use identical evaluation windows and token weighting.
- Small baselines rely on the ceiling operator; embeddings therefore use slightly larger ε to avoid spurious WARNs.

## Calibration (pilot-derived)

- Balanced tier uses $\epsilon_f = \{0.10, 0.08, 0.12, 0.12\}$ for
  `{ffn, attn, embed, other}` respectively (q95–q97 of null deltas).
- Conservative tightens to $\epsilon_f = \{0.06, 0.05, 0.07, 0.07\}$
  (q97–q99 of null deltas).
  Values are recorded in the packaged `tiers.yaml`
  (`invarlock._data.runtime/tiers.yaml`) and surfaced in certificates. Provide
  overrides via `INVARLOCK_CONFIG_ROOT/runtime/tiers.yaml` when needed.

*Example:* with `outliers_bare = 1` and ε = 0.10, the guard allows
`outliers_guarded = ceil((1 + 0.10) × 1) = 2`; a single extra outlier is
tolerated before a WARN.

## Calibration

Calibration values are derived from null-sweep runs and stored in the packaged
`tiers.yaml`. See the full calibration methodology in
[09-tier-v1-calibration.md](09-tier-v1-calibration.md).

To recalibrate, run null baselines (no edit) and compute per-family deltas
Δ(f) = g(f)/b(f) − 1, excluding cases with b(f) = 0. Set ε(f) to the q95–q99
quantile of Δ(f). For small families where discreteness matters (b(f) ∈ {0, 1}),
use a slightly larger ε to avoid spurious WARNs.

## Runtime Contract (certificate)

- Certificate reports `rmt.{outliers_bare,outliers_guarded,epsilon,epsilon_by_family,delta_per_family,delta_total,stable,status}`.
- Certificate lint verifies the inequality and marks violations; `validation.rmt_stable` reflects the ε‑rule gate.

## Observability

- `rmt.outliers_bare`, `rmt.outliers_guarded`, and `rmt.delta_per_family[*]`.
- `rmt.epsilon` (default) and `rmt.epsilon_by_family`.
- `rmt.status` / `rmt.stable` and `rmt.epsilon_violations` for pass/fail context.
- `resolved_policy.rmt.{margin,deadband,epsilon_by_family}` — resolved thresholds archived with the cert.

## Edge cases

- Small $b_f$: discreteness makes the ceiling operator important; use slightly larger ε for tiny families (e.g., embeddings).

## Background reading

- Pennington, J., & Worah, P. (2017). “Nonlinear Random Matrix Theory for Deep Learning.” *Advances in Neural Information Processing Systems (NeurIPS)*. <https://papers.nips.cc/paper/6857-nonlinear-random-matrix-theory-for-deep-learning>
- Martin, C. H., & Mahoney, M. W. (2021). “Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Learning.” *Journal of Machine Learning Research*, 22(165), 1–73. Preprint: <https://arxiv.org/abs/1810.01075>
