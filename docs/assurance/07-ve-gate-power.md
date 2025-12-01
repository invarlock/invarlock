# Variance Guard Predictive Gate (power & sidedness)

> **Plain language:** VE only proposes scales when the predictive paired
> ΔlogNLL shows a real improvement—Balanced needs a one-sided win, Conservative
> needs a two-sided push—and the certificate explains why VE stayed on or off.

## Claim

VE proposes scales only when the **predictive** paired ΔlogNLL CI excludes 0
with the tier's sidedness and the absolute mean effect exceeds
`min_effect_lognll`.

- **Balanced**: **one‑sided improvement** test. VE enables only when the
  predictive CI **upper bound** ≤ −`min_effect_lognll` and the mean Δ ≤
  −`min_effect_lognll`.
- **Conservative**: **two‑sided** test; VE enables only when the predictive CI
  lies strictly outside ±`min_effect_lognll` and the absolute mean Δ is at least
  `min_effect_lognll`.

Example (Balanced): with `min_effect_lognll = 9e-4`, a predictive estimate
Δ̄ = −0.002 with CI [−0.003, −0.001] enables VE because both the mean and the
CI upper bound beat −`min_effect_lognll`.

Example (Conservative): with `min_effect_lognll = 9e-4`, a predictive estimate
Δ̄ = −0.0013 with CI [−0.0021, −0.0010] enables VE because the entire CI lies
outside the interval [−`min_effect_lognll`, +`min_effect_lognll`]. A CI that
touches or sits within this interval (e.g., [−0.0015, −0.0002]) does not enable
VE.

## Derivation (power target)

Let paired Δ values on calibration windows have standard deviation
$\sigma_{\text{pred}}$ and count $n$. The CI half-width is approximately
`h ≈ z · σ_pred / √n`,
with $z = z_{0.95}$ (one‑sided) or $z_{0.975}$ (two‑sided). Choose
`min_effect_lognll ≈ h` to obtain ~50% power at the boundary; raise for
stricter tiers.

## Tier knobs (pilot coverage)

| Tier          | deadband | min_abs_adjust | max_scale_step | min_effect_lognll | predictive\_one\_sided | max\_adjusted\_modules |
|---------------|----------|----------------|----------------|-------------------|------------------------|------------------------|
| balanced      | 0.02     | 0.012          | 0.03           | 0.0009            | ✅ (one-sided)          | 1                      |
| conservative  | 0.03     | 0.02           | 0.015          | 0.0018            | ❌ (two-sided)          | 0                      |

Values come from the November 2025 pilot (see packaged `tiers.yaml`,
`TIER_POLICIES`) and maintain VE responsiveness without triggering false
positives.

> **Source of truth:** tier thresholds are drawn from the packaged `tiers.yaml`.
> Balanced sets `min_effect_lognll = 0.0009`; older pilot drafts that
> referenced `0.0005` have been retired to keep docs, configs, and certs
> consistent.

## Calibration

The `min_effect_lognll` values are derived from paired ΔlogNLL statistics on
calibration windows using the formula `min_effect ≈ z × σ_pred / √n` with the
appropriate z-quantile per tier. Calibrated values are stored in the packaged
`tiers.yaml`. See the full calibration methodology in
[09-tier-v1-calibration.md](09-tier-v1-calibration.md).

To recalibrate, run null baselines (no edit) and compute the paired Δ standard
deviation σ̂ across calibration windows. Use z = z₀.₉₅ for one-sided (Balanced)
or z = z₀.₉₇₅ for two-sided (Conservative), then set min_effect ≈ z × σ̂ / √n.

## Provenance & tap

- VE must evaluate A = **edited model (no VE)** and B = **virtual VE** on the
  **same windows**, drawn from the release evaluation schedule.
- The **tap** (i.e., the point in the model at which VE is applied/measured)
  must match the edited sublayer (e.g., **post‑`mlp.c_proj`, pre‑residual**);
  targets list those modules.

## Runtime Contract (certificate)

- Certificate records `variance.predictive_gate` with `{sided,min_effect,delta_ci,mean_delta,reason,evaluated}` and `ab_test.provenance` stating window IDs and seed for A/B.
- Lints reject enablement if CI contains 0 or if provenance is missing.

## Observability

- `variance.{ve_enabled,target_modules,proposed_scales}` — VE decision state and adjusted modules.
- `variance.predictive_gate.{sided,min_effect,delta_ci,mean_delta,reason}` — statistical outcome.
- `variance.ab_test.{seed,windows_used,provenance}` — reproducibility of the predictive A/B.
- `resolved_policy.variance.{min_effect_lognll,predictive_one_sided,max_adjusted_modules}` — tier knobs for the proof.

## Assumptions & Scope

- Predictive A/B runs reuse the same evaluation windows as the release schedule
  and are token-weighted identically.
- VE taps must target the edited modules (e.g., post `mlp.c_proj` for the
  edited projection); off-target taps invalidate the provenance check.
- Calibration statistics come from pilot null runs with the same window counts;
  different window budgets require recalibration of `min_effect_lognll`.

## References

- Wasserman, L. (2004). *All of Statistics: A Concise Course in Statistical Inference.* Springer. (See chapters on hypothesis testing and power analysis.)
- Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Brooks/Cole.
