# Variance Guard Predictive Gate (decision rule and sidedness)

> **Plain language:** VE only proposes scales when the reported predictive paired
> ΔlogNLL estimate meets the configured improvement rule—Balanced uses a
> one-sided interval and Conservative a two-sided interval—and the report
> explains why VE stayed on or off. Meeting that rule is not by itself proof of
> an out-of-sample improvement.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Define when the variance guard may enable scaling and how the predictive gate is audited. |
| **Audience** | Variance guard maintainers, calibration auditors, and release approvers checking VE evidence. |
| **Contract scope** | Predictive paired delta-log-loss gates, tier sidedness, minimum effect, and enablement provenance. |
| **Source of truth** | `src/invarlock/guards/variance*.py`, the packaged `runtime/tiers.yaml` policy resource, and variance assurance tests. |

## Claim

VE proposes scales only when the **predictive** paired ΔlogNLL CI upper bound
and mean effect are both negative and also meet or beat
−`min_effect_lognll`, using the tier's sidedness. With a zero min-effect,
zero itself does not pass: the upper bound and mean must be strictly negative.

- **Balanced**: **one‑sided improvement** test. VE enables only when the
  predictive CI **upper bound** and mean Δ are both `< 0` and both are ≤
  −`min_effect_lognll`.
- **Conservative**: **two‑sided** CI with $z = z_{0.975}$ and
  **improvement‑only** gating.
  VE enables only when the predictive CI **upper bound** and mean Δ are both
  negative and both are ≤ −`min_effect_lognll`. VE stays off for a positive
  interval; the gate records `regression_detected` only when the CI lower bound
  and mean are both at least +`min_effect_lognll`.

Example (Balanced): with `min_effect_lognll = 0.0`, a predictive
`mean_delta` of `-0.002` with CI `[-0.003, -0.001]` enables VE because both the
mean and the CI upper bound beat `-min_effect_lognll`.

Example (Conservative): with `min_effect_lognll = 0.016`, a predictive estimate
`mean_delta = -0.020` with CI `[-0.030, -0.017]` enables VE because the entire
CI lies outside the interval `[-min_effect_lognll, +min_effect_lognll]`. A CI
that touches or sits within this interval (e.g., `[-0.015, -0.002]`) does not
enable VE.

## Planning approximation

Let paired Δ values on calibration windows have standard deviation
$\sigma_{\text{pred}}$ and count $n$. The CI half-width is approximately:

$$
h \approx z \cdot \frac{\sigma_{\text{pred}}}{\sqrt{n}}
$$

Use $z = z_{0.95}$ for one-sided gates or $z = z_{0.975}$ for two-sided gates.
Choose `min_effect_lognll ≈ h` as an approximate sizing heuristic for boundary
cases; raise for stricter tiers.

## Tier knobs

| Tier          | deadband | min_abs_adjust | max_scale_step | min_effect_lognll | predictive\_one\_sided | max\_adjusted\_modules |
|---------------|----------|----------------|----------------|-------------------|------------------------|------------------------|
| balanced      | 0.02     | 0.012          | 0.03           | 0.0               | ✅ (one-sided)          | 1                      |
| conservative  | 0.03     | 0.02           | 0.015          | 0.016             | ❌ (two-sided)          | 0                      |

Values are stored in packaged `runtime/tiers.yaml`. They define behavior; the
public repository does not contain a representative study establishing their
false-enable rate or detection power under the chosen window budgets.

> **Source of truth:** tier thresholds are drawn from packaged
> `runtime/tiers.yaml`; overrides use
> `INVARLOCK_CONFIG_ROOT/runtime/tiers.yaml`.
>
> **Note:** `max_adjusted_modules = 0` means no module-count cap is enforced in
> the current VE scaling policy, not "adjust zero modules."

## Strict-verifier constraint

The runtime can leave VE disabled after a complete no-benefit experiment and
continue the edit. The current strict verifier is narrower: it requires
`variance.predictive_gate.evaluated=true` and `passed=true` for strict
acceptance, regardless of `variance.enabled`. Therefore a complete no-benefit
predictive-gate outcome is not currently acceptable as strict evidence. Treat
this as a product-contract constraint, not as evidence that VE must improve
every valid edit.

## Evidence status and local recalibration

The shipped `min_effect_lognll` values are operational policy defaults. The
half-width expression is a normal-approximation sizing heuristic, not the
implemented BCa interval and not evidence that the defaults achieve a stated
power. See [Tier Policy Values](09-tier-v1-calibration.md).

To propose a local replacement, run null baselines (no edit) and compute the paired Δ standard
deviation $\hat{\sigma}$ across calibration windows. Use $z = z_{0.95}$ for
one-sided gates (Balanced) or $z = z_{0.975}$ for two-sided gates
(Conservative), then use the half-width formula as an initial candidate. Measure
false enables and power on independent held-out null and edited runs before
adopting or describing the value as calibrated.

## Provenance & tap

- VE must evaluate A = **edited model (no VE)** and B = **virtual VE** on the
  **same windows**, drawn from the release evaluation schedule.
- The **tap** (i.e., the point in the model at which VE is applied/measured)
  must match the edited sublayer (e.g., **post‑`mlp.c_proj`, pre‑residual**);
  targets list those modules.

## Runtime Contract (report)

- report records `variance.predictive_gate` with `{evaluated,passed,reason,delta_ci,mean_delta}` and `variance.ab_test` with `{seed,windows_used,provenance}`; provenance states the window IDs for A/B.
- Tier knobs for sidedness and min-effect are recorded under `resolved_policy.variance.{predictive_one_sided,min_effect_lognll}`.
- Report verification rejects `variance.enabled = true` when the predictive
  gate did not pass, the predictive CI includes zero or misses the
  `min_effect_lognll` threshold, the mean Δ misses the same threshold, or A/B
  seed/window provenance is missing. Strict assurance always requires a
  complete predictive-gate result that can be independently replayed. A complete
  failing predictive-gate outcome blocks under `enforce` and remains observed under `observe`;
  missing, degraded, or monitor-only evidence blocks in either mode.

These checks validate report consistency. The offline verifier does not replay
the predictive A/B computation from model tensors and cannot establish that the
evaluation environment honestly generated or selected the recorded windows.

## Observability

- `variance.{enabled,target_modules,proposed_scales}` — VE decision state and adjusted modules.
- `variance.predictive_gate.{delta_ci,mean_delta,reason,passed}` — statistical outcome.
- `variance.ab_test.{seed,windows_used,provenance}` — reproducibility of the predictive A/B.
- `resolved_policy.variance.{min_effect_lognll,predictive_one_sided,max_adjusted_modules}` — tier knobs for the evidence check.

## Assumptions & Scope

- Predictive A/B runs reuse the same evaluation windows as the release schedule
  and are token-weighted identically.
- VE taps must target the edited modules (e.g., post `mlp.c_proj` for the
  edited projection); off-target taps invalidate the provenance check.
- The shipped threshold has no public universal calibration claim. Different
  model, edit, dataset, device, and window regimes require their own held-out
  evaluation if a statistical operating characteristic is claimed.

## References

- Wasserman, L. (2004). *All of Statistics: A Concise Course in Statistical Inference.* Springer. (See chapters on hypothesis testing and power analysis.)
- Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Brooks/Cole.
