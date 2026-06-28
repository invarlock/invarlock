# Spectral Guard False Positive Rate (FPR)

> **Plain language:** The spectral guard records a calibrated multiple-testing
> policy for per-family singular-value drift. Gaussian-tail FPR math applies to
> the families whose kappas were calibrated for that model. Published-basis
> no-op reports for newer families are null-behavior evidence, but their
> transferred caps are budgeted sentinels until a family-specific calibration
> re-derives κ. Low Balanced `embed`/`other` caps are operational sentinels,
> not standalone <=5% FPR claims.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Explain how spectral guard family caps map to false-positive-rate interpretation under modeled nulls. |
| **Audience** | Guard maintainers, calibration auditors, and contributors changing spectral policy. |
| **Contract scope** | Spectral z-score caps, multiple-testing policy, sentinel caps, and report observability. |
| **Source of truth** | `src/invarlock/guards/spectral*.py`, `runtime/tiers.yaml`, and spectral assurance tests. |

## Claim

The spectral guard monitors per-family z-scores and records the multiple-testing
policy needed to interpret WARNs under the chosen null-modeling assumptions.
For families whose kappas are calibrated against an approximately Gaussian null,
the two-sided tail probability gives the expected false-positive rate. Families
with intentionally low sentinel caps are still monitored and budgeted by
`max_caps`; cite them as operational thresholds rather than <=5% Gaussian-tail
claims.

## Derivation (sketch)

Per-family spectral monitoring computes z-scores

$$
z = \frac{s - \mu_f}{\sigma_f}
$$

for a spectral statistic $s$ (e.g., top singular value). A WARN is issued when
$|z| > \kappa_f$. Under a **modeled null** where $z \approx \mathcal{N}(0,1)$, the
per-module two-sided tail probability becomes

$$
p_{\text{tail}} \approx 2\big(1 - \Phi(\kappa_f)\big).
$$

Applying **Bonferroni** across the tested families controls the family-wise
error rate (FWER); applying **Benjamini–Hochberg (BH)** controls the expected
false-discovery proportion (FDR). Module-level WARN volume is budgeted
separately by `max_caps`. Balanced tiers choose BH (α=0.05, m=4 families);
Conservative tiers choose Bonferroni (α=0.000625, m=4 families). Document the
policy alongside $\kappa_f$ so auditors can recover the expected per-run WARN
rate.

## Assumptions & Scope

- Baseline runs provide $(\mu_f, \sigma_f)$ per family
  `f in {ffn, attn, embed, other}`; when $\sigma_f = 0$
  we fall back to the tier deadband δ.
- Only 2‑D weight matrices (FFN blocks, attention projections, embeddings) are
  evaluated; **1‑D LayerNorm parameters are explicitly excluded** from spectral
  monitoring. LayerNorm coverage is provided by invariants (presence checks)
  and activation‑based RMT (CI/Release); VE captures any aggregate performance shift.
- Balanced tier stores **Benjamini-Hochberg** metadata (`method = "bh"`, alpha =
  0.05, m = 4 families) with per-family caps `{ffn: 3.849, attn: 3.018, embed: 1.05,
  other: 0.0}`, `sigma_quantile = 0.95`, and `max_caps = 5`. Scope is `all`, so
  FFN, attention, embeddings, and other 2-D weights are all monitored. The
  Gaussian-tail FPR interpretation is defensible only for families with a
  matching null calibration basis. In the packaged pilot basis that means the
  GPT-2/BERT-calibrated high-kappa families; newly promoted published-basis
  causal LMs may expose `attn` cap hits in no-op reports and should treat the
  transferred attention cap as a budgeted sentinel until κ is recalibrated for
  that family. The lower `embed` and `other` caps are sentinel thresholds and
  can exceed a 5% Gaussian tail if interpreted alone.
- Conservative tier applies **Bonferroni** (`method = "bonferroni"`, α = 0.000625,
  m = 4) with caps `{ffn: 3.849, attn: 2.6, embed: 2.8, other: 2.8}`,
  `sigma_quantile = 0.90`, and `max_caps = 3`, keeping WARNs within the
  calibrated budget. Scope is `ffn` in the included tier policies, so only FFN
  blocks are actively budgeted under the Conservative spectral guard.
- Deadband δ suppresses flicker around the cap: Balanced records δ = 0.10,
  Conservative δ = 0.05, surfaced in reports via
  `spectral.summary.deadband`.
- reports expose the policy under
  `spectral.multiple_testing.{method,alpha,m}`,
  `spectral.summary.{sigma_quantile,max_caps,deadband}`, and
  `spectral.family_caps[*].kappa`.
- The FPR story is a calibration assumption under the chosen null model for the
  calibrated families, not a theorem about arbitrary transformer weights,
  transferred published-basis lanes, or sentinel thresholds.
- Empirical histograms of $z$ should be approximately standard normal; heavy
  tails → raise $\kappa_f$ or use robust $\sigma$ (MAD-scaled).

The deadband δ is a guardrail against flicker: relative changes within ±δ are
treated as neutral, so WARNs only fire when sustained growth exceeds both δ and
the family κ cap. Auditors can confirm the chosen δ directly in the report
summary.

## Runtime Contract (report)

- report exposes
  `spectral.summary.{sigma_quantile,deadband,modules_checked,max_caps,caps_exceeded}`,
  `spectral.family_caps`, and `spectral.families[family]` with `{max, mean,
  count, violations, kappa}`. `sigma_quantile` is the calibrated baseline
  percentile used to derive the reference target.
- Tier files document multiple-testing metadata and the mapping from
  $\kappa_f$ to modeled Gaussian tails for calibrated families. Transferred
  caps and sentinel caps should be audited as operational thresholds, not as
  FPR-controlled family caps.
- Policy metadata records the multiple-testing method
  (`spectral.multiple_testing`) and the cap limit (`spectral.max_caps`, mirrored
  in `spectral.summary.max_caps` where present).

## Observability

- `spectral.summary.{sigma_quantile,deadband,modules_checked,max_caps,caps_exceeded}`
- `spectral.family_caps[*].kappa` and per-family cap counters in
  `spectral.families[*]` (`violations` is the raw counter key for modules whose
  z-score crossed the family cap)
- `spectral.multiple_testing.{method,alpha,m}` and `spectral.max_caps`

### Worked example (Balanced tier)

- For FFN modules, `family_caps.ffn.kappa = 3.849`. Suppose a layer reports $z = 3.90$.
- report records a WARN, increments the raw
  `spectral.families[*].violations` cap counter for the affected family, and
  increments `spectral.caps_applied`.
- Balanced `max_caps = 5`. After the fifth WARN the guard continues to WARN;
  the sixth triggers `spectral.caps_exceeded=true` and the run aborts.
- Multiple-testing metadata shows `spectral.multiple_testing = {method: "bh",
  alpha: 0.05, m: 4}` so evidence readers can verify the published policy and compute
  modeled tails for the calibrated caps.

## Calibration

Calibration values are derived from null-sweep runs using the order-statistic
and parametric methods described in the tier calibration documentation
([09-tier-v1-calibration.md](09-tier-v1-calibration.md)). The calibrated κ
values are stored in the packaged `tiers.yaml`
(`runtime/tiers.yaml`; overrides use
`INVARLOCK_CONFIG_ROOT/runtime/tiers.yaml`).

To recalibrate, run null baselines (no edit) and collect per-family maximum
z-scores through the null-sweep summary tooling. The current summarizer
recommends κ(f) from the maximum observed family z-score plus a safety margin
and can lower the multiple-testing α if the observed any-warning rate exceeds
the target. Validate that subsequent null runs stay within the published
`max_caps` budget.

> *Basis column in Quality Gates tables: "point" = point estimate gate,
> "upper" = upper-bound gate, "point & upper" = both point and upper bounds must
> pass.*

## References

- Benjamini, Y., & Hochberg, Y. (1995). “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.” *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289–300. <https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>
- Dunn, O. J. (1961). “Multiple Comparisons among Means.” *Journal of the American Statistical Association*, 56(293), 52–64. <https://doi.org/10.1080/01621459.1961.10482090>
