# BCa Bootstrap for Paired Baseline Δlog‑Loss

> **Plain language:** Confidence intervals come from a paired, token‑weighted
> BCa bootstrap on Δlog‑loss; the ratio CI is just the exponentiated Δlog CI.
> When Δ is degenerate, or BCa’s acceleration term is undefined, the bootstrap
> helper falls back to a deterministic interval path. Reports record the
> configured bootstrap method/seed/replicate metadata. Preview/final drift is a
> separate independent-slice percentile bootstrap, not part of this paired BCa
> contract.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Document the paired token-weighted bootstrap method and its fallback behavior. |
| **Audience** | Statistics auditors, report-verifier maintainers, and contributors changing CI computation. |
| **Contract scope** | Identical-ID baseline/subject final-window delta-log-loss confidence intervals and ratio-CI conversion. |
| **Source of truth** | `src/invarlock/core/bootstrap.py`, `src/invarlock/reporting/report_primary_metric_analysis.py`, `src/invarlock/reporting/verify_bootstrap.py`, and bootstrap replay tests. |

## Claim

The implementation computes a paired, token‑weighted BCa interval on the
selected Δlog‑loss windows and yields a ratio interval by exponentiation.
When Δ is degenerate or acceleration is undefined, the implementation falls
back to a percentile CI or collapsed interval. Reports record bootstrap
configuration, but do not currently tag every effective BCa acceleration
fallback separately.

## Method (paired, token‑weighted)

- Resampling units: paired windows, sampled uniformly with replacement.
- Statistic: token‑weighted mean Δlog‑loss using each sampled window's `t_i`.
- Bootstrap: Bias‑Corrected and Accelerated (BCa), replicates `N` (α = 0.05 by default).
- Seeding: the recorded `dataset.windows.stats.bootstrap.seed` is the base seed;
  paired baseline replay uses that value plus the published offset `503`.

Given per‑window token counts `t_i` and log‑losses `ℓ_i^A`, `ℓ_i^B`, define

- `Δℓ_i = ℓ_i^B − ℓ_i^A` (paired per-window difference)
- For each replicate, draw `n` indices uniformly from the `n` paired windows,
  carry the corresponding `( Δℓ_i, t_i )` pairs, and recompute
  `sum(t_i * Δℓ_i) / sum(t_i)`.
- Compute CI `[L, U]` from those statistics using the BCa bias and acceleration
  adjustments. Probability-proportional-to-token resampling is not used.
- Perplexity ratio CI is `exp([L, U])`.

## Fallbacks

- Empty or no-pair input is rejected by the bootstrap helper and surfaced by the
  report pipeline as invalid/degraded pairing evidence.
- Degenerate Δ (all equal values or a single pair): the CI collapses to `[μ,
  μ]`, where `μ` is the token-weighted mean Δ. The paired baseline CI itself
  does not currently expose a dedicated degeneracy flag.
- Undefined acceleration (jackknife variance is zero): fall back to a percentile bootstrap CI; this acceleration fallback is not separately tagged in report metadata.

## Disjoint Preview/Final Companion Interval

Preview and final are separate, non-overlapping slices. The runtime computes
their mean log-loss difference by resampling each arm independently and taking
the difference of the two token-weighted means in every replicate. The emitted
method is `independent_percentile_delta_log`. There is no index alignment, no
per-window preview/final Δ array, and no paired BCa interpretation.
Its recorded replay seed is the base bootstrap seed plus the published offset
`97`; the strict verifier derives and checks that value.

This companion interval is recorded under
`dataset.windows.stats.preview_final_slice_delta_summary`; it is not the
baseline ratio interval stored in `primary_metric.ci`.

In a current run's low-level `report.json`, `metrics.logloss_delta_ci`
also carries this preview/final independent-slice interval. The evaluation
report builder recognizes the explicit slice-summary basis and does not
promote that interval to `primary_metric.ci`; it computes the latter from
identical-ID baseline/subject final windows. Reports without explicit basis
metadata are rejected.

## Runtime Contract (report)

- `primary_metric.ci` — Δlog‑loss CI (log space, ppl-like kinds)
- `primary_metric.display_ci` — ratio CI = `exp(primary_metric.ci)`
- Identity checks:
  - `primary_metric.display_ci == exp(primary_metric.ci)`
  - preview/final drift ratio is computed from `primary_metric.{preview,final}`
- `dataset.windows.stats.bootstrap.{replicates,seed,method,alpha}`
- `dataset.windows.stats.coverage.{preview,final}` — tier-floor window coverage enforcement
- `dataset.windows.stats.preview_final_slice_delta_summary.{mean,ci,basis,paired,ci_method,degenerate}`
- `dataset.windows.stats.bootstrap.{preview_final_delta_basis,preview_final_delta_method,preview_final_delta_seed}`

## Defaults & Tuning (tiers)

- Balanced: ≈ 180×180 windows, BCa replicates ≈ 1.2k.
- Conservative: ≈ 220×220 windows, BCa replicates ≈ 1.5k.

Record every adjustment and its seed in the accompanying evidence or change
record. CI/Release profiles enforce configured minima strictly when pairing is
established. Replicate and window floors are evidence-volume policies, not a
proof of nominal coverage.

## Notes

- Baseline/subject final-window pairing, disjoint preview/final ID sets, and
  zero configured sliding-window token overlap are separate requirements; see
  [Coverage & Pairing Plan](02-coverage-and-pairing.md).
- Small samples and near-degenerate data can trigger fallbacks or unstable
  intervals; the code tests deterministic mechanics, not field coverage.

## Assumptions & Scope

- Paired windows and token weighting are required for the log‑space identities
  to hold.
- A collapsed independent preview/final interval is recorded in
  `preview_final_slice_delta_summary.degenerate`; it is not treated as a failed
  paired baseline comparison.
- Percentile and collapsed intervals are fallback evidence surfaces for
  auditability; they should not be treated as stronger than a normal BCa
  interval.
- No general 95% coverage claim is made for arbitrary serial dependence,
  heterogeneous cluster sizes, adaptive window selection, or cherry-picked
  runs. Those properties require a sampling model and independent simulation or
  empirical coverage study for the intended workload.

Current reports use `preview_final_slice_delta_summary`; preview and final are
disjoint slices and therefore cannot support a paired label.

## References

- Efron, B. (1987). “Better Bootstrap Confidence Intervals.” *Journal of the American Statistical Association*, 82(397), 171–185. <https://doi.org/10.1080/01621459.1987.10478410>
- DiCiccio, T. J., & Efron, B. (1996). “Bootstrap Confidence Intervals.” *Statistical Science*, 11(3), 189–228. <https://projecteuclid.org/journals/statistical-science/volume-11/issue-3/Bootstrap-Confidence-Intervals/10.1214/ss/1032280214.full>
- Efron, B., & Narasimhan, B. (2021). “bcaboot: Bias Corrected Bootstrap Confidence Intervals.” R package vignette. <https://cran.r-project.org/web/packages/bcaboot/bcaboot.pdf>
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.* Chapman & Hall/CRC.
