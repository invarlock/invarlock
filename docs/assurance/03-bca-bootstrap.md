# BCa Bootstrap for Paired Δlog‑Loss (with Fallbacks)

> **Plain language:** Confidence intervals come from a paired, token‑weighted
> BCa bootstrap on Δlog‑loss; the ratio CI is just the exponentiated Δlog CI.
> When Δ is degenerate, or BCa’s acceleration term is undefined, we fall back
> safely.

## Claim

Paired, token‑weighted BCa on Δlog‑loss yields a ratio CI by exponentiation.
When Δ is degenerate or acceleration is undefined, the implementation falls
back safely (e.g., percentile CI or a collapsed interval).

## Method (paired, token‑weighted)

- Samples: paired Δlog‑loss per window, token‑weighted (`t_i`).
- Bootstrap: Bias‑Corrected and Accelerated (BCa), replicates `N` (α = 0.05 by default).
- Seeding: `meta.seeds.bootstrap` recorded for reproducibility.

Given per‑window token counts `t_i` and log‑losses `ℓ_i^A`, `ℓ_i^B`, define

- `Δℓ_i = ℓ_i^B − ℓ_i^A` (paired, token‑weighted)
- Compute CI `[L, U]` via BCa on `{Δℓ_i}` (resampled with probability `∝ t_i`).
- Perplexity ratio CI is `exp([L, U])`.

## Fallbacks

- Degenerate Δ (all equal, no pairs, or single pair): mark `degenerate=true`; CI collapses to `[μ, μ]` with `μ = mean(Δ)`.
- Undefined acceleration (jackknife variance is zero): fall back to a percentile bootstrap CI.

## Runtime Contract (certificate)

- `ppl.logloss_delta_ci` — Δlog‑loss CI (log space)
- `ppl.ratio_ci` — ratio CI = `exp(ppl.logloss_delta_ci)`
- Identity checks:
  - `ppl.preview_final_ratio == exp(weighted_mean(Δlog))`
  - `ppl.logloss_delta_ci` exponentiates to `ppl.ratio_ci`
- `ppl.stats.bootstrap.{replicates,seed,method}`
- `ppl.stats.paired_delta_summary.{mean,std,degenerate}`

## Defaults & Tuning (tiers)

- Balanced: ≈ 180×180 windows, BCa replicates ≈ 1.2k.
- Conservative: ≈ 220×220 windows, BCa replicates ≈ 1.5k.

Adjust only with an audit note; always record the seed. CI/Release profiles
enforce minima strictly when pairing is established.

## Notes

- Pairing and non‑overlap are required; see Coverage & Pairing Plan.
- BCa is numerically stable under typical window counts; for extreme small‑n, expect more frequent fallbacks.

## Assumptions & Scope

- Paired windows and token weighting are required for the log‑space identities
  to hold.
- Degenerate Δ cases are rare in practice at tier coverage; when they occur,
  the certificate records the fallback explicitly.

## References

- Efron, B. (1987). “Better Bootstrap Confidence Intervals.” *Journal of the American Statistical Association*, 82(397), 171–185. <https://www.jstor.org/stable/2289144>
- DiCiccio, T. J., & Efron, B. (1996). “Bootstrap Confidence Intervals.” *Statistical Science*, 11(3), 189–228. <https://projecteuclid.org/journals/statistical-science/volume-11/issue-3/Bootstrap-Confidence-Intervals/10.1214/ss/1032280214.full>
- Canty, A., & Davison, A. C. (2021). “bcaboot: Bias Corrected Bootstrap Confidence Intervals.” R package vignette. <https://cran.r-project.org/web/packages/bcaboot/bcaboot.pdf>
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.* Chapman & Hall/CRC.
