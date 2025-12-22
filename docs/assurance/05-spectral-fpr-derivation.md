# Spectral Guard False Positive Rate (FPR)

> **Plain language:** The spectral guard keeps each layer’s singular values
> close to the baseline so the per-run false positive rate (FPR) stays within
> the calibrated WARN budget.

## Claim

The spectral guard enforces the calibrated WARN budget by monitoring per-family
z-scores and limiting the per-run FPR.

## Derivation (sketch)

Per-family spectral monitoring computes z-scores

$$
z = \frac{s - \mu_f}{\sigma_f}
$$

for a spectral statistic $s$ (e.g., top singular value). A WARN is issued if
$|z| \ge \kappa_f$. Under a **null** where $z \sim \mathcal{N}(0,1)$, the
per-module two-sided tail probability becomes

$$
p_{\text{tail}} \approx 2\big(1 - \Phi(\kappa_f)\big).
$$

Applying **Bonferroni** across the $m_f$ modules controls the family-wise error
rate (FWER); applying **Benjamini–Hochberg (BH)** controls the expected
false-discovery proportion (FDR). Balanced tiers choose BH (α=0.05, m=4);
Conservative tiers choose Bonferroni (α=0.02, m=4). Document the policy
alongside $\kappa_f$ so auditors can recover the expected per-run WARN rate.

## Assumptions & Scope

- Baseline runs provide $(\mu_f, \sigma_f)$ per family $f \in
  \{\text{ffn}, \text{attn}, \text{embed}, \text{other}\}$; when $\sigma_f = 0$
  we fall back to the tier deadband δ.
- Only 2‑D weight matrices (FFN blocks, attention projections, embeddings) are
  evaluated; **1‑D LayerNorm parameters are explicitly excluded** from spectral
  monitoring. LayerNorm coverage is provided by invariants (presence checks)
  and activation‑based RMT (CI/Release); VE captures any aggregate performance shift.
- Balanced tier uses the **Benjamini–Hochberg** procedure (`method = "bh"`, α =
  0.05, m = 4 families) with per-family caps `{ffn: 3.834, attn: 3.423, embed: 3.1,
  other: 3.1}`, `sigma_quantile = 0.95`, and `max_caps = 5`, yielding ≤5% WARN
  rate on null runs (calibrated from the November 2025 pilot and stored in
  `tiers.yaml`). Scope is `all`, so FFN, attention, embeddings, and other 2‑D
  weights are all monitored.
- Conservative tier applies **Bonferroni** (`method = "bonferroni"`, α = 0.02,
  m = 4) with caps `{ffn: 2.3, attn: 2.6, embed: 2.8, other: 2.8}`,
  `sigma_quantile = 0.90`, and `max_caps = 3`, keeping WARNs near 2%. Scope is
  `ffn` in the shipped tier policies, so only FFN blocks are actively budgeted
  under the Conservative spectral guard.
- Deadband δ suppresses flicker around the cap: Balanced records δ = 0.10,
  Conservative δ = 0.05, surfaced in certificates via
  `spectral.summary.deadband`.
- Certificates expose the calibrations under
  `spectral.multiple_testing.{method,alpha,m}`,
  `spectral.summary.{sigma_quantile,max_caps,deadband}`, and
  `spectral.family_caps[*].kappa`.
- Empirical histograms of $z$ should be approximately standard normal; heavy
  tails → raise $\kappa_f$ or use robust $\sigma$ (MAD-scaled).

The deadband δ is a guardrail against flicker: relative changes within ±δ are
treated as neutral, so WARNs only fire when sustained growth exceeds both δ and
the family κ cap. Auditors can confirm the chosen δ directly in the certificate
summary.

## Runtime Contract (certificate)

- Certificate exposes
  `spectral.summary.{sigma_quantile,deadband,modules_checked,max_caps,caps_exceeded}`,
  `spectral.family_caps`, and `spectral.families[family]` with `{max, mean,
  count, violations, kappa}`. `sigma_quantile` is the calibrated baseline
  percentile used to derive the reference target. Legacy alias `contraction`
  may appear only in historical certificates.
- Tier files document FPR targets and mapping $\kappa_f \rightarrow$ expected WARNs.
- Policy metadata records the multiple-testing method
  (`spectral.multiple_testing`) and the cap limit (`spectral.max_caps`).

## Observability

- `spectral.summary.{sigma_quantile,deadband,modules_checked,max_caps,caps_exceeded}`
- `spectral.family_caps[*].kappa` and `spectral.families[*].{kappa,violations}`
- `spectral.multiple_testing.{method,alpha,m}` and `spectral.max_caps`

### Worked example (Balanced tier)

- For FFN modules, `family_caps.ffn.kappa = 3.834`. Suppose a layer reports $z = 3.90$.
- Certificate records a WARN in `spectral.families.ffn.violations += 1`; `spectral.caps_applied` increments.
- Balanced `max_caps = 5`. After the fifth WARN the guard continues to WARN;
  the sixth triggers `spectral.caps_exceeded=true` and the run aborts.
- Multiple-testing metadata shows `spectral.multiple_testing = {method: "bh",
  alpha: 0.05, m: 4}` so reviewers can verify the tier-wide correction.

## Calibration

Calibration values are derived from null-sweep runs using the order-statistic
and parametric methods described in the tier calibration documentation
([09-tier-v1-calibration.md](09-tier-v1-calibration.md)). The calibrated κ
values are stored in the packaged `tiers.yaml`
(`invarlock._data.runtime/tiers.yaml`).

To recalibrate, run null baselines (no edit) and collect per-module z-scores.
Allocate the WARN budget across families proportionally by module count, then
set κ(f) via order-statistic (the B(f)-th largest |z| in that family) or
parametric inversion of the tail probability. Add a small safety margin
(η ≈ 0.05–0.10) and validate that subsequent null runs stay within the budget.

> *Basis column in Quality Gates tables: "point" = point estimate gate,
> "upper" = upper-bound gate, "point & upper" = both point and upper bounds must
> pass.*

## References

- Benjamini, Y., & Hochberg, Y. (1995). “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.” *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289–300. <https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>
