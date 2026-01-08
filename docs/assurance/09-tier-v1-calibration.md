# Tier v1.0 Calibration (Pilot + Method)

> **Plain language:** This appendix has two roles:
> (1) the **pilot numbers** we measured for GPT-2 small, BERT base, and TinyLLaMA (Nov 2025) that underpin the **Balanced** and **Conservative** tiers; and
> (2) the **exact recipe** to recalibrate from scratch on your setup (weight-based Spectral κ, activation-based RMT ε, VE min-effect, and window sizing).
> Every knob is surfaced in run reports and certificates so reviewers can audit or recompute.
>
> For a key-by-key explanation of every value in the packaged tier file
> (`runtime/tiers.yaml`), see [Tier Policy Catalog](14-tier-policy-catalog.md).

---

## Spectral κ (z-caps) — Targets **and** Method

### What the tier ships with (pilot)

- **Balanced** per-family κ caps:
  `ffn: 3.849`, `attn: 3.423`, `embed: 3.1`, `other: 3.1`
  with **Benjamini–Hochberg (BH)** FDR control (`α=0.05`, `m=4` families), **deadband** `δ=0.10`, **scope: all** 2-D weight matrices (LayerNorm excluded), **no absolute clamp**, and **per-run WARN budget** `max_caps = 5`.
- **Conservative** tightens caps and budget:
  `ffn: 3.849`, `attn: 2.6`, `embed: 2.8`, `other: 2.8`, **Bonferroni** (`α=0.000625`), and `max_caps = 3`.

**Runtime visibility.** Certificates record per-family WARNs and effective caps under `spectral.*` (summary, multiple_testing, families, family_caps) and the resolved policy under `resolved_policy.spectral`.

### Window Minima Rationale (counts/power)

- The CI profile targets 200×200 non‑overlapping, paired windows with BCa replicates ≈ 1.2k. The Release profile targets 400×400 with ≈ 3.2k replicates. These counts follow a half‑width sizing rule on the paired Δlog‑loss CI (power ≈ 50% at the boundary for the chosen `min_effect_lognll`), verified on pilot runs.
- Release evidence must meet the requested counts; runs that under‑cover preview/final windows or bootstrap replicates fail certification in CI/Release profiles (see Coverage & Pairing Plan).

**Spectral calibration provenance.** Aggregated null-run stats are derived from
calibration runs. Local tooling can parse certificate JSON files (glob pattern
`**/cert_*.json`) to extract per-family z-scores and compute summary statistics
(mean, stdev, quantiles). Persist results in CSV format for reproducibility and
attach calibration certificates to change proposals.

---

### How to recalibrate κ on your machine (budget-aware)

> **Key idea.** Keep the **budget** `max_caps` fixed (e.g., 5 for Balanced); tune per-family κ so a clean baseline produces ≤ that many WARNs **per run** under BH. **Do not** enable an absolute clamp in Balanced.

1. **Gather per-module |z| by family.** From a baseline run, collect spectral z-scores
   $z(i) = \big(s(i) - \mu(f)\big)/\sigma(f)$ for each 2-D weight in family $f \in \{\text{ffn},\text{attn},\text{embed},\text{other}\}$.
   *(Tip: ensure the guard emits `final_z_scores` so you have module-level |z|.)*

2. **Allocate the WARN budget across families.** Let $m(f)$ be the module count in family $f$ and $M = \sum_{g} m(g)$ the total across families. With budget $B$ (Balanced: 5), assign
   $B(f) = \left\lfloor B \cdot \frac{m(f)}{M} + \tfrac{1}{2} \right\rfloor$

3. **Order-statistic recipe (recommended).** Sort $Z(f) = \{\, |z(i)| : i \in f\,\}$ in descending order; set $\kappa(f) = \max\big(\kappa^{\mathrm{default}}(f),\ Z(f)^{(B(f))}\big) + \eta$ with a small safety margin $\eta \in [0.05, 0.10]$ for robustness.

4. **Parametric alternative.** With two-sided tail $\mathrm{pTail}(\kappa)=2\big(1-\Phi(\kappa)\big)$ and target $m(f)\,\mathrm{pTail}(\kappa(f))\approx B(f)$, $\kappa(f) = \Phi^{-1}\left(1-\frac{B(f)}{2\,m(f)}\right)$ then add the same small margin.

5. **Keep these fixed (Balanced).** `multiple_testing: {method: bh, alpha: 0.05, m: 4}`, `deadband: 0.10`, `scope: all`, `max_caps: 5`, `max_spectral_norm: null`.

> **Spectral is weight-based.** z-tails are driven by weights, not evaluation windows; changing dataset seeds/windows **does not** move |z|. Prefer pooling per-module z across related baselines (e.g., 1B/3B/7B) rather than re-sampling windows.

---

## RMT ε (acceptance bands)

### What the tier ships with (pilot)

* **Balanced** ε per family: `{ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01}`
* **Conservative**: `{ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01}`

Acceptance rule per family $f$: with baseline edge‑risk $r_f^{\text{base}}$ and current edge‑risk $r_f^{\text{cur}}$,
$r_f^{\text{cur}} \le \left(1+\varepsilon(f)\right)\, r_f^{\text{base}}$.

**Runtime visibility.** Certificate fields under `rmt.*` report baseline/current edge‑risk, ε (default and by family), status, and `validation.rmt_stable`.

**RMT calibration provenance.** Aggregated null-run stats are derived from
calibration certificates. Local tooling can parse certificate JSON files to
extract `rmt.families.*.{edge_base,edge_cur,delta}` per family, and report
quantile summaries of Δ(f) = r_cur(f)/r_base(f) − 1 (skip cases with missing or
zero baseline).

---

### How to recalibrate ε

1. Run **null** baselines (no edit) and compute per-family deltas $\Delta(f) = r_{\text{cur}}(f)/r_{\text{base}}(f) - 1$ (skip cases with $r_{\text{base}}(f)=0$).
2. Set $\varepsilon(f) = \mathrm{Quantile}\big(\Delta(f);\ q\big)$ with $q \in [0.95, 0.99]$.
3. Use a slightly larger ε for tiny families (discreteness: $b(f)\in\{0,1\}$ matters).

---

## Variance Equalization (VE) — minimum effect

### What the tier ships with (pilot)

* **Balanced (one-sided, improvement-only)**: `min_effect_lognll = 0.0`
* **Conservative (two-sided, improvement-only)**: `min_effect_lognll = 0.016`

**Runtime visibility.** Recorded in certificates under `variance.predictive_gate` (CI, mean Δ, pass/fail reason) and under `resolved_policy.variance.{predictive_one_sided,min_effect_lognll}` (tier knobs).

**VE calibration provenance.** Summary stats are derived from calibration
certificates. Local tooling can parse certificate JSON files to extract
`variance.predictive_gate.{delta_ci,mean_delta}` and compute the paired Δ
standard deviation across runs.

---

### How to recalibrate min-effect

For paired ΔlogNLL with stdev $\hat{\sigma}$ over $n$ windows, $\text{min effect (logNLL)} \approx z \cdot \frac{\hat{\sigma}}{\sqrt{n}}$ with **Balanced** using one-sided $z = z_{0.95}$ and **Conservative** two-sided $z = z_{0.975}$. VE enables only if the predictive CI upper bound ≤ −`min_effect_lognll` and the mean Δ ≤ −`min_effect_lognll`; a CI entirely above +`min_effect_lognll` is treated as regression (VE stays off).

---

## Evaluation window sizing (coverage)

Pick preview/final counts so the **BCa half-width** on ΔlogNLL is within target:

$$
\text{half-width} \approx z \cdot \frac{\hat{\sigma}}{\sqrt{n}}
$$

* **Balanced pilot target:** ±0.001 on GPT-2/TinyLLaMA release profile (CI profile uses fewer windows).
* Sweep $n$ to find the “coverage vs cost” knee; enforce **non-overlap** (`stride = seq_len`) and reuse baseline window IDs for perfect pairing.

**Window sizing provenance.** Window counts are documented in preset configs
under `configs/tasks/*/ci_*.yaml` and `configs/tasks/*/release_*.yaml`.
**Runtime visibility.** Certificates expose window counts, coverage flags, and CI digests under `dataset.windows.stats` and `primary_metric`.

---

## “Fast path” recalibration (summary)

1. **Baseline (release, Balanced).** Run once and collect `final_z_scores`.
2. **Spectral κ.** Allocate budget ($B=5$) → per-family ($B(f)$); compute $\kappa(f)$ via order-statistic (or parametric) + margin; **keep** BH, deadband, scope, `max_caps`, and **no clamp**.
3. **RMT ε.** From null runs, set $\varepsilon(f)$ to the q95–q99 quantile of $\big(g(f)/b(f) - 1\big)$ per family (adjust for small $b(f)$).
4. **VE min-effect.** ($\approx z\,\hat{\sigma}/\sqrt{n}$) with tier-appropriate sidedness.
5. **Windows.** Size $n$ to hit the half-width target; enforce non-overlap and pairing.
6. **Trial via override.** Write calibrated values to a local override YAML (e.g., `configs/overrides/spectral_balanced_local.yaml`) and merge it into a local run preset under `guards:` instead of editing the global tier. Re-run baseline + edits; pre-screen gates; then build certificates.

---

> **Note.** These pilot numbers are defaults. Teams are encouraged to re-run
> calibration on their models/datasets/hardware and attach the resulting
> certificates and summary statistics to change proposals. The certificate
> fields make such updates auditable end-to-end.

## References

- Benjamini, Y., & Hochberg, Y. (1995). “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.” *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289–300. <https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>
