# Tier Policy v1 Calibration (Pilot + Method)

> **Plain language:** This appendix has two roles:
> (1) the **pilot calibration lanes** that underpin the **Balanced** and
> **Conservative** tiers; and
> (2) the **exact recipe** to recalibrate from scratch on your setup
> (weight-based Spectral κ, activation-based RMT ε, VE min-effect, and window
> sizing).
> Every knob is surfaced in run reports and reports so reviewers can audit or recompute.
> The public evidence floor is the packaged `published_basis` fixture set. That
> fixture set demonstrates the public report/evidence-pack contract; it is not
> the entire calibration corpus used to justify every numeric tier constant:
> `src/invarlock/_data/public_evidence/published_basis/gpt2/evaluation.report.json`,
> `src/invarlock/_data/public_evidence/published_basis/gpt2/evidence_pack_recipe.json`,
> `src/invarlock/_data/public_evidence/published_basis/gpt2/evidence_pack/`,
> `src/invarlock/_data/public_evidence/published_basis/bert/evaluation.report.json`,
> and `src/invarlock/_data/public_evidence/published_basis/bert/evidence_pack_recipe.json`.
>
> For a key-by-key explanation of every value in the packaged tier file
> (`src/invarlock/_data/runtime/tiers.yaml`; override path
> `INVARLOCK_CONFIG_ROOT/runtime/tiers.yaml`), see
> [Tier Policy Catalog](../reference/tier-policy-catalog.md).

---

## Spectral κ (z-caps) — Targets **and** Method

### What the tier ships with (pilot)

- **Balanced** per-family κ caps:
  `ffn: 3.849`, `attn: 3.018`, `embed: 1.05`, `other: 0.0`
  with **Benjamini–Hochberg (BH)** FDR control (`α=0.05`, `m=4` families), **deadband** `δ=0.10`, **scope: all** 2-D weight matrices (LayerNorm excluded), **no absolute clamp**, and **per-run WARN budget** `max_caps = 5`.
- **Conservative** tightens caps and budget:
  `ffn: 3.849`, `attn: 2.6`, `embed: 2.8`, `other: 2.8`, **Bonferroni** (`α=0.000625`), and `max_caps = 3`.

**Runtime visibility.** reports record per-family WARNs and effective caps under
`spectral.*` (summary, multiple_testing, families, family_caps) and the resolved
policy under `resolved_policy.spectral`.

### Window Minima Rationale (counts/power)

- The CI profile targets 240×240 non‑overlapping, paired windows with BCa
  replicates ≈ 1.2k. The Release profile targets 400×400 with ≈ 3.2k
  replicates. Tier floors remain lower guard rails (Balanced 180×180,
  Conservative 220×220) so profiles can request stricter counts. These counts
  follow a half‑width sizing rule on the paired Δlog‑loss CI (power ≈ 50% at the
  boundary for the chosen `min_effect_lognll`), verified on pilot runs.
- Release evidence must meet the requested counts; runs that under‑cover
  preview/final windows or bootstrap replicates fail evaluation in CI/Release
  profiles (see [Coverage & Pairing Plan](02-coverage-and-pairing.md)).

**Spectral calibration provenance.** Aggregated null-run stats are derived from
calibration runs. The repo ships the public published-basis reports and recipes
under `src/invarlock/_data/public_evidence/published_basis/{gpt2,bert}/`.
Local tooling can parse evaluation report JSON files (glob pattern
`**/evaluation.report.json`) and run reports to extract spectral evidence,
summarize per-family maximum z-scores, and recommend updated family caps and
multiple-testing α. Persist results in JSON/Markdown/CSV form with hashes for
reproducibility and attach calibration reports to change proposals.

---

### How to recalibrate κ on your machine (budget-aware)

> **Key idea.** Keep the **budget** `max_caps` fixed (e.g., 5 for Balanced);
> tune per-family κ so clean baselines stay inside that budget under the
> published multiple-testing policy. **Do not** enable an absolute clamp in
> Balanced.

1. **Gather spectral evidence.** From null/no-op runs, collect spectral guard
   evidence with per-family z-score summaries. Run reports may expose
   guard-level `final_z_scores` (or `extras.final_z_scores`); evaluation reports
   expose rendered spectral summaries such as `spectral.top_z_scores` when
   present.

2. **Summarize null sweeps.** Use the null-sweep calibration path
   (`invarlock advanced calibrate null-sweep`) or the underlying
   `summarize_null_sweep_reports` helper to compute:
   - `observed.family_max_z`
   - `observed.any_warning_rate`
   - `recommendations.family_caps`
   - `recommendations.multiple_testing`

3. **Cap recommendation.** The current summarizer recommends
   $\kappa(f) = \max_z(f) \times (1+\eta)$, rounded for report stability, where
   $\eta$ is the configured safety margin (default 0.05). If the observed
   any-warning rate is above target, it also recommends a lower
   multiple-testing α on a bounded grid.

4. **Parametric cross-check.** With two-sided tail
   $\mathrm{pTail}(\kappa)=2\big(1-\Phi(\kappa)\big)$, compare the proposed caps
   to modeled Gaussian tails for calibrated high-kappa families. Treat low
   Balanced `embed`/`other` caps as operational sentinels, not standalone
   Gaussian-tail claims.

5. **Keep these fixed (Balanced).** `multiple_testing: {method: bh, alpha: 0.05, m: 4}`, `deadband: 0.10`, `scope: all`, `max_caps: 5`, `max_spectral_norm: null`.

> **Spectral is weight-based.** z-tails are driven by weights, not evaluation windows; changing dataset seeds/windows **does not** move |z|. Prefer pooling per-module z across related baselines (e.g., 1B/3B/7B) rather than re-sampling windows.

### Worked Example: Recalibrating Spectral κ for a Custom GPT-2 Run

Suppose you ran a baseline and extracted z-scores from the report:

```bash
# Calibration-only / non-assurance example.
# Do not accept host-mode output as strict assurance evidence.
# 1. Run baseline
invarlock evaluate --allow-network --execution-mode host \
  --baseline gpt2 \
  --subject gpt2 \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --profile ci \
  --tier balanced \
  --out runs/baseline_calib \
  --report-out reports/baseline_calib

# 2. Extract z-scores from the baseline run report (example using jq)
jq '.guards[] | select(.name == "spectral") |
    (.final_z_scores // .extras.final_z_scores // .metrics.top_z_scores)' \
  runs/baseline_calib/source/*/report.json > z_scores.json
```

With 120 total modules distributed as: FFN=40, Attn=40, Embed=8, Other=32.

**Step-by-step κ calculation:**

1. **Summarize observed maxima.** Suppose the null-sweep summary reports 120
   total modules and the following per-family maxima:
   - `ffn`: 1.8
   - `attn`: 2.6
   - `embed`: 1.4
   - `other`: 1.1

2. **Apply margin.** With safety margin η=0.05, recommended κ values are:
   - κ(ffn) = 1.8 × 1.05 = **1.89**
   - κ(attn) = 2.6 × 1.05 = **2.73**
   - κ(embed) = 1.4 × 1.05 = **1.47**
   - κ(other) = 1.1 × 1.05 = **1.155**

3. **Review warning rate.** If observed any-warning rate exceeds the target,
   lower the multiple-testing α according to the null-sweep recommendation.

4. **Record module distribution for audit.** With budget B=5 and M=120 total modules:
   - B(ffn) = ⌊5 × 40/120 + 0.5⌋ = ⌊2.17⌋ = 2
   - B(attn) = ⌊5 × 40/120 + 0.5⌋ = 2
   - B(embed) = ⌊5 × 8/120 + 0.5⌋ = 1
   - B(other) = ⌊5 × 32/120 + 0.5⌋ = 1

5. **Write local override:**

   ```yaml
   # configs/overrides/spectral_local.yaml
   guards:
     spectral:
       family_caps:
         ffn: 1.89
         attn: 2.73
         embed: 1.5
         other: 1.2
   ```

6. **Re-run with override:**

   ```bash
   # Calibration-only / non-assurance example.
   # Do not accept host-mode output as strict assurance evidence.
   invarlock evaluate --allow-network --execution-mode host \
     --baseline gpt2 \
     --subject gpt2 \
     --preset configs/presets/causal_lm/wikitext2_512.yaml \
     --edit-config configs/overrides/spectral_local.yaml \
     --profile ci \
     --tier balanced
   ```

7. **Verify.** Check `spectral.caps_applied <= spectral.max_caps` and
   `spectral.caps_exceeded == false` (or the same values mirrored under
   `spectral.summary`) on clean baselines.

---

## RMT ε (acceptance bands)

### What the tier ships with (pilot)

* **Balanced** ε per family: `{ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01}`
* **Conservative**: `{ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01}`

Acceptance rule per family $f$: with baseline edge risk
$r_f^{\text{base}}$ and current edge risk $r_f^{\text{cur}}$:

$$
r_f^{\text{cur}} \le \left(1+\varepsilon(f)\right) r_f^{\text{base}}
$$

**Runtime visibility.** report fields under `rmt.*` report baseline/current edge‑risk, ε (default and by family), status, and `validation.rmt_stable`.

**RMT calibration provenance.** Aggregated null-run stats are derived from
calibration reports. The current repo does not ship a dedicated RMT ε
calibration CLI summarizer; recalibration is a manual/reviewer-audited procedure
using report JSON fields such as `rmt.families.*.{edge_base,edge_cur,delta}`.
Report quantile summaries of Δ(f) = r_cur(f)/r_base(f) − 1 and skip cases with
missing or zero baseline.

---

### How to recalibrate ε

1. Run **null** baselines (no edit) and compute per-family deltas
   $\Delta(f) = r_{\text{cur}}(f)/r_{\text{base}}(f) - 1$ (skip cases with
   $r_{\text{base}}(f)=0$).
2. Set $\varepsilon(f) = \mathrm{Quantile}(\Delta(f); q)$ with
   $q \in [0.95, 0.99]$.
3. Use a slightly larger ε for tiny families (discreteness: $b(f)\in\{0,1\}$ matters).

---

## Variance Equalization (VE) — minimum effect

### What the tier ships with (pilot)

* **Balanced (one-sided, improvement-only)**: `min_effect_lognll = 0.0`
* **Conservative (two-sided, improvement-only)**: `min_effect_lognll = 0.016`

**Runtime visibility.** Recorded in reports under `variance.predictive_gate` (CI, mean Δ, pass/fail reason) and under `resolved_policy.variance.{predictive_one_sided,min_effect_lognll}` (tier knobs).

**VE calibration provenance.** Summary stats are derived from calibration
reports. Local tooling can parse report JSON files to extract
`variance.predictive_gate.{delta_ci,mean_delta}` and compute the paired Δ
standard deviation across runs.

---

### How to recalibrate min-effect

For paired ΔlogNLL with standard deviation $\hat{\sigma}$ over $n$ windows:

$$
\text{min effect (logNLL)}
\approx
z \cdot \frac{\hat{\sigma}}{\sqrt{n}}
$$

Balanced uses one-sided $z = z_{0.95}$ and Conservative uses two-sided
$z = z_{0.975}$. VE enables only if the predictive CI upper bound is at most
`-min_effect_lognll` and the mean Δ is at most `-min_effect_lognll`; a CI
entirely above `+min_effect_lognll` is treated as regression, so VE stays off.

---

## Evaluation window sizing (coverage)

Pick preview/final counts so the **BCa half-width** on ΔlogNLL is within target:

$$
\text{half-width} \approx z \cdot \frac{\hat{\sigma}}{\sqrt{n}}
$$

* **Balanced pilot target:** ±0.001 on GPT-2 release profile (CI profile uses fewer windows).
* Sweep $n$ to find the “coverage vs cost” knee; enforce **non-overlap** (`stride = seq_len`) and reuse baseline window IDs for perfect pairing.

**Window sizing provenance.** Window counts are controlled by the selected runtime
profile (`--profile ...`), defined under `src/invarlock/_data/runtime/profiles/`.
Repo-only runnable presets under `configs/presets/` set small defaults for
unprofiled runs.
**Runtime visibility.** reports expose window counts, coverage flags, and CI digests under `dataset.windows.stats` and `primary_metric`.

---

## “Fast path” recalibration (summary)

1. **Baseline/null sweep (release, Balanced).** Collect guard-level
   `final_z_scores` or evaluation-report spectral summaries.
2. **Spectral κ.** Run the null-sweep summary, set κ from per-family max z plus
   safety margin, optionally lower α if warning rate is high; **keep** BH,
   deadband, scope, `max_caps`, and **no clamp** unless the change proposal says
   otherwise.
3. **RMT ε.** From null runs, set $\varepsilon(f)$ to the q95–q99 quantile of
   $g(f)/b(f) - 1$ per family (adjust when `b(f)` is small).
4. **VE min-effect.** Use $z\,\hat{\sigma}/\sqrt{n}$ with tier-appropriate sidedness.
5. **Windows.** Size $n$ to hit the half-width target; enforce non-overlap and pairing.
6. **Trial via override.** Write calibrated values to a local override YAML (e.g., `configs/overrides/spectral_balanced_local.example.yaml`, copied locally and edited) and merge it into a local run preset under `guards:` instead of editing the global tier. Re-run baseline + edits; pre-screen gates; then build reports.

---

> **Note.** These pilot numbers are defaults. Teams are encouraged to re-run
> calibration on their models/datasets/hardware and attach the resulting
> reports and summary statistics to change proposals. The report
> fields make such updates auditable end-to-end.

## See Also

- [Tier Policy Catalog](../reference/tier-policy-catalog.md) — Policy keys and where they appear in reports
- [Guards Reference](../reference/guards.md) — Guard configuration options
- [BCa Bootstrap](03-bca-bootstrap.md) — Primary-metric interval mechanics
- [Spectral False-Positive Control](05-spectral-fpr-derivation.md) — Multiple-testing and spectral cap rationale
- [RMT Epsilon Rule](06-rmt-epsilon-rule.md) — Activation edge-risk rule
- [VE Predictive Gate](07-ve-gate-power.md) — Variance-effect threshold sizing
- [Empirical Guard Evidence](17-empirical-guard-evidence.md) — Release evidence manifest scope

## References

- Benjamini, Y., & Hochberg, Y. (1995). “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.” *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289–300. <https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>
