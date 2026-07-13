# Tier Policy v1 Values and Recalibration Method

> **Plain language:** This appendix has two roles:
> (1) document the shipped **operational defaults** for the **Balanced** and
> **Conservative** tiers; and
> (2) give a recipe for proposing replacements on your setup
> (weight-based Spectral κ, activation-based RMT ε, VE min-effect, and window
> sizing).
> Every knob is surfaced in reports so readers can inspect the applied policy.
> The repository does **not** contain an independent, representative calibration
> corpus establishing false-positive, power, or confidence-interval coverage
> guarantees for every shipped number. Public evidence bundles demonstrate
> report and evidence-pack mechanics plus limited observed runs; they should not
> be read as population calibration.
>
> For a key-by-key explanation of every value in the packaged tier file
> (`runtime/tiers.yaml`; override path
> `INVARLOCK_CONFIG_ROOT/runtime/tiers.yaml`), see
> [Tier Policy Catalog](../reference/tier-policy-catalog.md).

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | State shipped policy values, their evidentiary limits, and a recipe for proposing locally evaluated replacements. |
| **Audience** | Calibration owners, release approvers, and contributors updating tier thresholds. |
| **Contract scope** | Balanced and Conservative tier values for Spectral kappa, RMT epsilon, VE min-effect, and window sizing. |
| **Source of truth** | Packaged `runtime/tiers.yaml` for behavior; independently reviewed calibration artifacts are required to support statistical interpretations. |

## Spectral κ (z-caps) — Targets **and** Method

### What the tier ships with

- **Balanced** per-family κ caps:
  `ffn: 3.849`, `attn: 3.018`, `embed: 1.05`, `other: 0.0`
  with a **BH-named candidate selector** (`α=0.05`, `m=4` configured
  families), **deadband** `δ=0.10`, **scope: all** 2-D weight matrices
  (LayerNorm excluded), **no absolute clamp**, and exploratory cap budget
  `max_caps = 5`.
- **Conservative** uses a different family-cap profile and a lower cap budget:
  `ffn: 3.849`, `attn: 2.6`, `embed: 2.8`, `other: 2.8`, a
  **Bonferroni-named candidate selector** (`α=0.000625`), and `max_caps = 3`.

The implementation combines that candidate selector with fixed per-family caps
and a cap budget. In the current external-baseline strict path, any selected
spectral violation blocks; `max_caps` governs exploratory local-baseline
behavior. The selector has not been shown to satisfy formal FDR or family-wise
error-rate control for the emitted decision. See
[Spectral Threshold Semantics](05-spectral-fpr-derivation.md).

**Runtime visibility.** reports record per-family WARNs and effective caps under
`spectral.*` (summary, multiple_testing, families, family_caps) and the resolved
policy under `resolved_policy.spectral`.

### Window Minima Rationale (counts/power)

- The CI profile targets 240×240 non‑overlapping, paired windows with BCa
  replicates ≈ 1.2k. The Release profile targets 400×400 with ≈ 3.2k
  replicates. Tier floors remain lower policy guard rails (Balanced 180×180,
  Conservative 220×220) so profiles can request stricter counts. These counts
  are engineering cost/evidence floors. The half-width expression below is a
  planning heuristic, not public evidence of power or nominal BCa coverage.
- CI/Release profiles request stricter counts than the base tier floors. The
  runtime/report gates enforce perfect pairing, zero overlap, and selected
  tier-floor minima; readers should compare requested profile counts to the
  recorded used counts when judging a release evidence package.

**Spectral evidence status.** The repository may be reviewed with an external
empirical-evidence bundle, but no in-tree corpus establishes the shipped caps as
calibrated population thresholds. Public-basis reports provide observed examples
and contract fixtures; they do not re-derive `runtime/tiers.yaml`. Local tooling can parse
evaluation report JSON files (glob pattern `**/evaluation.report.json`) and run
reports to extract spectral evidence, summarize per-family maximum z-scores,
and recommend updated family caps and multiple-testing α. Persist results in
JSON/Markdown/CSV form with hashes for reproducibility and attach calibration
reports to change proposals.

---

### How to recalibrate κ on your machine (budget-aware)

> **Key idea.** Keep the **budget** `max_caps` fixed (e.g., 5 for Balanced);
> tune per-family κ so the reviewed null sample stays inside that budget under
> the configured candidate-selection policy. **Do not** enable an absolute clamp in
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
   any-warning rate is above target, it may also report a lower
   candidate-selection α from a bounded grid. Treat an α change as a separate
   policy proposal; the worked cap-only procedure below keeps α fixed. Neither
   recommendation proves FDR/FWER control.

4. **Parametric cross-check.** With two-sided tail
   $\mathrm{pTail}(\kappa)=2\big(1-\Phi(\kappa)\big)$, compare the proposed caps
   to modeled Gaussian tails. This is only a diagnostic model check. Treat all
   caps as operational thresholds unless a representative, held-out null study
   supports a stated false-positive interpretation.

5. **Keep these fixed (Balanced).** `multiple_testing: {method: bh, alpha: 0.05, m: 4}`, `deadband: 0.10`, `scope: all`, `max_caps: 5`, `max_spectral_norm: null`.

> **Spectral is weight-based.** z-tails are driven by weights, not evaluation windows; changing dataset seeds/windows **does not** move |z|. Prefer pooling per-module z across related baselines (e.g., 1B/3B/7B) rather than re-sampling windows.

### Worked Example: Recalibrating Spectral κ for a Custom GPT-2 Run

Suppose you ran a baseline and extracted z-scores from the report:

```bash
# Calibration-only / non-assurance example.
# Do not accept host-mode output as strict assurance evidence.
# 1. Run baseline
invarlock evaluate --allow-network --execution-mode host \
  --assurance off \
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
   revise the cap proposal or open a separate candidate-selector policy change.
   Do not infer per-family budgets: the current runtime applies one global
   `max_caps` count after family selection.

4. **Write local override:** Start from
   `configs/overrides/spectral_balanced_local.example.yaml`, copy it for local
   editing, and update the proposed caps.

   ```yaml
   # Based on configs/overrides/spectral_balanced_local.example.yaml
   guards:
     spectral:
       family_caps:
         ffn: {kappa: 1.89}
         attn: {kappa: 2.73}
         embed: {kappa: 1.5}
         other: {kappa: 1.2}
   ```

5. **Re-run with override:** The command below uses the committed example path
   for reproducibility; replace it with your edited local copy when trialing new
   caps.

   ```bash
   # Calibration-only / non-assurance example.
   # Do not accept host-mode output as strict assurance evidence.
   invarlock evaluate --allow-network --execution-mode host \
     --assurance off \
     --baseline gpt2 \
     --subject gpt2 \
     --preset configs/presets/causal_lm/wikitext2_512.yaml \
     --edit-config configs/overrides/spectral_balanced_local.example.yaml \
     --profile ci \
     --tier balanced
   ```

6. **Review the result.** For exploratory local-baseline runs, check
   `spectral.caps_applied <= spectral.max_caps` and
   `spectral.caps_exceeded == false` (or their `spectral.summary` mirrors).
   For the current external-baseline strict path, any selected violation is a
   blocker even when it remains within `max_caps`.

---

## RMT ε (acceptance bands)

### What the tier ships with

* **Balanced** ε per family: `{ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01}`
* **Conservative**: `{ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01}`

Acceptance rule per family $f$: with baseline edge risk
$r_f^{\text{base}}$ and current edge risk $r_f^{\text{cur}}$:

$$
r_f^{\text{cur}} \le \left(1+\varepsilon(f)\right) r_f^{\text{base}}
$$

**Runtime visibility.** report fields under `rmt.*` report baseline/current edge‑risk, ε (default and by family), status, and `validation.rmt_stable`.

**RMT evidence status.** The shared `0.01` value is a policy default. The public
repository does not contain a representative corpus demonstrating that it has a
particular false-alarm rate across model families, layers, datasets, and devices.
The current repo does not ship a dedicated RMT ε
calibration CLI summarizer; recalibration is a manually audited procedure
using report JSON fields such as `rmt.families.*.{edge_base,edge_cur,delta}`.
Report quantile summaries of Δ(f) = r_cur(f)/r_base(f) − 1 and skip cases with
missing or zero baseline.

---

### How to recalibrate ε

1. Run **null** baselines (no edit) and compute per-family deltas
   $\Delta(f) = r_{\text{cur}}(f)/r_{\text{base}}(f) - 1$ (skip cases with
   $r_{\text{base}}(f)=0$).
2. On a training/calibration split, propose
   $\varepsilon(f) = \mathrm{Quantile}(\Delta(f); q)$ with
   $q \in [0.95, 0.99]$.
3. Report family sample counts and uncertainty; do not infer a special
   small-family correction from the edge-risk value itself.
4. Measure the selected threshold on independent held-out null and edited runs
   before attaching a false-alarm or detection-power interpretation.

---

## Variance Equalization (VE) — minimum effect

### What the tier ships with

* **Balanced (one-sided, improvement-only)**: `min_effect_lognll = 0.0`
* **Conservative (two-sided, improvement-only)**: `min_effect_lognll = 0.016`

**Runtime visibility.** Recorded in reports under `variance.predictive_gate` (CI, mean Δ, pass/fail reason) and under `resolved_policy.variance.{predictive_one_sided,min_effect_lognll}` (tier knobs).

**VE evidence status.** These values are operational policy choices. The public
repository does not contain a representative study establishing their power or
false-enable rate. Local tooling can parse report JSON files to extract
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

The $z$ expression is a planning approximation, not the implemented BCa
coverage calculation. A proposal can use one-sided $z = z_{0.95}$ for Balanced
and two-sided $z = z_{0.975}$ for Conservative as an initial sizing heuristic.
VE enables only if the predictive CI upper bound and mean Δ are both negative
and both are at most `-min_effect_lognll`. When the min-effect is zero, equality
to zero does not pass. A positive interval keeps VE off; the implementation
uses the `regression_detected` reason only when the interval lower bound and
mean are both at least `+min_effect_lognll`.

---

## Evaluation window sizing (coverage)

Pick preview/final counts so the **BCa half-width** on ΔlogNLL is within target:

$$
\text{half-width} \approx z \cdot \frac{\hat{\sigma}}{\sqrt{n}}
$$

* A local study may choose a target such as ±0.001, but the repository does not
  establish this target or the shipped count as generally calibrated.
* Sweep $n$ to find the “coverage vs cost” knee; enforce **non-overlap** (`stride = seq_len`) and reuse baseline window IDs for perfect pairing.

**Window sizing provenance.** Window counts are controlled by the selected runtime
profile (`--profile ...`), defined under `src/invarlock/_data/runtime/profiles/`.
Repo-only runnable presets under `configs/presets/` set small defaults for
unprofiled runs.
**Runtime visibility.** reports expose window counts, coverage flags, and CI digests under `dataset.windows.stats` and `primary_metric`.

---

## Recalibration proposal workflow

1. **Baseline/null sweep.** Collect guard-level
   `final_z_scores` or evaluation-report spectral summaries.
2. **Spectral κ candidate.** Run the null-sweep summary and set κ from
   per-family max z plus a safety margin. Keep the selector method/α, deadband,
   scope, `max_caps`, and no-clamp policy fixed unless the proposal explicitly
   evaluates those changes too.
3. **RMT ε candidate.** From a calibration split, set $\varepsilon(f)$ to a
   quantile, declared in advance, of $r_f^{cur}/r_f^{base} - 1$ for families with a
   positive baseline, then validate it on held-out runs.
4. **VE min-effect candidate.** Use $z\,\hat{\sigma}/\sqrt{n}$ only as an initial sizing heuristic.
5. **Windows.** Propose $n$ from the cost/half-width trade-off; enforce non-overlap and pairing.
6. **Trial via override.** Write proposed values to a local override YAML (e.g., `configs/overrides/spectral_balanced_local.example.yaml`, copied locally and edited) and merge it into a local run preset under `guards:` instead of editing the global tier. Re-run baseline + edits; pre-screen gates; then build reports.
7. **Hold out and document.** Evaluate false alarms and detection behavior on
   runs not used to choose the thresholds, including the intended device/model
   strata. Preserve the full run selection and failures to avoid cherry-picking.

---

> **Note.** These numbers are defaults, not universal calibrated constants.
> Teams should evaluate them on their models, datasets, edits, and hardware and
> attach the complete selected-run protocol, reports, failures, and summary
> statistics to change proposals. Report fields improve auditability but do not
> make the evaluation environment or its sampling choices independently trustworthy.

## See Also

- [Tier Policy Catalog](../reference/tier-policy-catalog.md) — Policy keys and where they appear in reports
- [Guards Reference](../reference/guards.md) — Guard configuration options
- [BCa Bootstrap](03-bca-bootstrap.md) — Primary-metric interval mechanics
- [Spectral False-Positive Control](05-spectral-fpr-derivation.md) — Multiple-testing and spectral cap rationale
- [RMT Epsilon Rule](06-rmt-epsilon-rule.md) — Activation edge-risk rule
- [VE Predictive Gate](07-ve-gate-power.md) — Variance-effect threshold sizing
- [Diagnostic Empirical Guard Artifact Inventory](17-empirical-guard-evidence.md) — Non-authoritative inventory scope

## References

- Benjamini, Y., & Hochberg, Y. (1995). “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.” *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289–300. <https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>
