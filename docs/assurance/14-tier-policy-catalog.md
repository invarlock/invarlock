# Tier Policy Catalog (runtime `tiers.yaml`)

> **Plain language:** The packaged `runtime/tiers.yaml` is the source of truth
> for tier defaults. Some values are **calibrated** from pilot/null runs (e.g.,
> Spectral κ, RMT ε, VE min‑effect). The rest are **policies** (explicit design
> choices like sample-size floors, deadbands, and caps). This page is the “why
> map”: for every key in `tiers.yaml`, it explains what it controls and where to
> point for the rationale.

## Where the tier file lives

- Packaged default: `src/invarlock/_data/runtime/tiers.yaml`
- Override: set `INVARLOCK_CONFIG_ROOT` and provide `runtime/tiers.yaml` under it
  (see `docs/reference/env-vars.md`).

## Tier scope

Balanced and Conservative are the supported safety tiers; Aggressive is
research‑oriented and explicitly outside the safety case (see
`docs/assurance/00-safety-case.md`).

## Key-by-key rationale (what + why)

### Primary-metric gates (`metrics.*`)

These are run-level acceptance gates applied when generating/verifying a
certificate (see “Quality Gates” in `docs/assurance/04-guard-contracts.md`).
For a concrete run, `invarlock report explain` prints the resolved thresholds,
token/example floors, and whether hysteresis applied.

#### `metrics.pm_ratio.*` (ppl-like kinds)

- `ratio_limit_base` *(policy)* — the baseline-relative gate for ppl-like
  primary metrics (`ratio_vs_baseline ≤ ratio_limit_base`, and when a CI exists,
  `ratio_ci.upper ≤ ratio_limit_base`). Rationale and tier intent are described
  in `docs/assurance/04-guard-contracts.md` (“Primary metric … ppl-like kinds”).
- `min_tokens` *(policy)* — minimum total tokens (preview + final) required
  before enforcing the ppl ratio gate. Rationale: prevents noisy PASS/FAIL on
  tiny samples; keeps CI smokes meaningful while still allowing small local
  demos.
- `min_token_fraction` *(policy)* — dataset-scale-aware floor: when the runner
  knows available tokens, the effective floor becomes
  `max(min_tokens, ceil(tokens_available * min_token_fraction))`. Rationale:
  avoids “passing” large datasets using an unrepresentative tiny subset.
- `hysteresis_ratio` *(policy)* — small additive slack on the ratio gate
  (`ratio_limit_base + hysteresis_ratio`). Rationale: avoids PASS/FAIL flapping
  when results hover near the boundary; certificates mark when hysteresis was
  needed (`validation.hysteresis_applied`).

#### `metrics.pm_tail.*` (Primary Metric Tail gate; ppl-like kinds)

This is a **tail-regression backstop** computed on paired **per-window**
ΔlogNLL samples vs the baseline (window-by-window `logloss_subject - logloss_baseline`,
matched by `window_id` on the **final** schedule). It is additive to the mean/CI
primary-metric gate: it catches “localized” heavy-tail regressions that can be
hidden by an acceptable mean ratio.

- `mode` *(policy)* — `off|warn|fail`.
  - `warn` (default): violations are recorded in the certificate but do **not**
    fail validation (`validation.primary_metric_tail_acceptable` stays true).
  - `fail`: violations fail validation and can trigger rollback in `invarlock run`
    (`rollback_reason = primary_metric_tail_failed`).
- `min_windows` *(policy)* — minimum paired windows required before evaluating
  thresholds. Underpowered runs set `primary_metric_tail.evaluated = false` and
  do not warn/fail.
- `quantile` *(policy)* — which percentile to monitor (default 0.95 → P95).
  Quantiles are computed **unweighted** with deterministic linear interpolation
  on sorted ΔlogNLL values.
- `quantile_max` *(policy/calibration target)* — maximum allowed ΔlogNLL at the
  selected quantile (e.g., `P95 ≤ 0.20`).
- `epsilon` *(policy)* — deadband for “tail mass”: `tail_mass = Pr[ΔlogNLL > ε]`.
- `mass_max` *(policy/calibration target)* — maximum allowed tail mass. Defaults
  to 1.0 (non-binding) until calibrated.

Visibility:

- Certificate evidence: `primary_metric_tail.{stats,policy,violations}`.
- Validation flag: `validation.primary_metric_tail_acceptable` (false only in `fail` mode).
- CLI: `invarlock report explain` prints “Gate: Primary Metric Tail (ΔlogNLL)”.

#### `metrics.accuracy.*` (accuracy kinds)

- `delta_min_pp` *(policy)* — minimum allowed Δaccuracy vs baseline (percentage
  points). Defaults per tier are stated in `docs/assurance/04-guard-contracts.md`
  (“accuracy kinds … defaults”).
- `min_examples` *(policy)* — minimum `n_final` required before enforcing the
  Δaccuracy gate. Rationale: avoids gating on too few examples.
- `min_examples_fraction` *(policy)* — dataset-scale-aware floor: when available
  examples are known, the effective floor becomes
  `max(min_examples, ceil(examples_available * min_examples_fraction))`.
- `hysteresis_delta_pp` *(policy)* — small slack on the Δaccuracy gate
  (`delta_min_pp - hysteresis_delta_pp`). Rationale: avoids flapping near the
  boundary; marked in certificates via `validation.hysteresis_applied`.

### Spectral guard (`spectral_guard.*`)

These are the weight-based stability thresholds. Calibrated values and the
multiple-testing procedure are documented in:

- `docs/assurance/05-spectral-fpr-derivation.md` (policy + FPR control)
- `docs/assurance/09-tier-v1-calibration.md` (pilot numbers + recalibration)

Keys:

- `sigma_quantile` *(calibrated)* — which baseline percentile defines the
  reference sigma target used for z-scoring.
- `deadband` *(policy)* — z-score deadband δ to suppress flicker (see
  `docs/assurance/05-spectral-fpr-derivation.md`).
- `scope` *(policy)* — which families are actively budgeted/monitored (e.g.,
  `all` vs `ffn`), described in `docs/assurance/05-spectral-fpr-derivation.md`.
- `max_caps` *(policy)* — per-run WARN/cap budget; exceeding this aborts in
  CI/Release (see `docs/assurance/05-spectral-fpr-derivation.md`).
- `max_spectral_norm` *(policy)* — optional absolute clamp. `null` means “no
  absolute clamp”; rely on relative z-caps and the WARN budget (see
  `docs/assurance/09-tier-v1-calibration.md` “Keep these fixed … no clamp”).
- `family_caps` *(calibrated)* — per-family κ caps (stored as raw floats in
  `tiers.yaml`; normalized to `{family: {kappa: ...}}` at runtime).
- `multiple_testing` *(policy)* — the correction procedure used to interpret
  κ across families (`bh`/`bonferroni`, α, m); see
  `docs/assurance/05-spectral-fpr-derivation.md`.

### RMT guard (`rmt_guard.*`)

The acceptance gate for activation edge-risk is the ε-band rule, documented in:

- `docs/assurance/06-rmt-epsilon-rule.md` (acceptance rule + calibration)
- `docs/assurance/09-tier-v1-calibration.md` (recalibration recipe)

Keys:

- `epsilon_by_family` *(calibrated)* — ε(f) per family for the acceptance band:
  `edge_cur(f) ≤ (1 + ε(f)) · edge_base(f)`.
- `epsilon_default` *(calibrated)* — fallback ε used when a family-specific
  value is missing.
- `deadband` *(policy)* — additional tolerance used by the RMT outlier
  diagnostics/correction path (separate from ε-band acceptance), aligning the
  “ignore small changes” behavior with other guards.
- `margin` *(policy)* — safety multiplier for the same outlier
  diagnostics/correction path; higher margins tolerate more deviation before
  flagging.

### Variance guard (`variance_guard.*`)

The predictive gate and min-effect semantics are documented in:

- `docs/assurance/07-ve-gate-power.md` (power + sidedness + tier knobs)
- `docs/assurance/09-tier-v1-calibration.md` (min-effect recalibration)

Keys:

- `predictive_gate` *(policy)* — when true, VE only enables if the predictive
  A/B gate passes (certificate records `variance.predictive_gate.*`).
- `predictive_one_sided` *(calibrated policy)* — one-sided improvement gate
  semantics (Balanced) vs two-sided CI (Conservative); see
  `docs/assurance/07-ve-gate-power.md`.
- `min_effect_lognll` *(calibrated)* — minimum absolute improvement required for
  VE enablement; derived from `z·σ̂/√n` per tier, see
  `docs/assurance/07-ve-gate-power.md`.
- `deadband` *(policy)* — ignores small proposed adjustments (prevents
  “flicker”/tiny rescalings).
- `min_abs_adjust` *(policy)* — absolute floor on per-module |scale − 1| before a
  proposed scale is considered.
- `max_scale_step` *(policy)* — per-module maximum |scale − 1| applied in a
  single run (caps correction aggressiveness).
- `topk_backstop` *(policy)* — backstop that allows selecting the top candidate
  scale when filtering would otherwise produce no usable scales.
- `max_adjusted_modules` *(policy)* — optional cap on how many modules receive a
  scale in one run (0 means “no cap”).
- `tap` *(policy)* — module-name pattern(s) that define where VE is allowed to
  attach. Rationale: the tap must match the edited sublayer for provenance and
  reproducibility; see “Provenance & tap” in `docs/assurance/07-ve-gate-power.md`.
