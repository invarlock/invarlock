# Guard Contracts & Statistical Primer

> **Plain language:** This handbook explains what each guard checks, the
> thresholds we enforce, and how those decisions appear in the report so
> readers can trace every PASS or FAIL.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Consolidate guard contracts, statistical assumptions, thresholds, and report evidence fields. |
| **Audience** | Guard maintainers, release approvers, and contributors changing guard policy or reporting. |
| **Contract scope** | Invariants, Spectral, RMT, Variance, bootstrap sanity, calibration expectations, and report observability. |
| **Source of truth** | Guard modules under `src/invarlock/guards/`, report guard renderers, packaged tiers, and assurance-contract tests. |

**Contents:**

- [1. Guard Contracts](#1-guard-contracts) — what each guard checks and how it fails
- [2. Statistical Method Primer](#2-statistical-method-primer) — paired Δlog perplexity and bootstrap CIs
- [3. Evaluation Slice Contract Requirements](#evaluation-slice-contract-requirements) — acceptance criteria for evaluation schedules
- [4. Reproducibility Kit](#4-reproducibility-kit) — how to reproduce a report
- [5. Device Tolerance Guidance](#5-device-tolerance-guidance) — expected drift across backends
- [6. Threshold Rationale (Defaults)](#6-threshold-rationale-defaults) — why the defaults are what they are
- [7. Scope Boundaries](#7-scope-boundaries) — where the guard contract applies
- [8. Coverage Reference](#8-coverage-reference) — tests that underpin this handbook

This handbook captures the evidence claims that underpin InvarLock's guard
pipeline. It consolidates guard contracts, statistical assumptions, and the
evidence limits that accompany the InvarLock assurance notes.

## 1. Guard Contracts

| Guard | Inputs | Check & Threshold | Failure behavior | Code reference |
|-------|--------|-------------------|-------------------|----------------|
| **Invariants** | Baseline and subject model weights, adapter metadata | Separate pre-edit and post-edit checks cover non-finite parameters/buffers, tokenizer/vocabulary alignment, and configured structural invariants | Strict assurance forces `strict_mode=true` and `on_fail=block`; outside strict mode, non-finite/tokenizer failures are fatal and remaining findings follow configured policy | `invarlock.guards.invariants` |
| **Spectral** | 2‑D layer weights (FFN, attention proj, embeddings) | Compute the spectral score under the fixed measurement contract; apply candidate selection, `abs(z) > κ_f`, and cap-budget rules. The thresholds are operational; the combined decision has no established FDR/FWER guarantee. | Complete selected findings block under `enforce` and remain visible under `observe`. Fatal, unsupported, degraded, or incomplete evidence blocks in either mode. | `invarlock.guards.spectral` |
| **RMT** | Sampled activations | Globally center and scale the sampled activation matrix, run the configured finite-iteration estimator, and compute per-module/family edge-risk scores; evaluate baseline-relative growth against the configured ε band. This is not covariance whitening. | Complete ε-band findings block under `enforce` and remain visible under `observe`; catastrophic primary-metric spikes are mandatory and gated separately (`spike_threshold` defaults to 2.0× for ppl-like metrics). | `invarlock.guards.rmt` |
| **Variance (VE)** | Paired ΔlogNLL on predictive windows | Enable VE only if the predictive CI upper bound and mean are both negative and also meet −`min_effect_lognll` (Balanced uses a one-sided CI; Conservative uses a two-sided CI). | VE stays disabled when the gate fails. A complete failing predictive-gate outcome blocks under `enforce` and remains visible under `observe`; incomplete or degraded evidence always blocks. | `invarlock.guards.variance` |
| **Bootstrap sanity** | Evaluation windows, token counts | Matching window IDs, zero overlap; BCa replicates ≥ selected tier floor | Abort or fail verification and surface reason | `invarlock.reporting.report_make` |

Reports record a report-level policy digest plus guard metrics. Spectral and RMT
carry explicit measurement-contract evidence, and variance may include a
variance-policy digest; these fields are mirrored under `resolved_policy.*` and
the `spectral`/`rmt`/`variance` blocks. Offline verification checks those
reported values and bindings; it does not replay tensor measurements from a
checkpoint and cannot detect a malicious report author that fabricated internally
consistent inputs without an independent rerun or attestation.

For the two guard formulas that are easiest to misread in a table:

$$
z = \frac{\hat{s} - \mu_f}{\sigma_f}
$$

where $\hat{s}$ is an iterative estimate of the largest singular value under the
spectral measurement contract.

$$
r = \frac{\hat{\sigma}_{\max}(A')}{\sigma_{\mathrm{MP}}(m,n)}
$$

where $A'$ is the centered and standardized activation matrix and
$\sigma_{\mathrm{MP}}(m,n)$ is the Marchenko-Pastur edge for the same shape.

### Invariants: what is checked

- No non‑finite tensors (NaN/Inf) in model parameters.
- Weight‑tying relationships preserved (e.g., tied embeddings/output projection).
- Embedding/output dimensions consistent with tokenizer and adapter descriptors.
- Expected LayerNorm modules present; shape sanity checks across layers.
- Tokenizer alignment: when both baseline and edited tokenizers are available, mismatches abort.

### Catastrophic limits and aborts

- Spike stop: a primary‑metric spike above the configured threshold (for
  ppl‑like metrics, `ratio > 2.0` by default) triggers a hard abort/fail
  independent of guard warnings.
- Pairing/coverage: preview/final counts must match, pairing must be 1.0, overlap 0.0 in CI/Release; violations abort evaluation.

### Invariants coverage checklist

The invariants guard has default fatal checks and policy-controlled structural
checks. In default non-strict monitor mode, only fatal invariant types block the run:

- **Non-finite tensors:** weights, buffers, or activations contain `NaN`/`Inf`.
- **Tokenizer alignment:** embedding and output projection dimensions disagree
  with the tokenizer vocabulary.

The following invariants default to warnings outside strict mode. Strict
assurance forces strict mode and blocking behavior, so these findings cannot be
accepted in a strict report:

- **Weight tying:** adapters that declare tied weights must expose identical
  tensors for each alias.
- **Shape compatibility:** edited modules preserve expected shapes (e.g.,
  attention head dims, FFN hidden widths) before the pipeline runs evaluation.
- **Checkpoint hygiene/evidence gaps:** missing or drifting structural evidence
  such as LayerNorm or positional-encoding checks is surfaced for audit.

**Deadband (δ)** is used only by the zero-variance fallback in the spectral
score calculation. When the recorded family standard deviation is positive,
the score is `(sigma - mean) / std` and the deadband does not buffer the family
cap. When `std == 0`, a relative change within ±δ maps to a neutral score and
larger changes are scaled by δ. The chosen δ is published as
`spectral.summary.deadband`.

**Caps and `max_caps`**: each selected module breach contributes to
`caps_applied`. Exploratory runs using a run-local baseline may continue while
`caps_applied ≤ max_caps`; exceeding the limit blocks. The current
external-baseline strict path records every selected finding. A complete
finding blocks under `enforce` and remains visible under `observe`, even within
the cap budget; missing or degraded evidence blocks in either mode. Reports
store the count and limit under `spectral.{caps_applied,max_caps}`. When
authority is `enforce`, the guard emits a blocking decision.

### Quality Gates (Acceptance)

- Primary metric (canonical gate in report):
  - ppl-like kinds (ppl_causal, ppl_mlm, ppl_seq2seq): require the canonical
    report point estimate
    `ratio_vs_baseline ≤ tier_limit + hysteresis_ratio` where base tier limits
    are 1.05 (Conservative), 1.10 (Balanced), 1.20 (Aggressive). The packaged
    `tiers.yaml` currently publishes `metrics.pm_ratio.hysteresis_ratio = 0.002`
    to avoid PASS/FAIL flapping at the boundary. The lower-level `ppl.ratio_ci`
    analysis path also checks its upper bound when that block is populated. If
    the run exceeds the base limit but passes only because of hysteresis, the
    report marks `validation.hysteresis_applied`. Gate flag:
    `validation.primary_metric_acceptable`.
  - accuracy kinds (accuracy): gate on Δ accuracy vs baseline
    (percentage points, recorded only in `delta_vs_baseline_pp`) with minimum
    coverage. Defaults
    (policy-controlled):
    - Balanced: Δ ≥ −1.0 pp and `n_final ≥ 200`
    - Conservative: Δ ≥ −0.5 pp and `n_final ≥ 200`
    - Aggressive: Δ ≥ −2.0 pp and `n_final ≥ 200`
    `metrics.accuracy.hysteresis_delta_pp` applies the same boundary-stability
    logic to the accuracy delta floor.
    Thresholds come from the tier configuration in the packaged
    `tiers.yaml` (see `metrics.accuracy` for each tier) and are surfaced at
    runtime under `resolved_policy.metrics.accuracy`. Strict verification also
    requires measured preview and final integer counts, binds
    `n_preview`/`n_final`, recomputes both accuracy points and the baseline
    percentage-point delta, and reconciles every present
    records/windows/coverage count surface. That detects internal forks; it does
    not authenticate labels or sampling.
- Primary metric tail (ppl-like kinds): a warn/fail gate on **per-window**
  ΔlogNLL vs the paired baseline. The tail statistic (default P95) must stay
  under `metrics.pm_tail.quantile_max`, and (optionally) the mass above ε must
  stay under `metrics.pm_tail.mass_max`. Gate flag: `validation.primary_metric_tail_acceptable`
  (only flips false when `metrics.pm_tail.mode = fail`).
- Preview→final drift: require the guarded run's final/preview ratio to stay
  inside the resolved profile band. The general default is 0.95–1.05; the
  packaged CI profile currently widens the upper bound to 1.07. Gate flag:
  `validation.preview_final_drift_acceptable`.
- Spectral stability: caps applied must not exceed the tier’s `max_caps`
  (default 5 for Balanced; 3 for Conservative). Gate flag: `validation.spectral_stable`.
- RMT ε‑band stability: per‑family activation edge risk must satisfy
  `edge_cur ≤ edge_base · (1+ε_f)` for each family with a non-zero baseline.
  Gate flag: `validation.rmt_stable`.
- Guard primary-metric impact: the direction-aware degradation between paired
  guarded and bare primary metrics must stay within its configured limit when
  evaluated. PPL uses relative increase; accuracy uses absolute drop. This is a
  model-quality gate, not an elapsed-time or compute measurement. Gate flag:
  `validation.guard_metric_impact_acceptable`.

Exceeding any gate flips the corresponding `validation.*` flag to false and the
report fails overall, **except** that the Primary Metric Tail gate can run
in `mode: warn` (staged rollout) where it emits a warning but keeps
`validation.primary_metric_tail_acceptable = true`. Catastrophic spikes are
handled during the run: the `spike_threshold` (default 2.0× PPL) triggers
immediate abort/fail regardless of other gates. See also
`src/invarlock/core/runner_runtime/finalize.py`.

**Sigma quantile (qσ)** controls the target sigma used for spectral monitoring.
Balanced uses `sigma_quantile = 0.95`, Conservative `0.90` (see
the packaged tiers configuration at
`runtime/tiers.yaml`; overrides use
`INVARLOCK_CONFIG_ROOT/runtime/tiers.yaml`). Reports expose this under
`spectral.summary.sigma_quantile`.
Per-family z-caps use $\kappa_f$; defaults are defined in the packaged tiers
configuration and summarized in the Threshold Rationale table below.

## 2. Statistical Method Primer

InvarLock evaluates edits using **paired Δlog perplexity** against the baseline:

> See [Quality Gates (Acceptance)](#quality-gates-acceptance) for the run-level thresholds the CLI enforces on these statistics.

$$
\Delta_i = \log(\text{PPL}_{\text{subject final}, i}) - \log(\text{PPL}_{\text{baseline final}, i})
$$

$$
\overline{\Delta} = \frac{\sum_i w_i \Delta_i}{\sum_i w_i},\quad
\text{ratio} = \exp(\overline{\Delta})
$$

All logarithms are natural (`ln`); see ln/log for the convention used across InvarLock.

Perplexity (PPL = exp(mean NLL)) uses the standard language-model
definition; see the
[Transformers perplexity guide](https://huggingface.co/docs/transformers/perplexity).

Preview→final drift is a separate guarded-run stability check; it does not
define the primary edited-vs-baseline ratio.

Primary metric confidence intervals use the **BCa bootstrap** (1.2k to 3.2k
replicates by profile, α=0.05). Paired windows are sampled uniformly as
clusters; each replicate recomputes the token-weighted mean from the selected
window/value-weight pairs. The half-width approximation for planning is:

$$
\text{half-width} \approx z \cdot \frac{\hat{\sigma}}{\sqrt{n}}
$$

Use `z = 1.96` for two-sided 95% intervals. Balanced tiers use one-sided CI for
VE gating; Conservative uses two-sided. VE predictive A/B evidence uses its own
predictive bootstrap surface recorded under `variance.predictive_gate`; do not
read primary-metric replicate floors as VE replicate counts. Here `α=0.05`
names the requested interval quantiles; the repository has not established 95%
coverage under arbitrary dependence, weighting, or adaptive selection.

**Bootstrap defaults**

- **Replicates:** floors are 1,200 (Balanced), 1,500 (Conservative), and 800
  (Aggressive). Release profile uses 3,200; tiny smoke profiles often use
  800-1,200.
- **Paired windows:** floors are 180/180 (Balanced), 220/220 (Conservative),
  140/140 (Aggressive); profiles may request higher counts.

These values are linted and surfaced in reports so readers can audit the
configured evidence volume. The tests do not establish their statistical power.

## Evaluation Slice Contract Requirements

An evaluation schedule is accepted when:

- `meta.tokenizer_hash`, provider digest, and token totals are present.
- Preview and final use disjoint ID sets. Within each arm, baseline and subject
  runs reuse the same IDs (pairing).
- Masked-token counts are non-zero for masked-LM baselines (see
  `tests/eval/metrics/test_metrics_masked_lm_paths.py`).
- Window overlap = 0, preview/final counts match, and coverage meets the
  selected tier floors; CI/Release profiles treat violations as hard errors
  during report assembly and verification (see
  `src/invarlock/reporting/report_make.py` and
  `src/invarlock/reporting/validation/report.py`).
- Predictive VE calibration windows are drawn from the same schedule; provenance
  appears under `variance.ab_test.provenance.window_ids`.

Baseline pairing schedules record the exact windows to preserve determinism.

## 4. Reproducibility Kit

To reproduce a report:

1. Persist the run config (`config.yaml`), `window_plan`, and `evaluation_windows`.
2. Record dataset/hash/tokenizer metadata (`invarlock report generate --run <run_report.json> --format json` already saves this).
3. Capture the seed bundle (`meta.seeds`) and policy digests.
4. Use `invarlock report generate --run <subject_report.json> --baseline-run-report <baseline_report.json> --format report`
   to regenerate the report; when seeds, config, and backend match, numeric
   evidence and provenance fields should match after normalizing volatile
   artifact paths and timestamps.

Explainers for each field live in [`docs/reference/reports.md`](../reference/reports.md).

## 5. Device Tolerance Guidance

The checker ships with the following operational tolerances relative to
CPU evidence. The repository does not contain a representative device corpus
showing that these are empirical upper bounds:

| Backend | Default review tolerance (vs CPU) | Notes |
|---------|------------------------|-------|
| CPU (float32) | baseline | Reference |
| MPS | ≤ 0.5% PM-ratio difference | Default review tolerance |
| CUDA (TensorFloat-32 off) | ≤ 1.0% PM-ratio difference | Default review tolerance; record deterministic settings |

Automate the check with:

```bash
python scripts/smoke/check_device_drift.py \
  artifacts/ci-pack-*/baseline_cpu/evaluation.report.json \
  artifacts/ci-pack-*/baseline_mps/evaluation.report.json \
  --tolerance 0.005
```

The regression lives in `tests/integration/scripts/test_device_drift_linter.py`
and is available for CI/release evidence packs. The repository tests the checker
on fixtures; real device drift fails fast only when CI or release evidence
provides comparable CPU/MPS/CUDA reports.

If drift exceeds these bands, investigate the numerical/runtime difference and
collect device-specific evidence. Do not automatically loosen VE or increase
window counts to make a failing comparison pass.

## 6. Threshold Rationale (Defaults)

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| PM ratio gate (Balanced) | subject final / baseline final ≤ 1.10 (+ published hysteresis) | Tier acceptance; exceeding the effective gate fails the run |
| PM ratio gate (Conservative) | subject final / baseline final ≤ 1.05 (+ published hysteresis) | Stricter release acceptance; exceeding the effective gate fails the run |
| Bootstrap α | 0.05 | Requested two-sided interval tail mass; nominal coverage is workload-dependent |
| Spectral κ | Balanced caps `{ffn: 3.849, attn: 3.018, embed: 1.05, other: 0.0}`; Conservative `{ffn: 3.849, attn: 2.6, embed: 2.8, other: 2.8}` (from `tiers.yaml`) | Operational family thresholds plus a cap budget; FDR/FWER not established |
| RMT ε | `{ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01}` | Operational relative-growth band; no general false-alarm calibration is shipped |
| VE min_effect | 0.0 (balanced), 0.016 (conservative) | Operational enablement thresholds; public power calibration is not shipped |

Detailed derivations are in the calibration appendix (`09-tier-v1-calibration.md`).

**Examples**

- **ε-band corner case:** if `rmt.families.attn.edge_base = 1.20` and
  `rmt.families.attn.epsilon = 0.01`, the guard allows
  `rmt.families.attn.edge_cur ≤ (1+0.01) × 1.20 = 1.212`.
- **Predictive gate:** on Balanced, if `mean_delta = -0.002` and the one-sided CI is
  `[-0.003, -0.001]`, VE enables (`mean_delta` and the CI upper bound both beat
  `-min_effect_lognll`).
- **Spectral caps:** in exploratory local-baseline mode, Balanced permits at
  most five selected caps (`max_caps = 5`); a sixth sets
  `spectral.summary.caps_exceeded = true`. In strict reports, a complete cap
  finding blocks under `enforce` and remains visible under `observe`.

## 7. Scope Boundaries

- Claims apply to configured evaluation slices; task-level accuracy requires
  task-specific evidence.
- Dataset shift or tokenizer changes invalidate pairing schedules.
- Adversarial robustness and gradient masking require separate evidence.
- CUDA kernels outside deterministic mode may exceed drift tolerances.
- Reference mask-based flows are conservative; stronger compression requires plugins.
- `contracts/support_matrix.json` records maintained lanes and their current
  evidence status; `docs/README.md#support-matrix` provides the readable table.
- Representative held-out calibration studies provide the empirical basis for
  family-specific FPR interpretations and tier changes.
- Contributions for additional model families are welcome; attach study reports
  and summary CSVs (typically written under `reports/calibration/` when running
  the calibration scripts) to change proposals or release artifacts.

## 8. Coverage Reference

The following tests underpin this handbook:

- tests/eval/test_assurance_contracts.py
- tests/eval/metrics/test_metrics_masked_lm_paths.py
- tests/edits/test_quant_rtn.py
- tests/cli/verify/test_verify.py: test_verify_command_passes

Run them collectively with `make test` or the narrower `make test-assurance`
target.

## References

- Evaluation math and paired ratios: [01-eval-math-derivation.md](01-eval-math-derivation.md)
- Paired BCa bootstrap details: [03-bca-bootstrap.md](03-bca-bootstrap.md)
- Spectral thresholds and candidate-selection semantics: [05-spectral-fpr-derivation.md](05-spectral-fpr-derivation.md)
- RMT ε‑rule and outlier bands: [06-rmt-epsilon-rule.md](06-rmt-epsilon-rule.md)
- VE predictive gate power and thresholds: [07-ve-gate-power.md](07-ve-gate-power.md)
- Perplexity background: [Hugging Face Transformers perplexity guide](https://huggingface.co/docs/transformers/perplexity)
