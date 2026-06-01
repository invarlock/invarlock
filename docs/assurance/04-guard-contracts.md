# Guard Contracts & Statistical Primer

> **Plain language:** This handbook explains what each guard checks, the
> thresholds we enforce, and how those decisions appear in the report so
> reviewers can trace every PASS or FAIL.

**Contents:**

- [1. Guard Contracts](#1-guard-contracts) — what each guard checks and how it fails
- [2. Statistical Method Primer](#2-statistical-method-primer) — paired Δlog perplexity and bootstrap CIs
- [3. Calibration & Evaluation Slice Requirements](#calibration-evaluation-slice-requirements) — acceptance criteria for evaluation schedules
- [4. Reproducibility Kit](#4-reproducibility-kit) — how to reproduce a report
- [5. Device Tolerance Guidance](#5-device-tolerance-guidance) — expected drift across backends
- [6. Threshold Rationale (Defaults)](#6-threshold-rationale-defaults) — why the defaults are what they are
- [7. Known Limitations](#7-known-limitations) — what the assurance case does not cover
- [8. Coverage Reference](#8-coverage-reference) — tests that underpin this handbook

This handbook captures the evidence claims that underpin InvarLock's guard
pipeline. It consolidates the guard contracts, statistical assumptions, and
calibration data that accompany the InvarLock assurance notes.

## 1. Guard Contracts

| Guard | Inputs | Check & Threshold | Failure behavior | Code reference |
|-------|--------|-------------------|-------------------|----------------|
| **Invariants** | Model weights, adapter metadata | Fatal invariants (non-finite scan, tokenizer alignment) plus structural checks (weight tying, embedding dims, layer norms) | Fatal invariant types block before evaluation; structural drift warns in monitor mode unless strict/block policy is configured | `invarlock.guards.invariants` |
| **Spectral** | 2‑D layer weights (FFN, attention proj, embeddings) | Compute the spectral z-score under the fixed measurement contract; pass while `abs(z) ≤ κ_f` under the published family caps. Gaussian-tail FPR applies to calibrated high-kappa families; low sentinel caps are operational thresholds. Optional degeneracy proxies (stable-rank drift, norm collapse) may add WARN/ABORT depending on policy. | WARN when `abs(z) > κ_f`; abort if the cap budget would exceed `max_caps` (and for configured fatal degeneracy thresholds). | `invarlock.guards.spectral` |
| **RMT** | Token‑weighted activations (sampled) | Compute a per‑module edge risk score on whitened activations under a fixed measurement contract; accept when baseline-relative growth stays within the calibrated ε-band per family. | report fails on ε‑band violations; catastrophic spikes in the primary metric are gated separately (`spike_threshold` = 2.0× for ppl‑like metrics). | `invarlock.guards.rmt` |
| **Variance (VE)** | Paired ΔlogNLL with calibration windows | Enable VE only if the predictive CI upper bound ≤ −`min_effect_lognll` **and** mean Δ ≤ −`min_effect_lognll` (Balanced uses one‑sided CI; Conservative uses two‑sided CI). A CI entirely above +`min_effect_lognll` is treated as regression and VE stays off. | VE disabled, guard records reason; edit continues | `invarlock.guards.variance` |
| **Bootstrap sanity** | Evaluation windows, token counts | Matching window IDs, zero overlap; BCa replicates ≥ requested | Abort evaluation and surface reason | `invarlock.reporting.report_make` |

Each guard logs its policy digest, metrics, and **measurement contract**; reports
mirror those fields under `resolved_policy.*` and `spectral`/`rmt`/`variance` blocks.

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

- Spike stop: a large primary‑metric spike (for ppl‑like metrics, ≥ 2.0× ratio) triggers a hard abort/fail independent of guard WARNs.
- Pairing/coverage: preview/final counts must match, pairing must be 1.0, overlap 0.0 in CI/Release; violations abort evaluation.

### Invariants coverage checklist

The invariants guard has default fatal checks and policy-controlled structural
checks. In default monitor mode, only fatal invariant types block the run:

- **Non-finite tensors:** weights, buffers, or activations contain `NaN`/`Inf`.
- **Tokenizer alignment:** embedding and output projection dimensions disagree
  with the tokenizer vocabulary.

The following invariants are still reported, but default to warnings unless the
guard is configured with strict mode or `on_fail=block`:

- **Weight tying:** adapters that declare tied weights must expose identical
  tensors for each alias.
- **Shape compatibility:** edited modules preserve expected shapes (e.g.,
  attention head dims, FFN hidden widths) before the pipeline runs evaluation.
- **Checkpoint hygiene/evidence gaps:** missing or drifting structural evidence
  such as LayerNorm or positional-encoding checks is surfaced for audit.

**Deadband (δ)** provides a z-score buffer that suppresses WARN “flicker” when
values hover near the cap. For example, if the relative change in a module’s
spectral norm is within ±0.10 (Balanced), the guard reports a neutral score.
The chosen δ is published in reports as `spectral.summary.deadband`.

**Caps and `max_caps`**: every time a module breaches its family cap the guard
records a cap. Runs may continue while `caps_applied ≤ max_caps`. Once the
limit is exceeded the guard emits a blocking decision, and the report stores
both the count and the limit under
`spectral.{caps_applied,max_caps}`.

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
    (percentage points) with minimum coverage. Defaults (policy‑controlled):
    - Balanced: Δ ≥ −1.0 pp and `n_final ≥ 200`
    - Conservative: Δ ≥ −0.5 pp and `n_final ≥ 200`
    - Aggressive: Δ ≥ −2.0 pp and `n_final ≥ 200`
    `metrics.accuracy.hysteresis_delta_pp` applies the same boundary-stability
    logic to the accuracy delta floor.
    Thresholds come from the calibrated tier configuration in the packaged
    `tiers.yaml` (see `metrics.accuracy` for each tier) and are surfaced at
    runtime under `resolved_policy.metrics.accuracy`.
- Primary metric tail (ppl-like kinds): a warn/fail gate on **per-window**
  ΔlogNLL vs the paired baseline. The tail statistic (default P95) must stay
  under `metrics.pm_tail.quantile_max`, and (optionally) the mass above ε must
  stay under `metrics.pm_tail.mass_max`. Gate flag: `validation.primary_metric_tail_acceptable`
  (only flips false when `metrics.pm_tail.mode = fail`).
- Preview→final drift: require 0.95–1.05 for the guarded run’s final/preview
  ratio. Gate flag: `validation.preview_final_drift_acceptable`.
- Spectral stability: caps applied must not exceed the tier’s `max_caps`
  (default 5 for Balanced; 3 for Conservative). Gate flag: `validation.spectral_stable`.
- RMT ε‑band stability: per‑family activation edge risk must satisfy
  `edge_cur ≤ edge_base · (1+ε_f)` for each family with a non-zero baseline.
  Gate flag: `validation.rmt_stable`.
- Guard overhead: guard/bare runtime overhead must stay within budget when
  evaluated. Gate flag: `validation.guard_overhead_acceptable`.

Exceeding any gate flips the corresponding `validation.*` flag to false and the
report fails overall, **except** that the Primary Metric Tail gate can run
in `mode: warn` (staged rollout) where it emits a warning but keeps
`validation.primary_metric_tail_acceptable = true`. Catastrophic spikes are
handled during the run: the `spike_threshold` (default 2.0× PPL) triggers
immediate abort/fail regardless of other gates. See also
`src/invarlock/core/runner_finalize.py`.

**Sigma quantile (qσ)** controls the target sigma used for spectral monitoring.
Balanced uses `sigma_quantile = 0.95`, Conservative `0.90` (see
the packaged tiers configuration at
`src/invarlock/_data/runtime/tiers.yaml`; overrides use
`INVARLOCK_CONFIG_ROOT/runtime/tiers.yaml`). Reports expose this under
`spectral.summary.sigma_quantile`.
Per-family z-caps use $\kappa_f$; defaults are defined in the packaged tiers
configuration and summarized in the Threshold Rationale table below.

## 2. Statistical Method Primer

InvarLock evaluates edits using **paired Δlog perplexity** against the baseline:

> See [Quality Gates (Acceptance)](#quality-gates-acceptance) for the run-level thresholds the CLI enforces on these statistics.

$$
\Delta_i =
\log(\text{PPL}_{\text{subject final}, i})
- \log(\text{PPL}_{\text{baseline final}, i})
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
replicates by profile, α=0.05). The half-width approximation for planning is:

$$
\text{half-width} \approx z \cdot \frac{\hat{\sigma}}{\sqrt{n}}
$$

Use `z = 1.96` for two-sided 95% intervals. Balanced tiers use one-sided CI for
VE gating; Conservative uses two-sided. VE predictive A/B evidence uses its own
predictive bootstrap surface recorded under `variance.predictive_gate`; do not
read primary-metric replicate floors as VE replicate counts.

**Bootstrap defaults**

- **Replicates:** floors are 1,200 (Balanced), 1,500 (Conservative), and 800
  (Aggressive). Release profile uses 3,200; tiny smoke profiles often use
  800-1,200.
- **Paired windows:** floors are 180/180 (Balanced), 220/220 (Conservative),
  140/140 (Aggressive); profiles may request higher counts.

These values are linted by `tests/eval/test_assurance_contracts.py` and surfaced
in reports so reviewers can audit reproducibility.

## Calibration Evaluation Slice Requirements

An evaluation schedule is accepted when:

- `meta.tokenizer_hash`, provider digest, and token totals are present.
- Preview/final windows share the same window IDs (pairing).
- Masked-token counts are non-zero for masked-LM baselines (see
  `tests/eval/test_metrics_masked_lm.py`).
- Window overlap = 0 and coverage ≥ requested counts; CI/Release profiles treat
  violations as hard errors during report assembly and verification (see
  `src/invarlock/reporting/report_make.py` and
  `src/invarlock/reporting/report_validation.py`).
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

Device drift budgets are calibrated relative to CPU evidence. We expect:

| Backend | Expected drift (vs CPU) | Notes |
|---------|------------------------|-------|
| CPU (float32) | baseline | Reference |
| MPS | ≤ 0.5% PM ratio | Uses Apple Accelerate; deterministic seeds supported |
| CUDA (TensorFloat-32 off) | ≤ 1.0% PM ratio | Enable deterministic algorithms; ensure `CUBLAS_WORKSPACE_CONFIG` set |

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

If drift exceeds these bands, re-tune VE thresholds or increase window counts.

## 6. Threshold Rationale (Defaults)

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| PM ratio gate (Balanced) | subject final / baseline final ≤ 1.10 (+ published hysteresis) | Tier acceptance; exceeding the effective gate fails the run |
| PM ratio gate (Conservative) | subject final / baseline final ≤ 1.05 (+ published hysteresis) | Stricter release acceptance; exceeding the effective gate fails the run |
| Bootstrap α | 0.05 | 95 % CI for ΔlogNLL |
| Spectral κ | Balanced caps `{ffn: 3.849, attn: 3.018, embed: 1.05, other: 0.0}`; Conservative `{ffn: 3.849, attn: 2.6, embed: 2.8, other: 2.8}` (from `tiers.yaml`) | Keeps WARN rate within the calibrated null budget |
| RMT ε | `{ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01}` | q95–q97 of null ratio (+ margin) |
| VE min_effect | 0.0 (balanced), 0.016 (conservative) | Calibrated from paired ΔlogNLL window sweeps |

Detailed derivations are in the calibration appendix (`09-tier-v1-calibration.md`).

**Examples**

- **ε-band corner case:** if `rmt.families.attn.edge_base = 1.20` and
  `rmt.families.attn.epsilon = 0.01`, the guard allows
  `rmt.families.attn.edge_cur ≤ (1+0.01) × 1.20 = 1.212`.
- **Predictive gate:** on Balanced, if `mean_delta = -0.002` and the one-sided CI is
  `[-0.003, -0.001]`, VE enables (`mean_delta` and the CI upper bound both beat
  `-min_effect_lognll`).
- **Spectral caps:** Balanced permits at most five caps (`max_caps = 5`). If the
  sixth violation fires, `spectral.summary.caps_exceeded = true` and the guard
  aborts the run.

## 7. Known Limitations

- Claims apply to evaluation slices only; task-level accuracy is not guaranteed.
- Dataset shift or tokenizer changes invalidate pairing schedules.
- No adversarial robustness or gradient masking claims.
- CUDA kernels outside deterministic mode may exceed drift tolerances.
- Reference mask-based flows are conservative; stronger compression requires plugins.
- Published assurance basis covers GPT-2 and BERT profiles.
- Additional supported-experimental lanes are defined in
  `contracts/support_matrix.json`; those lanes are not part of the published
  assurance basis until supporting artifacts are attached. Examples include
  Mistral 7B, Qwen2 7B, Qwen2.5 7B, and Qwen2.5 14B; the contract file remains
  authoritative and may include additional lanes.
- Contributions for additional model families are welcome; attach pilot reports
  and summary CSVs (typically written under `reports/calibration/` when running
  the calibration scripts) to change proposals or release artifacts.

## 8. Coverage Reference

The following tests underpin this handbook:

- tests/eval/test_assurance_contracts.py
- tests/eval/test_metrics_masked_lm.py
- tests/edits/test_quant_rtn.py
- tests/cli/test_verify.py: test_verify_command_passes

Run them collectively with `make test` or the narrower `make test-assurance`
target.

## References

- Evaluation math and paired ratios: [01-eval-math-derivation.md](01-eval-math-derivation.md)
- Paired BCa bootstrap details: [03-bca-bootstrap.md](03-bca-bootstrap.md)
- Spectral FPR and multiple-testing control: [05-spectral-fpr-derivation.md](05-spectral-fpr-derivation.md)
- RMT ε‑rule and outlier bands: [06-rmt-epsilon-rule.md](06-rmt-epsilon-rule.md)
- VE predictive gate power and thresholds: [07-ve-gate-power.md](07-ve-gate-power.md)
- Perplexity background: [Hugging Face Transformers perplexity guide](https://huggingface.co/docs/transformers/perplexity)
