# Guard Contracts & Statistical Primer

> **Plain language:** This handbook explains what each guard checks, the
> thresholds we enforce, and how those decisions appear in the certificate so
> reviewers can trace every PASS or FAIL.

This handbook captures the practical guarantees that underpin InvarLock’s guard
pipeline. It consolidates the guard contracts, statistical assumptions, and
calibration data that accompany the InvarLock assurance notes.

## 1. Guard Contracts

| Guard | Inputs | Check & Threshold | Failure behavior | Code reference |
|-------|--------|-------------------|-------------------|----------------|
| **Invariants** | Model weights, adapter metadata | Structural invariants (non-finite scan, weight tying, embedding dims, layer norms) | Abort edit if violated before evaluation | `invarlock.guards.invariants` |
| **Spectral** | 2‑D layer weights (FFN, attention proj, embeddings) | Compute $z = \frac{\hat{s} - \mu_f}{\sigma_f}$ where $\hat{s}$ is an **iterative estimate** of $\sigma_{\max}$ under a fixed measurement contract; require `abs(z) ≤ κ_f` calibrated for ≤5% WARN FPR. Optional degeneracy proxies (stable-rank drift, norm collapse) may add WARN/ABORT depending on policy. | WARN when cap applied; abort if cap would exceed `max_caps` (and for configured fatal degeneracy thresholds). | `invarlock.guards.spectral` |
| **RMT** | Token‑weighted activations (sampled) | Compute a per‑module **edge risk score** $r = \hat{\sigma}_{\max}(A') / \sigma_{\mathrm{MP}}(m,n)$ on whitened activations $A'$ under a fixed measurement contract; accept when baseline‑relative growth stays within the calibrated ε-band per family. | Certificate fails on ε‑band violations; catastrophic spikes in the primary metric are gated separately (`spike_threshold` = 2.0× for ppl‑like metrics). | `invarlock.guards.rmt` |
| **Variance (VE)** | Paired ΔlogNLL with calibration windows | Enable VE only if the predictive CI upper bound ≤ −`min_effect_lognll` **and** mean Δ ≤ −`min_effect_lognll` (Balanced uses one‑sided CI; Conservative uses two‑sided CI). A CI entirely above +`min_effect_lognll` is treated as regression and VE stays off. | VE disabled, guard records reason; edit continues | `invarlock.guards.variance` |
| **Bootstrap sanity** | Evaluation windows, token counts | Matching window IDs, zero overlap; BCa replicates ≥ requested | Abort certification and surface reason | `invarlock.reporting.certificate` |

Each guard logs its policy digest, metrics, and **measurement contract**; certificates
mirror those fields under `resolved_policy.*` and `spectral`/`rmt`/`variance` blocks.

### Invariants: what is checked

- No non‑finite tensors (NaN/Inf) in model parameters.
- Weight‑tying relationships preserved (e.g., tied embeddings/output projection).
- Embedding/output dimensions consistent with tokenizer and adapter descriptors.
- Expected LayerNorm modules present; shape sanity checks across layers.
- Tokenizer alignment: when both baseline and edited tokenizers are available, mismatches abort.

### Catastrophic limits and aborts

- Spike stop: a large primary‑metric spike (for ppl‑like metrics, ≥ 2.0× ratio) triggers a hard abort/rollback independent of guard WARNs.
- Pairing/coverage: preview/final counts must match, pairing must be 1.0, overlap 0.0 in CI/Release; violations abort certification.

### Invariants coverage checklist

The invariants guard fails fast when any of the following hold:

- **Non-finite tensors:** weights, buffers, or activations contain `NaN`/`Inf`.
- **Tokenizer alignment:** embedding and output projection dimensions disagree
  with the tokenizer vocabulary or tied-weight expectations.
- **Weight tying:** adapters that declare tied weights must expose identical
  tensors for each alias; mismatches trigger an abort.
- **Shape compatibility:** edited modules preserve expected shapes (e.g.,
  attention head dims, FFN hidden widths) before the pipeline runs evaluation.
- **Checkpoint hygiene:** missing mandatory tensors (layer norms, positional
  encodings) abort immediately to prevent undefined behavior.

**Deadband (δ)** provides a z-score buffer that suppresses WARN “flicker” when
values hover near the cap. For example, if the relative change in a module’s
spectral norm is within ±0.10 (Balanced), the guard reports a neutral score.
The chosen δ is published in certificates as `spectral.summary.deadband`.

**Caps and `max_caps`**: every time a module breaches its family cap the guard
records a cap. Runs may continue while `caps_applied ≤ max_caps`. Once the
limit is exceeded the guard returns `action = abort`, and the certificate
stores both the count and the limit under
`spectral.{caps_applied,max_caps}`.

### Quality Gates (Acceptance)

- Primary metric (canonical gate in certificate):
  - ppl-like kinds (ppl_causal, ppl_mlm, ppl_seq2seq): require
    `ratio_vs_baseline ≤ tier_limit` where tier limits are 1.05 (Conservative),
    1.10 (Balanced), 1.20 (Aggressive). When a ratio CI is present, the upper
    bound must also be ≤ the same limit. Gate flag: `validation.primary_metric_acceptable`.
  - accuracy kinds (accuracy, vqa_accuracy): gate on Δ accuracy vs baseline
    (percentage points) with minimum coverage. Defaults (policy‑controlled):
    - Balanced: Δ ≥ −1.0 pp and `n_final ≥ 200`
    - Conservative: Δ ≥ −0.5 pp and `n_final ≥ 200`
    - Aggressive: Δ ≥ −2.0 pp and `n_final ≥ 200`
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
certificate fails overall, **except** that the Primary Metric Tail gate can run
in `mode: warn` (staged rollout) where it emits a warning but keeps
`validation.primary_metric_tail_acceptable = true`. Catastrophic spikes are
handled during the run: the `spike_threshold` (default 2.0× PPL) triggers
immediate rollback regardless of other gates. See also
`src/invarlock/core/runner.py:1816`.

**Sigma quantile (qσ)** controls the target sigma used for spectral monitoring.
Balanced uses `sigma_quantile = 0.95`, Conservative `0.90` (see
the packaged tiers configuration at
`invarlock._data.runtime/tiers.yaml`). Certificates expose this under
`spectral.sigma_quantile`.
Per-family z-caps use $\kappa_f$; defaults are defined in the packaged tiers
configuration and summarized in the Threshold Rationale table below.

## 2. Statistical Method Primer

InvarLock evaluates edits using **paired Δlog perplexity**:

> See [Quality Gates (Acceptance)](#quality-gates-acceptance) for the run-level thresholds the CLI enforces on these statistics.

$$
\Delta_i = \log(\text{PPL}_{\text{final},i}) - \log(\text{PPL}_{\text{preview},i}),\quad
\overline{\Delta} = \frac{\sum_i w_i \Delta_i}{\sum_i w_i},\quad
\text{ratio} = \exp(\overline{\Delta})
$$

All logarithms are natural (`ln`); see ln/log for the convention used across InvarLock.

Perplexity (PPL = exp(mean NLL)) uses the standard language-model
definition—see the
Transformers perplexity guide.

Confidence intervals use the **BCa bootstrap** (1.2k to 3.2k replicates, α=0.05). The
half-width approximation for planning is `half_width ≈ z · σ̂ / √n` with
`z = 1.96` for two-sided 95% (balanced tiers use one-sided CI for VE gating; conservative uses two-sided).

**Bootstrap defaults**

- **Replicates:** floors are 1,200 (Balanced), 1,500 (Conservative), and 800
  (Aggressive). Release profile uses 3,200; tiny smoke profiles often use
  800-1,200.
- **Paired windows:** floors are 180/180 (Balanced), 220/220 (Conservative),
  140/140 (Aggressive); profiles may request higher counts.

These values are linted by `tests/eval/test_assurance_contracts.py` and surfaced
in certificates so reviewers can audit reproducibility.

## 3. Calibration & Evaluation Slice Requirements

An evaluation schedule is accepted when:

- `meta.tokenizer_hash`, provider digest, and token totals are present.
- Preview/final windows share the same window IDs (pairing).
- Masked-token counts are non-zero for masked-LM baselines (see
  `tests/eval/test_metrics_masked_lm.py`).
- Window overlap = 0 and coverage ≥ requested counts; CI/Release profiles treat
  violations as hard errors during the run (see `src/invarlock/core/runner.py`).
- Predictive VE calibration windows are drawn from the same schedule; provenance
  appears under `variance.ab_test.provenance.window_ids`.

Baseline pairing schedules record the exact windows to preserve determinism.

## 4. Reproducibility Kit

To reproduce a certificate:

1. Persist the run config (`config.yaml`), `window_plan`, and `evaluation_windows`.
2. Record dataset/hash/tokenizer metadata (`invarlock report --run <run_dir> --format json` already saves this).
3. Capture the seed bundle (`meta.seeds`) and policy digests.
4. Use `invarlock report --format cert` with the saved baseline/report combination
   to regenerate the certificate; when seeds, config, and backend match, the
   resulting certificate is bit-for-bit identical.

Explainers for each field live in `docs/reference/certificate-schema.md`.

## 5. Device Tolerance Guidance

The guards are calibrated on CPU/MPS. We expect:

| Backend | Expected drift (vs CPU) | Notes |
|---------|------------------------|-------|
| CPU (float32) | baseline | Reference |
| MPS | ≤ 0.5% PM ratio | Uses Apple Accelerate; deterministic seeds supported |
| CUDA (TensorFloat-32 off) | ≤ 1.0% PM ratio | Enable deterministic algorithms; ensure `CUBLAS_WORKSPACE_CONFIG` set |

Automate the check with:

```bash
python scripts/check_device_drift.py \
  artifacts/ci-pack-*/baseline_cpu/evaluation.cert.json \
  artifacts/ci-pack-*/baseline_mps/evaluation.cert.json \
  --tolerance 0.005
```

The regression lives in `tests/integration/scripts/test_device_drift_linter.py` and is wired
into CI so any drift beyond the documented band fails fast.

If drift exceeds these bands, re-tune VE thresholds or increase window counts.

## 6. Threshold Rationale (Defaults)

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| PM ratio gate (Balanced) | PM_final ≤ 1.10 × PM_preview | Tier acceptance; exceeding the gate fails the run |
| PM ratio gate (Conservative) | PM_final ≤ 1.05 × PM_preview | Stricter release acceptance; exceeding the gate fails the run |
| Bootstrap α | 0.05 | 95 % CI for ΔlogNLL |
| Spectral κ | Balanced caps `{ffn: 3.849, attn: 3.018, embed: 1.05, other: 0.0}`; Conservative `{ffn: 3.849, attn: 2.6, embed: 2.8, other: 2.8}` (from `tiers.yaml`) | Keeps WARN rate within the calibrated null budget |
| RMT ε | `{ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01}` | q95–q97 of null ratio (+ margin) |
| VE min_effect | 0.0 (balanced), 0.016 (conservative) | Calibrated from paired ΔlogNLL window sweeps |

Detailed derivations are in the calibration appendix (`09-tier-v1-calibration.md`).

**Examples**

- **ε-band corner case:** if `rmt.families.attn.edge_base = 1.20` and
  `rmt.families.attn.epsilon = 0.01`, the guard allows
  `rmt.families.attn.edge_cur ≤ (1+0.01) × 1.20 = 1.212`.
- **Predictive gate:** on Balanced, if Δ̄ = −0.002 and the one-sided CI is
  [−0.003, −0.001], VE enables (`mean_delta` and the CI upper bound both beat
  −`min_effect_lognll`).
- **Spectral caps:** Balanced permits at most five caps (`max_caps = 5`). If the
  sixth violation fires, `spectral.summary.caps_exceeded = true` and the guard
  aborts the run.

## 7. Known Limitations

- Guarantees apply to evaluation slices only; task-level accuracy is not certified.
- Dataset shift or tokenizer changes invalidate pairing schedules.
- No adversarial robustness or gradient masking guarantees.
- CUDA kernels outside deterministic mode may exceed drift tolerances.
- Reference mask-based flows are conservative; stronger compression requires plugins.
- Calibration data currently covers GPT-2, BERT, and TinyLLaMA profiles.
  Contributions for additional model families are welcome—attach pilot certs
  and summary CSVs (typically written under `reports/calibration/` when running
  the calibration scripts) to change proposals or release artifacts.

## 8. Coverage Reference

The following tests underpin this handbook:

- tests/eval/test_assurance_contracts.py
- tests/eval/test_metrics_masked_lm.py
- tests/edits/test_quant_rtn.py
- tests/cli/test_verify.py: test_verify_command_passes

Run them collectively with `make test` or `pytest -q -m "assurance"` where applicable.

## References

- Evaluation math and paired ratios: `01-eval-math-proof.md`
- Paired BCa bootstrap details: `03-bca-bootstrap.md`
- Spectral FPR and multiple-testing control: `05-spectral-fpr-derivation.md`
- RMT ε‑rule and outlier bands: `06-rmt-epsilon-rule.md`
- VE predictive gate power and thresholds: `07-ve-gate-power.md`
