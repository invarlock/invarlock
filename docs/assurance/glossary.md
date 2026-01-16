# Glossary

This glossary defines key terms used across InvarLock documentation and
certificates.

## Terms

### BCa Bootstrap
- **Definition:** Bias-corrected and accelerated bootstrap used to estimate
  confidence intervals for paired deltas.
- **Context:** Applied to paired log-loss deltas for primary metric gating.
- **Related terms:** Primary Metric, Window Pairing, Confidence Interval.
- **See also:** [BCa Bootstrap](03-bca-bootstrap.md)
- **Example:** "BCa bootstrap with 2000 reps on paired deltas."

### Compare & Certify (BYOE)
- **Definition:** Workflow that compares a subject model to a baseline without
  performing the edit inside InvarLock.
- **Context:** `invarlock certify --baseline ... --subject ...`.
- **Related terms:** Baseline, Subject Run, Certificate.
- **See also:** [Compare & Certify](../user-guide/compare-and-certify.md)
- **Example:** "BYOE: subject checkpoint produced externally."

### Four-Guard Pipeline
- **Definition:** The default guard chain: invariants, spectral, RMT, variance.
- **Context:** Core safety checks in `run` and `certify` flows.
- **Related terms:** Guard Chain, Guard Overhead.
- **See also:** [Guards](../reference/guards.md)
- **Example:** "Four-Guard Pipeline: invariants -> spectral -> RMT -> variance."

### Guard Chain (Canonical Order)
- **Definition:** Fixed execution order for guard preparation and evaluation.
- **Context:** Ensures deterministic, auditable guard outcomes.
- **Related terms:** Four-Guard Pipeline, Guard Overhead.
- **See also:** [Guards](../reference/guards.md)
- **Example:** "Guards execute in canonical order for reproducibility."

### kappa Threshold
- **Definition:** Per-family spectral cap used to flag abnormal z-scores.
- **Context:** `spectral.family_caps.*.kappa` in policy.
- **Related terms:** Spectral Cap, z-score.
- **See also:** [Spectral Guard](../reference/guards.md)
- **Example:** "kappa=2.8 for attention families."

### Policy Digest
- **Definition:** Stable hash summarizing resolved policy thresholds.
- **Context:** Stored in `policy_digest.thresholds_hash` and `policy_provenance`.
- **Related terms:** Tier Policy, Policy Overrides.
- **See also:** [Policy Provenance](11-policy-provenance.md)
- **Example:** "Digest changed after policy override."

### Primary Metric
- **Definition:** The canonical task metric used for gating (ppl or accuracy).
- **Context:** `metrics.primary_metric` in reports and certificates.
- **Related terms:** Primary Metric Tail, Window Pairing.
- **See also:** [Certificates](../reference/certificates.md)
- **Example:** "primary_metric.kind=ppl_causal".

### Primary Metric Tail
- **Definition:** Optional tail regression gate for ppl-like metrics.
- **Context:** `primary_metric_tail` block in certificates.
- **Related terms:** Primary Metric, BCa Bootstrap.
- **See also:** [Certificates](../reference/certificates.md)
- **Example:** "Tail gate warned on q95 mass."

### Spectral Cap
- **Definition:** Limit on spectral z-scores per family to flag instability.
- **Context:** Applied by the spectral guard to cap outliers.
- **Related terms:** kappa Threshold, z-score.
- **See also:** [Spectral FPR](05-spectral-fpr-derivation.md)
- **Example:** "Spectral cap exceeded for FFN family."

### Spectral Guard
- **Definition:** Guard that monitors spectral norms and z-scores for weights.
- **Context:** Emits `spectral` metrics and stability flags.
- **Related terms:** Four-Guard Pipeline, Spectral Cap.
- **See also:** [Guards](../reference/guards.md)
- **Example:** "Spectral guard stable with 0 caps applied."

### RMT epsilon Rule
- **Definition:** Random Matrix Theory epsilon band used for stability checks.
- **Context:** `rmt.epsilon` thresholds per family.
- **Related terms:** RMT Guard, kappa Threshold.
- **See also:** [RMT epsilon Rule](06-rmt-epsilon-rule.md)
- **Example:** "RMT epsilon band within policy."

### RMT Guard
- **Definition:** Guard that checks eigenvalue statistics against RMT bounds.
- **Context:** Emits `rmt` metrics and stability flags.
- **Related terms:** Four-Guard Pipeline, RMT epsilon Rule.
- **See also:** [Guards](../reference/guards.md)
- **Example:** "RMT guard stable with delta_total=0."

### Variance Effect (VE)
- **Definition:** Guard that tracks variance change in model activations.
- **Context:** Variance guard calibration and predictive gate.
- **Related terms:** Four-Guard Pipeline, Guard Overhead.
- **See also:** [VE Predictive Gate](07-ve-gate-power.md)
- **Example:** "VE predictive gate disabled for edit."

### Tier Policy
- **Definition:** Guard threshold preset (conservative, balanced, aggressive).
- **Context:** Resolved from `tiers.yaml` and applied during run/certify.
- **Related terms:** Policy Digest, Policy Overrides.
- **See also:** [Tier Policy Catalog](../reference/tier-policy-catalog.md)
- **Example:** "Tier Policy: balanced".

### Window Pairing
- **Definition:** Alignment of baseline and subject evaluation windows.
- **Context:** Required for paired gating and CI computation.
- **Related terms:** BCa Bootstrap, Primary Metric.
- **See also:** [Coverage & Pairing Plan](02-coverage-and-pairing.md)
- **Example:** "paired_windows=200; window_match_fraction=1.0".

### z-score
- **Definition:** Standardized deviation used in spectral guard scoring.
- **Context:** `spectral.top_z_scores` and family summaries.
- **Related terms:** Spectral Cap, kappa Threshold.
- **See also:** [Spectral FPR](05-spectral-fpr-derivation.md)
- **Example:** "max |z| = 2.1".

### Baseline
- **Definition:** Unedited reference run used for comparison and gating.
- **Context:** `baseline` report in Compare & Certify.
- **Related terms:** Subject Run, Window Pairing.
- **See also:** [Compare & Certify](../user-guide/compare-and-certify.md)
- **Example:** "baseline report.json".

### Subject Run
- **Definition:** Edited or target model run under evaluation.
- **Context:** `subject` report in Compare & Certify.
- **Related terms:** Baseline, Certificate.
- **See also:** [Compare & Certify](../user-guide/compare-and-certify.md)
- **Example:** "subject report.json".

### Guard Overhead
- **Definition:** Performance impact of guard checks vs bare control.
- **Context:** `guard_overhead` block in reports and certificates.
- **Related terms:** Four-Guard Pipeline, Timing Summary.
- **See also:** [Guard Overhead Method](10-guard-overhead-method.md)
- **Example:** "overhead_ratio=1.003".

### Measurement Contract
- **Definition:** Guard measurement procedure signature and digest.
- **Context:** `measurement_contract_hash` for spectral/RMT.
- **Related terms:** Policy Digest, Guard Chain.
- **See also:** [Guard Contracts](04-guard-contracts.md)
- **Example:** "contract hash matched baseline." 

### Provider Digest
- **Definition:** Dataset identity hash (ids/tokenizer/masking).
- **Context:** `provenance.provider_digest` ensures pairing parity.
- **Related terms:** Window Pairing, Tokenizer Hash.
- **See also:** [Coverage & Pairing Plan](02-coverage-and-pairing.md)
- **Example:** "provider_digest.ids_hash".

### Tokenizer Hash
- **Definition:** Stable hash of tokenizer settings and vocab.
- **Context:** Stored in `provenance.provider_digest` and dataset metadata.
- **Related terms:** Provider Digest, Window Pairing.
- **See also:** [Determinism Contracts](08-determinism-contracts.md)
- **Example:** "tokenizer hash mismatch triggers E002." 

### Certificate
- **Definition:** Structured evidence artifact summarizing a certification run.
- **Context:** `evaluation.cert.json` and rendered markdown/HTML.
- **Related terms:** Report, Evidence Bundle.
- **See also:** [Certificates](../reference/certificates.md)
- **Example:** "certificate schema_version=v1".

### Report
- **Definition:** Run-level artifact with metrics, guards, and metadata.
- **Context:** `report.json` generated by `run`.
- **Related terms:** Certificate, Evidence Bundle.
- **See also:** [Artifact Layout](../reference/artifacts.md)
- **Example:** "report.metrics.primary_metric".

### Evidence Bundle
- **Definition:** Set of files produced for audit (reports, certs, manifests).
- **Context:** `reports/cert/` output from `report --format cert`.
- **Related terms:** Report, Certificate.
- **See also:** [Artifact Layout](../reference/artifacts.md)
- **Example:** "manifest.json lists bundle files." 

### Timing Summary
- **Definition:** Consolidated timing breakdown for a certification run.
- **Context:** CLI timing output and telemetry fields.
- **Related terms:** Guard Overhead, Telemetry.
- **See also:** [Observability](../reference/observability.md)
- **Example:** "Timing Summary: model load, eval, cert gen." 

### Telemetry
- **Definition:** Performance and resource metrics emitted with certificates.
- **Context:** `telemetry.*` fields in certificates.
- **Related terms:** Timing Summary, Guard Overhead.
- **See also:** [Observability](../reference/observability.md)
- **Example:** "telemetry.memory_mb_peak".