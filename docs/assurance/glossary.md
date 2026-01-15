# Glossary

Technical terms used throughout the InvarLock documentation.

## Statistical Terms

**BCa (Bias-Corrected and Accelerated)**
Bootstrap confidence interval method that corrects for bias and skewness in
the bootstrap distribution. Used for paired ΔlogNLL intervals.
See [BCa Bootstrap](03-bca-bootstrap.md).

**CI (Confidence Interval)**
Range of values that contains the true parameter with specified probability
(typically 95%). InvarLock reports both point estimates and CIs.

**FDR (False Discovery Rate)**
Expected proportion of false positives among all positive findings. Controlled
via Benjamini-Hochberg procedure in spectral guard.

**FWER (Family-Wise Error Rate)**
Probability of making at least one false positive across a family of tests.
Controlled via Bonferroni correction in conservative tier.

## Guard Terms

**κ (kappa)**
Z-score cap for spectral guard. Modules with |z| > κ trigger warnings.
Calibrated per-family to control false positive rate.
See [Spectral FPR Derivation](05-spectral-fpr-derivation.md).

**ε (epsilon)**
Acceptance band for RMT guard. Subject edge-risk must satisfy
`edge_cur ≤ (1 + ε) × edge_base` per family.
See [RMT ε-Rule](06-rmt-epsilon-rule.md).

**VE (Variance Equalization)**
Guard that applies per-module scale corrections to reduce variance drift.
Enabled only when predictive A/B gate passes.
See [VE Gate Power](07-ve-gate-power.md).

**Deadband**
Small tolerance zone around thresholds to prevent flicker/flapping when
values hover near boundaries. Applied to spectral z-scores and VE adjustments.

**Measurement Contract**
Specification of estimator algorithm and sampling policy used by guards.
Recorded in certificates for reproducibility verification.

## Metric Terms

**PM (Primary Metric)**
The main quality metric (perplexity or accuracy) used to gate certification.
Ratio vs baseline is the key acceptance criterion.

**ΔlogNLL (Delta Log Negative Log-Likelihood)**
Per-window difference in log-loss between subject and baseline.
`exp(mean(ΔlogNLL)) ≈ ratio_vs_baseline` for ppl-like metrics.

**PPL (Perplexity)**
Exponentiated average negative log-likelihood. Lower is better.
`ppl = exp(mean(nll))` where nll is per-token loss.

**Primary Metric Tail**
Distribution of per-window ΔlogNLL values. Gate checks that tail quantiles
(P95, P99) don't exceed calibrated thresholds.

## Workflow Terms

**BYOE (Bring Your Own Edit)**
Recommended workflow where user provides pre-edited subject checkpoint.
InvarLock compares to baseline without applying any edit operation.

**Pairing**
Requirement that baseline and subject use identical evaluation windows
(same IDs, no overlap). Enforced in CI/Release profiles.

**Profile**
Preset configuration for window counts and bootstrap depth.
`ci` (200/200), `release` (400/400), `ci_cpu` (120/120).

**Tier**
Guard threshold bundle. `balanced` (standard), `conservative` (strict),
`aggressive` (research-only, outside safety case).

## Certificate Terms

**Policy Digest**
Hash summary of guard thresholds and configuration used for a run.
Enables detection of policy drift between certificates.

**Provenance**
Record of data sources, environment flags, and configuration used.
Stored under `certificate.provenance.*`.

**Validation Flags**
Boolean outcomes in `certificate.validation.*` indicating gate pass/fail:
`primary_metric_acceptable`, `spectral_stable`, `rmt_stable`, etc.

## Architecture Terms

**Adapter**
Interface that loads models and provides structure descriptions.
Examples: `hf_gpt2`, `hf_llama`, `hf_bert`.

**Guard**
Validation component that checks edit safety. Pipeline order:
invariants → spectral → RMT → variance.

**GuardChain**
Orchestration layer that sequences guard prepare/validate calls
and aggregates results into the certificate.

## See Also

- [Safety Case](00-safety-case.md) — Scope and guarantees
- [Guard Contracts](04-guard-contracts.md) — Detailed gate specifications
- [Tier Policy Catalog](../reference/tier-policy-catalog.md) — Policy key reference
