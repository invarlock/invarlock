# Spectral Threshold And Multiple-Testing Metadata

> **Plain language:** The spectral guard implements deterministic, operational
> thresholds. Reports record BH/Bonferroni-named selection metadata, but the
> current statistic and public evidence do not establish statistical FDR, FWER,
> or per-run false-positive guarantees. Treat cap crossings as review signals
> and the cap budget as policy, not as a calibrated probability statement.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Describe the implemented spectral statistic, threshold selection, and evidence limits. |
| **Audience** | Guard maintainers, evidence reviewers, and contributors changing spectral policy. |
| **Contract scope** | Family z-scores, cap crossings, family-selection metadata, and report observability. |
| **Source of truth** | `src/invarlock/guards/spectral_detection.py`, `src/invarlock/guards/spectral.py`, and the packaged `runtime/tiers.yaml` policy resource. |

## Implemented Statistic

For every monitored two-dimensional weight matrix, the guard estimates its top
singular value. During baseline capture it groups module values by family and
computes the cross-sectional family mean and population standard deviation. A
current module value is scored as

$$
z = \frac{s - \mu_f}{\sigma_f}.
$$

This is a standardized position within a heterogeneous family of modules. It is
not, by itself, a module-specific repeated-null statistic. When
`sigma_f == 0`, the implementation falls back to the module baseline ratio and
uses the configured deadband. When `sigma_f > 0`, the general z-score path does
not apply that relative-change deadband.

A module becomes a candidate when `abs(z) > kappa_f`. Reports record candidate
counts, selected counts, family caps, and the policy metadata.

## Multiple-Testing Metadata

The implementation converts candidate z-scores to two-sided standard-normal
tail values, keeps the minimum candidate p-value per family, and applies the
configured BH- or Bonferroni-named family selector. Important limitations are:

- candidate selection occurs before the family selector;
- a family p-value is the minimum across its candidate modules;
- the number and dependence of modules within a family are not corrected; and
- the cross-sectional z-score has not been shown to follow a standard-normal
  repeated-null distribution across supported model families.

Because those preconditions are not established, the metadata does **not**
currently justify statements that BH controls FDR or Bonferroni controls FWER
for an InvarLock run. The Gaussian tail calculation is a formula check only.

## Current Policy Values

The packaged tiers currently record:

- Balanced: caps `{ffn: 3.849, attn: 3.018, embed: 1.05, other: 0.0}`,
  selector metadata `{method: bh, alpha: 0.05, m: 4}`, and `max_caps = 5`.
- Conservative: caps `{ffn: 3.849, attn: 2.6, embed: 2.8, other: 2.8}`,
  selector metadata `{method: bonferroni, alpha: 0.000625, m: 4}`, and
  `max_caps = 3`.

These values are executable policy constants. In this repository snapshot they
must be cited as operational thresholds. The public no-op fixtures exercise
report and verifier behavior but do not re-derive the constants or establish a
false-positive rate.

## Runtime Contract

Reports expose:

- `spectral.summary.{sigma_quantile,deadband,modules_checked,max_caps,caps_exceeded}`
- `spectral.family_caps[*].kappa`
- `spectral.families[*].{max,mean,count,violations,kappa}`
- `spectral.multiple_testing.{method,alpha,m}`
- selection diagnostics such as family p-values where produced

`max_caps` is a policy budget. Passing it means the selected cap count stayed
within that budget in exploratory local-baseline behavior; it does not mean a
statistical error rate was achieved. The current external-baseline strict
runtime blocks on any selected violation, even when the count is within
`max_caps`. There is no separate strict-mode organization-policy switch that
turns this result into diagnostic-only evidence.

## What Tests Establish

Current automated tests establish configuration wiring, deterministic estimator
behavior, normal-tail formula arithmetic on independently generated normal
samples, and report extraction. They do not measure the deployed guard's null
warning rate, FDR, FWER, sensitivity, or calibration across model families.

## Evidence Needed For A Statistical Claim

A future FDR/FWER or false-positive claim needs, at minimum:

1. a null-generating protocol fixed in advance for each supported family;
2. module-level multiplicity and dependence handled in the tested procedure;
3. raw null-run artifacts and immutable hashes;
4. out-of-sample error-rate evaluation with uncertainty intervals; and
5. a clear mapping from those results to the exact shipped estimator and caps.

Until those artifacts are public and reproducible, recalibration output is a
local policy proposal rather than a published assurance result.

## Related Documentation

- [Guard Contracts](04-guard-contracts.md)
- [Tier Policy Values And Recalibration](09-tier-v1-calibration.md)
- [Diagnostic Empirical Guard Artifact Inventory](17-empirical-guard-evidence.md)
