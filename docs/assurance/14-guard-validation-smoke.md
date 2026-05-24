# Guard Validation Smoke

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Track the lightweight guard-validation evidence surface for spectral, RMT, and variance guards. |
| **Audience** | Maintainers, release reviewers, and calibration owners. |
| **Contract scope** | Deterministic synthetic smoke only; empirical model-family calibration remains a release-evidence requirement. |
| **Source of truth** | `scripts/guard_validation_smoke.py`, generated `artifacts/guard-validation/*`, and guard-specific assurance docs. |

## Maintainer Command

```bash
make guard-validation-smoke
```

The command writes:

- `artifacts/guard-validation/guard-validation-smoke.json`
- `artifacts/guard-validation/guard-validation-smoke.md`

The smoke estimates synthetic type-I error and power for the spectral, RMT, and
variance guard surfaces across several calibration-window counts. It is useful
for checking that the release evidence path exists and stays deterministic.

## Interpretation

The generated rows are not a substitute for real checkpoint validation. They
do not prove thresholds for GPT-2, LLaMA, Qwen, BERT, or any other model family.
They only provide a repeatable harness shape for:

- type-I error reporting
- power reporting
- calibration-window sensitivity
- model-family sensitivity placeholders
- injected-defect detection examples

Release reviewers should treat the smoke as a floor. Empirical artifacts for
real model families still belong in the release evidence bundle when a release
claims new or expanded guard calibration.

## Related Documentation

- [Spectral False-Positive Control](05-spectral-fpr-derivation.md)
- [RMT Epsilon Rule](06-rmt-epsilon-rule.md)
- [VE Predictive Gate](07-ve-gate-power.md)
- [Guard Contracts and Primer](04-guard-contracts.md)
