# Guard Validation Smoke

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Track the lightweight guard-validation evidence surface for spectral, RMT, and variance guards. |
| **Audience** | Maintainers, release reviewers, and calibration owners. |
| **Contract scope** | Deterministic synthetic smoke only; empirical model-family calibration remains a release-evidence requirement. |
| **Source of truth** | `scripts/smoke/guard_validation_smoke.py`, generated `artifacts/guard-validation/*`, and guard-specific assurance docs. |

## Maintainer Command

```bash
make guard-validation-smoke
```

The command writes:

- `artifacts/guard-validation/guard-validation-smoke.json`
- `artifacts/guard-validation/guard-validation-smoke.md`

Release evidence validation requires both artifacts through
`make release-evidence-check`.

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

## Non-Synthetic Evidence Paths

The repo also ships real-run evidence machinery that is separate from this
synthetic smoke:

- `make model-evidence-sweep` runs maintained shipped-model lanes through
  `scripts/model_evidence/model_evidence_sweep.py`.
- `scripts/model_evidence/run_model_evidence_remote.py` launches the same sweep on remote GPU
  hosts.
- `invarlock advanced calibrate null-sweep` and
  `invarlock advanced calibrate ve-sweep` emit empirical calibration artifacts.
- `scripts/evidence_packs/run_pack.sh` and `run_suite.sh` package maintainer
  evidence from real model/checkpoint runs.

Use `make empirical-guard-evidence-check` to validate a portable empirical
guard-evidence manifest when real evidence is attached for release review.
That checker does not replace `make guard-validation-smoke`; it validates the
separate non-synthetic artifact bundle.

## Related Documentation

- [Spectral False-Positive Control](05-spectral-fpr-derivation.md)
- [RMT Epsilon Rule](06-rmt-epsilon-rule.md)
- [VE Predictive Gate](07-ve-gate-power.md)
- [Guard Contracts and Primer](04-guard-contracts.md)
- [Empirical Guard Evidence](17-empirical-guard-evidence.md)
