# Guard Validation Smoke

> **Plain language:** The smoke command checks the synthetic guard-validation
> harness still runs and routes generated scores through production guard
> primitives. Real model-family evidence remains a separate
> release-evidence surface.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Track the lightweight guard-validation evidence surface for spectral, RMT, and variance guards. |
| **Audience** | Maintainers, release approvers, and calibration owners. |
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

The smoke estimates synthetic null-trigger and shifted-trigger rates across
several window counts. Each score is routed through a production primitive:
spectral violation summary, RMT epsilon-violation detection, or the variance
predictive gate outcome. The JSON records those import paths and roles. The
generator distributions are deliberately synthetic; these rates are useful for
implementation wiring and determinism checks, not threshold calibration.

The release checker reads both artifacts as immutable regular-file snapshots.
It rejects duplicate JSON keys, non-finite JSON values, symlinks, unexpected
v1 fields, invalid seed/replicate bounds, and source-identity mismatches. It
then independently regenerates every raw boolean outcome from the recorded
seed and current policy thresholds, recomputes counts and rates, verifies the
evidence digest, and requires the Markdown bytes to equal the canonical render.
This replay strengthens the wiring claim only; it does not turn synthetic
inputs into empirical guard evidence.

## Interpretation

The generated rows show that the three named production primitives execute
deterministically on declared synthetic inputs. They do not provide
model-family evidence, real-world type-I error, detection power, or threshold
calibration. Release approvers should treat the smoke as an implementation
floor only. Independently reviewable empirical artifacts belong in the release
evidence bundle whenever a release claims new or expanded guard or model-family
calibration.

## Non-Synthetic Evidence Paths

Real-run evidence remains separate from this synthetic smoke:

- The public evidence catalog declares the exact supported evaluation lanes and
  required artifacts.
- `invarlock evaluate` runs one resolved lane, and strict evidence-pack
  verification checks its catalog-bound artifacts.
- `invarlock advanced calibrate null-sweep` and
  `invarlock advanced calibrate ve-sweep` emit empirical calibration artifacts.

Multi-host scheduling and host lifecycle are external to the public repository.

Use `make empirical-guard-inventory-check` only to validate the shape of a
portable diagnostic artifact inventory. That command does not inspect artifact
contents and is not part of either release gate. Real empirical claims need a
separate content-aware study contract and independent review;
`make guard-validation-smoke` remains only the deterministic smoke floor.

## Related Documentation

- [Spectral False-Positive Control](05-spectral-fpr-derivation.md)
- [RMT Epsilon Rule](06-rmt-epsilon-rule.md)
- [VE Predictive Gate](07-ve-gate-power.md)
- [Guard Contracts and Primer](04-guard-contracts.md)
- [Diagnostic Empirical Guard Artifact Inventory](17-empirical-guard-evidence.md)
