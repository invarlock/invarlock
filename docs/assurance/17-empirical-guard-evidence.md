# Empirical Guard Evidence

> **Plain language:** Empirical guard evidence is the portable manifest layer
> that points reviewers to real model/checkpoint runs for spectral, RMT, and
> variance behavior.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Track non-synthetic guard evidence for spectral, RMT, and variance behavior on real model/checkpoint workflows. |
| **Audience** | Maintainers, release reviewers, and calibration owners. |
| **Contract scope** | Portable evidence manifests that point to real-run artifacts; not a substitute for the strict verifier report contract. |
| **Source of truth** | `scripts/release/evidence_contracts.py empirical`, `scripts/model_evidence/model_evidence_sweep.py`, calibration commands, and evidence-pack scripts. |

## Maintainer Command

```bash
make empirical-guard-evidence-check
```

By default, the checker reads:

```text
artifacts/guard-validation/empirical/manifest.json
```

Use `EMPIRICAL_GUARD_EVIDENCE_ROOT=<path>` when reviewing a bundle staged in a
different location.

## Real Evidence Producers

The empirical bundle is meant to reference artifacts produced by existing
non-synthetic workflows:

- `make model-evidence-sweep` or `scripts/model_evidence/model_evidence_sweep.py` for
  maintained shipped-model lanes.
- `scripts/model_evidence/run_model_evidence_remote.py` for remote GPU execution of the same
  model-evidence sweep.
- `invarlock advanced calibrate null-sweep` for empirical spectral null
  behavior.
- `invarlock advanced calibrate ve-sweep` for variance-effect sweep behavior.
- `scripts/evidence_packs/run_pack.sh` and `run_suite.sh` for packaged
  maintainer evidence from real model/checkpoint runs.

The synthetic guard-validation smoke remains the minimum deterministic release
floor. Empirical evidence is required when a release claims new or expanded
guard calibration, model-family calibration, or support promotion beyond the
currently published basis.

## Manifest Contract

An empirical bundle uses this shape:

```json
{
  "schema": "invarlock/empirical-guard-evidence-v1",
  "source_commands": [
    "make model-evidence-sweep MODEL_EVIDENCE_ARGS='--slug tiny_gpt2_canary'",
    "invarlock advanced calibrate null-sweep --config configs/calibration/null_sweep_ci.yaml",
    "invarlock advanced calibrate ve-sweep --config configs/calibration/rmt_ve_sweep_ci.yaml"
  ],
  "guard_rows": [
    {
      "guard": "spectral",
      "evidence_kind": "calibration_null_sweep",
      "status": "empirical",
      "model_family": "gpt2",
      "artifact": "calibration/null_sweep_report.json"
    },
    {
      "guard": "rmt",
      "evidence_kind": "model_evidence_sweep",
      "status": "empirical",
      "model_family": "gpt2",
      "artifact": "model-evidence/summary.json"
    },
    {
      "guard": "variance",
      "evidence_kind": "calibration_ve_sweep",
      "status": "empirical",
      "model_family": "gpt2",
      "artifact": "calibration/ve_sweep_report.json"
    }
  ],
  "model_family_rows": [
    {
      "model_family": "gpt2",
      "status": "observed",
      "artifact": "families/gpt2.json"
    }
  ]
}
```

Artifacts are relative to the manifest root and must be present in the bundle.
The checker rejects synthetic-only rows, missing required guards, missing model
family coverage, absolute artifact paths, and paths that escape the evidence
root.

## Interpretation

Passing the empirical checker means the release bundle contains portable
manifest references that self-declare non-synthetic evidence with the required
guard coverage. The checker validates manifest shape, required guard/model rows,
declared evidence kinds/statuses, command markers, and relative nonempty
artifact paths. It does not parse artifact contents, authenticate the producer,
or prove every threshold is statistically final, and it does not replace
`invarlock verify --assurance strict` for strict report acceptance.

## Related Documentation

- [Guard Validation Smoke](16-guard-validation-smoke.md)
- [Spectral False-Positive Control](05-spectral-fpr-derivation.md)
- [RMT Epsilon Rule](06-rmt-epsilon-rule.md)
- [VE Predictive Gate](07-ve-gate-power.md)
- [Tier Policy v1 Calibration](09-tier-v1-calibration.md)
- [Calibration Reference](../reference/calibration.md)
- [Evidence Packs](../user-guide/evidence-packs.md)
