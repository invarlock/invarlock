# Diagnostic Empirical Guard Artifact Inventory

> **Plain language:** This is a portable inventory of files claimed to come
> from model/checkpoint runs for spectral, RMT, and variance behavior. Its
> checker validates inventory shape and paths only. A successful check is not
> evidence that a run happened, that an artifact is authentic, or that any
> statistical or release claim is justified.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Inventory artifacts that are candidates for separate empirical review. |
| **Audience** | Maintainers and calibration reviewers. |
| **Contract scope** | Non-authoritative path and shape validation. This command is not called by `make release-evidence-check` or `make release-preflight` and cannot authorize a release or calibration claim. |
| **Source of truth** | `scripts/release/evidence_contracts.py empirical-inventory` defines inventory validation; the listed command outputs remain separate. |

## Maintainer Command

```bash
make empirical-guard-inventory-check
```

By default, the checker reads:

```text
artifacts/guard-validation/empirical/manifest.json
```

Use `EMPIRICAL_GUARD_INVENTORY_ROOT=<path>` when reviewing a bundle staged in a
different location.

## Evidence-Generating Commands

The diagnostic inventory can reference artifacts produced by these public
commands:

- `invarlock evaluate` for one fully resolved catalog lane.
- `invarlock advanced evidence-pack verify` for catalog-bound portable
  evidence.
- `invarlock advanced calibrate null-sweep` for empirical spectral null
  behavior.
- `invarlock advanced calibrate ve-sweep` for variance-effect sweep behavior.

Commands that load or evaluate a model use the configured runtime container
boundary by default. Host execution is a separate explicit opt-in and does not
gain authority from inclusion in this diagnostic inventory.

The synthetic guard-validation smoke remains a deterministic
production-primitive wiring check, not field-effectiveness evidence. New or
expanded guard calibration, model-family calibration, or guard-effectiveness claims
require a separate authoritative study contract and independent review. This
inventory does not satisfy that requirement.

The inventory may also list no-op catalog-evaluation reports under
`families/*.json`. Inventory membership does not establish that
those summaries are authentic or adequate null-behavior evidence. A
family-specific false-positive claim requires a representative held-out null
study whose protocol was fixed in advance. Merely generating a null-sweep
artifact or a recommended κ does not establish the claimed error rate.

## Manifest Contract

The diagnostic inventory uses this shape:

```json
{
  "schema": "invarlock/empirical-guard-inventory-v1",
  "authority": "diagnostic_inventory",
  "source_commands": [
    "invarlock evaluate --config resolved-config.yaml --report-out reports/eval",
    "invarlock advanced evidence-pack verify --help",
    "invarlock advanced calibrate null-sweep --config configs/calibration/null_sweep_ci.yaml",
    "invarlock advanced calibrate ve-sweep --config configs/calibration/rmt_ve_sweep_ci.yaml"
  ],
  "guard_rows": [
    {
      "guard": "spectral",
      "evidence_kind": "calibration_null_sweep",
      "status": "indexed",
      "model_family": "gpt2",
      "artifact": "calibration/null_sweep_report.json"
    },
    {
      "guard": "rmt",
      "evidence_kind": "catalog_evaluation",
      "status": "indexed",
      "model_family": "gpt2",
      "artifact": "catalog-evidence/summary.json"
    },
    {
      "guard": "variance",
      "evidence_kind": "calibration_ve_sweep",
      "status": "indexed",
      "model_family": "gpt2",
      "artifact": "calibration/ve_sweep_report.json"
    }
  ],
  "model_family_rows": [
    {
      "model_family": "gpt2",
      "status": "indexed",
      "artifact": "families/gpt2.json"
    }
  ]
}
```

```json
{
  "model_family_rows": [
    {
      "model_family": "mistral_7b",
      "status": "indexed",
      "artifact": "families/mistral_7b.json"
    }
  ]
}
```

Artifacts are relative to the manifest root and must be present in the bundle.
The checker rejects synthetic-only rows, missing required guards, missing model
family coverage, absolute artifact paths, and paths that escape the evidence
root. It also rejects duplicate manifest keys and symlinked artifact files.

## Interpretation

A zero exit code means only that the diagnostic inventory has the required
guard/model rows, recognized evidence-kind labels, command strings, and
relative nonempty artifact paths. The JSON result always reports
`"authoritative": false` and `"scope": "artifact_inventory_only"`.

The checker does not parse the referenced artifacts, hash or authenticate them,
bind them to source/model/dataset identities, replay an evaluation, validate a
sampling protocol, or evaluate statistical adequacy. It is deliberately absent
from both release gates. Release or calibration authority requires a separate
content-aware contract that validates those properties and is named explicitly
by the claim being reviewed.

Additional manifest keys are not validated by this checker and cannot
strengthen its verdict. In particular, a self-declared calibration summary or
model count does not establish artifact hashes, corpus completeness, or
re-derivation of tier constants unless a separate checker validates those
properties.

## Related Documentation

- [Guard Validation Smoke](16-guard-validation-smoke.md)
- [Spectral False-Positive Control](05-spectral-fpr-derivation.md)
- [RMT Epsilon Rule](06-rmt-epsilon-rule.md)
- [VE Predictive Gate](07-ve-gate-power.md)
- [Tier Policy v1 Calibration](09-tier-v1-calibration.md)
- [Calibration Reference](../reference/calibration.md)
- [Evidence Packs](../user-guide/evidence-packs.md)
