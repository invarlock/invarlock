# Tier Policy Tuning CLI (Calibration)

> Scope note: this page covers **Tier Policy Tuning** via `invarlock advanced calibrate ...`.
> It outputs `tiers_patch_*.yaml` recommendations for a reviewed tier-policy
> override or the packaged source tier file
> (`runtime/tiers.yaml`, the logical packaged-resource path).
> Catalog-bound evidence packs retain the exact preset and resolved runtime
> config used by evaluation; see [Evidence Packs](../user-guide/evidence-packs.md).

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Run policy-tuning sweeps that produce workload-scoped candidate thresholds for review. |
| **Audience** | Operators recalibrating tier policies for additional model families or revised guard contracts. |
| **Primary commands** | `invarlock advanced calibrate null-sweep`, `invarlock advanced calibrate ve-sweep`. |
| **Requires** | `invarlock[hf]` for HF workflows; base config YAML for each sweep type. |
| **Network** | Offline by default; use `--allow-network` on calibration commands when a sweep needs model or dataset downloads. |
| **Source of truth** | `src/invarlock/cli/commands/calibrate.py`, `src/invarlock/calibration.py`. |

Smoke-sized configs are also shipped for maintainers who want to exercise the
calibration command surface without a full policy-tuning exercise:
`configs/calibration/null_sweep_smoke.yaml` and
`configs/calibration/rmt_ve_sweep_smoke.yaml`. These are intended for smoke
coverage and operational validation, not for published calibration evidence.

The command is a recommendation harness, not a certification procedure. Its
output describes the supplied models, edits, datasets, devices, and seeds; it
does not establish population-level false-positive, FDR, FWER, or power
guarantees for other workloads.

## Quick Start

The commands below use the runtime container by default. Add
`--allow-host-execution` only for host-side calibration workflows that
intentionally bypass that boundary.

```bash
# Run a spectral null sweep to recommend candidate κ/alpha settings
invarlock advanced calibrate null-sweep \
  --allow-network \
  --config configs/calibration/null_sweep_ci.yaml \
  --out reports/calibration/null_sweep \
  --tier balanced --tier conservative \
  --n-seeds 10

# Run a VE sweep to recommend a candidate min_effect_lognll
invarlock advanced calibrate ve-sweep \
  --allow-network \
  --config configs/calibration/rmt_ve_sweep_ci.yaml \
  --out reports/calibration/ve_sweep \
  --tier balanced --tier conservative \
  --n-seeds 10
```

For smoke-only runs, swap the configs above for the shipped smoke configs and
keep the run small:

```bash
invarlock advanced calibrate null-sweep \
  --allow-network \
  --config configs/calibration/null_sweep_smoke.yaml \
  --out reports/calibration/null_sweep_smoke

invarlock advanced calibrate ve-sweep \
  --allow-network \
  --config configs/calibration/rmt_ve_sweep_smoke.yaml \
  --out reports/calibration/ve_sweep_smoke
```

## Concepts

- **Policy-tuning sweeps**: Run multiple seeds/tiers to build empirical distributions
  for threshold recommendations.
- **Null sweep**: Uses no-op runs to measure observed spectral warnings and
  recommend κ caps and an α setting for that run set. The target is an observed
  run-level rate, not a proof of false-positive control.
- **VE sweep**: Uses a real model modification (e.g., `quant_rtn`
  quantize/dequantize simulation) to measure variance guard
  predictive gate behavior and recommend `min_effect_lognll`.
- **Artifacts**: Each sweep emits JSON (machine), CSV (spreadsheet), Markdown
  (human), and a `tiers_patch_*.yaml` recommendation file.
- **Artifact names**: The current experimental command emits the names below.
  Review tooling may consume them, but the calibration command is outside the
  stable public CLI contract.

## Catalog Lanes and Included Configs

The maintained evaluation lanes are the `maintained_catalog` rows in
`contracts/support_matrix.json`, with the readable table in
`docs/README.md#support-matrix`. Each lane has an included preset and calibration
configuration. Multimodal configurations using `vision_text` materialize their
pinned manifest before evaluation.

`public_evidence/catalog_evidence_index.json` lists current empirical artifacts.
The initial status is **Evidence not yet created**; completed lanes move to
**Available** as their current run and verification artifacts are published.
Family-specific calibration evidence can then be reviewed alongside the tier
configuration it supports.

### Policy-Tuning Sweep → Tier Policy Flow

```text
  ┌──────────────────┐
  │ Base Config YAML │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ policy tuning CLI│
  │ (null/ve sweep)  │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Per-seed reports │
  │ (runs/<tier>/...)│
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐      ┌─────────────────────┐
  │ Sweep artifacts  │ ───► │ tiers_patch_*.yaml  │
  │ (JSON/CSV/MD)    │      │ (review + merge)    │
  └──────────────────┘      └─────────────────────┘
```

## Reference

### Policy-Tuning Commands

| Command | Purpose | Key outputs |
| --- | --- | --- |
| `invarlock advanced calibrate null-sweep` | Recommend spectral κ/alpha from supplied null runs. | `null_sweep_report.json`, `tiers_patch_spectral_null.yaml` |
| `invarlock advanced calibrate ve-sweep` | Recommend VE min_effect_lognll from supplied edit runs. | `ve_sweep_report.json`, `tiers_patch_variance_ve.yaml` |

### null-sweep

Runs a null (no-op edit) sweep and recommends spectral κ/alpha from the
observed run set.

**Usage:** `invarlock advanced calibrate null-sweep --config <CONFIG> --out <OUT> [options]`

| Option | Default | Description |
| --- | --- | --- |
| `--config` | `configs/calibration/null_sweep_ci.yaml` | Base null-sweep YAML (noop edit). |
| `--out` | `reports/calibration/null_sweep` | Output directory for calibration artifacts. |
| `--tier` | All tiers | Tier(s) to evaluate (repeatable). |
| `--seed` | `--seed-start` + range | Seed(s) to run (repeatable). Overrides `--n-seeds`/`--seed-start`. |
| `--n-seeds` | `10` | Number of seeds to run. |
| `--seed-start` | `42` | Starting seed. |
| `--profile` | `ci` | Run profile (`ci`, `release`, `ci_cpu`, `dev`). |
| `--device` | Auto | Device override. |
| `--safety-margin` | `0.05` | Safety margin applied to κ recommendations. |
| `--target-any-warning-rate` | `0.01` | Target run-level spectral warning rate under the null. |

**Outputs:**

- `null_sweep_report.json` — Machine-readable sweep summary with per-tier recommendations.
- `null_sweep_runs.csv` — Per-run metrics (max z-scores, candidate counts, etc.).
- `null_sweep_summary.md` — Human-readable Markdown summary.
- `tiers_patch_spectral_null.yaml` — Recommended `spectral_guard` settings for `tiers.yaml`.

### ve-sweep

Runs VE predictive-gate sweeps and recommends `min_effect_lognll` per tier.

**Usage:** `invarlock advanced calibrate ve-sweep --config <CONFIG> --out <OUT> [options]`

| Option | Default | Description |
| --- | --- | --- |
| `--config` | `configs/calibration/rmt_ve_sweep_ci.yaml` | Base VE sweep YAML (quant_rtn edit). |
| `--out` | `reports/calibration/ve_sweep` | Output directory for calibration artifacts. |
| `--tier` | All tiers | Tier(s) to evaluate (repeatable). |
| `--seed` | `--seed-start` + range | Seed(s) to run (repeatable). Overrides `--n-seeds`/`--seed-start`. |
| `--n-seeds` | `10` | Number of seeds to run. |
| `--seed-start` | `42` | Starting seed. |
| `--window` | `6, 8, 12, 16` | Variance calibration window counts (repeatable). |
| `--target-enable-rate` | `0.05` | Target expected VE enable rate (predictive-gate lower bound). |
| `--profile` | `ci` | Run profile (`ci`, `release`, `ci_cpu`, `dev`). |
| `--device` | Auto | Device override. |
| `--safety-margin` | `0.0` | Safety margin applied to min_effect recommendations. |

**Outputs:**

- `ve_sweep_report.json` — Machine-readable sweep summary with per-tier recommendations.
- `ve_sweep_runs.csv` — Per-run metrics (predictive gate deltas, CI widths, etc.).
- `ve_power_curve.csv` — Mean CI width per (tier, windows) for power analysis.
- `ve_sweep_summary.md` — Human-readable Markdown summary.
- `tiers_patch_variance_ve.yaml` — Recommended `variance_guard` settings for `tiers.yaml`.

### Applying recommendations

After a sweep, review its scope and merge the `tiers_patch_*.yaml` into an
`INVARLOCK_CONFIG_ROOT/runtime/tiers.yaml` override or the source policy at
`runtime/tiers.yaml`:

```bash
# Review recommendations
cat reports/calibration/null_sweep/tiers_patch_spectral_null.yaml

# Merge into tiers.yaml (manual review recommended)
# The patch contains only the keys being updated:
#   balanced:
#     spectral_guard:
#       family_caps: { ... }
#       multiple_testing: { alpha: ... }
```

## Troubleshooting

- **Missing config files**: Ensure calibration configs exist under `configs/calibration/`.
- **Sweep failures**: Check individual run reports under `<out>/runs/<tier>/seed_*`.
- **Unexpected recommendations**: Review the safety margin and target rate parameters.

## Observability

- Sweep artifacts record config, profile, tiers, and run count. Those are
  report-recorded fields, not authenticated provenance or execution attestation.
- Per-run reports are preserved under `<out>/runs/` for debugging.
- Power curves (VE sweep) help assess sample size requirements.

## Related Documentation

- [CLI Reference](cli.md)
- [Tier Policy Catalog](tier-policy-catalog.md)
- [Guards](guards.md)
- [Tier v1 Calibration (Assurance)](../assurance/09-tier-v1-calibration.md)
