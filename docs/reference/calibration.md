# Calibration CLI

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Run calibration sweeps to empirically derive guard thresholds and tier policy recommendations. |
| **Audience** | Operators recalibrating tier policies for new model families or updated guard contracts. |
| **Primary commands** | `invarlock calibrate null-sweep`, `invarlock calibrate ve-sweep`. |
| **Requires** | `invarlock[hf]` for HF workflows; base config YAML for each sweep type. |
| **Network** | Offline by default; enable per command with `INVARLOCK_ALLOW_NETWORK=1`. |
| **Source of truth** | `src/invarlock/cli/commands/calibrate.py`, `src/invarlock/calibration/`. |

## Quick Start

```bash
# Run spectral null-sweep (noop edit) to calibrate κ/alpha
invarlock calibrate null-sweep \
  --config configs/calibration/null_sweep_ci.yaml \
  --out reports/calibration/null_sweep \
  --tier balanced --tier conservative \
  --n-seeds 10

# Run VE sweep (quant_rtn edit) to calibrate min_effect_lognll
invarlock calibrate ve-sweep \
  --config configs/calibration/rmt_ve_sweep_ci.yaml \
  --out reports/calibration/ve_sweep \
  --tier balanced --tier conservative \
  --n-seeds 10
```

## Concepts

- **Calibration sweeps**: Run multiple seeds/tiers to build empirical distributions
  for threshold recommendations.
- **Null sweep**: Uses a no-op edit to measure baseline spectral behavior and
  derive false-positive-controlled κ caps and α levels.
- **VE sweep**: Uses a real edit (e.g., `quant_rtn`) to measure variance guard
  predictive gate behavior and recommend `min_effect_lognll`.
- **Artifacts**: Each sweep emits JSON (machine), CSV (spreadsheet), Markdown
  (human), and a `tiers_patch_*.yaml` recommendation file.

### Sweep → Tier policy flow

```
  ┌──────────────────┐
  │ Base Config YAML │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  calibrate CLI   │
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
  │ (JSON/CSV/MD)    │      │ (copy → tiers.yaml) │
  └──────────────────┘      └─────────────────────┘
```

## Reference

### Command Index

| Command | Purpose | Key outputs |
| --- | --- | --- |
| `invarlock calibrate null-sweep` | Calibrate spectral κ/alpha from null (noop) runs. | `null_sweep_report.json`, `tiers_patch_spectral_null.yaml` |
| `invarlock calibrate ve-sweep` | Calibrate VE min_effect_lognll from real edit runs. | `ve_sweep_report.json`, `tiers_patch_variance_ve.yaml` |

### null-sweep

Runs a null (no-op edit) sweep and calibrates spectral κ/alpha empirically.

**Usage:** `invarlock calibrate null-sweep --config <CONFIG> --out <OUT> [options]`

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

**Usage:** `invarlock calibrate ve-sweep --config <CONFIG> --out <OUT> [options]`

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

After a sweep, merge the `tiers_patch_*.yaml` into your `runtime/tiers.yaml`:

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

- Sweep artifacts include full provenance (config, profile, tiers, run count).
- Per-run reports are preserved under `<out>/runs/` for debugging.
- Power curves (VE sweep) help assess sample size requirements.

## Related Documentation

- [CLI Reference](cli.md)
- [Tier Policy Catalog](tier-policy-catalog.md)
- [Guards](guards.md)
- [Tier v1 Calibration (Assurance)](../assurance/09-tier-v1-calibration.md)
