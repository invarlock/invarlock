# Tier Policy Tuning CLI (Calibration)

> Scope note: this page covers **Tier Policy Tuning** via `invarlock advanced calibrate ...`.
> It outputs `tiers_patch_*.yaml` recommendations for a reviewed tier-policy
> override or the packaged source tier file (`runtime/tiers.yaml`).
> For evidence-pack run-scoped preset derivation (`CALIBRATION_RUN -> GENERATE_PRESET`),
> see [Evidence Pack Internals](../user-guide/evidence-packs-internals.md).

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Run policy-tuning sweeps to empirically derive guard thresholds and tier policy recommendations. |
| **Audience** | Operators recalibrating tier policies for additional model families or revised guard contracts. |
| **Primary commands** | `invarlock advanced calibrate null-sweep`, `invarlock advanced calibrate ve-sweep`. |
| **Requires** | `invarlock[hf]` for HF workflows; base config YAML for each sweep type. |
| **Network** | Offline by default; use `--allow-network` on calibration commands when a sweep needs model or dataset downloads. |
| **Source of truth** | `src/invarlock/cli/commands/calibrate.py`, `src/invarlock/calibration.py`. |

Smoke-sized configs are also shipped for maintainers who want to exercise the
calibration command surface without a full policy-tuning campaign:
`configs/calibration/null_sweep_smoke.yaml` and
`configs/calibration/rmt_ve_sweep_smoke.yaml`. These are intended for smoke
coverage and operational validation, not for published calibration evidence.

## Quick Start

The commands below use the runtime container by default. Add
`--allow-host-execution` only for host-side calibration workflows that
intentionally bypass that boundary.

```bash
# Run spectral null-sweep (noop edit) to calibrate κ/alpha
invarlock advanced calibrate null-sweep \
  --allow-network \
  --config configs/calibration/null_sweep_ci.yaml \
  --out reports/calibration/null_sweep \
  --tier balanced --tier conservative \
  --n-seeds 10

# Run VE sweep (quant_rtn simulation edit) to calibrate min_effect_lognll
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
- **Null sweep**: Uses a no-op edit to measure baseline spectral behavior and
  derive false-positive-controlled κ caps and α levels.
- **VE sweep**: Uses a real model modification (e.g., `quant_rtn`
  quantize/dequantize simulation) to measure variance guard
  predictive gate behavior and recommend `min_effect_lognll`.
- **Artifacts**: Each sweep emits JSON (machine), CSV (spreadsheet), Markdown
  (human), and a `tiers_patch_*.yaml` recommendation file.
- **Artifact contract**: The file names above are treated as stable public
  outputs and may be consumed directly by verification, review, and policy-pack
  workflows.

## Published Basis vs Included Configs

The published assurance basis is the set of `published_basis` rows in
`contracts/support_matrix.json`, with the readable grouping in
`docs/README.md#support-matrix`. The repo also includes pilot calibration
configs for prepared candidate lanes under `configs/calibration/`, but those
configs are not part of the published assurance basis until supporting artifacts
are attached. Multimodal calibration configs that use `vision_text` expect the
referenced local manifest to be materialized before the sweep runs.

The empirical guard manifest also indexes no-op published-basis reports for the
modern promoted families as null-behavior evidence. Those reports are useful
calibration inputs, but they do not update the packaged tier constants by
themselves. Until a family-specific null sweep re-derives κ, transferred
attention caps should be interpreted as budgeted sentinels rather than
Gaussian-tail FPR claims for that family.

Guard-value evidence is a separate claim from calibration. The Mistral 7B
package at `public_evidence/published_basis/mistral_7b/guard_value_demo/`
publishes PM-pass, baseline-relative spectral, RMT, and variance/VE cases from
clean confirmation reruns. That package demonstrates added guard value for the
selected edits, but it does not by itself re-derive tier constants.

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
  │ (JSON/CSV/MD)    │      │ (copy → tiers.yaml) │
  └──────────────────┘      └─────────────────────┘
```

## Reference

### Policy-Tuning Commands

| Command | Purpose | Key outputs |
| --- | --- | --- |
| `invarlock advanced calibrate null-sweep` | Calibrate spectral κ/alpha from null (noop) runs. | `null_sweep_report.json`, `tiers_patch_spectral_null.yaml` |
| `invarlock advanced calibrate ve-sweep` | Calibrate VE min_effect_lognll from real edit runs. | `ve_sweep_report.json`, `tiers_patch_variance_ve.yaml` |

### null-sweep

Runs a null (no-op edit) sweep and calibrates spectral κ/alpha empirically.

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

After a sweep, merge the `tiers_patch_*.yaml` into a reviewed
`runtime/tiers.yaml` override or the source tier policy:

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
