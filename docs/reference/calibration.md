# Calibration Reference

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Run calibration sweeps to derive tier policy values for spectral κ and VE min_effect. |
| **Audience** | Operators recalibrating guard thresholds for custom models or tiers. |
| **Commands** | `invarlock calibrate null-sweep`, `invarlock calibrate ve-sweep` |
| **Outputs** | JSON reports, CSV run data, Markdown summaries, and tiers.yaml patch files. |
| **Requires** | `invarlock[hf]` for model loading; `invarlock[guards]` for guard math. |
| **Network** | Offline by default; set `INVARLOCK_ALLOW_NETWORK=1` for model/dataset downloads. |
| **Source of truth** | `src/invarlock/cli/commands/calibrate.py` |

## Quick Start

```bash
# Calibrate spectral κ caps using null (noop) runs
invarlock calibrate null-sweep \
  --config configs/calibration/null_sweep_ci.yaml \
  --out reports/calibration/null_sweep \
  --profile ci \
  --n-seeds 10

# Calibrate VE min_effect_lognll using edit runs
invarlock calibrate ve-sweep \
  --config configs/calibration/rmt_ve_sweep_ci.yaml \
  --out reports/calibration/ve_sweep \
  --profile ci \
  --n-seeds 10
```

## Concepts

- **Null sweep**: Runs a noop edit multiple times with different seeds to observe
  baseline spectral z-score distributions. Used to calibrate κ caps that achieve
  the target false-positive rate (FPR).
- **VE sweep**: Runs an actual edit (typically `quant_rtn`) with varied calibration
  window counts to observe predictive gate CI widths. Used to calibrate
  `min_effect_lognll` that achieves the target enable rate under the null hypothesis.
- **Tiers patch**: Both commands emit a `tiers_patch_*.yaml` file containing
  recommended policy updates that can be merged into `runtime/tiers.yaml`.

### When to recalibrate

- Adding support for a new model family (LLaMA, Mistral, etc.)
- Changing evaluation window counts significantly
- Adjusting bootstrap replicates or CI levels
- After major guard algorithm changes

## Reference

### null-sweep

```bash
invarlock calibrate null-sweep [OPTIONS]
```

Run a null (noop edit) sweep across tiers and seeds to calibrate spectral κ caps.

| Option | Default | Description |
| --- | --- | --- |
| `--config` | `configs/calibration/null_sweep_ci.yaml` | Base sweep config (noop edit). |
| `--out` | `reports/calibration/null_sweep` | Output directory for artifacts. |
| `--tier` | All tiers | Tier(s) to evaluate (repeatable). |
| `--seed` | — | Explicit seed(s) (repeatable). Overrides `--n-seeds`. |
| `--n-seeds` | `10` | Number of seeds to run. |
| `--seed-start` | `42` | Starting seed when auto-generating. |
| `--profile` | `ci` | Run profile (`ci`, `release`, `ci_cpu`). |
| `--device` | Auto | Device override (`cpu`, `cuda`, `mps`). |
| `--safety-margin` | `0.05` | Margin added to κ recommendations. |
| `--target-any-warning-rate` | `0.01` | Target run-level spectral warning rate. |

**Output artifacts:**

| File | Format | Description |
| --- | --- | --- |
| `null_sweep_report.json` | JSON | Full sweep results with per-tier summaries. |
| `null_sweep_runs.csv` | CSV | Per-run data (caps applied, max z-scores by family). |
| `null_sweep_summary.md` | Markdown | Human-readable summary table. |
| `tiers_patch_spectral_null.yaml` | YAML | Recommended `spectral_guard` policy updates. |

**Example output structure:**

```text
reports/calibration/null_sweep/
├── configs/
│   ├── null_balanced_42.yaml
│   ├── null_balanced_43.yaml
│   └── ...
├── runs/
│   ├── balanced/seed_42/
│   ├── balanced/seed_43/
│   └── ...
├── null_sweep_report.json
├── null_sweep_runs.csv
├── null_sweep_summary.md
└── tiers_patch_spectral_null.yaml
```

### ve-sweep

```bash
invarlock calibrate ve-sweep [OPTIONS]
```

Run VE predictive-gate sweeps to calibrate `min_effect_lognll` per tier.

| Option | Default | Description |
| --- | --- | --- |
| `--config` | `configs/calibration/rmt_ve_sweep_ci.yaml` | Base sweep config (edit enabled). |
| `--out` | `reports/calibration/ve_sweep` | Output directory for artifacts. |
| `--tier` | All tiers | Tier(s) to evaluate (repeatable). |
| `--seed` | — | Explicit seed(s) (repeatable). Overrides `--n-seeds`. |
| `--n-seeds` | `10` | Number of seeds to run. |
| `--seed-start` | `42` | Starting seed when auto-generating. |
| `--window` | `6, 8, 12, 16` | Calibration window counts (repeatable). |
| `--target-enable-rate` | `0.05` | Target VE enable rate under the null. |
| `--profile` | `ci` | Run profile (`ci`, `release`, `ci_cpu`). |
| `--device` | Auto | Device override (`cpu`, `cuda`, `mps`). |
| `--safety-margin` | `0.0` | Margin applied to min_effect recommendations. |

**Output artifacts:**

| File | Format | Description |
| --- | --- | --- |
| `ve_sweep_report.json` | JSON | Full sweep results with per-tier summaries and power curve. |
| `ve_sweep_runs.csv` | CSV | Per-run data (predictive gate metrics, CI widths). |
| `ve_power_curve.csv` | CSV | Mean CI width by tier and window count. |
| `ve_sweep_summary.md` | Markdown | Human-readable summary table. |
| `tiers_patch_variance_ve.yaml` | YAML | Recommended `variance_guard` policy updates. |

**Example output structure:**

```text
reports/calibration/ve_sweep/
├── configs/
│   ├── ve_balanced_w6_42.yaml
│   ├── ve_balanced_w8_42.yaml
│   └── ...
├── runs/
│   ├── balanced/windows_6/seed_42/
│   ├── balanced/windows_8/seed_42/
│   └── ...
├── ve_sweep_report.json
├── ve_sweep_runs.csv
├── ve_power_curve.csv
├── ve_sweep_summary.md
└── tiers_patch_variance_ve.yaml
```

### Interpreting Results

**Null sweep summary:**

```markdown
| Tier | Runs | Any-warning rate | α (recommended) |
|---|---:|---:|---:|
| balanced | 30 | 0.033 | 0.000167 |
| conservative | 30 | 0.000 | 0.000100 |
```

- **Any-warning rate**: Fraction of runs where any spectral cap was applied.
- **α (recommended)**: Significance level for multiple-testing correction.

**VE sweep summary:**

```markdown
| Tier | Runs | Recommended min_effect_lognll | Expected enable rate |
|---|---:|---:|---:|
| balanced | 120 | 0.000000 | 0.050 |
| conservative | 120 | 0.016000 | 0.050 |
```

- **min_effect_lognll**: Minimum improvement required for VE to enable.
- **Expected enable rate**: Fraction of runs where VE would enable.

### Applying Recommendations

After running calibration sweeps:

1. Review the generated `tiers_patch_*.yaml` files.
2. Merge recommendations into your local `runtime/tiers.yaml` override:

   ```yaml
   # runtime/tiers.yaml (override)
   balanced:
     spectral_guard:
       family_caps:
         ffn: 3.85
         attn: 3.02
     variance_guard:
       min_effect_lognll: 0.0
   ```

3. Set `INVARLOCK_CONFIG_ROOT` to point to your override directory.
4. Re-run a validation sweep to confirm the new thresholds.

## Troubleshooting

- **Sweep takes too long**: Reduce `--n-seeds` or use `--profile ci_cpu` for fewer windows.
- **High warning rate in null sweep**: Increase `--safety-margin` or tighten κ caps.
- **VE never enables**: Check that `min_effect_lognll` isn't set too high for your edit.
- **Missing run outputs**: Check that the base config file exists and has valid model/dataset settings.

## Observability

- All runs produce standard `report.json` files under `runs/`.
- Summary reports include timestamps and config references for reproducibility.
- CSV files can be imported into spreadsheets for further analysis.

## Related Documentation

- [Tier Policy Catalog](tier-policy-catalog.md) — Policy keys and rationale
- [Tier v1 Calibration (Assurance)](../assurance/09-tier-v1-calibration.md) — Methodology
- [Guards](guards.md) — Guard configuration and tuning
- [CLI Reference](cli.md) — Other CLI commands
