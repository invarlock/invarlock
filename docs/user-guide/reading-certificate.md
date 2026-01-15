# Reading a Certificate (v1)

This guide walks through the key sections of a v1 certificate and how to
interpret each field. Certificates are JSON files emitted by `invarlock certify`
or `invarlock report --format cert`.

## Certificate Structure Overview

A certificate contains these major sections:

| Section | Purpose |
| --- | --- |
| `meta` | Model, device, and run metadata |
| `dataset` | Provider, window counts, and pairing stats |
| `primary_metric` | Quality measurement vs baseline |
| `primary_metric_tail` | Per-window tail analysis (ppl-like only) |
| `validation` | Boolean gate outcomes |
| `resolved_policy` | Guard thresholds used for this run |
| `artifacts` | Paths to reports and event logs |

## Primary Metric

The `primary_metric` block is the core quality measurement:

```json
{
  "primary_metric": {
    "kind": "ppl_causal",
    "unit": "ppl",
    "direction": "lower",
    "preview": 42.18,
    "final": 43.10,
    "ratio_vs_baseline": 1.022,
    "display_ci": [0.995, 1.050]
  }
}
```

**Key fields:**

- `kind` — Metric type: `ppl_causal`, `ppl_mlm`, `ppl_seq2seq`, `accuracy`
- `preview` / `final` — Point estimates on preview and final windows
- `ratio_vs_baseline` — Subject ÷ baseline (ppl) or delta (accuracy)
- `display_ci` — 95% confidence interval on the ratio

**Interpreting the ratio:**

- For ppl-like metrics: `ratio < 1.10` typically passes balanced tier
- For accuracy: `delta > -1.0 pp` typically passes

## Primary Metric Tail (ppl-like only)

When evaluating perplexity, the certificate includes per-window tail analysis:

```json
{
  "primary_metric_tail": {
    "mode": "warn",
    "evaluated": true,
    "passed": true,
    "warned": false,
    "stats": {
      "n": 200,
      "epsilon": 0.0001,
      "q95": 0.02,
      "q99": 0.04,
      "max": 0.06,
      "tail_mass": 0.03
    },
    "policy": {
      "mode": "warn",
      "min_windows": 50,
      "quantile": 0.95,
      "quantile_max": 0.20
    }
  }
}
```

**Key fields:**

- `mode` — `warn` (advisory) or `fail` (blocks validation)
- `stats.q95` — 95th percentile of per-window ΔlogNLL vs baseline
- `stats.tail_mass` — Fraction of windows exceeding ε threshold
- `policy.quantile_max` — Maximum allowed q95 value

## Validation Flags

The `validation` object contains boolean gate outcomes:

```json
{
  "validation": {
    "primary_metric_acceptable": true,
    "primary_metric_tail_acceptable": true,
    "preview_final_drift_acceptable": true,
    "guard_overhead_acceptable": true,
    "invariants_pass": true,
    "spectral_stable": true,
    "rmt_stable": true
  }
}
```

All flags must be `true` for verification to pass. The flag names are part of
a schema allow-list; unknown keys cause validation failure.

## Dataset and Pairing

The `dataset` block records evaluation data provenance:

```json
{
  "dataset": {
    "provider": "wikitext2",
    "seq_len": 512,
    "windows": {
      "preview": 200,
      "final": 200,
      "seed": 42,
      "stats": {
        "window_match_fraction": 1.0,
        "window_overlap_fraction": 0.0,
        "paired_windows": 200
      }
    }
  }
}
```

**Pairing requirements (CI/Release):**

- `window_match_fraction` must be `1.0`
- `window_overlap_fraction` must be `0.0`
- `paired_windows` must be > 0

## Measurement Contracts

Guard evidence includes measurement contracts for reproducibility:

```json
{
  "resolved_policy": {
    "spectral": {
      "measurement_contract": {
        "estimator": "power_iter",
        "iters": 4,
        "init": "ones"
      },
      "sigma_quantile": 0.95,
      "deadband": 0.10
    },
    "rmt": {
      "measurement_contract": {
        "estimator": "power_iter",
        "iters": 3
      },
      "epsilon_by_family": {
        "ffn": 0.01,
        "attn": 0.01
      }
    }
  }
}
```

In CI/Release, verification enforces that baseline and subject used matching
contracts (`spectral_measurement_contract_match = true`).

## Provenance and Policy Digest

Certificates record policy configuration for audit:

```json
{
  "policy_digest": {
    "policy_version": "v1",
    "tier_policy_name": "balanced",
    "thresholds_hash": "d49f15ade7d54beb",
    "changed": false
  },
  "provenance": {
    "provider_digest": "wikitext2:validation:512:42",
    "env_flags": ["INVARLOCK_DEDUP_TEXTS"]
  }
}
```

## Confidence Label

When present, `confidence` summarizes statistical reliability:

```json
{
  "confidence": {
    "label": "High",
    "basis": "CI width < 0.05, stable across windows",
    "width": 0.055,
    "unstable": false
  }
}
```

- **High** — Narrow CI, no stability warnings
- **Medium** — Acceptable CI, minor variance
- **Low** — Wide CI or stability concerns

## Verifying a Certificate

Use the CLI to validate schema, pairing, and ratio math:

```bash
# Basic verification
invarlock verify reports/cert/evaluation.cert.json

# With profile enforcement
invarlock verify reports/cert/evaluation.cert.json --profile ci

# JSON output for scripting
invarlock verify reports/cert/evaluation.cert.json --json
```

## Related Documentation

- [Certificate Schema (v1)](../reference/certificate-schema.md) — Full schema with all fields
- [Safety Case (Assurance)](../assurance/00-safety-case.md) — What the certificate claims actually mean
- [Guard Contracts (Assurance)](../assurance/04-guard-contracts.md) — How gates and thresholds work
- [Exporting Certificates (HTML)](../reference/exporting-certificates-html.md) — Rendering for review
- [Troubleshooting](troubleshooting.md) — Common errors and solutions
