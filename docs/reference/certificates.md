# Certificates

This document consolidates all certificate-related reference material: schema,
telemetry fields, and HTML export.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Define the v1 certificate contract, telemetry fields, and export formats. |
| **Audience** | Operators verifying certificates and tool authors parsing them. |
| **Schema version** | `schema_version = "v1"` (PM-only). |
| **Source of truth** | `invarlock.reporting.certificate_schema.CERTIFICATE_JSON_SCHEMA`. |

## Table of Contents

- [Quick Start](#quick-start)
- [Certificate Layout](#certificate-layout)
  - [Safety Dashboard Interpretation](#safety-dashboard-interpretation)
- [Schema](#schema)
  - [Minimal v1 Certificate Example](#minimal-v1-certificate-example)
  - [Schema Summary](#schema-summary-validator-view)
  - [Required vs Optional Blocks](#required-vs-optional-blocks)
  - [Primary Metric Tail Gate](#primary-metric-tail-gate-optional)
- [Telemetry Fields](#telemetry-fields)
- [HTML Export](#html-export)
- [Troubleshooting](#troubleshooting)
- [Related Documentation](#related-documentation)

---

## Quick Start

```bash
# Generate a certificate from a run report
invarlock report --run runs/subject/report.json --baseline runs/baseline/report.json --format cert

# Validate certificate structure
invarlock verify reports/cert/evaluation.cert.json

# Inspect telemetry fields
jq '.telemetry' reports/cert/evaluation.cert.json

# Export to HTML
invarlock report html -i reports/cert/evaluation.cert.json -o reports/cert/evaluation.html
```

## Certificate Layout

The markdown certificate is structured to highlight safety outcomes first:

- **Safety Dashboard**: one-line PASS/FAIL + core gates (primary metric, drift, invariants, spectral, RMT, overhead).
- **Quality Gates**: table of canonical gating checks with measured values.
- **Safety Check Details**: invariants, spectral stability, RMT health, and pairing snapshots.
- **Primary Metric**: task-specific metric summary with CI + baseline comparison.
- **Guard Observability**: compact summaries with expandable guard details.
- **Policy Configuration**: tier + digest summary with resolved policy details in `<details>`.
- **Appendix**: environment, inference diagnostics, and variance guard details.

### Safety Dashboard Interpretation

| Row | Meaning | Action |
| --- | --- | --- |
| Overall | Aggregate PASS/FAIL of canonical gates | If FAIL, inspect the matching gate row |
| Primary Metric | Ratio/Δpp vs baseline | Confirm within tier threshold |
| Drift | Final/preview ratio | Check device stability, dataset drift |
| Invariants/Spectral/RMT | Guard status | Expand guard details for failures |
| Overhead | Guarded vs bare PM | Only present if overhead is evaluated |

## Evidence Flow

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        EVIDENCE FLOW                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌───────────────┐                                                     │
│   │  BASELINE RUN │                                                     │
│   │   report.json │                                                     │
│   └───────┬───────┘                                                     │
│           │                                                             │
│           ▼                                                             │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │                      SUBJECT RUN                               │    │
│   │ ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │    │
│   │ │ model.meta  │  │ dataset.data │  │ guards[].metrics       │ │    │
│   │ │ ─────────── │  │ ──────────── │  │ ────────────────────── │ │    │
│   │ │ model_id    │  │ provider     │  │ invariants.passed      │ │    │
│   │ │ adapter     │  │ seq_len      │  │ spectral.summary       │ │    │
│   │ │ device      │  │ windows.stats│  │ rmt.families           │ │    │
│   │ │ seeds       │  │ paired_count │  │ variance.enabled       │ │    │
│   │ └─────────────┘  └──────────────┘  └────────────────────────┘ │    │
│   │                                                                │    │
│   │ ┌─────────────────┐  ┌───────────────────────────────────────┐│    │
│   │ │ metrics         │  │ policy_resolved                       ││    │
│   │ │ ─────────────── │  │ ─────────────────────────────────────── ││    │
│   │ │ primary_metric  │  │ spectral.*, rmt.*, variance.*          ││    │
│   │ │ ratio_vs_base   │  │ tier_policy_name                       ││    │
│   │ │ display_ci      │  │ thresholds_hash                        ││    │
│   │ └─────────────────┘  └───────────────────────────────────────┘│    │
│   └────────────────────────────────┬──────────────────────────────┘    │
│                                    │                                    │
│                                    ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                    make_certificate()                          │   │
│   │ baseline_report + subject_report → evaluation.cert.json        │   │
│   └────────────────────────────────┬───────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                    invarlock verify                            │   │
│   │ schema + pairing + ratio math + measurement contracts          │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Schema

### Concepts

- **Schema stability**: v1 is a PM-only contract; breaking changes require a
  schema-version bump.
- **Validation allow-list**: only specific `validation.*` flags are accepted by
  the schema validator.
- **Baseline pairing**: certificates assume paired windows; verification enforces
  pairing in CI/Release profiles.

### Provenance Map

| Certificate block | Sourced from report | Verify checks |
| --- | --- | --- |
| `meta` | `report.meta` | Schema only. |
| `dataset` / `evaluation_windows` | `report.data`, `report.dataset.windows.stats` | Pairing + count checks. |
| `primary_metric` | `report.metrics.primary_metric` | Ratio + drift band (CI/Release). |
| `spectral` / `rmt` / `variance` | `report.guards[]` | Measurement contracts (CI/Release). |
| `provenance.provider_digest` | `report.provenance.provider_digest` | Required in CI/Release. |

### Minimal v1 Certificate Example

The example below shows a realistic, PM‑only certificate envelope. It follows
the current validator in `invarlock.reporting.certificate_schema` and the
fields produced by `invarlock.assurance.make_certificate`.

```json
{
  "schema_version": "v1",
  "run_id": "20251013T012233Z-quant8-balanced",
  "meta": {
    "model_id": "gpt2",
    "adapter": "hf_gpt2",
    "device": "cpu",
    "seeds": {
      "python": 1337,
      "numpy": 1337,
      "torch": 1337
    }
  },
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
        "paired_windows": 200,
        "coverage": {
          "preview": { "used": 200 },
          "final": { "used": 200 }
        }
      }
    }
  },
  "primary_metric": {
    "kind": "ppl_causal",
    "unit": "ppl",
    "direction": "lower",
    "preview": 42.18,
    "final": 43.10,
    "ratio_vs_baseline": 1.02,
    "display_ci": [1.00, 1.05]
  },
  "primary_metric_tail": {
    "mode": "warn",
    "evaluated": true,
    "passed": true,
    "warned": false,
    "violations": [],
    "policy": {
      "mode": "warn",
      "min_windows": 50,
      "quantile": 0.95,
      "quantile_max": 0.20,
      "epsilon": 0.0001,
      "mass_max": 1.0
    },
    "stats": {
      "n": 200,
      "epsilon": 0.0001,
      "q95": 0.02,
      "q99": 0.04,
      "max": 0.06,
      "tail_mass": 0.03
    },
    "source": "paired_baseline.final"
  },
  "validation": {
    "primary_metric_acceptable": true,
    "primary_metric_tail_acceptable": true,
    "preview_final_drift_acceptable": true,
    "guard_overhead_acceptable": true
  },
  "policy_digest": {
    "policy_version": "v1",
    "tier_policy_name": "balanced",
    "thresholds_hash": "d49f15ade7d54beb",
    "hysteresis": {
      "ppl": 0.002
    },
    "min_effective": 0.0,
    "changed": false
  },
  "artifacts": {
    "events_path": "runs/quant8/20251013_012233/events.jsonl",
    "report_path": "runs/quant8/20251013_012233/report.json"
  },
  "plugins": {
    "adapters": [],
    "edits": [],
    "guards": []
  }
}
```

**Notes:**

- `schema_version` is a string and must be `"v1"` for the current format.
- `run_id` is a short, opaque identifier; certificates treat it as a stable
  string key.
- `primary_metric` is the **canonical** place for PM values.
- The `validation` object holds boolean flags; only a small allow‑list of
  keys is recognized by the validator.

### Schema Summary (Validator View)

The v1 validator uses a JSON Schema (draft 2020‑12) embedded in
`CERTIFICATE_JSON_SCHEMA`. The schema is intentionally permissive around new
fields while enforcing a small, stable core:

**Required top‑level fields:**

- `schema_version` — must equal `"v1"`.
- `run_id` — non‑empty string (minimum length 4).
- `meta` — object (model/device/seeds; validator does not fix sub‑shape).
- `dataset` — object with at least:
  - `provider`: string
  - `seq_len`: integer ≥ 1
  - `windows.preview`: integer ≥ 0
  - `windows.final`: integer ≥ 0
  - `windows.stats`: object (paired-window stats and coverage)
- `artifacts` — object (paths to `report.json`, `events.jsonl`, etc.).
- `plugins` — object listing discovered adapters/edits/guards.
- `primary_metric` — object (canonical primary metric snapshot).

**Primary metric block (required):**

- `primary_metric.kind`: string (e.g., `"ppl_causal"`, `"accuracy"`).
- `primary_metric.preview` / `primary_metric.final`: numbers.
- `primary_metric.ratio_vs_baseline`: number.
- `primary_metric.display_ci`: two‑element numeric array `[lo, hi]`.
- Additional optional fields: `unit`, `direction`, `ci`, `gating_basis`,
  `aggregation_scope`, `estimated`, etc.

**Validation flags:**

- `validation` is an object of booleans; the allow‑list is loaded from
  `contracts/validation_keys.json` when present, or from a small default set.
- Common flags:
  - `primary_metric_acceptable`
  - `primary_metric_tail_acceptable`
  - `preview_final_drift_acceptable`
  - `guard_overhead_acceptable`
  - `invariants_pass`
  - `spectral_stable`
  - `rmt_stable`
  - `hysteresis_applied`
  - `moe_observed`
  - `moe_identity_ok`
- The validator rejects certificates that contain non‑boolean values under
  any of these keys.

**Policy and structure:**

- `policy_digest` — small summary of tier policy thresholds and whether they
  changed relative to the baseline.
- `resolved_policy` — snapshot of effective guard policies (spectral, rmt,
  variance, metrics).
- `policy_provenance` — tier label, overrides, and digest.
- `structure` — structural deltas and compression diagnostics (optional).

**Confidence (optional):**

- `confidence` — object with:
  - `label`: `"High" | "Medium" | "Low"`.
  - `basis`: string description of the confidence basis.
  - Optional numeric fields: `width`, `threshold`, `unstable` flag, etc.

The full machine‑readable schema is available at runtime via
`invarlock.reporting.certificate_schema.CERTIFICATE_JSON_SCHEMA`.

### Certificate → Verify Matrix

| Certificate block | Derived from | Verify checks |
| --- | --- | --- |
| `meta` | `report.meta` | Schema only. |
| `dataset` / `evaluation_windows` | `report.data`, `report.dataset.windows.stats` | Pairing + count checks. |
| `primary_metric` | `report.metrics.primary_metric` | Ratio + drift band (CI/Release). |
| `validation` | `report.metrics` + policy thresholds | Schema allow‑list only. |
| `spectral` / `rmt` / `variance` | `report.guards[]` | Measurement contracts (CI/Release). |
| `guard_overhead` | `report.guard_overhead` | Required in Release unless skipped. |
| `provenance.provider_digest` | `report.provenance.provider_digest` | Required in CI/Release. |

### Required vs Optional Blocks

| Key | Required | Source | Stability |
| --- | --- | --- | --- |
| `schema_version` | Yes | `CERTIFICATE_SCHEMA_VERSION` | PM-only v1 |
| `run_id` | Yes | Run metadata | Stable |
| `meta` | Yes | `report.meta` | Stable |
| `dataset` | Yes | `report.dataset` + windows stats | Stable |
| `primary_metric` | Yes | `report.metrics.primary_metric` | Stable |
| `artifacts` | Yes | Run artifact paths | Stable |
| `plugins` | Yes | Plugin discovery snapshot | Stable |
| `validation` | Optional | Gate outcomes | Allow-list evolves |
| `policy_digest` / `resolved_policy` | Optional | Tier policies | Calibrated changes |
| `primary_metric_tail` | Optional | Paired ΔlogNLL tail gate | ppl-like only |
| `structure` / `confidence` / `system_overhead` / `provenance` | Optional | Best-effort evidence | May evolve |

### Primary Metric Tail Gate (optional)

For ppl-like metrics with paired per-window logloss, certificates may include
`primary_metric_tail`, which records tail summaries of per-window ΔlogNLL vs the
baseline and the tail-gate evaluation outcome:

- `primary_metric_tail.stats` — deterministic quantiles (`q50/q90/q95/q99`),
  `max`, and `tail_mass = Pr[ΔlogNLL > ε]`.
- `primary_metric_tail.policy` — resolved `metrics.pm_tail` policy (mode,
  quantile, thresholds, floors).
- `primary_metric_tail.violations` — structured reasons when thresholds are exceeded.
- `validation.primary_metric_tail_acceptable` — remains `true` in `warn` mode;
  flips `false` only when `mode=fail` and a violation is evaluated.

---

## Telemetry Fields

Telemetry values are copied from `report.json` into certificates and always
include the execution device. CPU telemetry sweeps are collected via
`scripts/run_cpu_telemetry.sh`.

| JSON Pointer | Meaning | Notes |
| --- | --- | --- |
| `/telemetry/device` | Execution device (`cpu`, `mps`, `cuda`). | Mirrors `meta.device`. |
| `/telemetry/latency_ms_per_tok` | Mean latency per token. | ms/token. |
| `/telemetry/memory_mb_peak` | Peak resident memory. | MiB. |
| `/telemetry/preview_total_tokens` | Tokens processed in preview. | Derived from windows. |
| `/telemetry/final_total_tokens` | Tokens processed in final. | Derived from windows. |
| `/telemetry/throughput_tok_per_s` | Average throughput. | Present when available. |

**Observability:**

- `report.json` contains `metrics.latency_ms_per_tok` and `metrics.memory_mb_peak`.
- `telemetry.summary_line` is emitted when `INVARLOCK_TELEMETRY=1`.

---

## HTML Export

The HTML renderer converts the Markdown certificate into structured HTML
tables (via the `markdown` library when available) and preserves the same
numeric values (ratios, CIs, deltas). When the dependency is unavailable, the
renderer falls back to a `<pre>` block. Use `--embed-css` (default) to inline
a minimal stylesheet for standalone use, including status badges and
print-friendly rules.

### CLI

```bash
invarlock report html -i <cert.json> -o <out.html>
```

**Flags:**

- `--embed-css/--no-embed-css` — inline stylesheet (default: embed)
- `--force` — overwrite existing output

### Python API

```python
from invarlock.reporting.html import render_certificate_html

html = render_certificate_html(certificate)
```

---

## Troubleshooting

### Schema Issues

- **Schema validation fails**: check `schema_version` and required top-level
  fields (`run_id`, `meta`, `dataset`, `artifacts`, `primary_metric`).
- **Unexpected validation keys**: ensure `validation.*` keys match the allow-list
  in `certificate_schema`.

### Telemetry Issues

- **Telemetry missing**: ensure the run completed successfully and check
  `report.metrics` for latency/memory values.

### HTML Export Issues

- **Missing certificate**: generate one first via `invarlock report --format cert`.
- **HTML missing styles**: omit `--no-embed-css` or apply custom CSS downstream.

---

## Observability

- `validation.*`, `resolved_policy.*`, and `policy_digest.*` capture policy state.
- `primary_metric_tail` appears only for ppl-like metrics with paired windows.
- The rendered HTML is derived from the Markdown report. If values look wrong,
  inspect the underlying `evaluation.cert.json`.
- The Markdown certificate is a human-readable view (starts with a Safety Dashboard + Contents); the JSON certificate is the canonical evidence artifact.

---

## Related Documentation

- [CLI Reference](cli.md)
- [Artifact Layout](artifacts.md)
- [Safety Case](../assurance/00-safety-case.md) — What the certificate guarantees
- [Reading a Certificate](../user-guide/reading-certificate.md) — User-oriented guide
