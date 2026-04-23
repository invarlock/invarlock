# reports

This document consolidates all report-related reference material: schema,
telemetry fields, and HTML export.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Define the v1 report contract, telemetry fields, and export formats. |
| **Audience** | Operators verifying reports and tool authors parsing them. |
| **Schema version** | `schema_version = "v1"` (PM-only). |
| **Source of truth** | `invarlock.reporting.report_schema.REPORT_JSON_SCHEMA`. |

## Table of Contents

- [Quick Start](#quick-start)
- [report Layout](#report-layout)
  - [Executive Summary Interpretation](#executive-summary-interpretation)
- [Schema](#schema)
  - [Minimal v1 report Example](#minimal-v1-report-example)
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
# Generate a report from a run report
invarlock report generate \
  --run runs/subject/report.json \
  --baseline-run-report runs/baseline/report.json \
  --format report

# Validate an container-backed report bundle
invarlock verify reports/eval/evaluation.report.json
# expects reports/eval/runtime.manifest.json next to the report

# Explain a bundle directly from report provenance
invarlock report explain --evaluation-report reports/eval/evaluation.report.json

# Inspect telemetry fields
jq '.telemetry' reports/eval/evaluation.report.json

# Export to HTML
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
```

Artifact model:

| Artifact | Produced by | Primary consumers |
| --- | --- | --- |
| `evaluation.report.json` | `invarlock evaluate`, `invarlock report generate --format report` | `invarlock verify`, `invarlock report html`, `invarlock report validate`, `invarlock report explain --evaluation-report`, `invarlock advanced runtime-verify` |
| `report.json` | Baseline/subject run directories under `runs/...` | `invarlock report generate`, `invarlock report explain --subject-report ... --baseline-report ...` |

## report Layout

The markdown report is structured to highlight evaluation outcomes first:

Container-backed evaluations emit `runtime.manifest.json` next to
`evaluation.report.json`. Archive and verify them together.

The HTML export keeps that same body content but adds a browser shell with:

- summary chips for the overall status, primary-metric kind, and linked-run readiness
- quick links for the major report sections
- anchored section headings so reviews can deep-link directly into the report

- **Executive Summary**: one-line PASS/FAIL + compact gate table (primary metric, drift, invariants, spectral, RMT, overhead).
- **Quality Gates**: table of canonical gating checks with measured values.
- **Guard Check Details**: invariants, spectral stability, RMT health, and pairing snapshots.
- **Primary Metric**: task-specific metric summary with CI + baseline comparison.
- **Guard Observability**: compact summaries with expandable guard details.
- **Policy Configuration**: tier + digest summary with resolved policy details in `<details>`.
- **Appendix**: environment, inference diagnostics, and variance guard details.

### Executive Summary Interpretation

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
│                        ┌───────────────┐                                │
│                        │  BASELINE RUN │                                │
│                        │   report.json │                                │
│                        └───────┬───────┘                                │
│                                │                                        │
│                                ▼                                        │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │                      SUBJECT RUN                              │     │
│   │ ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │     │
│   │ │ model.meta  │  │ dataset.data │  │ guards[].metrics       │ │     │
│   │ │ ─────────── │  │ ──────────── │  │ ────────────────────── │ │     │
│   │ │ model_id    │  │ provider     │  │ invariants.passed      │ │     │
│   │ │ adapter     │  │ seq_len      │  │ spectral.summary       │ │     │
│   │ │ device      │  │ windows.stats│  │ rmt.families           │ │     │
│   │ │ seeds       │  │ paired_count │  │ variance.enabled       │ │     │
│   │ └─────────────┘  └──────────────┘  └────────────────────────┘ │     │
│   │                                                               │     │
│   │ ┌─────────────────┐  ┌──────────────────────────────────────┐ │     │
│   │ │ metrics         │  │ policy_resolved                      │ │     │
│   │ │ ─────────────── │  │ ──────────────────────────────────── │ │     │
│   │ │ primary_metric  │  │ spectral.*, rmt.*, variance.*        │ │     │
│   │ │ ratio_vs_base   │  │ tier_policy_name                     │ │     │
│   │ │ display_ci      │  │ thresholds_hash                      │ │     │
│   │ └─────────────────┘  └──────────────────────────────────────┘ │     │
│   └────────────────────────────────┬──────────────────────────────┘     │
│                                    │                                    │
│                                    ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐    │
│   │                    make_report()                               │    │
│   │ baseline_report + subject_report → evaluation.report.json      │    │
│   └────────────────────────────────┬───────────────────────────────┘    │
│                                    │                                    │
│                                    ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐    │
│   │                    invarlock verify                            │    │
│   │ schema + pairing + ratio math + measurement contracts +        │    │
│   │ runtime-manifest provenance                                    │    │
│   └────────────────────────────────────────────────────────────────┘    │
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
- **Baseline pairing**: reports assume paired windows; verification enforces
  pairing in CI/Release profiles.

### Provenance Map

| report block | Sourced from report | Verify checks |
| --- | --- | --- |
| `meta` | `report.meta` | Schema only. |
| `dataset` / `evaluation_windows` | `report.data`, `report.dataset.windows.stats` | Pairing + count checks; `dataset.hash.source` records whether hashes came from explicit preview/final hashes, explicit token IDs, or config fallback. |
| `primary_metric` | `report.metrics.primary_metric` | Ratio + drift band (CI/Release). |
| `spectral` / `rmt` / `variance` | `report.guards[]` | Measurement contracts (CI/Release); `rmt.mode` surfaces the active RMT measurement path. |
| `provenance.provider_digest` | `report.provenance.provider_digest` | Required in CI/Release. |

### Minimal v1 report Example

The example below shows a realistic, PM‑only report envelope. It follows
the validator in `invarlock.reporting.report_schema` and the
fields produced by `invarlock.reporting.make_report`.

```json
{
  "schema_version": "v1",
  "run_id": "20251013T012233Z-quant8-balanced",
  "meta": {
    "model_id": "gpt2",
    "adapter": "hf_causal",
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

- `schema_version` is a string and must be `"v1"` for the v1 format.
- `run_id` is a short, opaque identifier; reports treat it as a stable
  string key.
- `primary_metric` is the **canonical** place for PM values.
- The `validation` object holds boolean flags; only a small allow‑list of
  keys is recognized by the validator.

### Schema Summary (Validator View)

The v1 validator uses a JSON Schema (draft 2020‑12) embedded in
`REPORT_JSON_SCHEMA`. The schema is intentionally permissive around new
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

- `validation` is an object of booleans; allowed keys come from
  `contracts/validation_keys.json`, and report validation fails closed when that
  contract is missing or malformed.
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
- The validator rejects reports that contain non‑boolean values under
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
`invarlock.reporting.report_schema.REPORT_JSON_SCHEMA`.

### report → Verify Matrix

| report block | Derived from | Verify checks |
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
| `schema_version` | Yes | `REPORT_SCHEMA_VERSION` | PM-only v1 |
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

For ppl-like metrics with paired per-window logloss, reports may include
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

Telemetry values are copied from `report.json` into reports and always
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
- `dataset.hash.source` distinguishes content-derived, provider-derived, and config-derived dataset hashes.
- `rmt.mode` and `rmt.measurement_contract_hash` show which RMT measurement contract produced the report evidence.

---

## HTML Export

The HTML renderer converts the Markdown report into structured HTML
tables (via the `markdown` library when available) and preserves the same
numeric values (ratios, CIs, deltas). When the dependency is unavailable, the
renderer falls back to a `<pre>` block. Use `--embed-css` (default) to inline
a minimal stylesheet for standalone use, including status badges and
print-friendly rules.

### CLI

```bash
invarlock report html -i <evaluation.report.json> -o <out.html>
```

**Flags:**

- `--embed-css/--no-embed-css` — inline stylesheet (default: embed)
- `--force` — overwrite existing output

### Python API

```python
from invarlock.reporting.html import render_report_html

html = render_report_html(report)
```

---

## Troubleshooting

### Schema Issues

- **Schema validation fails**: check `schema_version` and required top-level
  fields (`run_id`, `meta`, `dataset`, `artifacts`, `primary_metric`).
- **Unexpected validation keys**: ensure `validation.*` keys match the allow-list
  in `report_schema`.

### Telemetry Issues

- **Telemetry missing**: ensure the run completed successfully and check
  `report.metrics` for latency/memory values.

### HTML Export Issues

- **Missing report**: generate one first via `invarlock report --format report`.
- **HTML missing styles**: omit `--no-embed-css` or apply custom CSS later in your publishing layer.

---

## Observability

- `validation.*`, `resolved_policy.*`, and `policy_digest.*` capture policy state.
- `primary_metric_tail` appears only for ppl-like metrics with paired windows.
- The rendered HTML is derived from the Markdown report. If values look wrong,
  inspect the underlying `evaluation.report.json`.
- The Markdown report is a human-readable view that starts with the Executive Summary; the JSON report is the canonical evidence artifact.

---

## Related Documentation

- [CLI Reference](cli.md)
- [Artifact Layout](artifacts.md)
- [Assurance Case](../assurance/00-assurance-case.md) — What the report covers
- [Reading a report](../user-guide/reading-report.md) — User-oriented guide
