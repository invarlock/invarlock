# Certificate Schema (v1)

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Define the v1 certificate contract emitted by InvarLock. |
| **Audience** | Operators verifying certificates and tool authors parsing them. |
| **Schema version** | `schema_version = "v1"` (PM-only). |
| **Source of truth** | `invarlock.reporting.certificate_schema.CERTIFICATE_JSON_SCHEMA`. |

## Quick Start

```bash
# Generate a certificate from a run report
invarlock report --run runs/subject/report.json --baseline runs/baseline/report.json --format cert

# Validate certificate structure
invarlock verify reports/cert/evaluation.cert.json
```

## Concepts

- **Schema stability**: v1 is a PM-only contract; breaking changes require a
  schema-version bump.
- **Validation allow-list**: only specific `validation.*` flags are accepted by
  the schema validator.
- **Baseline pairing**: certificates assume paired windows; verification enforces
  pairing in CI/Release profiles.

### Provenance map

| Certificate block | Sourced from report | Verify checks |
| --- | --- | --- |
| `meta` | `report.meta` | Schema only. |
| `dataset` / `evaluation_windows` | `report.data`, `report.dataset.windows.stats` | Pairing + count checks. |
| `primary_metric` | `report.metrics.primary_metric` | Ratio + drift band (CI/Release). |
| `spectral` / `rmt` / `variance` | `report.guards[]` | Measurement contracts (CI/Release). |
| `provenance.provider_digest` | `report.provenance.provider_digest` | Required in CI/Release. |

## Reference

This page anchors the certificate contract that InvarLock emits in the
**PM‑only v1** format. It focuses on:

1. A **minimal example certificate** that matches the current validator.
2. A **schema summary** describing required top‑level fields and key sections.

> Certificates are versioned. The material below describes
> `schema_version = "v1"`. Any change that alters required fields or semantics
> must bump the major version.

---

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

Notes

- `schema_version` is a string and must be `"v1"` for the current format.
- `run_id` is a short, opaque identifier; certificates treat it as a stable
  string key.
- `primary_metric` is the **canonical** place for PM values.
- The `validation` object holds boolean flags; only a small allow‑list of
  keys is recognized by the validator (see below).

---

### Schema Summary (Validator View)

The v1 validator uses a JSON Schema (draft 2020‑12) embedded in
`CERTIFICATE_JSON_SCHEMA`. The schema is intentionally permissive around new
fields while enforcing a small, stable core:

- **Required top‑level fields**
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

- **Primary metric block (required)**
  - `primary_metric.kind`: string (e.g., `"ppl_causal"`, `"accuracy"`).
  - `primary_metric.preview` / `primary_metric.final`: numbers.
  - `primary_metric.ratio_vs_baseline`: number.
  - `primary_metric.display_ci`: two‑element numeric array `[lo, hi]`.
  - Additional optional fields: `unit`, `direction`, `ci`, `gating_basis`,
    `aggregation_scope`, `estimated`, etc.

- **Validation flags**
  - `validation` is an object of booleans; the allow‑list is loaded from
    `contracts/validation_keys.json` when present, or from a small default
    set in code.
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
    any of these keys; the allow‑list may expand when
    `contracts/validation_keys.json` is present.

- **Policy and structure**
  - `policy_digest` — small summary of tier policy thresholds and whether they
    changed relative to the baseline.
  - `resolved_policy` — snapshot of effective guard policies (spectral, rmt,
    variance, metrics).
  - `policy_provenance` — tier label, overrides, and digest (`policy_digest`
    mirror).
  - `structure` — structural deltas and compression diagnostics (optional).

- **Confidence (optional)**
  - `confidence` — object with:
    - `label`: `"High" | "Medium" | "Low"`.
    - `basis`: string description of the confidence basis.
    - Optional numeric fields: `width`, `threshold`, `unstable` flag, etc.

The full machine‑readable schema is available at runtime via
`invarlock.reporting.certificate_schema.CERTIFICATE_JSON_SCHEMA`. Use that
dict directly for tooling that needs strict validation.

### Certificate → Verify matrix

| Certificate block | Derived from | Verify checks |
| --- | --- | --- |
| `meta` | `report.meta` | Schema only. |
| `dataset` / `evaluation_windows` | `report.data`, `report.dataset.windows.stats` | Pairing + count checks. |
| `primary_metric` | `report.metrics.primary_metric` | Ratio + drift band (CI/Release). |
| `validation` | `report.metrics` + policy thresholds | Schema allow‑list only. |
| `spectral` / `rmt` / `variance` | `report.guards[]` | Measurement contracts (CI/Release). |
| `guard_overhead` | `report.guard_overhead` | Required in Release unless skipped. |
| `provenance.provider_digest` | `report.provenance.provider_digest` | Required in CI/Release. |

### Required vs optional blocks

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

### Primary Metric Tail gate (optional)

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

## Troubleshooting

- **Schema validation fails**: check `schema_version` and required top-level
  fields (`run_id`, `meta`, `dataset`, `artifacts`, `primary_metric`).
- **Unexpected validation keys**: ensure `validation.*` keys match the allow-list
  in `certificate_schema`.

## Observability

- `validation.*`, `resolved_policy.*`, and `policy_digest.*` capture policy state.
- `primary_metric_tail` appears only for ppl-like metrics with paired windows.

## Related Documentation

- [CLI Reference](cli.md)
- [Certificate Telemetry](certificate-telemetry.md)
- [Artifact Layout](artifacts.md)
