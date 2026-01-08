# Certificate Schema (v1)

This page anchors the certificate contract that InvarLock emits in the
**PM‑only v1** format. It focuses on:

1. A **minimal example certificate** that matches the current validator.
2. A **schema summary** describing required top‑level fields and key sections.
3. A short **evolution note** for readers migrating from older, ppl‑centric layouts.

> Certificates are versioned. The material below describes
> `schema_version = "v1"`. Any change that alters required fields or semantics
> must bump the major version.

---

## Minimal v1 Certificate Example

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
  "validation": {
    "primary_metric_acceptable": true,
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
- `primary_metric` is the **canonical** place for PM values; ppl‑like
  top-level `ppl` blocks are no longer required.
- The `validation` object holds boolean flags; only a small allow‑list of
  keys is recognized by the validator (see below).

---

## Schema Summary (Validator View)

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
    - `preview_final_drift_acceptable`
    - `guard_overhead_acceptable`
    - `invariants_pass`
    - `spectral_stable`
    - `rmt_stable`
  - The validator rejects certificates that contain non‑boolean values under
    any of these keys.

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

---

## Evolution from ppl‑centric schemas

Earlier experimental schemas exposed top‑level `ppl`, `spectral`, `rmt`, and
`variance` blocks and treated ppl as the only primary metric. The v1
PM‑only schema:

- moves primary metric details under `primary_metric`,
- relies on `validation` flags and `policy_digest` for gates and policies,
- keeps guard‑specific blocks (`spectral`, `rmt`, `variance`) available but
  **optional** from the validator’s perspective.

For existing integrations:

- Prefer `primary_metric.{preview,final,ratio_vs_baseline,display_ci}` over
  older `ppl_*` fields.
- Read gates from `validation.*` and thresholds from `policy_digest` /
  `resolved_policy`.
- Treat additional sections (MoE, system overhead, telemetry, etc.) as
  optional extensions that may appear in future minor releases.
