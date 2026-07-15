# reports

This document consolidates all report-related reference material: schema,
telemetry fields, and HTML export.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Define the v1 report contract, telemetry fields, and export formats. |
| **Audience** | Operators verifying reports and tool authors parsing them. |
| **Schema version** | `schema_version = "v1"` (minimal schema core; CI, release, and strict assurance add profile-specific requirements). |
| **Source of truth** | `invarlock.reporting.report_schema.REPORT_JSON_SCHEMA`. |

## Table of Contents

- [Quick Start](#quick-start)
- [report Layout](#report-layout)
  - [Executive Summary Interpretation](#executive-summary-interpretation)
- [Report Outline](report-outline.md)
- [Schema](#schema)
  - [Minimal v1 report Example](#minimal-v1-report-example)
  - [Schema Summary](#schema-summary-validator-view)
  - [Required vs Optional Blocks](#required-vs-optional-blocks)
  - [Primary Metric Tail Gate](#primary-metric-tail-gate-optional)
- [Telemetry Fields](#telemetry-fields)
- [HTML Export](#html-export)
- [CI and Registry Exports](#ci-and-registry-exports)
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

# Validate a strict container-backed report using independent acceptance inputs
invarlock verify \
  --profile ci \
  --assurance strict \
  --baseline runs/baseline/report.json \
  --policy-pack acceptance-policy-pack.json \
  --expected-runtime-image-digest "$EXPECTED_RUNTIME_IMAGE_DIGEST" \
  reports/eval/evaluation.report.json
# expects reports/eval/runtime.manifest.json next to the report

# Explain a bundle directly from the evaluation report
invarlock report explain --evaluation-report reports/eval/evaluation.report.json

# Inspect telemetry fields
jq '.telemetry' reports/eval/evaluation.report.json

# Export to HTML
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html

# Export CI/model-registry handoff artifacts
invarlock report export -i reports/eval/evaluation.report.json --format mlflow-tags
invarlock report export -i reports/eval/evaluation.report.json --format model-card-md
invarlock report export -i reports/eval/evaluation.report.json --format release-review-md
```

Artifact model:

| Artifact | Produced by | Primary consumers |
| --- | --- | --- |
| `evaluation.report.json` | `invarlock evaluate`, `invarlock report generate --format report` | `invarlock verify`, `invarlock report html`, `invarlock report export`, `invarlock report validate`, `invarlock report explain --evaluation-report`, `invarlock advanced runtime-verify` |
| `report.json` | Baseline/subject run directories under `runs/...` | `invarlock report generate`, `invarlock report explain --subject-report ... --baseline-report ...` |

`report explain --evaluation-report` reads `evaluation.report.json` directly.
Raw subject and baseline `report.json` files are still useful when you need to
regenerate the paired evaluation bundle or inspect low-level run telemetry, but
portable fixtures that ship only `evaluation.report.json`,
`runtime.manifest.json`, and evidence metadata can still be rendered, schema
validated, and explained. Strict verification additionally requires the
complete raw baseline and independently maintained policy pack.

Strict reports bind each model through one typed `model_identity` object in
`meta`, `subject_ref`, and `baseline_ref`. A remote identity contains its
immutable revision; a local identity contains its checkpoint-tree `sha256`.
The verifier requires exact object equality between metadata and references.

## report Layout

The markdown report is structured to highlight evaluation outcomes first:

Renderers use the shared renderer-neutral [Report Outline](report-outline.md).
The outline groups report evidence as Decision, Primary Metric, Policy Gates,
Guard Signals, optional
Benchmark Comparison, Evidence And Provenance, and Technical Appendix.

Container-backed evaluations emit `runtime.manifest.json` next to
`evaluation.report.json`. Archive and verify them together.

The sibling manifest and its report hash establish bundle consistency, not
evidence-source authenticity or execution attestation. Strict verification also
requires a complete raw baseline, an independently maintained policy pack, and
`--expected-runtime-image-digest` from independent policy channels. The digest
comparison checks the manifest's image claim, but a compromised evaluation environment can
still fabricate a consistent bundle naming that digest.

The HTML export renders that shared outline directly. It adds:

- a summary ledger row for **report-local gates**, subject model, baseline
  model/run, metric, and guard warnings
- an unconditional `REPORT-LOCAL / UNVERIFIED RENDER` notice: rendering a JSON
  report does not independently verify its bytes, provenance, policy inputs, or
  report-authored assurance fields
- a sticky brand/theme row with a light/dark toggle
- quick links for the outline sections, with hash anchors and the active
  section highlight aligned to the sticky row while scrolling
- task-aware primary-metric wording, including ratio output for ppl-like tasks
  and percentage-point deltas for accuracy tasks
- guard-warning detail tables when baseline-relative warning data is present
- an optional Benchmark Comparison section when benchmark/scenario data is
  embedded in the report
- capped appendix previews for raw policy/plugin/artifact blocks, with
  `evaluation.report.json` remaining the complete audit artifact

The embedded stylesheet follows the current InvarLock site Ledger ink token
map: warm paper/ink in light mode, warm-black/cream in dark mode, blue as the
brand accent, oxblood as the editorial signal, and green/red/yellow reserved
for verdict states.

- **Decision**: PASS/FAIL, evidence mode, subject model, baseline model/run, edit, primary metric, and warning count.
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
| Guard Metric Impact | Direction-aware degradation of the guarded metric vs a paired bare control | Only present when the paired comparison was evaluated |

## Evidence Flow

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        EVIDENCE FLOW                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   baseline/report.json + subject/report.json                            │
│                         │                                               │
│                         ▼                                               │
│   report builder: pair windows, recompute metrics, apply local policy   │
│                         │                                               │
│                         ▼                                               │
│   evaluation.report.json + runtime.manifest.json                        │
│                         │                                               │
│                         ├──────────────▶ invarlock report html           │
│                         │                 (review rendering)            │
│                         ▼                                               │
│   invarlock verify  ◀── independent verifier inputs                     │
│                         retained raw baseline report                    │
│                         acceptance policy pack                          │
│                         expected runtime-image digest                   │
│                         │                                               │
│                         ▼                                               │
│   exit 0 + JSON verification result, or nonzero + diagnostics           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Schema

### Concepts

- **Schema stability**: v1 has a stable core around primary metric,
  dataset/window metadata, artifacts, plugins, and report identity. Optional
  policy, guard, provenance, telemetry, and confidence blocks are additive
  unless moved into the required core, which requires a schema-version bump.
- **Validation allow-list**: only specific `validation.*` flags are accepted by
  the schema validator.
- **Baseline pairing**: baseline and subject final windows use identical IDs;
  verification enforces that pairing in CI/Release profiles. Preview and final
  are disjoint slices and their drift interval uses independent resampling.

### Recorded Metadata Map

| report block | Sourced from report | Verify checks |
| --- | --- | --- |
| `meta` | `report.meta` | Schema only. |
| `dataset` / `evaluation_windows` | `report.data`, `report.dataset.windows.stats` | Pairing + count checks; `dataset.hash.source` records whether hashes came from explicit preview/final hashes, explicit token IDs, or config fallback. |
| `primary_metric` | `report.metrics.primary_metric` | Ratio + drift band (CI/Release). |
| `spectral` / `rmt` / `variance` | `report.guards[]` | Measurement contracts (CI/Release); `rmt.mode` surfaces the active RMT measurement path. |
| `provenance.provider_digest` | `report.provenance.provider_digest` | Required in CI/Release. |

These fields are report-recorded evidence. Schema, digest, and consistency
checks can detect malformed or internally inconsistent artifacts; they do not
authenticate the evidence source. Evidence-pack signatures require an independently
trusted signer fingerprint, and runtime image matching is not execution
attestation.

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
        "bootstrap": {
          "replicates": 2000,
          "seed": 1337,
          "method": "bca_paired_delta_log",
          "preview_final_delta_basis": "independent_disjoint_slices",
          "preview_final_delta_method": "independent_percentile_delta_log",
          "preview_final_delta_seed": 1434
        },
        "preview_final_slice_delta_summary": {
          "mean": 0.0216,
          "ci": [-0.01, 0.05],
          "basis": "independent_disjoint_slices",
          "paired": false,
          "ci_method": "independent_percentile_delta_log",
          "ci_reason": null,
          "preview_windows": 200,
          "final_windows": 200,
          "degenerate": false,
          "degenerate_reason": null
        },
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
  "guard_metric_impact": {
    "metric_kind": "ppl_causal",
    "direction": "lower",
    "degradation_basis": "relative_increase",
    "bare_value": 43.00,
    "guarded_value": 43.10,
    "bare_facts": {
      "weighted_logloss_sum": 376.1200115693562,
      "token_count": 100,
      "example_ids_digest": "d0bca111f8628137adc4c16f123496dcdd1d590d06cb5d9acd68b39fe656fb97"
    },
    "guarded_facts": {
      "weighted_logloss_sum": 376.3522997109702,
      "token_count": 100,
      "example_ids_digest": "d0bca111f8628137adc4c16f123496dcdd1d590d06cb5d9acd68b39fe656fb97"
    },
    "bare_report": {
      "primary_metric": {"kind": "ppl_causal", "final": 43.0},
      "final": {
        "logloss": [3.7612001156935624],
        "token_counts": [100],
        "window_ids": [0]
      }
    },
    "degradation": 0.0023255814,
    "degradation_limit": 0.01,
    "display_value": 0.23255814,
    "display_unit": "percent",
    "source": "paired_control",
    "schedule_digest": "d566dfb39f20d549d1c0684e94949c71",
    "checks": {
      "metric_kind_matches": true,
      "measurements_valid": true,
      "guard_metric_impact": true,
      "arm_facts_replay": true
    },
    "diagnostics": [],
    "evaluated": true,
    "passed": true
  },
  "validation": {
    "primary_metric_acceptable": true,
    "primary_metric_tail_acceptable": true,
    "preview_final_drift_acceptable": true,
    "guard_metric_impact_acceptable": true
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

Here `paired_windows` counts baseline/subject final pairs.
`window_match_fraction` summarizes matching against the supplied baseline
pairing context (both slices when that context contains both), while
`window_overlap_fraction` records token-window overlap implied by sequence
length and stride. It is not an ID-pairing measure. Raw
`evaluation_windows.preview.window_ids` and
`evaluation_windows.final.window_ids` provide the separate evidence that the
preview/final ID sets are disjoint. The configured `bootstrap.method` names the
paired baseline/subject method; the explicit `preview_final_delta_*` metadata
and `preview_final_slice_delta_summary` identify the separate independent-slice
drift interval.

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
- `run_id` — non‑empty string (minimum length 1).
- `meta` — object (model/device/seeds; validator does not fix sub‑shape).
- `dataset` — object with at least:
  - `provider`: string
  - `seq_len`: integer ≥ 0
  - `windows.preview`: integer ≥ 0
  - `windows.final`: integer ≥ 0
  - `windows.stats`: object (paired-window stats and coverage)
- `artifacts` — object (paths to `report.json`, `events.jsonl`, etc.).
- `plugins` — object listing discovered adapters/edits/guards.
- `primary_metric` — object (canonical primary metric snapshot).

**Primary metric block (object required, only `kind` required by schema):**

- `primary_metric.kind`: string (e.g., `"ppl_causal"`, `"accuracy"`).
- `primary_metric.preview` / `primary_metric.final`: numbers when available.
- `primary_metric.n_preview` / `primary_metric.n_final`: evaluated example
  counts for both accuracy arms; strict verification requires positive explicit
  integers reconciled with the classification and coverage blocks.
- `primary_metric.counts_source`: identifies measured vs synthetic count provenance.
- `primary_metric.estimated`: marks whether the metric/count surface is estimated;
  strict accuracy requires measured, non-estimated evidence.
- `primary_metric.ratio_vs_baseline`: multiplicative ratio for PPL-like metrics;
  it is invalid for accuracy reports.
- `primary_metric.delta_vs_baseline_pp`: canonical accuracy change in percentage
  points, computed as `100 × (final - baseline_final)`.
- `primary_metric.display_ci`: two‑element numeric array `[lo, hi]` when available.
- Additional optional fields: `unit`, `direction`, `ci`, `gating_basis`,
  `aggregation_scope`, `estimated`, etc.

**Guard metric impact block (required for Release and strict assurance):**

- `metric_kind`, `direction`, and `degradation_basis` identify the registered
  metric semantics. PPL-like metrics are lower-is-better with
  `relative_increase`; accuracy is higher-is-better with `absolute_drop`.
- `bare_value` and `guarded_value` are the paired arm measurements.
- `degradation` is `guarded_value / bare_value - 1` for PPL-like metrics and
  `bare_value - guarded_value` for accuracy. Positive is worse; negative is an
  improvement.
- `degradation_limit` is the maximum allowed degradation in the selected basis.
- `display_value` and `display_unit` are presentation-only: percent for relative
  PPL change and percentage points for accuracy change. Verification recomputes
  both the canonical and display values.
- `source` records how the comparison was obtained, while `schedule_digest`
  binds it to the paired example/window schedule. `checks` and `diagnostics`
  expose the gate result and any failure reason.
- `evaluated` and `passed` must both be true. Strict verification also requires
  valid bound arm evidence and rejects a skip, non-finite value, registry
  mismatch, or inconsistent derived field.

**Validation flags:**

- `validation` is an object of booleans; allowed keys come from
  `contracts/validation_keys.json`, and report validation fails closed when that
  contract is missing or malformed.
- Common flags:
  - `primary_metric_acceptable`
  - `primary_metric_tail_acceptable`
  - `preview_final_drift_acceptable`
  - `guard_metric_impact_acceptable`
  - `guard_warnings_present`
  - `guard_warning_policy_acceptable`
  - `invariants_pass`
  - `spectral_stable`
  - `rmt_stable`
  - `hysteresis_applied`
  - `moe_observed`
  - `moe_identity_ok`
- The validator rejects reports that contain non‑boolean values under
  any of these keys.

**Guard warnings (optional):**

- `guard_warnings.present`: `true` when the subject has guard-signal movement
  relative to the baseline while the hard policy may still pass.
- `guard_warnings.warning_count`: number of warning records.
- `guard_warnings.warnings[]`: structured warnings with `guard`, `kind`,
  optional `family`/`module`, `baseline`, `subject`, `policy_gate`, and
  `message`.
- Warnings are advisory by default. `invarlock verify --warning-policy fail`
  treats any warning as a verification failure.

**Guard measurement evidence:**

- `rmt.measurement_contract` records the RMT measurement mode/configuration
  used for the report and is cross-checked against resolved policy evidence.
- `spectral.families[*].max`, `spectral.families[*].mean`,
  `spectral.families[*].count`, `spectral.families[*].violations`, and
  `spectral.families[*].kappa` are the per-family summary fields when spectral
  measurements are present.

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
| `validation` | `report.metrics` + policy thresholds | Schema allow-list, release-required values, and primary-metric policy recomputation. |
| `guard_warnings` | Baseline/subject guard evidence | Advisory by default; fail only under strict warning policy. |
| `spectral` / `rmt` / `variance` | `report.guards[]` | Measurement contracts (CI/Release). |
| `guard_metric_impact` | `report.guard_metric_impact` | Recomputes the registered metric-specific degradation and display value; Release requires evaluated, passing, paired evidence and skips fail. |
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
| `policy_digest` / `resolved_policy` | Optional | Tier policies | Policy changes |
| `guard_metric_impact` | Release/strict | Paired bare/guarded primary-metric evidence | Direction-aware degradation contract |
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
`scripts/smoke/run_cpu_telemetry.sh`.

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

The HTML renderer builds a browser-readable report from the shared
renderer-neutral [Report Outline](report-outline.md). It does not depend on the
Markdown renderer or the optional `markdown` Python package. Use `--embed-css`
(default) to inline the standalone stylesheet; use `--no-embed-css` only when
an external publishing system supplies its own styles.

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

## CI and Registry Exports

`invarlock report export` converts an existing `evaluation.report.json` into
small handoff artifacts for systems that already own CI, registry, model-card,
or release-review workflows.

```bash
invarlock report export \
  --evaluation-report reports/eval/evaluation.report.json \
  --format mlflow-tags \
  --policy-profile ci \
  --verify-result reports/eval/invarlock-verify.json \
  --output reports/eval/mlflow-tags.json
```

| Format | Output | Purpose |
| --- | --- | --- |
| `mlflow-tags` | JSON with `tags` and report artifact path | Set registry tags and log the report as an MLflow artifact from an MLflow-enabled environment. |
| `model-card-md` | Markdown block | Paste InvarLock evidence into a Hugging Face model card or equivalent model README. |
| `release-review-md` | Markdown packet | Attach pass/fail, baseline/subject identity, report hash, policy profile, and acceptance checklist to release review. |

These exports summarize regression evidence only. They do not change verifier
semantics, replace `invarlock verify`, or provide deployment approval.

Without `--verify-result`, an export labels its status `report_local_pass` or
`report_local_fail`. That is a rendering of the submitted report's gate fields,
not an independent verifier result.

Common options:

- `--policy-profile`: profile label to use when the report does not record one.
- `--report-url`: public report URL for Markdown exports.
- `--evidence-url`: public evidence-pack URL for Markdown exports.
- `--verify-result`: path to `invarlock verify --json` output. When supplied,
  the exporter strictly parses `verify-v1`, requires exactly one result item
  whose `id` resolves to the report, and requires the item's receipt
  `subject_report_sha256` to match the exact report bytes being exported.
  Duplicate keys, non-finite values, string booleans, malformed items, stale
  report paths, and stale receipt digests are rejected. Current receipts are
  explicitly unsigned, so a valid receipt is labelled
  `receipt_bound_untrusted` with a separate claimed verifier outcome; it never
  creates an independently verified pass badge.
- `--force`: overwrite an existing output file.

---

## Troubleshooting

### Schema Issues

- **Schema validation fails**: check `schema_version` and required top-level
  fields (`run_id`, `meta`, `dataset`, `artifacts`, `primary_metric`).
- **Unexpected validation keys**: ensure `validation.*` keys match the allow-list
  in `contracts/validation_keys.json`.

### Telemetry Issues

- **Telemetry missing**: ensure the run completed successfully and check
  `report.metrics` for latency/memory values.

### HTML Export Issues

- **Missing report**: generate one first via
  `invarlock report generate --run <subject report.json> --baseline-run-report <baseline report.json> --format report -o <output-dir>`.
- **HTML missing styles**: omit `--no-embed-css` or apply custom CSS later in your publishing layer.

---

## Observability

- `validation.*`, `resolved_policy.*`, and `policy_digest.*` capture policy state.
- `primary_metric_tail` appears only for ppl-like metrics with paired windows.
- HTML and Markdown are both rendered views of the shared report outline. If
  values look wrong, inspect the underlying `evaluation.report.json`.
- The JSON report is the canonical evidence artifact, but it remains
  report-supplied unless authenticated and collected inside a trusted
  evaluation boundary.

---

## Related Documentation

- [CLI Reference](cli.md)
- [Artifact Layout](artifacts.md)
- [Assurance Case](../assurance/00-assurance-case.md) — What the report covers
- [Reading a report](../user-guide/reading-report.md) — User-oriented guide
