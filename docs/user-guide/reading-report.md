# Reading a report (v1)

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Understand and interpret InvarLock v1 reports. |
| **Audience** | Reviewers validating evaluation evidence. |
| **Key sections** | Executive Summary, Quality Gates, Primary Metric, Provenance, Measurement contracts. |
| **Validation** | Use `invarlock verify <evaluation.report.json>` to check schema, pairing, and required runtime provenance via `runtime.manifest.json`. |
| **Source of truth** | [reports](../reference/reports.md) for full schema. |

This guide highlights the key sections of a v1 report and how to
interpret them.

Browser-first reading order for the HTML export:

```text
1. Summary chips
2. Quick links rail
3. Executive Summary
4. Quality Gates
5. Provenance / Policy details
```

The HTML shell is intentionally thin. It adds navigation and orientation, but
the evidence still comes from the same canonical report body and should be
re-checked with `invarlock verify`.

- Executive Summary
  - First-screen summary of overall PASS/FAIL plus the compact gate table (primary metric, drift, invariants, guards, overhead when evaluated).
- Summary chips (HTML shell)
  - Browser-only overview of overall status, primary-metric kind, and whether the bundle still links back to both run reports for `report explain --evaluation-report`.
- Quick links rail (HTML shell)
  - Browser-only navigation for jumping to Executive Summary, gates, provenance, and appendix sections without scrolling through the whole report.
- Primary Metric row
  - Shows the task‑appropriate metric (ppl_* or accuracy), its point estimates,
    and paired CI. The ratio/Δpp vs baseline drives the gate.
- Primary Metric Tail row (when present)
  - Shows tail regression vs baseline for ppl-like metrics using per-window
    ΔlogNLL (e.g., P95 and tail mass above ε). Default policy is `mode: warn`
    (does not fail the report); `mode: fail` sets
    `validation.primary_metric_tail_acceptable = false`.
- System Overhead row (when available)
  - Latency and throughput stats appear separate from quality and reflect the guarded run.
- pPL identity (ppl families)
  - Confirms `exp(mean Δlog)` ≈ `ratio_vs_baseline`; Δlog CI maps to ratio CI
    when reported.
- Provenance
  - Provider/environment/policy digests: `provider_digest`
    (ids/tokenizer/masking), `env_flags`, and `policy_digest` with thresholds
    snapshot.
  - `dataset.hash.source` tells you whether dataset hashes were derived from
    explicit preview/final hashes, explicit token IDs, or a config fallback.
- Policy Configuration
  - Human-readable tier/digest plus collapsible resolved policy YAML; full details remain in `evaluation.report.json`.
- Measurement contract
  - `resolved_policy.spectral.measurement_contract` /
    `resolved_policy.rmt.measurement_contract` pin the estimator + sampling
    procedure used by guards.
  - `rmt.mode` makes the active RMT measurement path reviewer-visible; public
    reports emit `activation_edge_risk`.
  - `spectral.measurement_contract_hash` / `rmt.measurement_contract_hash` are
    compact digests for audit and baseline pairing.
  - In CI/Release, `invarlock verify` enforces baseline/subject pairing (`*_measurement_contract_match = true`).
- Confidence label
  - High/Medium/Low based on CI width and stability; see thresholds and `unstable` flag.

Tip: Use `invarlock verify` to recheck schema, pairing, ratio math, and the
adjacent `runtime.manifest.json`.

### Executive Summary Interpretation

- **Overall** mirrors the canonical gate allow-list. A FAIL means at least one gate failed.
- **Primary Metric** shows ratio/Δpp vs baseline; compare to tier thresholds in the gate table.
- **Drift** is final/preview; large drift usually indicates dataset/device instability.
- **Overhead** appears only when guard overhead is evaluated; skipped in some profiles.

## Related Documentation

- [reports](../reference/reports.md) — Full v1 schema reference, telemetry, and HTML export
- [Assurance Case](../assurance/00-assurance-case.md) — Report claim scope
- [CLI Reference](../reference/cli.md) — `invarlock verify` command details
