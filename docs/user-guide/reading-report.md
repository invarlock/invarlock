# Reading a report (v1)

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Understand and interpret InvarLock v1 reports. |
| **Audience** | Reviewers validating evaluation evidence. |
| **Key sections** | Decision, Primary Metric, Policy Gates, Guard Signals, Evidence And Provenance, Technical Appendix. |
| **Validation** | Use `invarlock verify <evaluation.report.json>` to check schema, pairing, and required runtime provenance via `runtime.manifest.json`. |
| **Source of truth** | [reports](../reference/reports.md) for full schema. |

This guide highlights the key sections of a v1 report and how to
interpret them.

Browser-first reading order for the HTML export:

```text
1. Summary ledger row
2. Sections rail
3. Decision
4. Primary Metric
5. Policy Gates
6. Guard Signals
7. Evidence And Provenance
8. Technical Appendix
```

The HTML export renders the shared report outline directly. The evidence still
comes from `evaluation.report.json` and should be re-checked with
`invarlock verify`.

- Decision
  - First-screen summary of overall PASS/FAIL, evidence mode, subject model,
    baseline model/run, adapter, edit, primary metric, and guard-warning count.
- Summary ledger row
  - Browser overview of verdict, subject, baseline, primary-metric kind, and
    guard warnings.
- Sections rail
  - Browser navigation for jumping to the canonical outline sections without
    scrolling through the whole report. In HTML, the active section is
    highlighted using the same measured sticky-row offset as hash navigation.
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
- Guard Warnings (when present)
  - Shows baseline-relative guard-signal changes that are still inside the hard
    policy budget. These are warnings by default, not verification failures.
  - Use `invarlock verify --fail-on-warnings <evaluation.report.json>` when your
    workflow wants any guard warning to fail the verification step.
- pPL identity (ppl families)
  - Confirms `exp(mean Δlog)` ≈ `ratio_vs_baseline`; Δlog CI maps to ratio CI
    when reported.
- Provenance
  - Provider/environment/policy digests: `provider_digest`
    (ids/tokenizer/masking), `env_flags`, and `policy_digest` with thresholds
    snapshot.
  - `dataset.hash.source` tells you whether dataset hashes were derived from
    explicit preview/final hashes, explicit token IDs, or a config fallback.
- Technical Appendix
  - Capped previews of verbose policy, plugin, and artifact blocks. Full details
    remain in `evaluation.report.json`.
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

`invarlock report explain --evaluation-report` needs the evaluation bundle to
link back to raw subject and baseline `report.json` files. Public evidence
fixtures may omit those raw run reports while still being valid for
`verify`, `report html`, and `report validate`.

### Decision Interpretation

- **Overall** mirrors the canonical gate allow-list. A FAIL means at least one gate failed.
- **Primary Metric** shows ratio/Δpp vs baseline; compare to tier thresholds in the gate table.
- **Drift** is final/preview; large drift usually indicates dataset/device instability.
- **Guard Warnings** mean the edit moved a guard signal relative to the
  baseline while remaining within hard policy. They become failures only under
  strict warning mode.
- **Overhead** appears only when guard overhead is evaluated; skipped in some profiles.

## Related Documentation

- [reports](../reference/reports.md) — Full v1 schema reference, telemetry, and HTML export
- [Assurance Case](../assurance/00-assurance-case.md) — Report claim scope
- [CLI Reference](../reference/cli.md) — `invarlock verify` command details
