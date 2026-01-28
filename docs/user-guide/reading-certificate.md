# Reading a Certificate (v1)

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Understand and interpret InvarLock v1 certificates. |
| **Audience** | Reviewers validating certification evidence. |
| **Key sections** | Evaluation Dashboard, Quality Gates, Primary Metric, Provenance, Measurement contracts. |
| **Validation** | Use `invarlock verify <cert.json>` to check schema and pairing. |
| **Source of truth** | [Certificates](../reference/certificates.md) for full schema. |

This guide highlights the key sections of a v1 certificate and how to
interpret them.

- Evaluation Dashboard
  - First-screen summary of overall PASS/FAIL plus key gates (primary metric, drift, invariants, guards, overhead when evaluated).
- Primary Metric row
  - Shows the task‑appropriate metric (ppl_* or accuracy), its point estimates,
    and paired CI. The ratio/Δpp vs baseline drives the gate.
- Primary Metric Tail row (when present)
  - Shows tail regression vs baseline for ppl-like metrics using per-window
    ΔlogNLL (e.g., P95 and tail mass above ε). Default policy is `mode: warn`
    (does not fail the certificate); `mode: fail` sets
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
- Policy Configuration
  - Human-readable tier/digest plus collapsible resolved policy YAML; full details remain in `evaluation.cert.json`.
- Measurement contract
  - `resolved_policy.spectral.measurement_contract` /
    `resolved_policy.rmt.measurement_contract` pin the estimator + sampling
    procedure used by guards.
  - `spectral.measurement_contract_hash` / `rmt.measurement_contract_hash` are
    compact digests for audit and baseline pairing.
  - In CI/Release, `invarlock verify` enforces baseline/subject pairing (`*_measurement_contract_match = true`).
- Confidence label
  - High/Medium/Low based on CI width and stability; see thresholds and `unstable` flag.

Tip: Use `invarlock verify` to recheck schema, pairing, and ratio math.

### Evaluation Dashboard Interpretation

- **Overall** mirrors the canonical gate allow-list. A FAIL means at least one gate failed.
- **Primary Metric** shows ratio/Δpp vs baseline; compare to tier thresholds in the gate table.
- **Drift** is final/preview; large drift usually indicates dataset/device instability.
- **Overhead** appears only when guard overhead is evaluated; skipped in some profiles.

## Related Documentation

- [Certificates](../reference/certificates.md) — Full v1 schema reference, telemetry, and HTML export
- [Safety Case](../assurance/00-safety-case.md) — What the certificate does and does not guarantee
- [CLI Reference](../reference/cli.md) — `invarlock verify` command details
