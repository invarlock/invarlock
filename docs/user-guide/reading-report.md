# Reading a report (v1)

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Understand and interpret InvarLock v1 reports. |
| **Audience** | Readers validating evaluation evidence. |
| **Key sections** | Decision, Primary Metric, Policy Gates, Guard Signals, Evidence And Provenance, Technical Appendix. |
| **Validation** | Use `invarlock verify`; strict verification also requires the complete raw baseline, acceptance policy pack, and independent runtime-image digest. |
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

The HTML and Markdown exports render the shared report outline directly. Every
render includes a `REPORT-LOCAL / UNVERIFIED RENDER` notice: its visible gate
status comes from the submitted `evaluation.report.json`, not an independent
check of report bytes, provenance, policy inputs, or report-authored assurance
fields. Re-check the evidence with `invarlock verify`.

## What PASS Means

The HTML report and `evaluate` summary can show `PASS` when report-local policy
gates pass. For a generated strict report, that is provisional:
`assurance.report_local_verdict=pass` appears alongside
`assurance.verdict=pending_verifier` and
`assurance.verified_assurance_verdict=pending`.

A strict verifier acceptance is different. It requires `invarlock verify
--assurance strict ...` to exit `0` after checking the submitted report against
the complete raw baseline, authorized policy pack, independently obtained image
digest, and sibling runtime manifest. That result means the supplied evidence
satisfies the current InvarLock contract for the identified baseline, subject,
dataset, pairing plan, and acceptance inputs. It does not establish evidence-source
honesty, actual container execution, checkpoint origin, representative
sampling, downstream safety, or general model quality.

For strict verification, confirm that the command supplied:

- the exact raw baseline `report.json` retained from evaluation;
- an independently maintained policy pack;
- an expected runtime-image digest obtained independently of the report.

The submitted report cannot create independence by building the policy pack
itself or copying the digest out of `runtime.manifest.json`. The verifier caller
or CI
owner must control authorization and the digest source. `invarlock advanced
policy build` serializes policy and optional approval metadata but does not
confer approval. See [Policy-pack build and
verification](../reference/contracts.md#policy-packs) and the
[Runtime Provenance Guide](../security/runtime-provenance-guide.md).

The adjacent runtime manifest binds declared runtime metadata to the report.
Matching an expected digest checks the manifest's image claim, not whether that
image actually executed.

## Evidence Maturity

| Surface | Empirical maturity | Current strict behavior | How to read it |
| --- | --- | --- | --- |
| Paired primary metric | **Implemented, recomputed gate** | Must satisfy the configured paired regression policy. | The main behavioral regression decision; field sensitivity depends on the selected data, metric, and thresholds. |
| Invariants | **Stable blocking guard** | Structural and non-finite findings block. | Fail-closed integrity evidence. |
| Spectral | **Operational diagnostic** | Complete findings block under `enforce` and remain visible under `observe`. | Investigate baseline-relative weight movement within calibrated policy scope. |
| RMT | **Experimental diagnostic** | Complete epsilon findings block under `enforce` and remain visible under `observe`. | Treat activation edge-risk as scoped supporting evidence. |
| Variance/VE | **Experimental intervention** | The predictive gate must be evaluated; complete failing predictive-gate outcomes block under `enforce` and remain visible under `observe`. | Treat A/B-gated remediation evidence as workload-specific. |

These labels describe interpretation maturity. They do not change current
CI/release evidence requirements or create separate CLI modes. Observation
authority never permits missing, unsupported, degraded, or monitor-only evidence.

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
  - Use `invarlock verify --warning-policy fail <evaluation.report.json>` when
    your workflow wants any guard warning to fail the verification step.
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
  - `rmt.mode` makes the active RMT measurement path visible to readers; public
    reports emit `activation_edge_risk`.
  - `spectral.measurement_contract_hash` / `rmt.measurement_contract_hash` are
    compact digests for audit and baseline pairing.
  - In CI/Release, `invarlock verify` enforces baseline/subject pairing (`*_measurement_contract_match = true`).
- Confidence label
  - High/Medium/Low based on CI width and stability; see thresholds and `unstable` flag.

Tip: Use `invarlock verify` to recheck schema, pairing, ratio math, and the
adjacent `runtime.manifest.json`.

`invarlock report explain --evaluation-report` reads `evaluation.report.json`
directly. Public evidence fixtures may omit raw subject and baseline
`report.json` files while remaining useful for `report html`, schema validation,
and `report explain`. They cannot satisfy the current strict verifier without
the complete raw baseline and acceptance policy inputs.

### Decision Interpretation

- **Overall** mirrors the canonical gate allow-list. A FAIL means at least one gate failed.
- **Primary Metric** shows ratio/Δpp vs baseline; compare to tier thresholds in the gate table.
- **Drift** is final/preview; large drift usually indicates dataset/device instability.
- **Guard Warnings** mean the edit moved a guard signal relative to the
  baseline while remaining within hard policy. They become failures only under
  strict warning mode.
- **Guard Metric Impact** appears only when the paired model-quality comparison
  is evaluated; it is skipped in some profiles. It does not measure runtime or
  resource cost.

## Related Documentation

- [reports](../reference/reports.md) — Full v1 schema reference, telemetry, and HTML export
- [Assurance Case](../assurance/00-assurance-case.md) — Report claim scope
- [CLI Reference](../reference/cli.md) — `invarlock verify` command details
