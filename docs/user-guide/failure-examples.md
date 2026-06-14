# Failure Examples

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Show common strict-assurance and verifier failure shapes. |
| **Audience** | Operators debugging `evaluate`, `verify`, and report-review outcomes. |
| **Contract scope** | Examples target current strict assurance behavior and report v1 fields. |
| **Related** | [Troubleshooting](troubleshooting.md) covers numbered error codes (`E001`–`E601`). |
| **Source of truth** | `src/invarlock/reporting/verify_contract.py`, `src/invarlock/core/assurance_contract.py`, `docs/assurance/14-trust-model.md`. |

These examples show how to read common non-pass outcomes. They are intentionally
small and focus on report and verifier behavior. For numbered pipeline error
codes (pairing, primary metric, verification), see [Troubleshooting](troubleshooting.md).

## Manipulated Report

**Symptom:**

```text
verify failed: assurance.display_ci_identity_failed
```

**Meaning:** a report field such as `display_ci`, `ci`, or
`ratio_vs_baseline` no longer matches the paired log-space contract.

**Action:** discard the report, regenerate it from the original run artifacts,
and verify again.

## Wrong Guard Order

**Symptom:**

```text
verify failed: assurance.canonical_guard_chain_not_enforced
```

**Meaning:** the report did not use
`invariants -> spectral -> rmt -> variance -> invariants`.

**Action:** rerun with the default strict assurance plan or mark the report
non-assurance.

## Missing Runtime Manifest

**Symptom:**

```text
verify failed: runtime_manifest_missing
```

**Meaning:** the verifier cannot verify the runtime provenance required by strict
assurance.

**Action:** rerun in container mode and keep `runtime.manifest.json` with the
evaluation report. See [Runtime Provenance Guide](../security/runtime-provenance-guide.md).

## Unsupported Model Family

**Symptom:**

```json
{
  "supported": false,
  "reason": "no_supported_rmt_modules",
  "assurance_blocking": true
}
```

**Meaning:** at least one guard cannot produce evidence for the selected model
family under the strict claim.

**Action:** use a supported lane, add the missing adapter/guard support and
tests, or treat the run as exploratory. See
[Model Family Catalog](../reference/model-family-catalog.md) for the current
support inventory.

## Guard Catches Clean Primary Metric

**Reproduce:**

```bash
invarlock verify --profile release --assurance strict \
  public_evidence/caught_regressions/spectral_guard_failure/evaluation.report.json
```

**Symptom:**

```text
Release verification requires validation.spectral_stable == true
spectral did not pass
```

**Meaning:** the primary metric can still look acceptable while a guard blocks
release. In the shipped fixture, `primary_metric.ratio_vs_baseline` is `1.0`,
but the spectral guard records a release-blocking weight-geometry violation.

**Action:** inspect the guard diagnostics, regenerate the edited checkpoint, or
route the run to exploratory/non-assurance review. See
[Public Evidence Walkthrough](public-evidence-walkthrough.md).

For a real model-run counterpart, inspect
`public_evidence/published_basis/mistral_7b/guard_value_demo/guard_value_summary.json`.
That artifact includes a Mistral 7B targeted spectral probe where the primary
metric passes, but the evidence-pack guard-value comparison records a new FFN
spectral cap relative to the published noop basis. It also includes a matched
attention 1.12x negative control that passes PM without adding a new cap.

## Development Fallback

**Symptom:**

```json
{
  "assurance": {
    "mode": "off",
    "fallback_fields_used": true
  }
}
```

**Meaning:** a development report repaired or synthesized evidence for
usability. That output can help debugging but cannot pass strict assurance.

**Action:** rerun with strict assurance after fixing the missing source
evidence.

## Related Documentation

- [Trust Model](../assurance/14-trust-model.md)
- [Strict Assurance Checklist](../assurance/15-strict-assurance-checklist.md)
- [Reading a Report](reading-report.md)
- [Troubleshooting](troubleshooting.md)
- [Runtime Provenance Guide](../security/runtime-provenance-guide.md)
