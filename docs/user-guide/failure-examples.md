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
support status.

## Guard Diagnostics

A paired primary metric and checkpoint guards answer different review questions. A report can preserve its task metric while a spectral, RMT, invariant, or variance policy requests additional review.

Run the verifier against a current evaluation bundle:

```bash
invarlock verify --profile release --assurance strict \
  --baseline PATH/TO/baseline.report.json \
  --policy-pack PATH/TO/policy-pack.json \
  --expected-runtime-image-digest sha256:REVIEWED_DIGEST \
  PATH/TO/evaluation.report.json
```

Use the emitted diagnostic code and guard payload to locate the affected module family, compare it with the baseline, and decide whether to adjust the edited checkpoint or the independently maintained policy.

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
