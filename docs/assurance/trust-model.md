# Trust Model

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Define the public meaning of an InvarLock strict assurance pass. |
| **Audience** | Release reviewers, CI owners, operators accepting report evidence. |
| **Contract scope** | Current strict assurance behavior, claim set `invarlock-weight-edit-regression-v1`, report v1. |
| **Source of truth** | `src/invarlock/core/assurance_contract.py`, `src/invarlock/reporting/verify_contract.py`, `docs/assurance/00-assurance-case.md`, `docs/reference/reports.md`. |

## Quick Start

```bash
# Require every input report to claim and satisfy strict assurance.
invarlock verify --assurance strict reports/eval/evaluation.report.json
```

Strict verification also expects the sibling `runtime.manifest.json` for
container-backed evidence. See [Runtime Provenance Guide](../security/runtime-provenance-guide.md)
for the manifest contract.

## What A Strict Pass Means

A strict pass means one configured edited checkpoint comparison did not
violate the InvarLock weight-edit regression contract for the selected
baseline, subject, dataset windows, tier, profile, and runtime policy.

The current strict assurance contract requires:

- `assurance.mode = strict`
- `assurance.claim_set = invarlock-weight-edit-regression-v1`
- CI or release profile
- balanced or conservative tier
- canonical guard chain: `invariants -> spectral -> rmt -> variance -> invariants`
- complete guard evidence for every chain stage
- no synthesized, repaired, fallback, degraded, or monitor-only evidence
- strict paired-window counts and zero overlap
- primary metric log-space/display-space CI identity
- tokenizer/provider parity
- verified runtime provenance
- no unsupported guard/model status with `assurance_blocking = true`

## What A Strict Pass Does Not Mean

A strict pass does not mean:

- the model is safe, aligned, or free of content harms
- the baseline model is trustworthy
- downstream tasks outside the configured evaluation windows are preserved
- prompt injection, jailbreak, toxicity, bias, or deployment security risks
  are addressed
- host, GPU, dependency, or network isolation is complete
- unsupported model families have been validated by implication

## Report Statuses

Strict reports include a top-level `assurance` section. Generated reports record
the intended strict claim and leave runtime provenance verification pending
until `invarlock verify` checks the sibling `runtime.manifest.json`. Reviewers
should treat a strict pass as the combination of report-local strict shape and a
verified runtime-provenance result, not as a successful command exit alone.

| Report field | Required strict value |
| --- | --- |
| `mode` | `strict` |
| `verdict` | `pass` |
| `claim_set` | `invarlock-weight-edit-regression-v1` |
| `canonical_guard_chain_enforced` | `true` |
| `fallback_fields_used` | `false` |
| `runtime_provenance_verified` | `false` in generated reports; verifier confirms separately |
| `runtime_provenance_verification_status` | `pending` in generated reports |
| `blocking_reasons` | empty list |

The verifier JSON result must then include
`verification.runtime_provenance.status = verified`.

### Example (report fragment)

```json
{
  "assurance": {
    "mode": "strict",
    "verdict": "pass",
    "claim_set": "invarlock-weight-edit-regression-v1",
    "canonical_guard_chain_enforced": true,
    "fallback_fields_used": false,
    "runtime_provenance_verified": false,
    "runtime_provenance_verification_status": "pending",
    "blocking_reasons": []
  }
}
```

### Example (verifier fragment)

```json
{
  "results": [
    {
      "verification": {
        "runtime_provenance": {
          "status": "verified",
          "verified": true,
          "skipped": false,
          "issues": []
        }
      }
    }
  ]
}
```

## Development Reports

Development and exploratory reports may still be useful for debugging. They
are not assurance-grade unless the report claims and verifies strict
assurance. Common non-strict shapes are catalogued in
[Failure Examples](../user-guide/failure-examples.md).

## Related Documentation

- [Assurance Case Overview](00-assurance-case.md)
- [Strict Assurance Checklist](strict-assurance-checklist.md) — Reviewer checklist
- [Reports Reference](../reference/reports.md) — Full v1 schema
- [Runtime Provenance Guide](../security/runtime-provenance-guide.md) — Manifest requirements
- [Failure Examples](../user-guide/failure-examples.md) — Common non-pass shapes
- [One Run Lifecycle](../reference/one-run-lifecycle.md) — Where strict mode is enforced
- [Alternatives Comparison](../reference/alternatives-comparison.md) — Scope boundaries against adjacent tools
