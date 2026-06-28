# Trust Model

> **Plain language:** A strict pass means one configured weight-edit regression
> comparison met the declared report and provenance contract. Its scope is that
> configured comparison and the associated report/provenance evidence.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Define the public meaning of an InvarLock strict assurance pass. |
| **Audience** | Release approvers, CI owners, operators accepting report evidence. |
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

Evidence-pack verification is separate from report verification. A signed
evidence pack validates manifest integrity; signer authenticity requires pinning
with `--expected-fingerprint` or a local trust store.

Release exports such as `invarlock report export --format release-review-md`
are presentation wrappers around the report and verifier result. They do not
create a new assurance mode; acceptance still depends on `invarlock verify` and
the runtime/evidence-pack checks described here.

## What A Strict Pass Means

A strict pass means one configured edited checkpoint comparison did not
violate the InvarLock weight-edit regression contract for the selected
baseline, subject, dataset windows, tier, profile, and runtime policy.
The result is scoped to that configured comparison and its report/provenance
evidence.

The current strict assurance contract requires:

- `assurance.mode = strict`
- `assurance.claim_set = invarlock-weight-edit-regression-v1`
- CI or release profile
- balanced or conservative tier
- canonical guard chain: `invariants -> spectral -> rmt -> variance -> invariants`
- complete guard evidence for every guard name in the canonical chain; the
  single `invariants` evidence block covers both pre/post invariant stages in
  the current report contract
- zero synthesized, repaired, fallback, degraded, or monitor-only evidence
- strict paired-window counts and zero overlap
- primary metric log-space/display-space CI identity
- tokenizer/provider parity
- verified runtime provenance
- zero unsupported guard/model statuses with `assurance_blocking = true`

## Strict Pass Scope

A strict pass covers the configured evidence surface:

- the selected baseline, subject, dataset windows, tier, profile, and runtime
  policy
- the report-local strict-assurance shape and guard evidence
- verified runtime provenance for the generated report
- signer authenticity when the signer fingerprint is pinned or matched through a
  trusted local store
- the published support tier for the model family or adapter lane under review

Adjacent review domains include content safety, alignment, prompt-security,
deployment security, host isolation, dependency isolation, and model families
outside the published support basis.

## Report Statuses

Strict reports include a top-level `assurance` section. Generated reports record
the intended strict claim and leave runtime provenance verification pending
until `invarlock verify` checks the sibling `runtime.manifest.json`. Evidence readers
should require the combination of report-local strict shape and a verified
runtime-provenance result.

| Report field | Required strict value |
| --- | --- |
| `mode` | `strict` |
| `verdict` | `pending_verifier` in generated reports; verifier success is required for acceptance |
| `report_local_verdict` | `pass` |
| `verified_assurance_verdict` | `pending` in generated reports |
| `claim_set` | `invarlock-weight-edit-regression-v1` |
| `canonical_guard_chain_enforced` | `true` |
| `fallback_fields_used` | `false` |
| `runtime_provenance_verified` | `false` in generated reports; verifier confirms separately |
| `runtime_provenance_verification_status` | `pending` in generated reports |
| `blocking_reasons` | empty list |

The verifier JSON result must then include
`results[*].verification.runtime_provenance.status = "verified"`.

## Evidence Pack Signer Authenticity

Package-native evidence packs can include `manifest.signature.json`, an Ed25519
signature over `manifest.json`. The verifier always derives and reports the
signing-key fingerprint when a signature is present. That check provides
tamper evidence for the manifest and checksum chain.

Authenticity is stronger: evidence readers must decide which signing keys they accept.
For distributable evidence, require one of:

- `invarlock advanced evidence-pack verify <dir> --expected-fingerprint sha256:<64-hex-chars>`
- `invarlock advanced evidence-pack verify <dir> --trust-store <json>`
- `~/.config/invarlock/trusted-signers.json` containing accepted fingerprints

An unpinned signature should be treated as trust-on-first-use evidence for
integrity review. Publisher authenticity requires an accepted fingerprint from
`--expected-fingerprint`, `--trust-store`, or the local trusted-signers file.

### Example (report fragment)

```json
{
  "assurance": {
    "mode": "strict",
    "verdict": "pending_verifier",
    "report_local_verdict": "pass",
    "verified_assurance_verdict": "pending",
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

Development and exploratory reports may still be useful for debugging. Reports
become assurance-grade when they claim and verify strict assurance. Common
non-strict shapes are catalogued in [Failure Examples](../user-guide/failure-examples.md).

## Related Documentation

- [Assurance Case Overview](00-assurance-case.md)
- [Strict Assurance Checklist](15-strict-assurance-checklist.md) — Acceptance checklist
- [Reports Reference](../reference/reports.md) — Full v1 schema
- [Runtime Provenance Guide](../security/runtime-provenance-guide.md) — Manifest requirements
- [Failure Examples](../user-guide/failure-examples.md) — Common non-pass shapes
- [One Run Lifecycle](../reference/one-run-lifecycle.md) — Where strict mode is enforced
- [Alternatives Comparison](../reference/alternatives-comparison.md) — Scope boundaries against adjacent tools
