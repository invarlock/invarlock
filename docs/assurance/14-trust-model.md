# Trust Model

> **Plain language:** A strict pass means one configured weight-edit regression
> comparison met the declared report and provenance contract. Its scope is that
> configured comparison and the associated report/provenance evidence.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Define the public meaning of an InvarLock strict assurance pass. |
| **Audience** | Release approvers, CI owners, operators accepting report evidence. |
| **Contract scope** | Current strict assurance behavior, claim set `invarlock-weight-edit-regression-v2`, report v1. |
| **Source of truth** | `src/invarlock/core/assurance_contract.py`, `src/invarlock/reporting/verify_contract.py`, `docs/assurance/00-assurance-case.md`, `docs/reference/reports.md`. |

## Quick Start

```bash
# The expected digest must come from reviewed policy, not the submitted manifest.
TRUSTED_RUNTIME_IMAGE_DIGEST='sha256:REPLACE_WITH_REVIEWED_64_HEX_DIGEST'
BASELINE_RUN_REPORT='/path/to/retained/baseline/report.json'
ACCEPTANCE_POLICY_PACK='/path/to/acceptance/policy-pack.json'
invarlock verify --profile release --assurance strict \
  --baseline "$BASELINE_RUN_REPORT" \
  --policy-pack "$ACCEPTANCE_POLICY_PACK" \
  --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST" \
  reports/eval/evaluation.report.json
```

Strict verification also expects a policy pack supplied independently by the
verifier caller and the sibling `runtime.manifest.json` for container-backed evidence.
The policy pack must exactly match the policy resolved in the report; the
pack's digest also binds its compatibility metadata. Strict verification
requires `compatibility.dataset_identity` and reconciles those caller-selected
provider, dataset, configuration, revision, and split coordinates with the
report. The submitted report cannot authorize itself. See
[Runtime Provenance Guide](../security/runtime-provenance-guide.md) for the
manifest contract.

Evidence-pack verification is separate from report verification. A signed
evidence pack validates manifest integrity; signer authenticity requires pinning
with `--expected-fingerprint` or a local trust store. Signer pinning does not pin
the runtime image: strict nested report verification additionally requires
`--expected-runtime-image-digest`.

Release exports such as `invarlock report export --format release-review-md`
are presentation wrappers around the report and verifier result. They do not
create a new assurance mode; acceptance still depends on `invarlock verify` and
the runtime/evidence-pack checks described here.

## What A Strict Pass Means

A strict pass means one submitted, configured edited-checkpoint comparison is
internally consistent with the InvarLock weight-edit regression contract for the selected
baseline, subject, dataset windows, tier, profile, and runtime policy.
The result is scoped to that configured comparison and its report/provenance
evidence.

The current strict assurance contract requires:

- `assurance.mode = strict`
- `assurance.claim_set = invarlock-weight-edit-regression-v2`
- an exact `resolved_policy.guard_authority` mapping, mirrored in
  `assurance.guard_authority`, for spectral, RMT, and variance findings
- CI or release profile
- balanced or conservative tier
- canonical guard chain: `invariants -> spectral -> rmt -> variance -> invariants`
- complete guard evidence for every guard name in the canonical chain; the
  single `invariants` evidence block covers both pre/post invariant stages in
  the current report contract
- zero synthesized, repaired, fallback, degraded, or monitor-only evidence
- strict paired-window counts, disjoint preview/final IDs, and zero configured
  sliding-window overlap
- primary metric log-space/display-space CI identity
- a complete independently supplied noop baseline run report, with raw metric
  replay and tokenizer/provider/model/dataset parity
- a report-bound runtime manifest and an independently supplied expected image
  digest matching the manifest declaration
- no unsupported guard status accepted as passing evidence
- measured, passing guard-metric-impact evidence; an explicit skip does not satisfy
  current strict assurance
- invariant, primary-metric, drift, and guard-metric-impact findings always
  block; spectral, RMT, and variance findings block when their policy authority
  is `enforce`
- `observe` authority changes only the acceptance effect of a complete guard
  finding. It does not permit skipped, unsupported, degraded,
  monitor-only, incomplete, or non-replayable evidence.

Frozen v1 reports remain verifiable with implicit all-`enforce` authority. A
v1 report or policy pack cannot declare `guard_authority`.

All shipped v2 tiers also default spectral, RMT, and variance authority to
`enforce`. An `observe` value is a deliberate, independently authorized policy
override for a complete measured finding; it is not inferred from guard
maturity or report contents.

## Strict Pass Scope

A strict pass covers the configured evidence surface:

- the selected baseline, subject, dataset windows, tier, profile, and runtime
  policy
- the report-local strict-assurance shape and guard evidence
- report/manifest binding plus an independently supplied expected runtime-image digest
  that matches the manifest's **claimed** image identity

Evidence-pack signer authentication and support-matrix classification are
separate review results. Report verification does not authenticate an
evidence-pack signer, and it does not establish that a model or adapter lane is
listed in the maintained catalog.

Adjacent review domains include content safety, alignment, prompt-security,
deployment security, host isolation, dependency isolation, and model families
outside the maintained catalog.

The external image pin does **not** cryptographically attest actual container
execution. A compromised evaluation environment can fabricate an internally consistent report
and manifest that name the caller's expected digest without running that
image. Execution truth requires a separately trusted rerun, transparency-backed
build/deployment provenance, remote attestation, or an equivalent external
control. Signer authentication identifies who signed the bundle; it does not
make the submitted bundle's factual assertions true.

## Report Statuses

Strict reports include a top-level `assurance` section. Generated reports record
the intended strict claim and leave runtime provenance verification pending
until `invarlock verify` checks the sibling `runtime.manifest.json`. Readers
should require the combination of report-local strict shape, manifest binding,
and an independently pinned image digest.

| Report field | Required strict value |
| --- | --- |
| `mode` | `strict` |
| `verdict` | `pending_verifier` in generated reports; verifier success is required for acceptance |
| `report_local_verdict` | `pass` |
| `verified_assurance_verdict` | `pending` in generated reports |
| `claim_set` | `invarlock-weight-edit-regression-v2` |
| `guard_authority` | exact mirror of `resolved_policy.guard_authority` |
| `canonical_guard_chain_enforced` | `true` |
| `fallback_fields_used` | `false` |
| `runtime_provenance_verified` | `false` in generated reports; verifier confirms separately |
| `runtime_provenance_verification_status` | `pending` in generated reports |
| `blocking_reasons` | empty list |

The verifier JSON result must then include
`results[*].verification.runtime_provenance.status = "expected_image_digest_matched"`,
`binding_verified = true`, and `expected_digest_matched = true`. A valid manifest
without an external pin is reported as `manifest_bound` and does not satisfy
strict assurance.

`expected_digest_matched` states only that a value supplied by the verifier
caller matched the manifest. It does not establish where that value came from
or mean the verifier independently observed or attested execution. Likewise, the report-side
`runtime_provenance_verified` field is status plumbing, not an execution claim.

## Evidence Pack Signer Authenticity

Package-native evidence packs can include `manifest.signature.json`, an Ed25519
signature over `manifest.json`. The verifier always derives and reports the
signing-key fingerprint when a signature is present. That check provides
tamper evidence for the manifest and checksum chain.

Authenticity is stronger: readers must decide which signing keys they accept.
Set `TRUSTED_SIGNER_FINGERPRINT` from an independently controlled publisher-key
record, not by extracting the fingerprint from the submitted pack. For
distributable evidence, require one of:

- `invarlock advanced evidence-pack verify <dir> --expected-fingerprint "$TRUSTED_SIGNER_FINGERPRINT"`
- `invarlock advanced evidence-pack verify <dir> --trust-store <json>`
- `~/.config/invarlock/trusted-signers.json` containing accepted fingerprints

An unpinned signature should be treated as trust-on-first-use evidence for
integrity review. Publisher authenticity requires an accepted fingerprint from
`--expected-fingerprint`, `--trust-store`, or the local trusted-signers file.
This authenticates the evidence publisher, not the runtime image. Verifier callers
must establish both anchors separately.

### Example (report fragment)

This fragment shows an explicit policy override from the shipped all-`enforce`
default. The independently supplied policy pack must authorize the same exact
mapping.

```json
{
  "assurance": {
    "mode": "strict",
    "verdict": "pending_verifier",
    "report_local_verdict": "pass",
    "verified_assurance_verdict": "pending",
    "claim_set": "invarlock-weight-edit-regression-v2",
    "guard_authority": {
      "spectral": "enforce",
      "rmt": "observe",
      "variance": "observe"
    },
    "canonical_guard_chain_enforced": true,
    "fallback_fields_used": false,
    "runtime_provenance_verified": false,
    "runtime_provenance_verification_status": "pending",
    "blocking_reasons": []
  },
  "resolved_policy": {
    "guard_authority": {
      "spectral": "enforce",
      "rmt": "observe",
      "variance": "observe"
    }
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
          "status": "expected_image_digest_matched",
          "verified": true,
          "binding_verified": true,
          "expected_digest_matched": true,
          "trust_status": "expected_image_digest_matched",
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
become eligible for strict acceptance only after the report, baseline, policy,
and runtime-provenance checks succeed; external trust limits above still apply.
Common non-strict shapes are catalogued in
[Failure Examples](../user-guide/failure-examples.md).

## Related Documentation

- [Assurance Case Overview](00-assurance-case.md)
- [Strict Assurance Checklist](15-strict-assurance-checklist.md) — Acceptance checklist
- [Reports Reference](../reference/reports.md) — Full v1 schema
- [Runtime Provenance Guide](../security/runtime-provenance-guide.md) — Manifest requirements
- [Failure Examples](../user-guide/failure-examples.md) — Common non-pass shapes
- [One Run Lifecycle](../reference/one-run-lifecycle.md) — Where strict mode is enforced
- [Alternatives Comparison](../reference/alternatives-comparison.md) — Scope boundaries against adjacent tools
