# Runtime Provenance Guide

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Explain the runtime-provenance evidence required for strict assurance. |
| **Audience** | Release approvers, security auditors, operators packaging evidence bundles. |
| **Contract scope** | Current strict assurance behavior, runtime manifest schema v1. |
| **Required artifact** | `runtime.manifest.json` adjacent to every container-backed `evaluation.report.json`. |
| **Source of truth** | `src/invarlock/runtime_verify.py`, `src/invarlock/runtime_provenance.py`, `contracts/runtime_manifest.schema.json`, `docs/security/threat-model.md`. |

Runtime evidence has two distinct trust levels:

- **Manifest bound**: the report hash matches a schema-valid sibling manifest.
  This detects report/manifest drift, but every runtime field is still a
  report assertion.
- **Image-digest pinned**: the declared image digest also matches an expected
  digest supplied independently by the verifier caller or CI policy. This is the
  minimum runtime trust level accepted by strict report assurance.

Neither level proves that a container actually executed. Independent execution
attestation remains an external control. In particular, a compromised evaluation environment
can fabricate both report and manifest, bind their hashes correctly, and claim
the caller's expected image digest without having executed that image.

## Quick Start

```bash
# Obtain this value from a reviewed release policy, deployment configuration,
# or other source independent of the submitted runtime.manifest.json.
TRUSTED_RUNTIME_IMAGE_DIGEST='sha256:REPLACE_WITH_REVIEWED_64_HEX_DIGEST'
BASELINE_RUN_REPORT='/path/to/retained/baseline/report.json'
ACCEPTANCE_POLICY_PACK='/path/to/acceptance/policy-pack.json'

# Validate the report/manifest binding and the independent image pin together.
invarlock verify --profile release --assurance strict \
  --baseline "$BASELINE_RUN_REPORT" \
  --policy-pack "$ACCEPTANCE_POLICY_PACK" \
  --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST" \
  reports/eval/evaluation.report.json

# Validate the runtime manifest directly against its report.
invarlock advanced runtime-verify \
  --report reports/eval/evaluation.report.json \
  --manifest reports/eval/runtime.manifest.json \
  --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST"
```

The default `invarlock evaluate --execution-mode container` flow emits
`runtime.manifest.json` next to `evaluation.report.json` automatically. Host
execution forfeits strict assurance and should be verified explicitly with
`verify --runtime-provenance host --assurance off`.

## Strict Runtime Requirements

- Container execution is required for strict assurance.
- The verifier caller must supply `--expected-runtime-image-digest` from an
  independent trust source. Reading the expected value from the submitted
  manifest is circular and does not satisfy strict assurance.
- The verifier caller must supply the complete retained noop baseline run report
  with `--baseline`; a metric fragment cannot establish raw metric or
  model/dataset/tokenizer/provider parity.
- The verifier caller must supply an authorized policy pack with `--policy-pack`;
  strict verification does not accept thresholds selected by the report.
- `runtime.manifest.json` must be present with the report artifacts.
- Host execution and unverified provenance force the report out of strict
  assurance.

## What The Manifest Records

The runtime manifest records the declared execution mode, runtime
tool, image reference/digest, command context, and policy allowances. The
verifier checks schema and report-hash binding. It does not query a container
engine, registry, transparency log, or remote attestation service.

Matching the external digest therefore answers a limited identity question:
"does this manifest claim the image the caller expected?" It does not answer
"did this image execute this evaluation?" A separately trusted execution
record or rerun is necessary for that stronger statement.

Kernel-level isolation, cloud tenancy, GPU firmware integrity, and baseline
model trust are outside the manifest boundary. See [Threat Model](threat-model.md)
for the wider set of assumptions and out-of-scope concerns.

## Recommended Review

Before accepting release evidence:

- [ ] Obtain the expected runtime digest independently of the evidence bundle.
- [ ] Obtain the policy pack from the independently supplied policy channel.
- [ ] Verify with `--assurance strict --baseline ... --policy-pack ... --expected-runtime-image-digest ...`.
- [ ] Confirm JSON output reports `status=expected_image_digest_matched` and
  `expected_digest_matched=true`.
- [ ] Confirm host execution is absent from strict reports.
- [ ] Confirm network, remote-code, and third-party plugin allowances match
  the release policy.
- [ ] Keep the manifest, report, logs, and wheel/sdist hashes together in the
  release bundle.
- [ ] Treat evidence-pack signer pinning and runtime-image pinning as separate
  trust decisions; one does not substitute for the other.

The JSON field `expected_digest_matched` means that the manifest matched an
expected digest supplied by the caller. It does not establish where the caller
obtained that value and is not a remote-attestation verdict.

## Related Documentation

- [Threat Model](threat-model.md)
- [Security Architecture](architecture.md)
- [Security Best Practices](best-practices.md)
- [Trust Model](../assurance/14-trust-model.md)
- [Strict Assurance Checklist](../assurance/15-strict-assurance-checklist.md)
- [Reports Reference](../reference/reports.md)
- [CLI Reference](../reference/cli.md) — `verify` and `advanced runtime-verify`
- [Public Contracts](../reference/contracts.md) — manifest schema location
