# Runtime Provenance Guide

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Explain the runtime-provenance evidence required for strict assurance. |
| **Audience** | Release reviewers, security reviewers, operators packaging evidence bundles. |
| **Contract scope** | Current strict assurance behavior, runtime manifest schema v1. |
| **Required artifact** | `runtime.manifest.json` adjacent to every container-backed `evaluation.report.json`. |
| **Source of truth** | `src/invarlock/core/runtime_manifest_verify.py`, `src/invarlock/runtime_provenance.py`, `contracts/runtime_manifest.schema.json`, `docs/security/threat-model.md`. |

Strict assurance requires runtime provenance. The verifier must be able to
connect an evaluation report to the runtime environment that produced it.

## Quick Start

```bash
# Validate runtime provenance and assurance contract together.
invarlock verify --assurance strict reports/eval/evaluation.report.json

# Validate the runtime manifest directly against its report.
invarlock advanced runtime-verify \
  --report reports/eval/evaluation.report.json \
  --manifest reports/eval/runtime.manifest.json
```

The default `invarlock evaluate --execution-mode container` flow emits
`runtime.manifest.json` next to `evaluation.report.json` automatically. Host
execution forfeits strict assurance and should be verified explicitly with
`verify --runtime-provenance host --assurance off`.

## Strict Runtime Requirements

- Container execution is required for strict assurance.
- Runtime image provenance must be digest-pinned unless the report is
  explicitly non-assurance.
- `runtime.manifest.json` must be present with the report artifacts.
- Host execution and unverified provenance force the report out of strict
  assurance.

## What The Manifest Proves

The runtime manifest records the execution mode, runtime tool, image
reference, image digest or local-image allowance, command context, and policy
allowances used for the evaluation.

It does not prove kernel-level isolation, cloud tenancy, GPU firmware
integrity, or that the baseline model itself is trustworthy. See
[Threat Model](threat-model.md) for the wider set of assumptions and
out-of-scope concerns.

## Recommended Review

Before accepting release evidence:

- [ ] Verify the report with `invarlock verify --assurance strict`.
- [ ] Confirm the runtime image reference is pinned to a digest.
- [ ] Confirm host execution is absent from strict reports.
- [ ] Confirm network, remote-code, and third-party plugin allowances match
  the release policy.
- [ ] Keep the manifest, report, logs, and wheel/sdist hashes together in the
  release bundle.

## Related Documentation

- [Threat Model](threat-model.md)
- [Security Architecture](architecture.md)
- [Security Best Practices](best-practices.md)
- [Trust Model](../assurance/14-trust-model.md)
- [Strict Assurance Checklist](../assurance/15-strict-assurance-checklist.md)
- [Reports Reference](../reference/reports.md)
- [CLI Reference](../reference/cli.md) — `verify` and `advanced runtime-verify`
- [Public Contracts](../reference/contracts.md) — manifest schema location
