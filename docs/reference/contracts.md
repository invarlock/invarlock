# Public Contracts

## Overview

This page documents the stable public contracts that InvarLock exposes for
reports, verification, proof packs, calibration artifacts, and policy packs.
These contracts are intended to be consumed as-is by automation, review, and
auditing workflows.

The public contract surface covers:

- `evaluation.report.json` semantics and report schema validation
- `invarlock verify` JSON and exit semantics, including runtime-manifest
  attestation for attested outputs via `runtime.manifest.json`
- proof-pack manifest format and strict verification rules
- plugin ABI compatibility rules
- adapter capability metadata
- runtime tiers/profiles and calibration artifact semantics
- policy digests, policy provenance, and policy-pack verification

## Machine-readable contract files

| Contract | Path | Purpose |
| --- | --- | --- |
| Support matrix | `contracts/support_matrix.json` | Normalized support tiers and public evidence references |
| Model family catalog | `contracts/model_family_catalog.json` | Broader inventory for declared support, code-level coverage, usage-only checkpoints, and recommended additions |
| Adapter capabilities | `contracts/adapter_capabilities.json` | Snapshot/restore, guard coverage, runtime limits, extras |
| Plugin compatibility | `contracts/plugin_compatibility.json` | Core ABI policy and failure mode |
| Runtime manifest | `contracts/runtime_manifest.schema.json` | Runtime attestation schema for `runtime.manifest.json` sidecars |
| Proof-pack manifest | `contracts/proof_pack_manifest.schema.json` | Portable pack manifest schema for `verify_pack.sh`, including builder/subject/material attestation fields |
| Policy pack | `contracts/policy_pack.schema.json` | Build/verify contract for Git-native policy packs |
| Validation keys | `contracts/validation_keys.json` | Allow-list for report validation flags |
| Console labels | `contracts/console_labels.json` | Stable report console labels |
| Metric kinds | `contracts/metric_kinds.json` | Stable metric kind catalog for report surfaces |

These JSON files are shipped in installed wheels under
`invarlock/_data/contracts/*.json`. The logical public contract names remain
`contracts/<name>.json`, and `invarlock.public_contracts` resolves them from the
repo checkout when present or from packaged wheel data otherwise.

## CLI surfaces

The CLI exposes these contracts directly:

- `invarlock verify --json`
- `invarlock plugins adapters --json`
- `invarlock doctor --json`
- `invarlock proof-pack verify --json`
- `invarlock policy build`
- `invarlock policy verify`
- `scripts/proof_packs/verify_pack.sh --strict`

The first six surfaces are available from installed packages. The repo shell
verifier remains available for proof-pack workflow maintainers, but pure wheel
installs can now verify packs with `invarlock proof-pack verify`.

For support-related automation, `plugins adapters --json` and `doctor --json`
now expose both the strict `support_matrix` contract and the broader
`model_family_catalog` contract.

## Policy packs

Policy packs are Git-native artifacts that bind:

- `resolved_policy`
- ordered `overrides`
- a deterministic `policy_digest`
- compatibility metadata
- optional approval metadata

Build and verify them with:

```bash
invarlock policy build \
  --resolved-policy resolved_policy.json \
  --overrides overrides.json \
  --compatibility compatibility.json \
  --out policy-pack.json

invarlock policy verify policy-pack.json --json
```
