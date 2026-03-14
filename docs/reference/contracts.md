# Public Contracts

## Overview

This page documents the stable public contracts that InvarLock exposes for
reports, verification, proof packs, calibration artifacts, and policy packs.
These contracts are intended to be consumed as-is by automation, review, and
auditing workflows.

The public contract surface covers:

- `evaluation.report.json` semantics and report schema validation
- `invarlock verify` JSON and exit semantics
- proof-pack manifest format and strict verification rules
- plugin ABI compatibility rules
- adapter capability metadata
- runtime tiers/profiles and calibration artifact semantics
- policy digests, policy provenance, and policy-pack verification

## Machine-readable contract files

| Contract | Path | Purpose |
| --- | --- | --- |
| Support matrix | `contracts/support_matrix.json` | Normalized support tiers and public evidence references |
| Adapter capabilities | `contracts/adapter_capabilities.json` | Snapshot/restore, guard coverage, runtime limits, extras |
| Plugin compatibility | `contracts/plugin_compatibility.json` | Core ABI policy and failure mode |
| Proof-pack manifest | `contracts/proof_pack_manifest.schema.json` | Portable pack manifest schema for `verify_pack.sh` |
| Policy pack | `contracts/policy_pack.schema.json` | Build/verify contract for Git-native policy packs |
| Validation keys | `contracts/validation_keys.json` | Allow-list for report validation flags |
| Console labels | `contracts/console_labels.json` | Stable report console labels |

## CLI surfaces

The CLI exposes these contracts directly:

- `invarlock verify --json`
- `invarlock plugins adapters --json`
- `invarlock doctor --json`
- `invarlock policy build`
- `invarlock policy verify`
- `scripts/proof_packs/verify_pack.sh --strict`

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
