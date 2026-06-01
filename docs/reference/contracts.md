# Public Contracts

## Overview

This page documents the stable public contracts that InvarLock exposes for
reports, verification, evidence packs, calibration artifacts, and policy packs.
These contracts are intended to be consumed as-is by automation, review, and
auditing workflows.

InvarLock is pre-1.0 as a package, but the core evidence-artifact surfaces are
versioned and intended to be stable within their declared contract versions.

The public contract surface covers:

- `evaluation.report.json` semantics and report schema validation
- `invarlock verify` JSON and exit semantics, including runtime-manifest
  provenance for container-backed outputs via `runtime.manifest.json`
- evidence-pack manifest format and strict verification rules
- plugin ABI compatibility rules
- adapter capability metadata
- runtime tiers/profiles and calibration artifact semantics
- policy digests, policy provenance, and policy-pack verification

## Versioned sub-contracts

| Sub-contract | Version field | Current value | Canonical source |
| --- | --- | --- | --- |
| Report schema | `evaluation.report.json.schema_version` | `v1` | `invarlock.reporting.report_schema.REPORT_JSON_SCHEMA` |
| Evidence-pack format | `manifest.json.format` | `evidence-pack-v1` | `contracts/evidence_pack_manifest.schema.json` |
| Verifier output | `invarlock verify --json.format_version` | `verify-v1` | `contracts/verify_output.schema.json` |
| Runtime manifest | `runtime.manifest.json.verifier_contract_version` | `runtime-manifest-v1` | `contracts/runtime_manifest.schema.json` |
| CLI stability policy | policy identifier | `cli-stability-v1` | `docs/reference/cli.md` |
| Adapter/model support tiers | `support_matrix.support_tiers[]` | `published_basis`, `supported_experimental`, `community_experimental` | `contracts/support_matrix.json` |

Compatibility rules:

- Within a `v1` contract, new optional fields may be added and consumers should
  ignore fields they do not understand.
- Removing a required field, renaming a field, changing a field type, or
  changing pass/fail semantics requires a new version value.
- Report, evidence-pack, runtime-manifest, and verifier validators fail closed
  on mismatched explicit version fields.
- Optional report blocks can graduate into the required core only with a report
  schema version bump.

## Machine-readable contract files

| Contract | Path | Purpose |
| --- | --- | --- |
| Support matrix | `contracts/support_matrix.json` | Normalized support tiers and public evidence references |
| Model family catalog | `contracts/model_family_catalog.json` | Broader inventory for declared support, code-level coverage, usage-only checkpoints, and recommended additions |
| Adapter capabilities | `contracts/adapter_capabilities.json` | Snapshot/restore, guard coverage, runtime limits, extras |
| Plugin compatibility | `contracts/plugin_compatibility.json` | Core ABI policy and failure mode |
| Runtime manifest | `contracts/runtime_manifest.schema.json` | Runtime provenance schema for `runtime.manifest.json` sidecars |
| Verify output | `contracts/verify_output.schema.json` | JSON output schema for `invarlock verify --json` |
| Evidence-pack manifest | `contracts/evidence_pack_manifest.schema.json` | Portable pack manifest schema for `verify_pack.sh`, including builder/subject/material signed provenance fields |
| Policy pack | `contracts/policy_pack.schema.json` | Build/verify contract for Git-native policy packs |
| Validation keys | `contracts/validation_keys.json` | Allow-list for report validation flags |
| Console labels | `contracts/console_labels.json` | Stable report console labels |
| Metric kinds | `contracts/metric_kinds.json` | Stable metric kind catalog for report surfaces |

These JSON files are included in installed wheels under
`invarlock/_data/contracts/*.json`. The logical public contract names remain
`contracts/<name>.json`, and `invarlock.public_contracts` resolves them from the
repo checkout when present or from packaged wheel data otherwise.

The public contract catalog exposes the list-shaped files as first-class
entries too: `validation_keys`, `console_labels`, and `metric_kinds` are
surfaced by `invarlock.public_contracts.contract_catalog()` and embedded in the
JSON payloads emitted by `invarlock doctor --json` and `invarlock advanced
plugins ... --json`.

## CLI surfaces

The CLI exposes these contracts directly:

- `invarlock verify --json`
- `invarlock advanced runtime-verify --json`
- `invarlock advanced plugins list --json`
- `invarlock advanced plugins adapters --json`
- `invarlock doctor --json`
- `invarlock advanced evidence-pack verify --json`
- `invarlock advanced policy build`
- `invarlock advanced policy verify --json`
- `scripts/evidence_packs/verify_pack.sh --pack <dir> --strict --report-assurance strict`

The first eight surfaces are available from installed packages. The low-level
`invarlock advanced runtime-verify` command is the package-native
runtime-manifest verifier used for direct report/manifest checks. The repo
shell verifier remains available for evidence-pack workflow maintainers, and
pure wheel installs can verify packs with `invarlock advanced evidence-pack
verify`.

Third-party plugins are fail-closed on ABI declaration: adapters, edits, and
guards must declare `INVARLOCK_CORE_ABI`, and the value must match the exact
core ABI published in `contracts/plugin_compatibility.json`.

For support-related automation, `plugins adapters --json` and `doctor --json`
expose both the strict `support_matrix` contract and the broader
`model_family_catalog` contract, plus the `validation_keys`, `console_labels`,
and `metric_kinds` entries from the public contract catalog.

The versioned JSON surfaces are intentionally explicit:

- `invarlock doctor --json` emits `format_version: "doctor-v1"`
- `invarlock verify --json` emits `format_version: "verify-v1"`
- `invarlock advanced runtime-verify --json` emits
  `format_version: "runtime-verify-v1"`
- `invarlock advanced plugins list --json` and
  `invarlock advanced plugins adapters --json` emit
  `format_version: "plugins-v1"`
- `invarlock advanced policy verify --json` emits
  `format_version: "policy-pack-verify-v1"`
- `invarlock advanced evidence-pack verify --json` emits
  `format_version: "evidence-pack-verify-v1"` and nests the bundled report
  verification result under `verify.format_version: "verify-v1"`

The CLI stability policy covers command names, documented options, exit-code
meaning, and the required fields of the listed JSON envelopes. Commands under
`advanced` remain outside the core user loop, but the JSON surfaces listed here
are public automation contracts.

## Adapter support tiers

Adapter availability and public assurance support are separate concepts.
`contracts/adapter_capabilities.json` describes whether an adapter can load,
snapshot, restore, and expose guard-compatible modules. `contracts/support_matrix.json`
describes the public support tier for a model/runtime/adapter lane.

| Tier | Meaning |
| --- | --- |
| `published_basis` | Maintained public evidence floor with report, runtime-manifest, and evidence-pack provenance where available. |
| `supported_experimental` | Repo ships preset, calibration/config, targeted tests, and smoke/evidence paths, but no published-basis fixture set is claimed. |
| `community_experimental` | Adapter/runtime path is usable for community experimentation without a maintained public evidence basis. |

Policy packs that declare `compatibility.support_tiers` must use one of those
three tier values.

## Packaged public contract data

The maintained public contract data ships in two places:

- installed wheels, under `invarlock/_data/contracts/*.json`
- source tags in the repository

Repo tags and installed wheels are the only maintained public contract
carriers.

The support-matrix published-basis evidence paths remain logical
`public_evidence/published_basis/...` references. Installed wheels resolve those
logical paths from packaged files under
`invarlock/_data/public_evidence/published_basis/...`, so installed packages can
render and verify the shipped published-basis `evaluation.report.json` examples
with their sibling `runtime.manifest.json` files, and load the paired
`evidence_pack_recipe.json` data without cloning the repo. The GPT-2
published-basis lane also packages a signed `evidence_pack/` directory for
strict offline evidence-pack verification.

## Policy packs

Policy packs are Git-native artifacts that bind:

- `resolved_policy`
- ordered `overrides`
- a deterministic `policy_digest`
- compatibility metadata
- optional approval metadata

Build and verify them with:

```bash
invarlock advanced policy build \
  --resolved-policy resolved_policy.json \
  --overrides overrides.json \
  --compatibility compatibility.json \
  --out policy-pack.json

invarlock advanced policy verify policy-pack.json --json
```
