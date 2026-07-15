# Public Contracts

## Overview

This page documents the stable public contracts that InvarLock exposes for
reports, verification, evidence packs, runtime-provider evidence, calibration
artifacts, and policy packs. These contracts are intended to be consumed as-is
by automation, review, and auditing workflows.

InvarLock is pre-1.0 as a package, but the core evidence-artifact surfaces are
versioned and intended to be stable within their declared contract versions.

The public contract surface covers:

- `evaluation.report.json` semantics and report schema validation
- `invarlock verify` JSON and exit semantics, including runtime-manifest
  provenance for container-backed outputs via `runtime.manifest.json`
- evidence-pack manifest format and strict verification rules
- plugin ABI compatibility rules
- adapter capability metadata
- runtime-provider capabilities, artifact identities, receipts, scoring
  observations, behavioral schedules, and paired-claim receipts
- runtime tiers/profiles and calibration artifact semantics
- policy digests, policy provenance, and policy-pack verification

## Versioned sub-contracts

| Sub-contract | Version field | Current value | Canonical source |
| --- | --- | --- | --- |
| Report schema | `evaluation.report.json.schema_version` | `v1` | `invarlock.reporting.report_schema.REPORT_JSON_SCHEMA` |
| Evidence-pack format | `manifest.json.format` | `evidence-pack-v1` | `contracts/evidence_pack_manifest.schema.json` |
| Verifier output | `invarlock verify --json.format_version` | `verify-v1` | `contracts/verify_output.schema.json` |
| Runtime manifest | `runtime.manifest.json.verifier_contract_version` | `runtime-manifest-v1` | `contracts/runtime_manifest.schema.json` |
| Runtime-provider manifest | `runtime.manifest.json.verifier_contract_version` | `runtime-manifest-v2` | `contracts/runtime_manifest_v2.schema.json` |
| Runtime-provider capabilities | `format_version` | `runtime-provider-capabilities-v1` | `contracts/runtime_provider_capabilities.json` |
| Model artifact identity | `format_version` | `invarlock/model-artifact-identity-v1` | `contracts/model_artifact_identity.schema.json` |
| Runtime-provider receipt | `format_version` | `invarlock/runtime-provider-receipt-v1` | `contracts/runtime_provider_receipt.schema.json` |
| Runtime scoring observation | `format_version` | `invarlock/runtime-scoring-observation-v1` | `contracts/runtime_scoring_observation.schema.json` |
| Runtime behavioral schedule | `format_version` | `invarlock/runtime-behavioral-schedule-v1` | `contracts/runtime_behavioral_schedule.schema.json` |
| Runtime behavioral claim receipt | `format_version` | `invarlock/runtime-behavioral-claim-receipt-v1` | `contracts/runtime_behavioral_claim_receipt.schema.json` |
| CLI stability policy | policy identifier | `cli-stability-v1` | `docs/reference/cli.md` |
| Support matrix | `support_matrix.format_version` | `support-matrix-v2` | `contracts/support_matrix.json` |
| Model-family catalog | `model_family_catalog.format_version` | `model-family-catalog-v2` | `contracts/model_family_catalog.json` |
| Policy pack | `policy_pack.format` | `policy-pack-v2` | `contracts/policy_pack.schema.json` |
| Behavioral policy pack | `policy_pack.format` | `policy-pack-v3` | `contracts/policy_pack.schema.json` |
| Public-evidence index | `catalog_evidence_index.format_version` | `public-evidence-index-v2` | `public_evidence/catalog_evidence_index.json` |
| Lane support tiers | `support_matrix.support_tiers[]` | `maintained_catalog`, `supported_experimental`, `community_experimental` | `contracts/support_matrix.json` |

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
| Model family catalog | `contracts/model_family_catalog.json` | Declared support, code-level coverage, usage-only checkpoints, and recommended additions |
| Model classification | `contracts/model_classification.json` | Lifecycle classification for cataloged, backlog, blocked, smoke-only, usage-only, and out-of-scope model status |
| Adapter capabilities | `contracts/adapter_capabilities.json` | Snapshot/restore, guard coverage, runtime limits, extras |
| Plugin compatibility | `contracts/plugin_compatibility.json` | Core ABI policy and failure mode |
| Runtime manifest v1 | `contracts/runtime_manifest.schema.json` | Runtime provenance schema for ordinary evaluation report sidecars |
| Runtime manifest v2 | `contracts/runtime_manifest_v2.schema.json` | Strict runtime-provider side manifest binding report, config, receipt, observation, artifact identity, schedule, and policy |
| Runtime-provider capabilities | `contracts/runtime_provider_capabilities.json` | Closed provider ABI and supported artifact formats, precision modes, and device kinds |
| Model artifact identity | `contracts/model_artifact_identity.schema.json` | Path-free identities for Hugging Face snapshots, GGUF files, and TensorRT-LLM engine bundles |
| Runtime-provider receipt | `contracts/runtime_provider_receipt.schema.json` | Provider execution receipt binding capabilities, artifact identity, image, device observation, and scoring observation |
| Runtime scoring observation | `contracts/runtime_scoring_observation.schema.json` | Canonical policy-scored behavioral records and aggregate metric |
| Runtime behavioral schedule | `contracts/runtime_behavioral_schedule.schema.json` | Canonical ordered inputs shared by separately produced provider sides |
| Runtime behavioral claim receipt | `contracts/runtime_behavioral_claim_receipt.schema.json` | Positive digest-only receipt from directed baseline/subject replay |
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

The public contract catalog exposes the model lifecycle ledger and list-shaped
files as first-class entries too: `model_classification`, `validation_keys`,
`console_labels`, and `metric_kinds` are surfaced by
`invarlock.public_contracts.contract_catalog()` and embedded in the JSON payloads
emitted by `invarlock doctor --json` and `invarlock advanced plugins ... --json`.

## CLI surfaces

The CLI exposes these contracts directly:

- `invarlock verify --json`
- `invarlock advanced runtime-verify --json`
- `invarlock advanced plugins list --json`
- `invarlock advanced plugins adapters --json`
- `invarlock advanced plugins runtime-providers --json`
- `invarlock doctor --json`
- `invarlock advanced evidence-pack verify --json`
- `invarlock advanced evidence-catalog validate --json`
- `invarlock advanced policy build`
- `invarlock advanced policy verify --json`
- `invarlock advanced runtime-behavior build-schedule --json`
- `invarlock advanced runtime-behavior prepare-binding --json`
- `invarlock advanced runtime-behavior build-policy --json`
- `invarlock advanced runtime-behavior run-side --json`
- `invarlock advanced runtime-behavior verify-pair --json`
- `scripts/evidence_packs/verify_pack.sh --pack <dir> --strict
  --report-assurance strict --policy-pack <acceptance-policy-pack.json>
  --expected-runtime-image-digest
  "$EXPECTED_RUNTIME_IMAGE_DIGEST"`

The listed package commands are available from installed packages. The low-level
`invarlock advanced runtime-verify` command is the package-native
runtime-manifest verifier used for direct report/manifest checks. The repo
shell verifier remains available for evidence-pack workflow maintainers, and
pure wheel installs can verify packs with `invarlock advanced evidence-pack
verify`.
Strict nested report verification requires a signed/checksummed baseline
mapping inside the pack, a matching independently supplied policy pack, and an
expected runtime-image digest from channels independent of the submitted pack.

Third-party plugins are fail-closed on ABI declaration: adapters, edits, and
guards must declare `INVARLOCK_CORE_ABI`, and the value must match the exact
core ABI published in `contracts/plugin_compatibility.json`.

For support-related automation, `plugins adapters --json` and `doctor --json`
expose both the strict `support_matrix` contract and the broader
`model_family_catalog` contract. Lifecycle decisions such as `cataloged`,
`backlog`, `blocked`, `usage_only`, and `out_of_scope` live in
`model_classification`; update that manifest and rerun `make contracts-check`
to refresh support surfaces. The same JSON surfaces also include the `validation_keys`,
`console_labels`, and `metric_kinds` entries from the public contract catalog.

The versioned JSON surfaces are intentionally explicit:

- `invarlock doctor --json` emits `format_version: "doctor-v1"`
- `invarlock verify --json` emits `format_version: "verify-v1"`
  and each loaded result includes an additive unsigned
  `verification.receipt` (format version **invarlock&#46;verify-receipt&#46;v1**) with the exact
  subject/baseline SHA-256 values, provider digests, verifier version, and
  normalized caller inputs. `signed=false` means the receipt is descriptive;
  authenticate the containing evidence pack separately. `report export` checks
  that receipt's subject digest against the exact bytes it exports, but labels
  it `receipt_bound_untrusted` rather than treating it as a verified pass.
- `invarlock advanced runtime-verify --json` emits
  `format_version: "runtime-verify-v1"`
- `invarlock advanced plugins list --json`,
  `invarlock advanced plugins adapters --json`, and
  `invarlock advanced plugins runtime-providers --json` emit
  `format_version: "plugins-v2"`
- `invarlock advanced policy verify --json` emits
  `format_version: "policy-pack-verify-v1"`
- `invarlock advanced evidence-pack verify --json` emits
  `format_version: "evidence-pack-verify-v1"` and nests the bundled report
  verification result under `verify.format_version: "verify-v1"`. Its
  diagnostic `--skip-verify` mode returns nonzero status `8`, `ok=false`,
  `integrity_ok=true`, and `reports_verified=false`; it is not a report
  verification result and is rejected in strict, CI, or release contexts.
- `invarlock advanced evidence-catalog validate --json` emits
  `format_version: "evidence-catalog-validate-v1"`.

Runtime-provider inventory items distinguish an importable connector from an
executed backend: `connector_status` reports connector readiness,
`backend_delivery` reports the Python-extra or OCI-image delivery boundary, and
the metadata-only command always reports `runtime_qualification: not_probed`.
Running a platform qualification does not mutate the inventory; review its
result separately. Plugin maturity and `strict_assurance_allowed` describe
connector ownership and strict-contract eligibility, not successful execution
or qualification of a backend, image, platform, or model artifact. Built-in rows
also name their `invarlock.runtime_providers` entry-point group and key.

The experimental `invarlock advanced runtime-behavior` commands also emit
versioned envelopes: `build-schedule --json` uses
`runtime-behavior-build-schedule-cli-v1`, `prepare-binding --json` uses
`runtime-behavior-prepare-binding-cli-v1`, `build-policy --json` uses
`runtime-behavior-build-policy-cli-v1`, `run-side --json` uses
`runtime-behavior-run-side-cli-v1`, and `verify-pair --json` uses
`runtime-behavior-verify-pair-cli-v1`. Those command names, options, exit-code
meanings, and JSON envelopes remain experimental. Their format versions identify
the emitted shapes but do not make those CLI envelopes stable. The underlying
schedule, binding, policy, side-evidence, observation, provider-receipt, runtime
manifest, and paired-claim artifact schemas remain versioned public contracts.

The general CLI stability policy covers only the surfaces classified as stable
in [CLI Reference](cli.md#stable-vs-experimental-commands). It excludes the
experimental `runtime-behavior` command group and its envelopes.

See [Runtime Providers](runtime-providers.md) for the installed provider
inventory, required runtime images, and the complete side-production and paired
verification journey.

## Lane support tiers and plugin maturity

Adapter availability and public assurance support are separate concepts.
`contracts/adapter_capabilities.json` describes whether an adapter can load,
snapshot, restore, and expose guard-compatible modules. `contracts/support_matrix.json`
describes the public support tier for a model/runtime/adapter lane.

| Tier | Meaning |
| --- | --- |
| `maintained_catalog` | Maintained catalog evidence lane; availability is reported separately by `evidence_status`. |
| `supported_experimental` | Maintained adapter, preset, configuration, test, and smoke path. |
| `community_experimental` | Adapter and runtime path available for community evaluation. |

`policy-pack-v2` and `policy-pack-v3` inputs that declare
`compatibility.support_tiers` must use one of those three current tier values.
Verification still accepts frozen `policy-pack-v1` artifacts that authorize the
historical `published_basis` tier. The general policy builder and defaults emit
`policy-pack-v2` with `maintained_catalog`; the directed runtime-behavioral
builder emits `policy-pack-v3`.

`maintained_catalog` is a stable compatibility identifier for lane eligibility;
it does not mean that evidence already exists. For each lane,
`evidence_status` and `evidence_status_label` state whether current evidence is
available.

The `support_tier` field in `plugins-v2` inventory rows is a separate
plugin-maturity vocabulary. It does not authorize a model lane, report evidence
availability, backend execution, or runtime qualification:

| Plugin maturity | Meaning |
| --- | --- |
| `core_supported` | Shipped first-party implementation on the core plugin surface. |
| `first_party_experimental` | Shipped and maintained first-party integration whose runtime or workflow remains experimental. |
| `optional_backend_loader` | Shipped connector for an optional external backend dependency. |
| `validation_simulation` | Shipped simulation used to validate contracts and decision paths rather than claim a production transformation. |
| `internal_baseline_edit` | Shipped no-change edit used for baseline and compatibility workflows. |
| `demo_only` | Illustrative plugin outside strict-assurance use. |
| `third_party` | Externally supplied plugin with no first-party support or qualification claim. |

`strict_assurance_allowed` is orthogonal to plugin maturity. It records whether
the plugin class is eligible to participate in a strict contract; it is not a
statement that a particular model, backend, image, platform, or run was
qualified.

## Packaged public contract data

The maintained public contract data ships in two places:

- installed wheels, under `invarlock/_data/contracts/*.json`
- source tags in the repository

Repo tags and installed wheels are the maintained public contract carriers.
Full evidence packs are separate immutable GitHub Release assets referenced by
the compact index; they are evidence carriers, not contract authorities.

Source tags and installed wheels ship the same compact current-evidence index
at `invarlock/_data/public_evidence/catalog_evidence_index.json`. An empty index
uses `status=not_created` and the label **Evidence not yet created**. Completed
lanes add hash-bound artifact entries as their current evidence becomes
available. Each externalized entry records the release-asset URL, archive
SHA-256, byte size, archive root, and logical path.

Immutable assets published before this terminology change retain
`public_evidence/published_basis/...` inside their archive bindings. That string
identifies a historical carrier path only; it is not a current support tier.

The 31 entries currently listed in the compact index are strictly verified
frozen-v1 packs accepted through the current verifier's explicit compatibility
path. Their availability does not imply that they contain or exercise v2
`guard_authority` fields.

## Policy packs

Policy packs are Git-native artifacts that bind:

- `resolved_policy`
- ordered `overrides`
- compatibility metadata, including the independently maintained dataset identity
- a deterministic `policy_digest` covering the resolved policy, ordered
  overrides, and compatibility metadata
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
