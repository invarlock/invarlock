# Evidence Catalog

## Purpose

InvarLock's evidence catalog defines the maintained evaluation lanes used to
exercise its public model, adapter, input, and verification surfaces. The
catalog is useful before and after a run:

- before a run, it fixes the model, adapter, preset, input materialization,
  execution policy, and required artifact roles;
- after a run, it provides the identity against which the resulting evidence
  pack is verified.

The machine-readable source is `contracts/evidence_catalog_v1.json`. The
support table and current evidence state live in
`contracts/support_matrix.json`.

## Current status

The current index contains strictly verified evidence for 31 of the 39
maintained lanes. Those rows are labeled **Available** and link to their
evidence packs and verification receipts. The other 8 rows retain the label
**Evidence not yet created** until current-contract evidence is published.

Each available row has a corresponding entry in
`public_evidence/published_basis_index.json`. The compact index is carried by
source tags and installed wheels; the complete 31-lane evidence tree is the
hash-bound GitHub Release asset recorded by every artifact entry.

## Creating and verifying current evidence

Maintainers produce one lane at a time with the repository command:

```bash
python scripts/model_evidence/run_catalog_lane.py --help
```

The command derives evaluation behavior from the catalog and independently
resolved inputs, performs strict report and pack verification, and writes a
signed pack to a caller-selected staging directory. It accepts the acceptance
policy, runtime-image digest, source identity, and evidence signing key from the
caller.
The complete invocation and execution boundary are documented in
`scripts/evidence_packs/README.md`.

To publish completed results, stage the verified packs under
`public_evidence/published_basis/`, update each lane from `not_created` to
`available`, build the archive, and run
`scripts/checks/sync_packaged_public_evidence.py` with the external asset URL,
SHA-256, size, and archive root. Commit the compact source and packaged indexes;
upload the full archive as the corresponding GitHub Release asset.

## What an available lane contains

Each catalog entry declares the artifacts required for publication. The common
set includes:

| Artifact | Role |
| --- | --- |
| `evaluation.report.json` | Canonical paired comparison and guard evidence. |
| `runtime.manifest.json` | Runtime identity bound to the evaluation report. |
| `final_verdict.json` | Verified lane outcome. |
| `source_repo.json` | Source revision and source-bundle identity. |
| `resolved-inputs.json` | Exact model and dataset inputs. |
| `resolved-config.yaml` | Effective configuration used by evaluation. |
| `preset.yaml` | Catalog-selected preset. |
| `baseline.report.json` | Independent baseline used for recomputation. |
| `policy-pack.json` | Independently maintained acceptance policy. |

Vision-text lanes also bind the materialized record schedule used by the
evaluation.

## Review workflow

Download the release asset named in the index, verify its recorded SHA-256 and
size, and unpack it at the repository root. Then inspect the catalog and an
available pack with the public CLI:

```bash
invarlock advanced evidence-catalog validate \
  contracts/evidence_catalog_v1.json --json

invarlock advanced evidence-pack inspect PATH --json
invarlock advanced evidence-pack verify PATH --strict --json
```

For a complete published set, `verify-set` checks every retained pack against
the catalog and independent source, image, and signer anchors.

## Using the catalog for your own checkpoint

The same workflow applies to a checkpoint produced by quantization, pruning,
adapter merge, fine-tuning, or another editing system:

1. choose the closest maintained lane and preset;
2. run `invarlock evaluate` with the baseline and edited subject;
3. retain the raw baseline, resolved inputs, resolved configuration, and
   runtime manifest;
4. run `invarlock verify` with independently supplied policy and runtime inputs;
5. render `evaluation.html` for human review.

The resulting evidence is scoped to the chosen checkpoints, paired records,
configuration, and policy, and records the inputs needed to reproduce and audit
the verification decision.
