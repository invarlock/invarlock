# Public Evidence

This directory contains the compact current-evidence index for the maintained
evaluation lanes in `contracts/evidence_catalog_v1.json`. Full evidence packs
are distributed as the GitHub Release asset named and hash-bound by
`published_basis_index.json`.

The catalog and support matrix remain useful before a lane has been run: they
define the model, adapter, preset, input materialization, execution policy, and
required artifact roles. Evidence appears here only after the corresponding run
has completed and passed the current verification contract.

## Evidence status

| Status | Meaning |
| --- | --- |
| `not_created` | The maintained catalog lane is defined for evaluation; current evidence has not yet been created. |
| `available` | Current evidence is listed in the compact index and available from its hash-bound release asset. |

The status for every maintained lane is recorded in
`contracts/support_matrix.json`. Until a current artifact is available, the
human-readable label is **Evidence not yet created**.

## Artifact shape

Each available lane's archive entry supplies the artifact roles declared by its
catalog entry, including the evaluation report, runtime manifest, final
verdict, source and input provenance, resolved configuration, preset,
independent baseline, and policy pack. The index records each logical path,
artifact digest and size, plus the containing archive's URL, digest, size, and
root.

Current artifacts are additive: publishing one lane changes only that lane's
status and index entry. The remaining catalog rows continue to read
`not_created`.

## Verification

Download the release asset recorded in `published_basis_index.json`, verify its
SHA-256 and size, and unpack it at the repository root. Then use the public
verifier commands to inspect an available artifact:

```bash
invarlock advanced evidence-pack inspect PATH --json
invarlock advanced evidence-pack verify PATH --strict --json
```

The canonical JSON artifacts are the review source of truth. HTML reports are
rendered views of the same evaluation result. Maintainers can also verify the
download automatically with:

```bash
python scripts/checks/check_public_evidence.py --fetch-external-assets
```
