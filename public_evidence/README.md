# Public Evidence

This directory contains current evidence artifacts for the maintained
evaluation lanes in `contracts/evidence_catalog_v1.json`.

The catalog and support matrix remain useful before a lane has been run: they
define the model, adapter, preset, input materialization, execution policy, and
required artifact roles. Evidence appears here only after the corresponding run
has completed and passed the current verification contract.

## Evidence status

| Status | Meaning |
| --- | --- |
| `not_created` | The maintained catalog lane is defined for evaluation; current evidence has not yet been created. |
| `available` | Current evidence artifacts are present and listed by the public evidence index. |

The status for every maintained lane is recorded in
`contracts/support_matrix.json`. Until a current artifact is available, the
human-readable label is **Evidence not yet created**.

## Artifact shape

Each available lane supplies the artifact roles declared by its catalog entry,
including the evaluation report, runtime manifest, final verdict, source and
input provenance, resolved configuration, preset, independent baseline, and
policy pack.

Current artifacts are additive: publishing one lane changes only that lane's
status and index entry. The remaining catalog rows continue to read
`not_created`.

## Verification

Use the public verifier commands to inspect an available artifact:

```bash
invarlock advanced evidence-pack inspect PATH --json
invarlock advanced evidence-pack verify PATH --strict --json
```

The canonical JSON artifacts are the review source of truth. HTML reports are
rendered views of the same evaluation result.
