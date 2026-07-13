# Evidence Scope

An integration example produces regression evidence for one configured
baseline-vs-subject comparison. The subject may be a materialized checkpoint or
the same checkpoint loaded through a runtime adapter.

The evidence is strongest when the workflow runs in the default container-backed
strict mode and the generated `evaluation.report.json` verifies against an
independent raw baseline, acceptance policy pack, and runtime-image digest while
binding its sibling `runtime.manifest.json`.

## Example Status Labels

| Label | Meaning |
| --- | --- |
| `runnable` | The target README provides commands that generate an evaluation report, verifier JSON, and HTML report in the documented environment. |
| `reference-pattern` | The README shows how to attach existing reports to CI or registry surfaces without producing a new subject checkpoint. |
| `exploratory-host` | The target workflow runs in host mode with `--assurance off`; useful for local dependency setup and backend investigation. |
| `compatibility-investigation` | The target artifact cannot yet be loaded or verified through the documented InvarLock path; the README records the blocker. |

## Run Lanes

Use consistent run lanes in target READMEs:

| Artifact lane label | Meaning |
| --- | --- |
| `cpu-host-off` | Local optional-dependency bring-up with `--lane host --device cpu`; this is available only when the backend can run the selected comparison without CUDA. |
| `cuda-host-off` | Local CUDA dependency bring-up with `--lane host --device cuda`; useful for backend validation before strict evidence. |
| `cuda-container-strict` | CUDA-host runtime manifest, provenance evidence, and strict verifier assurance with `--lane cuda`. |

When a lane is unavailable, state the concrete backend reason instead of
omitting the lane.

## Maintainer-Facing Wording

Use scoped language:

> This example produces regression evidence for one configured baseline
> checkpoint versus one configured subject path.

Keep public claims tied to the generated artifacts, the selected model family,
the adapter, the dataset/window plan, and the verifier result.

`source_matrix.json` is the source-controlled contract for README strict-lane
requirements. It records the target runner, strict lane, runtime image source,
expected verifier and runtime provenance status, and required core and
target-specific sidecars for a current run. Its v1 schema is closed:
the top-level object, each target entry, and nested expectation objects accept
only their documented fields. An unfiltered validation requires every canonical
strict target exactly once; focused `--targets` validation still checks the full
shape of every matrix entry before selecting a lane.

The validator reads the matrix, report, runtime manifest, acceptance baseline,
and acceptance policy as single regular-file snapshots. It rejects symlinks,
duplicate keys, non-finite JSON values, and concurrent replacement or mutation.
It replays strict verification from those exact bytes, checks the required
sidecar set and closed runtime-observation schema, and requires the runtime
quantization proof to match the backend inventory exactly. This establishes a
live loaded-model type inventory for supported module-backed adapters; it does
not establish packed checkpoint storage, kernel performance, or transformation
history.

## Strict Evidence Checklist

- `invarlock evaluate` ran with `--assurance strict` or the default strict mode.
- The default container execution path produced `runtime.manifest.json`.
- `invarlock verify --profile ci|release --assurance strict --baseline <retained baseline report.json> --policy-pack <acceptance policy pack.json> --expected-runtime-image-digest ...`
  passed for `evaluation.report.json`, using an image pin obtained independently
  of the submitted runtime manifest.
- `verify.json` was generated from the same report with `--json`.
- `evaluation.html` was rendered from the same report.
- `source_matrix.json` has an entry for every README that documents a strict
  integration lane.
- `validate_source_matrix_artifacts.py` replays strict verification with the
  acceptance baseline, policy pack, and image digest before treating that run as
  current evidence.
- `lane_artifact.json` records `cuda-container-strict`.
- `run_command.txt` records the wrapper, evaluate, verify, and render commands.
- `run_summary.txt` records verifier and runtime-provenance status.
- Target-specific provenance sidecars are retained in the lane output when they
  apply: `checkpoint_refs.json`, `external_edit_summary.json`,
  `adapter_runtime_summary.json`, `training_receipt.json`,
  `training_binding.json`, `training_evidence_proof.json`,
  `training_profile_snapshot.json`, and `fixture_summary.json`.
- Quantized-adapter lanes include `backend_inventory.json` when adapter
  provenance is available.
- Strict module-backed quantized lanes additionally include
  `runtime_quantization_proof.json`. It records recognized live runtime types
  for the selected adapter and backend; it is not evidence of packed storage
  or another checkpoint transformation.
- Shared strict evidence records the manifest-declared runtime image digest and
  records verifier output showing a match to a separately supplied image pin.
- Each integration example README records required or optional dependencies and
  backend limitations when they apply.
