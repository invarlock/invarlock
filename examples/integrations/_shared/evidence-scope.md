# Evidence Scope

An integration example produces regression evidence for one configured
baseline-vs-subject comparison. The subject may be a materialized checkpoint or
the same checkpoint loaded through a runtime adapter.

The evidence is strongest when the workflow runs in the default container-backed
strict mode and the generated `evaluation.report.json` verifies with its sibling
`runtime.manifest.json`.

## Example Status Labels

| Label | Meaning |
| --- | --- |
| `runnable` | The target README provides commands that generate an evaluation report, verifier JSON, and HTML report in the documented environment. |
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

## Strict Evidence Checklist

- `invarlock evaluate` ran with `--assurance strict` or the default strict mode.
- The default container execution path produced `runtime.manifest.json`.
- `invarlock verify --assurance strict` passed for `evaluation.report.json`.
- `verify.json` was generated from the same report with `--json`.
- `evaluation.html` was rendered from the same report.
- `lane_artifact.json` records `cuda-container-strict`.
- `run_command.txt` records the wrapper, evaluate, verify, and render commands.
- `run_summary.txt` records verifier and runtime-provenance status.
- Quantized-adapter lanes include `backend_inventory.json` when adapter
  provenance is available.
- Shared strict evidence records or references the runtime image digest.
- Each integration example README records required or optional dependencies and
  backend limitations when they apply.
