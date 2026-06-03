# Integration Examples

This directory contains first-party, optional examples for attaching InvarLock
regression evidence to model-edit workflows owned by external tools.

Each example is intentionally small enough to inspect quickly and writes
generated outputs outside tracked source files. Model weights, large reports,
and downloaded datasets belong outside the repository.

## Current Status

The shared scaffold is active. Each example directory owns its status,
prerequisites, commands, and generated artifact list.

## Shared Assets

- `_shared/evidence-scope.md` defines the claim boundary for integration
  examples.
- `_shared/expected-artifacts.md` lists the artifacts each runnable example
  should produce.
- `source_matrix.json` binds explicit strict-evidence README claims to the
  target runner, runtime image source, lane label, and required sidecars.
- `_shared/preflight.sh` contains shared host-lane preflight and artifact-lane
  labeling helpers.
- `_shared/run_invarlock_compare.sh` is a reusable baseline-vs-subject wrapper
  for HF-loadable checkpoints and adapter-backed subject paths.
- `_runtime_images/` contains example-only CUDA image definitions for optional
  quant backends. These images are not the regular InvarLock runtime images.

## Example Lifecycle

1. Confirm the optional backend and adapter status with `invarlock doctor` and
   `invarlock advanced plugins list --json`.
2. Create or reference the subject path, whether that is a materialized
   checkpoint or a runtime adapter loading mode.
3. Document and, where possible, run `cuda-container-strict` as the primary
   evidence path. Host lanes are secondary comparison paths: `cuda-host-off`
   for host CUDA setup and `cpu-host-off` for non-CUDA setup when that backend
   actually supports CPU. The user-facing shortcuts remain `--lane cuda` and
   `--lane host`; host lanes should pass an explicit `--device cpu` or
   `--device cuda` when comparing lanes. Optional quant examples should use the
   narrowest matching image under `_runtime_images/`; dense examples should use
   the standard InvarLock CUDA runtime.
4. Run `invarlock evaluate` against the baseline and subject.
5. Run `invarlock verify --json` and render `evaluation.html`.
6. Record the output paths and any backend limitations in the example README.

Use these examples as public, reproducible reference flows when discussing
integrations with upstream projects. Each README should make the runnable status
explicit: `runnable`, `exploratory-host`, or `compatibility-investigation`.
