# Integration Examples

This directory contains first-party, optional examples for attaching InvarLock
regression evidence to model-edit workflows owned by external tools.

Each target example should stay small enough to review in a pull request and
should point generated outputs to local ignored directories. Model weights,
large reports, and downloaded datasets belong outside the repository.

## Current Status

The shared scaffold is active. Target directories land one at a time after
their runnable path or compatibility blocker is understood, and each target
README owns its status, prerequisites, commands, and generated artifact list.

Keep this overview stable unless a shared integration contract changes. Do not
update it only to announce a new target directory.

## Shared Assets

- `_shared/evidence-scope.md` defines the claim boundary for integration
  examples.
- `_shared/expected-artifacts.md` lists the review artifacts each runnable
  example should produce.
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
   review path. Host lanes are secondary comparison paths: `cuda-host-off` for
   host CUDA bring-up and `cpu-host-off` for non-CUDA local bring-up when that
   backend actually supports CPU. The user-facing shortcuts remain `--lane cuda`
   and `--lane host`; host lanes should pass an explicit `--device cpu` or
   `--device cuda` when comparing lanes. Optional quant examples should use the
   narrowest matching image under `_runtime_images/`; dense examples should use
   the regular CUDA runtime.
4. Run `invarlock evaluate` against the baseline and subject.
5. Run `invarlock verify --json` and render `evaluation.html`.
6. Record the output paths and any backend limitations in the target README.

Use target examples as public review aids before opening external issues or
docs pull requests. The target README should make the runnable status explicit:
`runnable`, `exploratory-host`, or `compatibility-investigation`.
