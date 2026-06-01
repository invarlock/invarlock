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
- `_shared/run_invarlock_compare.sh` is a reusable baseline-vs-subject wrapper
  for HF-loadable paths.

## Example Lifecycle

1. Confirm the optional backend and adapter status with `invarlock doctor` and
   `invarlock advanced plugins list --json`.
2. Create or reference a subject checkpoint from the external tool.
3. Run `invarlock evaluate` against the baseline and subject.
4. Run `invarlock verify --json` and render `evaluation.html`.
5. Record the output paths and any backend limitations in the target README.

Use target examples as public review aids before opening external issues or
docs pull requests. The target README should make the runnable status explicit:
`runnable`, `exploratory-host`, or `compatibility-investigation`.
