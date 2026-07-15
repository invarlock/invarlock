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

- `_shared/evidence-scope.md` defines the evidence scope for integration
  examples.
- `_shared/expected-artifacts.md` lists the artifacts each runnable example
  should produce.
- `source_matrix.json` binds explicit strict-evidence README claims to the
  target runner, runtime image source, lane label, verifier expectation, and
  required core and target-specific sidecars.
- `_shared/preflight.sh` contains shared host-lane preflight and artifact-lane
  labeling helpers.
- `_shared/run_invarlock_compare.sh` is a reusable baseline-vs-subject wrapper
  for HF-loadable checkpoints and adapter-backed subject paths.
- `public_e2e/` demonstrates a handoff from caller-supplied current evidence to
  verifier, HTML, MLflow tag, model-card, release-review, and CI summary
  artifacts.
- `design_partner_diagnostic/` provides the reviewer-owned case template for a
  single strict baseline-versus-transformed-subject diagnostic. The checked
  runbook lives in `docs/user-guide/design-partner-diagnostic.md`.
- `ci_registry/` shows how to attach current report verification, HTML, MLflow
  tags, Hugging Face model-card evidence, and release-review packets to
  existing CI and registry workflows.
- `_shared/validate_source_matrix_artifacts.py` checks generated strict-lane
  artifact directories against `source_matrix.json`.
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
5. Run `invarlock verify --json --baseline ... --policy-pack ...
   --expected-runtime-image-digest ...` using independently reviewed inputs,
   then render `evaluation.html`.
6. Validate generated strict-lane artifacts against `source_matrix.json` before
   presenting a new run as verified strict evidence:

   ```bash
   python3 examples/integrations/_shared/validate_source_matrix_artifacts.py \
     --targets <target> \
     --baseline-report /path/to/raw-baseline-report.json \
     --policy-pack /path/to/acceptance-policy-pack.json \
     --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST"
   ```

   Validate targets separately when their acceptance inputs differ.

7. Record the output paths and any backend limitations in the example README.

Use these examples as public, reproducible reference flows when discussing
integrations with upstream projects. Each README should make the status
explicit: `runnable`, `reference-pattern`, `exploratory-host`, or
`compatibility-investigation`.
