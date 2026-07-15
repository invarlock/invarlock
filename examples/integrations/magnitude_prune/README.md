# Magnitude-Prune Integration Example

Status: `runnable`. A `cuda-container-strict` result requires independent
acceptance inputs and the successful current run described below.

This example shows how to attach InvarLock regression evidence to a checkpoint
created by an external pruning workflow. It zeros a deterministic low-magnitude
slice of eligible tensors in `sshleifer/tiny-gpt2`, saves the resulting
HF-loadable subject directory, and then compares that subject against the
baseline with the shared integration wrapper.

The example keeps pruning as upstream subject generation. InvarLock evaluates
the produced subject checkpoint under the selected dataset window, tier,
profile, guard policy, and runtime policy.

## Prerequisites

Install InvarLock with the Hugging Face stack:

```bash
python -m pip install "invarlock[hf]"
```

From a repository checkout, an existing `.venv` created with the HF extras is
also fine:

```bash
uv sync --extra hf
```

## Run

### Lane Support

| Artifact lane label | Command shape | Notes |
| --- | --- | --- |
| `cuda-container-strict` | `--lane cuda` | Primary evidence path with the standard InvarLock CUDA runtime image. |
| `cuda-host-off` | `--lane host --device cuda` | Secondary local CUDA comparison path without strict container evidence. |
| `cpu-host-off` | `--lane host --device cpu` | Secondary local non-CUDA bring-up for the saved magnitude-pruned checkpoint. |

Host lanes run prerequisite preflight before materialization and evaluation. The
`cuda-host-off` lane checks `torch.cuda.is_available()` before the backend run.

### cuda-container-strict lane

Build the standard InvarLock CUDA runtime image, then run this lane on a CUDA
host with that image configured:

```bash
make runtime-image-cuda

TRUSTED_RUNTIME_IMAGE_DIGEST='sha256:REPLACE_WITH_REVIEWED_64_HEX_DIGEST'
INVARLOCK_ACCEPTANCE_BASELINE_REPORT=/path/to/raw-baseline-report.json \
INVARLOCK_ACCEPTANCE_POLICY_PACK=/path/to/acceptance-policy-pack.json \
INVARLOCK_RUNTIME_IMAGE=invarlock-runtime:cuda-local \
INVARLOCK_EXPECTED_RUNTIME_IMAGE_DIGEST="$TRUSTED_RUNTIME_IMAGE_DIGEST" \
uv run --extra hf \
examples/integrations/magnitude_prune/run_tiny_magnitude_prune.sh \
  --allow-network \
  --force \
  --lane cuda
```

The runner defaults to the `release` profile so the strict verification path has
enough evaluation tokens for a stable primary-metric verdict. Obtain the
trusted digest independently from reviewed build/release policy. The matching
digest in `runtime.manifest.json` is a manifest claim, not the source of the
verifier pin.

This strict lane is scoped to the configured tiny magnitude-pruned checkpoint
and runtime image. Rerun the strict lane for the target runtime before using the
artifact as shared integration evidence.

### cpu-host-off lane

From the repository root:

```bash
uv run --extra hf \
examples/integrations/magnitude_prune/run_tiny_magnitude_prune.sh \
  --allow-network \
  --force \
  --lane host \
  --device cpu
```

Use this lane for local dependency setup and non-CUDA compatibility runs.

For `cuda-host-off` evaluation, use the same command with `--device cuda`.

## Evidence Boundary

The subject checkpoint is materialized before the InvarLock comparison. The
strict lane covers the configured baseline-vs-subject evaluation, `hf_causal`
adapter load, guard evidence, runtime manifest, and verifier result for that
produced subject. The magnitude-prune step is represented by
`external_edit_summary.json` and checkpoint hashes.

## Outputs

The runner writes generated outputs under local output directories:

| Path | Role |
| --- | --- |
| `models/tiny-gpt2-magnitude-pruned/` | HF-loadable magnitude-pruned subject checkpoint. |
| `artifacts/tiny-magnitude-prune-fixture/tiny_causal_text.jsonl` | Deterministic local text fixture for evaluation. |
| `artifacts/tiny-magnitude-prune-fixture/preset.yaml` | Generated preset pointing at the local fixture. |
| `artifacts/tiny-magnitude-prune-fixture/fixture_summary.json` | Fixture parameters and file hashes. |
| `reports/tiny-magnitude-prune/<artifact-lane>/evaluation.report.json` | Canonical verifier input. |
| `reports/tiny-magnitude-prune/<artifact-lane>/verify.json` | Machine-readable verifier result. |
| `reports/tiny-magnitude-prune/<artifact-lane>/runtime.manifest.json` | Runtime provenance sidecar. |
| `reports/tiny-magnitude-prune/<artifact-lane>/evaluation.html` | Human-readable report. |
| `reports/tiny-magnitude-prune/<artifact-lane>/lane_artifact.json` | Canonical artifact-lane label and effective runtime settings. |
| `reports/tiny-magnitude-prune/<artifact-lane>/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-magnitude-prune/<artifact-lane>/run_summary.txt` | Concise success or failure status, lane label, verifier status, runtime provenance status, and primary output paths. |
| `reports/tiny-magnitude-prune/<artifact-lane>/checkpoint_refs.json` | Baseline and subject checkpoint references. |
| `reports/tiny-magnitude-prune/<artifact-lane>/external_edit_summary.json` | Magnitude-prune metadata and checkpoint file hashes. |
| `reports/tiny-magnitude-prune/<artifact-lane>/fixture_summary.json` | Evaluation fixture parameters and file hashes. |

A successful run ends with the shared completion block documented in
`examples/integrations/_shared/README.md#expected-run-output`. If a run fails,
check the prerequisite message first, then inspect
`reports/tiny-magnitude-prune/<artifact-lane>/run_command.txt`.

The subject materializer writes a non-zero pruning delta and fails if the
pruning step does not change any floating tensors.
