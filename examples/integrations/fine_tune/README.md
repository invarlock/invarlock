# Fine-Tune Integration Example

Status: `runnable`; strict container evidence is verified on CUDA for this tiny
fine-tuned checkpoint example with the standard InvarLock CUDA runtime image.

This example shows how to attach InvarLock regression evidence to a checkpoint
created by an external fine-tuning workflow. It runs one deterministic tiny
training step for `sshleifer/tiny-gpt2`, saves the resulting HF-loadable subject
directory, and then compares that subject against the baseline with the shared
integration wrapper.

The example keeps fine-tuning as upstream subject generation. InvarLock
evaluates the produced subject; it does not certify training quality, edit
success, locality, robustness, or safety.

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
| `cpu-host-off` | `--lane host --device cpu` | Secondary local non-CUDA bring-up for the saved fine-tuned checkpoint. |

Host lanes run prerequisite preflight before materialization and evaluation. The
`cuda-host-off` lane checks `torch.cuda.is_available()` before the backend run.

### cuda-container-strict lane

Build the standard InvarLock CUDA runtime image, then run this lane on a CUDA
host with that image configured:

```bash
make runtime-image-cuda

INVARLOCK_RUNTIME_IMAGE=invarlock-runtime:cuda-local \
uv run --extra hf \
examples/integrations/fine_tune/run_tiny_fine_tune.sh \
  --allow-network \
  --force \
  --lane cuda
```

The runner defaults to the `release` profile so the strict verification path has
enough evaluation tokens for a stable primary-metric verdict. Use the
digest-pinned image reference recorded in `runtime.manifest.json` when the
strict container artifact will be shared externally.

This strict lane is scoped to the configured tiny fine-tuned checkpoint and
runtime image. Rerun the strict lane for the target runtime before using the
artifact as shared integration evidence.

### cpu-host-off lane

From the repository root:

```bash
uv run --extra hf \
examples/integrations/fine_tune/run_tiny_fine_tune.sh \
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
produced subject. The fine-tune step is represented by
`external_edit_summary.json` and checkpoint hashes.

## Outputs

The runner writes generated outputs under local output directories:

| Path | Role |
| --- | --- |
| `models/tiny-gpt2-fine-tuned/` | HF-loadable fine-tuned subject checkpoint. |
| `artifacts/tiny-fine-tune-fixture/tiny_causal_text.jsonl` | Deterministic local text fixture for evaluation. |
| `artifacts/tiny-fine-tune-fixture/preset.yaml` | Generated preset pointing at the local fixture. |
| `artifacts/tiny-fine-tune-fixture/fixture_summary.json` | Fixture parameters and file hashes. |
| `reports/tiny-fine-tune/<artifact-lane>/evaluation.report.json` | Canonical verifier input. |
| `reports/tiny-fine-tune/<artifact-lane>/verify.json` | Machine-readable verifier result. |
| `reports/tiny-fine-tune/<artifact-lane>/runtime.manifest.json` | Runtime provenance sidecar. |
| `reports/tiny-fine-tune/<artifact-lane>/evaluation.html` | Human-readable report. |
| `reports/tiny-fine-tune/<artifact-lane>/lane_artifact.json` | Canonical artifact-lane label and effective runtime settings. |
| `reports/tiny-fine-tune/<artifact-lane>/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-fine-tune/<artifact-lane>/run_summary.txt` | Concise success or failure status, lane label, verifier status, runtime provenance status, and primary output paths. |
| `reports/tiny-fine-tune/<artifact-lane>/checkpoint_refs.json` | Baseline and subject checkpoint references. |
| `reports/tiny-fine-tune/<artifact-lane>/external_edit_summary.json` | Fine-tune metadata and checkpoint file hashes. |
| `reports/tiny-fine-tune/<artifact-lane>/fixture_summary.json` | Evaluation fixture parameters and file hashes. |

A successful run ends with the shared completion block documented in
`examples/integrations/_shared/README.md#expected-run-output`. If a run fails,
check the prerequisite message first, then inspect
`reports/tiny-fine-tune/<artifact-lane>/run_command.txt`.

The subject materializer writes a non-zero parameter delta and fails if the
training step does not change any floating tensors.

## Public Evidence Anchor

The repository also ships a small public fine-tune BYOE fixture:

```bash
invarlock verify --profile release --assurance strict \
  public_evidence/byoe_examples/fine_tune_byoe/evaluation.report.json
```

Use that fixture as the stable public reference when the local example
environment cannot materialize a fresh fine-tuned subject.
