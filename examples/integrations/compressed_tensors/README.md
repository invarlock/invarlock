# compressed-tensors Checkpoint Integration Example

Status: `runnable`; strict container evidence is verified for the
`cuda-container-strict` lane on CUDA for this tiny compressed-tensors example.
`cuda-host-off` and `cpu-host-off` lanes are supported.

This example shows how to attach InvarLock regression evidence to a Hugging Face
causal checkpoint saved in the `compressed-tensors` packed checkpoint format. It
creates a tiny dense Llama-style HF baseline, creates a matching packed
compressed-tensors subject checkpoint, then compares the baseline through
`hf_causal` against the subject loaded through InvarLock's `hf_ct` adapter.

The example keeps compressed-tensors in the example environment rather than the
core InvarLock install. Its scope is pre-quantized checkpoint-load regression
evidence; compressed-kernel speedup validation is outside this example.
`llmcompressor` is covered here as tooling that can produce compatible
compressed-tensors checkpoints.

## Prerequisites

Install InvarLock with the compressed-tensors optional stack in the same example
environment:

```bash
python -m pip install "invarlock[compressed-tensors]"
```

From a repository checkout, `uv` can provide the optional stack for the run:

```bash
uv run --extra compressed-tensors python -c "import compressed_tensors"
```

## Run

### Lane Support

| Artifact lane label | Command shape | Notes |
| --- | --- | --- |
| `cuda-container-strict` | `--lane cuda` | Primary evidence path with the example-specific compressed-tensors image. |
| `cuda-host-off` | `--lane host --device cuda` | Secondary local CUDA comparison path without strict container evidence. |
| `cpu-host-off` | `--lane host --device cpu` | Secondary local non-CUDA bring-up when the installed compressed-tensors backend supports CPU. |

Host lanes run prerequisite preflight before fixture preparation and
evaluation. The `cuda-host-off` lane checks `torch.cuda.is_available()` before
the backend run.

### cuda-container-strict lane

Build and check the example-specific compressed-tensors image, then run this lane on
a CUDA host with that image configured:

```bash
examples/integrations/_runtime_images/build_example_runtime_image.sh cuda-compressed-tensors
examples/integrations/_runtime_images/smoke_example_runtime_image.sh cuda-compressed-tensors

INVARLOCK_RUNTIME_IMAGE=invarlock-example-runtime:cuda-compressed-tensors \
uv run --extra compressed-tensors \
  examples/integrations/compressed_tensors/run_tiny_hf_ct.sh \
  --allow-network \
  --force \
  --lane cuda
```

Use the digest-pinned image reference recorded in `runtime.manifest.json` when
the strict container artifact will be shared externally.
This strict lane is scoped to the configured tiny `hf_ct` packed-checkpoint
subject and image. Rerun the strict lane for the target runtime before using the
artifact as shared integration evidence.

### cpu-host-off lane

From the repository root:

```bash
uv run --extra compressed-tensors \
  examples/integrations/compressed_tensors/run_tiny_hf_ct.sh \
  --allow-network \
  --force \
  --lane host \
  --device cpu
```

Use this lane for local dependency setup and non-CUDA compatibility runs when the
compressed-tensors backend supports the selected host.

For `cuda-host-off` evaluation, use the same command with `--device cuda`.

## Evidence Boundary

The subject checkpoint is materialized before the InvarLock comparison. The
strict lane covers the configured baseline-vs-subject evaluation, `hf_ct`
adapter load, guard evidence, runtime manifest, and verifier result for that
produced subject. The compressed-tensors materialization step is represented by
`adapter_runtime_summary.json` and checkpoint hashes.

## Outputs

The runner writes generated outputs under local output directories:

| Path | Role |
| --- | --- |
| `models/tiny-llama-hf-ct-baseline/` | Deterministic dense HF checkpoint used as the baseline. |
| `models/tiny-llama-hf-ct-subject/` | Matching compressed-tensors packed subject checkpoint loaded by `hf_ct`. |
| `artifacts/tiny-hf-ct/tiny_causal_text.jsonl` | Deterministic local text fixture for evaluation. |
| `artifacts/tiny-hf-ct/preset.yaml` | Generated preset pointing at the local fixture. |
| `artifacts/tiny-hf-ct/fixture_summary.json` | Fixture parameters and file hashes. |
| `reports/tiny-hf-ct/<artifact-lane>/evaluation.report.json` | Canonical verifier input. |
| `reports/tiny-hf-ct/<artifact-lane>/verify.json` | Machine-readable verifier result. |
| `reports/tiny-hf-ct/<artifact-lane>/evaluation.html` | Human-readable report. |
| `reports/tiny-hf-ct/<artifact-lane>/backend_inventory.json` | compressed-tensors backend version and module inventory when exposed. |
| `reports/tiny-hf-ct/<artifact-lane>/lane_artifact.json` | Canonical artifact-lane label and effective runtime settings. |
| `reports/tiny-hf-ct/<artifact-lane>/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-hf-ct/<artifact-lane>/run_summary.txt` | Concise success or failure status, lane label, verifier status, runtime provenance status, and primary output paths. |
| `reports/tiny-hf-ct/<artifact-lane>/checkpoint_refs.json` | Baseline and subject checkpoint references. |
| `reports/tiny-hf-ct/<artifact-lane>/adapter_runtime_summary.json` | `hf_ct` adapter metadata, packed tensor inventory, quantization settings, and file hashes. |

A successful run ends with the shared completion block documented in
`examples/integrations/_shared/README.md#expected-run-output`. If a run fails,
check the prerequisite message first, then inspect
`reports/tiny-hf-ct/<artifact-lane>/run_command.txt`.

The subject checkpoint is produced with `compressed_tensors.compressors.ModelCompressor`
and contains packed weight tensors plus the HF `quantization_config` metadata.
Transformers may decompress those tensors for inference on the selected stack,
so the stable example claim is checkpoint-load regression evidence.
The shell runner relies on InvarLock report persistence to emit
`backend_inventory.json` when adapter provenance is available.
