# compressed-tensors Checkpoint Integration Example

Status: `diagnostic runtime compatibility`. `hf_ct` can load genuine packed
compressed-tensors checkpoints, but it is intentionally ineligible for strict
assurance until InvarLock has a dedicated packed-storage artifact proof.
`cuda-host-off` and `cpu-host-off` lanes are supported for diagnostic work.

This example shows how to attach InvarLock regression evidence to a Hugging Face
causal checkpoint saved in the `compressed-tensors` packed checkpoint format. It
creates a tiny dense Llama-style HF baseline, creates a matching packed
compressed-tensors subject checkpoint, then compares the baseline through
`hf_causal` against the subject loaded through InvarLock's `hf_ct` adapter.

The example keeps compressed-tensors in the example environment rather than the
core InvarLock install. Its scope is diagnostic pre-quantized checkpoint-load
regression work; compressed-kernel speedup validation and strict artifact
assurance are outside this example.
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

### Optional CUDA image for diagnostic checks

For isolated CUDA backend import and load diagnostics, build and smoke the
matching example image:

```bash
examples/integrations/_runtime_images/build_example_runtime_image.sh cuda-compressed-tensors
examples/integrations/_runtime_images/smoke_example_runtime_image.sh cuda-compressed-tensors
```

The image tag is `invarlock-example-runtime:cuda-compressed-tensors`. It can
help reproduce an environment issue, but it does not turn this example into a
strict lane or establish packed-storage provenance.

## Run

### Lane Support

| Artifact lane label | Command shape | Notes |
| --- | --- | --- |
| `cuda-container-strict` | unavailable | The runner rejects this lane before setup because packed-storage proof is not implemented. |
| `cuda-host-off` | `--lane host --device cuda` | Secondary local CUDA comparison path without strict container evidence. |
| `cpu-host-off` | `--lane host --device cpu` | Secondary local non-CUDA bring-up when the installed compressed-tensors backend supports CPU. |

Host lanes run prerequisite preflight before fixture preparation and
evaluation. The `cuda-host-off` lane checks `torch.cuda.is_available()` before
the backend run.

### Strict assurance is unavailable

The runner rejects `--lane cuda` and any strict-assurance request before it
creates a fixture or runs a comparison. A `compressed-tensors` config and a
backend inventory do not prove that the stored packed tensors correspond to
the model claimed by an evidence artifact. Do not use this example as strict
or release evidence until a dedicated packed-storage artifact proof is added.

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
diagnostic lane covers the configured baseline-vs-subject evaluation and
`hf_ct` adapter load for that produced subject. The compressed-tensors
materialization step is represented by `adapter_runtime_summary.json` and
checkpoint hashes, but those evaluation-time facts are not a packed-storage
artifact proof.

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

A successful diagnostic run ends with the shared completion block documented in
`examples/integrations/_shared/README.md#expected-run-output`. If a run fails,
check the prerequisite message first, then inspect
`reports/tiny-hf-ct/<artifact-lane>/run_command.txt`.

The subject checkpoint is produced with `compressed_tensors.compressors.ModelCompressor`
and contains packed weight tensors plus the HF `quantization_config` metadata.
Transformers may decompress those tensors for inference on the selected stack,
so the stable example claim is diagnostic checkpoint-load compatibility only.
The shell runner relies on InvarLock report persistence to emit
`backend_inventory.json` when adapter provenance is available.
