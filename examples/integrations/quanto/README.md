# Quanto Runtime Integration Example

Status: `runnable`; `cuda-container-strict`, `cuda-host-off`, and `cpu-host-off`
lanes are supported.

This example shows how to attach InvarLock regression evidence to a Hugging
Face causal checkpoint loaded through InvarLock's `hf_quanto` adapter. It creates a
tiny local Llama-style HF checkpoint, uses that checkpoint as the baseline, then
uses the same checkpoint as the subject loaded through `hf_quanto`, where Quanto
quantization is requested through the Transformers load-time quantization
configuration.

The example is source-tree only. It does not add Quanto to the core InvarLock
install.

## Prerequisites

Install InvarLock with the Quanto optional stack in the same example environment:

```bash
python -m pip install "invarlock[quanto]"
```

From a repository checkout, `uv` can provide the optional stack for the run:

```bash
uv run --extra quanto python -c "import optimum.quanto"
```

## Run

## Lane Support

| Artifact lane label | Command shape | Notes |
| --- | --- | --- |
| `cuda-container-strict` | `--lane cuda` | Primary review path with the example-only Quanto image. |
| `cuda-host-off` | `--lane host --device cuda` | Secondary local CUDA comparison path without strict container evidence. |
| `cpu-host-off` | `--lane host --device cpu` | Secondary local non-CUDA bring-up when the installed Quanto backend supports CPU. |

Host lanes run prerequisite preflight before fixture preparation and
evaluation. The `cuda-host-off` lane checks `torch.cuda.is_available()` before
the backend run.

### cuda-container-strict lane

Build and smoke the example-only Quanto image, then run this lane on a CUDA host
with that image configured:

```bash
examples/integrations/_runtime_images/build_example_runtime_image.sh cuda-quanto
examples/integrations/_runtime_images/smoke_example_runtime_image.sh cuda-quanto

INVARLOCK_RUNTIME_IMAGE=invarlock-example-runtime:cuda-quanto \
uv run --extra quanto \
  examples/integrations/quanto/run_tiny_hf_quanto.sh \
  --allow-network \
  --force \
  --lane cuda
```

Use the digest-pinned image reference recorded in `runtime.manifest.json` when
the strict container artifact will be shared for review.

### cpu-host-off lane

From the repository root:

```bash
uv run --extra quanto \
  examples/integrations/quanto/run_tiny_hf_quanto.sh \
  --allow-network \
  --force \
  --lane host \
  --device cpu
```

Use this lane for local dependency bring-up and non-CUDA smoke runs when the
installed Quanto backend supports the selected host.

For `cuda-host-off` evaluation, use the same command with `--device cuda`.

## Outputs

The runner writes generated outputs under ignored local directories:

| Path | Role |
| --- | --- |
| `models/tiny-llama-hf-quanto-baseline/` | Deterministic tiny HF checkpoint used by both sides of the comparison. |
| `artifacts/tiny-hf-quanto/tiny_causal_text.jsonl` | Deterministic local text fixture for evaluation. |
| `artifacts/tiny-hf-quanto/preset.yaml` | Generated preset pointing at the local fixture. |
| `artifacts/tiny-hf-quanto/fixture_summary.json` | Fixture parameters and file hashes. |
| `reports/tiny-hf-quanto/evaluation.report.json` | Canonical verifier input. |
| `reports/tiny-hf-quanto/verify.json` | Machine-readable verifier result. |
| `reports/tiny-hf-quanto/evaluation.html` | Human-readable report. |
| `reports/tiny-hf-quanto/backend_inventory.json` | Quanto backend version and quantized module inventory when exposed. |
| `reports/tiny-hf-quanto/lane_artifact.json` | Canonical artifact-lane label and effective runtime settings. |
| `reports/tiny-hf-quanto/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-hf-quanto/run_summary.txt` | Concise success or failure status, lane label, and primary output paths. |
| `reports/tiny-hf-quanto/checkpoint_refs.json` | Baseline and subject checkpoint references. |
| `reports/tiny-hf-quanto/adapter_runtime_summary.json` | `hf_quanto` runtime adapter metadata, quantization settings, and file hashes. |

A successful run ends with the shared completion block documented in
`examples/integrations/_shared/README.md#expected-run-output`. If a run fails,
check the prerequisite message first, then inspect
`reports/tiny-hf-quanto/run_command.txt`.

The example uses Quanto runtime quantization through the HF load path, so the
subject remains an HF-loadable checkpoint plus adapter runtime configuration
rather than a Quanto-only checkpoint format.
