# GPTQModel Integration Example

Status: `runnable`; strict container evidence verified on CUDA.

This example shows how to attach InvarLock regression evidence to a checkpoint
quantized by GPTQModel. It creates a deterministic tiny Llama-style Hugging Face
baseline, quantizes that checkpoint with GPTQModel, and compares the quantized
subject through InvarLock's `hf_gptq` adapter.

The example is source-tree only. It does not add GPTQModel to the core
InvarLock install.

## Prerequisites

Install InvarLock with the GPTQ optional stack in the same example environment:

```bash
python -m pip install "invarlock[gptq]"
```

From a repository checkout, `uv` can provide the optional stack for the run:

```bash
uv run --extra gptq python -c "import gptqmodel"
```

## Run

## Lane Support

| Artifact lane label | Command shape | Notes |
| --- | --- | --- |
| `cuda-container-strict` | `--lane cuda` | Primary review path with the example-only GPTQModel/AWQ image. |
| `cuda-host-off` | `--lane host --device cuda` | Secondary host CUDA comparison path; requires host CUDA plus the same GPTQModel/Triton prerequisites. |
| `cpu-host-off` | `--lane host --device cpu` | Secondary local bring-up path when the host GPTQModel/Triton prerequisites support CPU. |

Host lanes run prerequisite preflight before model materialization and
evaluation. GPTQModel host lanes require a C compiler and matching Python
development headers because Triton may compile runtime helpers during loading.

### cuda-container-strict lane

Build and smoke the example-only GPTQModel/AWQ image, then run this lane on a
CUDA host with that image configured:

```bash
examples/integrations/_runtime_images/build_example_runtime_image.sh cuda-gptqmodel
examples/integrations/_runtime_images/smoke_example_runtime_image.sh cuda-gptqmodel

INVARLOCK_RUNTIME_IMAGE=invarlock-example-runtime:cuda-gptqmodel \
uv run --extra gptq \
  examples/integrations/gptqmodel/run_tiny_gptqmodel.sh \
  --allow-network \
  --force \
  --lane cuda
```

Strict container evidence should use the digest-pinned image reference recorded
in `runtime.manifest.json` when the artifact is being shared for review.

### cpu-host-off lane

From the repository root:

```bash
uv run --extra gptq \
  examples/integrations/gptqmodel/run_tiny_gptqmodel.sh \
  --allow-network \
  --force \
  --lane host \
  --device cpu
```

The default path uses `--execution-mode host --assurance off` because GPTQModel
runtime loading is platform-dependent. Use this lane for local dependency
bring-up; non-CUDA execution depends on the installed GPTQModel backend. In
host mode the runner sets `TORCHDYNAMO_DISABLE=1` unless you already set it,
which avoids platform-local Torch compile failures during tiny smoke runs. It
still runs the InvarLock evaluator, verifier, backend inventory, and HTML
renderer.

For `cuda-host-off` evaluation, use the same command with `--device cuda`.

## Outputs

The runner writes generated outputs under ignored local directories:

| Path | Role |
| --- | --- |
| `models/tiny-llama-baseline/` | Deterministic tiny HF baseline checkpoint. |
| `models/tiny-llama-gptq-4bit/` | GPTQModel-quantized subject checkpoint. |
| `artifacts/tiny-gptqmodel-fixture/tiny_causal_text.jsonl` | Deterministic local text fixture for evaluation and calibration. |
| `artifacts/tiny-gptqmodel-fixture/preset.yaml` | Generated preset pointing at the local fixture. |
| `artifacts/tiny-gptqmodel-fixture/fixture_summary.json` | Fixture parameters and file hashes. |
| `reports/tiny-gptqmodel/evaluation.report.json` | Canonical verifier input. |
| `reports/tiny-gptqmodel/verify.json` | Machine-readable verifier result. |
| `reports/tiny-gptqmodel/evaluation.html` | Human-readable report. |
| `reports/tiny-gptqmodel/backend_inventory.json` | GPTQModel backend version and quantized module inventory when exposed. |
| `reports/tiny-gptqmodel/lane_artifact.json` | Canonical artifact-lane label and effective runtime settings. |
| `reports/tiny-gptqmodel/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-gptqmodel/run_summary.txt` | Concise success or failure status, lane label, and primary output paths. |
| `reports/tiny-gptqmodel/checkpoint_refs.json` | Baseline and subject checkpoint references. |
| `reports/tiny-gptqmodel/external_edit_summary.json` | GPTQModel quantization metadata and checkpoint file hashes. |

A successful run ends with the shared completion block documented in
`examples/integrations/_shared/README.md#expected-run-output`. If a run fails,
check the prerequisite message first, then inspect
`reports/tiny-gptqmodel/run_command.txt`.

The helper fails if GPTQModel does not produce a quantized checkpoint
configuration or if the subject cannot be loaded back through GPTQModel.
