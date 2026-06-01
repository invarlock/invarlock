# GPTQModel Integration Example

Status: `runnable`; strict container runtime provenance verified on CUDA.

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

From the repository root:

```bash
uv run --extra gptq \
  examples/integrations/gptqmodel/run_tiny_gptqmodel.sh \
  --allow-network \
  --force
```

The default path uses `--execution-mode host --assurance off` because GPTQModel
runtime loading is platform-dependent. In host mode the runner sets
`TORCHDYNAMO_DISABLE=1` unless you already set it, which avoids platform-local
Torch compile failures during tiny smoke runs. It still runs the
InvarLock evaluator, verifier, backend inventory, and HTML renderer.

For a CUDA/container run that completes with runtime provenance verification,
provide the quant runtime image, switch the runner to container mode, and keep
the primary assurance verdict off:

```bash
INVARLOCK_RUNTIME_IMAGE=invarlock-runtime:cuda-quant \
uv run --extra gptq \
  examples/integrations/gptqmodel/run_tiny_gptqmodel.sh \
  --allow-network \
  --force \
  --execution-mode container \
  --assurance off \
  --runtime-provenance container \
  --device cuda
```

Use a digest-pinned runtime image when the artifact is being shared for review.
The `--assurance strict` path is reserved for quantized-checkpoint guard
contract work until spectral and variance coverage for GPTQ module targets is
expanded.

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
| `reports/tiny-gptqmodel/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-gptqmodel/checkpoint_refs.json` | Baseline and subject checkpoint references. |
| `reports/tiny-gptqmodel/external_edit_summary.json` | GPTQModel quantization metadata and checkpoint file hashes. |

The helper fails if GPTQModel does not produce a quantized checkpoint
configuration or if the subject cannot be loaded back through GPTQModel.
