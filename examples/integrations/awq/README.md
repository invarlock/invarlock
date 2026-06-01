# AWQ Integration Example

Status: `runnable` on CUDA hosts.

This example shows how to attach InvarLock regression evidence to a checkpoint
quantized with GPTQModel's AWQ flow. It creates a deterministic small
Llama-style Hugging Face baseline with AWQ-compatible layer widths, quantizes
that checkpoint as AWQ, and compares the quantized subject through InvarLock's
`hf_awq` adapter.

The example is source-tree only. It does not add GPTQModel or CUDA libraries to
the core InvarLock install.

## Prerequisites

Run this example on a CUDA host. GPTQModel's AWQ quantization path requires CUDA
for materialization. The script defaults to the `torch_awq` backend so the tiny
checkpoint exercises a portable AWQ load path rather than an architecture-tuned
kernel.

Install InvarLock with the AWQ optional stack in the same example environment:

```bash
python -m pip install "invarlock[awq]"
```

From a source checkout with `uv`:

```bash
uv run --extra awq python -c "import gptqmodel"
```

## Run

```bash
uv run --extra awq \
  examples/integrations/awq/run_tiny_awq.sh \
  --allow-network \
  --force
```

The default path uses `--execution-mode host --assurance off` because the AWQ
runtime depends on the selected CUDA host and installed GPTQModel wheel. To
exercise a different AWQ backend, pass `--awq-backend VALUE`.

For strict container-backed evidence, run on a CUDA host with the quant runtime
image configured and pass:

```bash
uv run --extra awq \
  examples/integrations/awq/run_tiny_awq.sh \
  --allow-network \
  --force \
  --execution-mode container \
  --assurance strict
```

## Generated Artifacts

The default run writes ignored local artifacts under this directory:

| Path | Purpose |
| --- | --- |
| `models/tiny-llama-baseline/` | Deterministic tiny HF baseline checkpoint. |
| `models/tiny-llama-awq-4bit/` | GPTQModel AWQ-quantized subject checkpoint. |
| `artifacts/tiny-awq-fixture/tiny_causal_text.jsonl` | Deterministic local text fixture for evaluation and calibration. |
| `artifacts/tiny-awq-fixture/preset.yaml` | Generated preset pointing at the local fixture. |
| `artifacts/tiny-awq-fixture/fixture_summary.json` | Fixture parameters and file hashes. |
| `reports/tiny-awq/evaluation.report.json` | Canonical verifier input. |
| `reports/tiny-awq/verify.json` | Machine-readable verifier result. |
| `reports/tiny-awq/evaluation.html` | Human-readable report. |
| `reports/tiny-awq/backend_inventory.json` | GPTQModel backend version and AWQ module inventory when exposed. |
| `reports/tiny-awq/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-awq/checkpoint_refs.json` | Baseline and subject checkpoint references. |
| `reports/tiny-awq/external_edit_summary.json` | AWQ quantization metadata and checkpoint file hashes. |

The helper fails if CUDA is unavailable, if GPTQModel does not expose a
quantized checkpoint configuration, or if the subject cannot be loaded back
through the Transformers AWQ loader with the selected backend.
