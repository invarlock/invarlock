# hf_bnb Bitsandbytes Runtime-Load Integration Example

Status: `runnable`; strict container evidence verified on CUDA.

This example shows how to attach InvarLock regression evidence to a subject
loaded through the built-in `hf_bnb` adapter. By default it creates a tiny
local Llama-style checkpoint, compares that checkpoint as a normal `hf_causal`
baseline against the same checkpoint loaded with bitsandbytes 8-bit runtime
quantization, and records backend inventory alongside the evaluation report.

The subject is a runtime-loaded model, not a saved HF export. Use this example
when the integration point is "load this checkpoint through bitsandbytes and
compare the resulting subject" rather than "publish a new checkpoint directory."

The example is source-tree only. It does not add bitsandbytes to the core
InvarLock install.

## Prerequisites

Install InvarLock with the Hugging Face and GPU extras in the same example
environment:

```bash
python -m pip install "invarlock[hf,gpu]"
```

From a repository checkout, `uv` can provide the optional stack for the run:

```bash
uv run --extra hf --extra gpu python -c "import bitsandbytes"
```

## Run

From the repository root:

```bash
uv run --extra hf --extra gpu \
  examples/integrations/hf_bnb/run_tiny_hf_bnb_8bit.sh \
  --allow-network \
  --force
```

The default path uses `--execution-mode host --assurance off` because
bitsandbytes runtime support is platform-dependent. It still runs the
InvarLock evaluator, verifier, backend inventory, and HTML renderer.

For a strict CUDA/container run, provide the quant runtime image and switch the
runner to container mode:

```bash
INVARLOCK_RUNTIME_IMAGE=invarlock-runtime:cuda-quant \
uv run --extra hf --extra gpu \
  examples/integrations/hf_bnb/run_tiny_hf_bnb_8bit.sh \
  --allow-network \
  --force \
  --execution-mode container \
  --assurance strict \
  --device cuda
```

Strict container evidence should use a digest-pinned runtime image when the
artifact is being shared for review.

## Outputs

The runner writes generated outputs under ignored local directories:

| Path | Role |
| --- | --- |
| `models/tiny-llama-bnb-baseline/` | Deterministic tiny HF checkpoint used as both dense baseline and bitsandbytes runtime-loaded subject. |
| `artifacts/tiny-hf-bnb-8bit/tiny_causal_text.jsonl` | Deterministic local text fixture for the CI-sized comparison. |
| `artifacts/tiny-hf-bnb-8bit/preset.yaml` | Generated preset pointing at the local fixture. |
| `artifacts/tiny-hf-bnb-8bit/fixture_summary.json` | Fixture parameters and file hashes. |
| `reports/tiny-hf-bnb-8bit/evaluation.report.json` | Canonical verifier input. |
| `reports/tiny-hf-bnb-8bit/verify.json` | Machine-readable verifier result. |
| `reports/tiny-hf-bnb-8bit/evaluation.html` | Human-readable report. |
| `reports/tiny-hf-bnb-8bit/backend_inventory.json` | bitsandbytes backend version, quantized module types, and smoke results. |
| `reports/tiny-hf-bnb-8bit/run_command.txt` | Wrapper, evaluate, verify, and render commands. |

The generated preset uses local JSONL data so the evaluation data path is
offline after fixture creation. `--allow-network` is only needed for the HF
model files when they are not already cached.
