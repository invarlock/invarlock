# torchao Int8 Export Integration Example

Status: `runnable`; strict container evidence verified on CUDA.

This example shows how to attach InvarLock regression evidence to a checkpoint
created by an external `torchao` quantization workflow. It creates a tiny local
Llama-style HF baseline, applies `torchao` int8 weight-only quantization, exports
a dequantized HF-loadable subject checkpoint, and compares the exported subject
against the generated baseline with the shared integration wrapper.

The direct `torchao` tensor-subclass model is not treated as a portable HF
checkpoint in this example. The materializer records a native quantized-save
probe in `external_edit_summary.json`; the runnable path is the HF-loadable
export produced after dequantizing the quantized weights.

The example is source-tree only. It does not add `torchao` to the core InvarLock
install.

## Prerequisites

Install InvarLock with the Hugging Face stack and add `torchao` to the same
example environment:

```bash
python -m pip install "invarlock[hf]" torchao
```

From a repository checkout, an existing `.venv` with `invarlock[hf]` is also
fine:

```bash
.venv/bin/python -m pip install torchao
```

## Run

From the repository root:

```bash
examples/integrations/torchao_int8_export/run_tiny_torchao_int8_export.sh \
  --allow-network \
  --force
```

The default compare path is strict/container-backed. For host-only bring-up or
dependency debugging, run the same example with explicit host mode:

```bash
examples/integrations/torchao_int8_export/run_tiny_torchao_int8_export.sh \
  --allow-network \
  --force \
  --execution-mode host \
  --assurance off
```

The runner defaults to the `release` profile so the strict verification path has
enough evaluation tokens for a stable primary-metric verdict.
Set `INVARLOCK_RUNTIME_IMAGE` and `INVARLOCK_RUNTIME_IMAGE_DIGEST` when the
strict container artifact will be shared for review.

## Outputs

The runner writes generated outputs under ignored local directories:

| Path | Role |
| --- | --- |
| `models/tiny-llama-baseline/` | Deterministic tiny HF baseline checkpoint. |
| `models/tiny-llama-torchao-int8-export/` | HF-loadable subject exported from the `torchao` quantization pass. |
| `artifacts/tiny-torchao-int8-export-fixture/tiny_causal_text.jsonl` | Deterministic local text fixture for evaluation. |
| `artifacts/tiny-torchao-int8-export-fixture/preset.yaml` | Generated preset pointing at the local fixture. |
| `artifacts/tiny-torchao-int8-export-fixture/fixture_summary.json` | Fixture parameters and file hashes. |
| `reports/tiny-torchao-int8-export/evaluation.report.json` | Canonical verifier input. |
| `reports/tiny-torchao-int8-export/verify.json` | Machine-readable verifier result. |
| `reports/tiny-torchao-int8-export/evaluation.html` | Human-readable report. |
| `reports/tiny-torchao-int8-export/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-torchao-int8-export/checkpoint_refs.json` | Baseline and subject checkpoint references. |
| `reports/tiny-torchao-int8-export/external_edit_summary.json` | `torchao` export metadata, save-boundary probe, deltas, and file hashes. |

The materializer fails if `torchao` does not produce quantized tensor-backed
weights or if the exported subject has no measurable weight delta.

## Public Evidence Anchor

The repository also ships a small quantization-style public fixture:

```bash
invarlock verify --profile release --assurance strict \
  public_evidence/real_runs/tiny_gpt2_quant_rtn/evaluation.report.json
```

Use that fixture as the stable public reference when the local example
environment does not have `torchao` installed.
