# PEFT LoRA-Merge Integration Example

Status: `runnable`; strict container evidence verified on CUDA.

This example shows how to attach InvarLock regression evidence to a checkpoint
created by an external PEFT LoRA merge. It materializes a tiny deterministic
LoRA adapter for `sshleifer/tiny-gpt2`, merges it into a HF-loadable subject
directory, and then compares that subject against the baseline with the shared
integration wrapper.

The example is source-tree only. It does not add PEFT to the core InvarLock
install.

## Prerequisites

Install InvarLock with the Hugging Face stack and add PEFT to the same example
environment:

```bash
python -m pip install "invarlock[hf]" peft
```

From a repository checkout, an existing `.venv` with `invarlock[hf]` is also
fine:

```bash
.venv/bin/python -m pip install peft
```

## Run

From the repository root:

```bash
examples/integrations/peft_lora/run_tiny_peft_lora.sh \
  --allow-network \
  --force
```

The default compare path is strict/container-backed. For host-only bring-up or
dependency debugging, run the same example with explicit host mode:

```bash
examples/integrations/peft_lora/run_tiny_peft_lora.sh \
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
| `models/tiny-gpt2-peft-lora-merged/` | HF-loadable merged subject checkpoint. |
| `artifacts/tiny-peft-lora-fixture/tiny_causal_text.jsonl` | Deterministic local text fixture for evaluation. |
| `artifacts/tiny-peft-lora-fixture/preset.yaml` | Generated preset pointing at the local fixture. |
| `artifacts/tiny-peft-lora-fixture/fixture_summary.json` | Fixture parameters and file hashes. |
| `reports/tiny-peft-lora/evaluation.report.json` | Canonical verifier input. |
| `reports/tiny-peft-lora/verify.json` | Machine-readable verifier result. |
| `reports/tiny-peft-lora/evaluation.html` | Human-readable report. |
| `reports/tiny-peft-lora/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-peft-lora/checkpoint_refs.json` | Baseline and subject checkpoint references. |
| `reports/tiny-peft-lora/external_edit_summary.json` | PEFT merge metadata and checkpoint file hashes. |

The subject materializer writes a non-zero LoRA delta and fails if the merged
checkpoint does not change the target attention weights.
When PEFT is installed into a broad quantization environment, the materializer
keeps this dense LoRA path isolated from optional GPTQModel/AWQ dispatch.

## Public Evidence Anchor

The repository also ships a small public LoRA-merge BYOE fixture:

```bash
invarlock verify --profile release --assurance strict \
  public_evidence/byoe_examples/lora_merge_byoe/evaluation.report.json
```

Use that fixture as the stable public reference when the local example
environment does not have PEFT installed.
