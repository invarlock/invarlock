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

## Lane Support

| Artifact lane label | Command shape | Notes |
| --- | --- | --- |
| `cuda-container-strict` | `--lane cuda` | Primary review path with the regular CUDA runtime image. |
| `cuda-host-off` | `--lane host --device cuda` | Secondary local CUDA comparison path without strict container evidence. |
| `cpu-host-off` | `--lane host --device cpu` | Secondary local non-CUDA bring-up for the merged dense checkpoint. |

Host lanes run prerequisite preflight before materialization and evaluation. The
`cuda-host-off` lane checks `torch.cuda.is_available()` before the backend run.

### cuda-container-strict lane

Build the regular CUDA runtime image, then run this lane on a CUDA host with
that image configured. This example evaluates a merged dense checkpoint, so it
does not need the quant example images.

```bash
make runtime-image-cuda

INVARLOCK_RUNTIME_IMAGE=invarlock-runtime:cuda-local \
examples/integrations/peft_lora/run_tiny_peft_lora.sh \
  --allow-network \
  --force \
  --lane cuda
```

The runner defaults to the `release` profile so the strict verification path has
enough evaluation tokens for a stable primary-metric verdict.
Use the digest-pinned image reference recorded in `runtime.manifest.json` when
the strict container artifact will be shared for review.

### cpu-host-off lane

From the repository root:

```bash
examples/integrations/peft_lora/run_tiny_peft_lora.sh \
  --allow-network \
  --force \
  --lane host \
  --device cpu
```

Use this lane for local dependency bring-up and non-CUDA smoke runs.

For `cuda-host-off` evaluation, use the same command with `--device cuda`.

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
| `reports/tiny-peft-lora/lane_artifact.json` | Canonical artifact-lane label and effective runtime settings. |
| `reports/tiny-peft-lora/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-peft-lora/run_summary.txt` | Concise success or failure status, lane label, and primary output paths. |
| `reports/tiny-peft-lora/checkpoint_refs.json` | Baseline and subject checkpoint references. |
| `reports/tiny-peft-lora/external_edit_summary.json` | PEFT merge metadata and checkpoint file hashes. |

A successful run ends with the shared completion block documented in
`examples/integrations/_shared/README.md#expected-run-output`. If a run fails,
check the prerequisite message first, then inspect
`reports/tiny-peft-lora/run_command.txt`.

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
