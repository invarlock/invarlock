# PEFT LoRA-Merge Integration Example

Status: `runnable`; strict container evidence is verified on CUDA for this tiny
PEFT LoRA-merge example with the standard InvarLock CUDA runtime image.

This example shows how to attach InvarLock regression evidence to a checkpoint
created by an external PEFT LoRA merge. It materializes a tiny deterministic
LoRA adapter for `sshleifer/tiny-gpt2`, merges it into a HF-loadable subject
directory, and then compares that subject against the baseline with the shared
integration wrapper.

The example keeps PEFT in the example environment rather than the core
InvarLock install.

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

If the checkout environment was created by `uv sync`, install PEFT into that
environment with:

```bash
uv pip install --python .venv/bin/python peft
```

From a source checkout, you can also keep the optional dependency scoped to the
example command:

```bash
uv run --extra hf --with peft \
  examples/integrations/peft_lora/run_tiny_peft_lora.sh --help
```

## Run

### Lane Support

| Artifact lane label | Command shape | Notes |
| --- | --- | --- |
| `cuda-container-strict` | `--lane cuda` | Primary evidence path with the standard InvarLock CUDA runtime image. |
| `cuda-host-off` | `--lane host --device cuda` | Secondary local CUDA comparison path without strict container evidence. |
| `cpu-host-off` | `--lane host --device cpu` | Secondary local non-CUDA bring-up for the merged dense checkpoint. |

Host lanes run prerequisite preflight before materialization and evaluation. The
`cuda-host-off` lane checks `torch.cuda.is_available()` before the backend run.

### cuda-container-strict lane

Build the standard InvarLock CUDA runtime image, then run this lane on a CUDA host with
that image configured. This example evaluates a merged dense checkpoint, so it
does not need the quant example images.

```bash
make runtime-image-cuda

INVARLOCK_RUNTIME_IMAGE=invarlock-runtime:cuda-local \
uv run --extra hf --with peft \
examples/integrations/peft_lora/run_tiny_peft_lora.sh \
  --allow-network \
  --force \
  --lane cuda
```

The runner defaults to the `release` profile so the strict verification path has
enough evaluation tokens for a stable primary-metric verdict.
Use the digest-pinned image reference recorded in `runtime.manifest.json` when
the strict container artifact will be shared externally.
This strict lane is scoped to the configured tiny merged dense checkpoint and
runtime image. Rerun the strict lane for the target runtime before using the
artifact as shared integration evidence.

### cpu-host-off lane

From the repository root:

```bash
uv run --extra hf --with peft \
examples/integrations/peft_lora/run_tiny_peft_lora.sh \
  --allow-network \
  --force \
  --lane host \
  --device cpu
```

Use this lane for local dependency setup and non-CUDA compatibility runs.

For `cuda-host-off` evaluation, use the same command with `--device cuda`.

## Evidence Boundary

The subject checkpoint is materialized before the InvarLock comparison. The
strict lane covers the configured baseline-vs-subject evaluation, `hf_causal`
adapter load, guard evidence, runtime manifest, and verifier result for that
produced subject. The LoRA merge step is represented by
`external_edit_summary.json` and checkpoint hashes.

## Outputs

The runner writes generated outputs under local output directories:

| Path | Role |
| --- | --- |
| `models/tiny-gpt2-peft-lora-merged/` | HF-loadable merged subject checkpoint. |
| `artifacts/tiny-peft-lora-fixture/tiny_causal_text.jsonl` | Deterministic local text fixture for evaluation. |
| `artifacts/tiny-peft-lora-fixture/preset.yaml` | Generated preset pointing at the local fixture. |
| `artifacts/tiny-peft-lora-fixture/fixture_summary.json` | Fixture parameters and file hashes. |
| `reports/tiny-peft-lora/<artifact-lane>/evaluation.report.json` | Canonical verifier input. |
| `reports/tiny-peft-lora/<artifact-lane>/verify.json` | Machine-readable verifier result. |
| `reports/tiny-peft-lora/<artifact-lane>/evaluation.html` | Human-readable report. |
| `reports/tiny-peft-lora/<artifact-lane>/lane_artifact.json` | Canonical artifact-lane label and effective runtime settings. |
| `reports/tiny-peft-lora/<artifact-lane>/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-peft-lora/<artifact-lane>/run_summary.txt` | Concise success or failure status, lane label, verifier status, runtime provenance status, and primary output paths. |
| `reports/tiny-peft-lora/<artifact-lane>/checkpoint_refs.json` | Baseline and subject checkpoint references. |
| `reports/tiny-peft-lora/<artifact-lane>/external_edit_summary.json` | PEFT merge metadata and checkpoint file hashes. |

A successful run ends with the shared completion block documented in
`examples/integrations/_shared/README.md#expected-run-output`. If a run fails,
check the prerequisite message first, then inspect
`reports/tiny-peft-lora/<artifact-lane>/run_command.txt`.

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
