# torchao Int8 Runtime Integration Example

Status: `runnable`; strict container evidence is verified on CUDA for this tiny
`hf_torchao` runtime-load example with the example-only TorchAO image.

This example shows how to attach InvarLock regression evidence to a Hugging Face
causal checkpoint loaded through InvarLock's `hf_torchao` adapter. It creates a
tiny local Llama-style HF checkpoint, uses that checkpoint as the baseline, then
uses the same checkpoint as the subject loaded through `hf_torchao`, where
`torchao` int8 weight-only quantization is applied at adapter load time.

The checkpoint save-boundary probe in `adapter_runtime_summary.json` is
supporting metadata. The runnable evidence path is the `hf_torchao` subject
adapter path recorded in `run_command.txt` and `evaluation.report.json`.

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

### Lane Support

| Artifact lane label | Command shape | Notes |
| --- | --- | --- |
| `cuda-container-strict` | `--lane cuda` | Primary review path with the example-only TorchAO image. |
| `cuda-host-off` | `--lane host --device cuda` | Secondary local CUDA comparison path without strict container evidence. |
| `cpu-host-off` | `--lane host --device cpu` | Secondary local non-CUDA bring-up for `hf_torchao`. |

Host lanes run prerequisite preflight before fixture preparation and
evaluation. The `cuda-host-off` lane checks `torch.cuda.is_available()` before
the backend run.

### cuda-container-strict lane

Build and smoke the example-only TorchAO image, then run this lane on a CUDA
host with that image configured:

```bash
examples/integrations/_runtime_images/build_example_runtime_image.sh cuda-torchao
examples/integrations/_runtime_images/smoke_example_runtime_image.sh cuda-torchao

INVARLOCK_RUNTIME_IMAGE=invarlock-example-runtime:cuda-torchao \
examples/integrations/torchao_int8_runtime/run_tiny_hf_torchao_int8.sh \
  --allow-network \
  --force \
  --lane cuda
```

The runner defaults to the `release` profile so the strict verification path has
enough evaluation tokens for a stable primary-metric verdict.
Use the digest-pinned image reference recorded in `runtime.manifest.json` when
the strict container artifact will be shared for review.

This strict lane proves the configured tiny HF checkpoint loaded through the
`hf_torchao` adapter. It does not claim blanket strict support for every
external torchao tensor-subclass wrapper or model shape; rerun the strict lane
for the target runtime before using the result as outreach evidence.

### cpu-host-off lane

From the repository root:

```bash
examples/integrations/torchao_int8_runtime/run_tiny_hf_torchao_int8.sh \
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
| `models/tiny-llama-hf-torchao-baseline/` | Deterministic tiny HF checkpoint used by both sides of the comparison. |
| `artifacts/tiny-hf-torchao-int8/tiny_causal_text.jsonl` | Deterministic local text fixture for evaluation. |
| `artifacts/tiny-hf-torchao-int8/preset.yaml` | Generated preset pointing at the local fixture. |
| `artifacts/tiny-hf-torchao-int8/fixture_summary.json` | Fixture parameters and file hashes. |
| `reports/tiny-hf-torchao-int8/<artifact-lane>/evaluation.report.json` | Canonical verifier input. |
| `reports/tiny-hf-torchao-int8/<artifact-lane>/verify.json` | Machine-readable verifier result. |
| `reports/tiny-hf-torchao-int8/<artifact-lane>/evaluation.html` | Human-readable report. |
| `reports/tiny-hf-torchao-int8/<artifact-lane>/backend_inventory.json` | torchao backend version and quantized module inventory when adapter provenance is available. |
| `reports/tiny-hf-torchao-int8/<artifact-lane>/lane_artifact.json` | Canonical artifact-lane label and effective runtime settings. |
| `reports/tiny-hf-torchao-int8/<artifact-lane>/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-hf-torchao-int8/<artifact-lane>/run_summary.txt` | Concise success or failure status, lane label, verifier status, runtime provenance status, and primary output paths. |
| `reports/tiny-hf-torchao-int8/<artifact-lane>/checkpoint_refs.json` | Baseline and subject checkpoint references. |
| `reports/tiny-hf-torchao-int8/<artifact-lane>/adapter_runtime_summary.json` | `hf_torchao` runtime adapter metadata, quantization probe, and file hashes. |

A successful run ends with the shared completion block documented in
`examples/integrations/_shared/README.md#expected-run-output`. If a run fails,
check the prerequisite message first, then inspect
`reports/tiny-hf-torchao-int8/<artifact-lane>/run_command.txt`.

The preparer fails if `torchao` does not produce quantized tensor-backed weights
or if runtime quantization has no measurable weight delta.
`backend_inventory.json` is emitted by InvarLock report persistence when adapter
provenance is available; the shell runner does not write that sidecar directly.

## Public Evidence Anchor

The repository also ships a small quantization-style public fixture:

```bash
invarlock verify --profile release --assurance strict \
  public_evidence/real_runs/tiny_gpt2_quant_rtn/evaluation.report.json
```

Use that fixture as the stable public reference when the local example
environment does not have `torchao` installed.
