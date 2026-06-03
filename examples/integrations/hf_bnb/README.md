# hf_bnb Bitsandbytes Runtime-Load Integration Example

Status: `runnable`; strict container evidence is verified on CUDA for this tiny
bitsandbytes runtime-load example with the example-only bitsandbytes image.

This example shows how to attach InvarLock regression evidence to a subject
loaded through the built-in `hf_bnb` adapter. By default it creates a tiny
local Llama-style checkpoint, compares that checkpoint as a normal `hf_causal`
baseline against the same checkpoint loaded with bitsandbytes 8-bit runtime
quantization, and records backend inventory alongside the evaluation report.

Use this example when the integration point is "load this checkpoint through
bitsandbytes and compare the resulting subject" rather than "publish a new
checkpoint directory." The subject is a runtime-loaded model for that comparison
path.

The example keeps bitsandbytes in the example environment rather than the core
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

### Lane Support

| Artifact lane label | Command shape | Notes |
| --- | --- | --- |
| `cuda-container-strict` | `--lane cuda` | Primary evidence path with the example-specific bitsandbytes image. |
| `cuda-host-off` | `--lane host --device cuda` | Secondary local CUDA comparison path without strict container evidence. |
| `cpu-host-off` | `--lane host --device cpu` | Secondary local non-CUDA bring-up when the installed bitsandbytes backend supports it. |

Host lanes run prerequisite preflight before model preparation and evaluation.
The `cuda-host-off` lane checks `torch.cuda.is_available()` before the backend run.

### cuda-container-strict lane

Build and check the example-specific bitsandbytes image, then run this lane on a
CUDA host with that image configured:

```bash
examples/integrations/_runtime_images/build_example_runtime_image.sh cuda-bnb
examples/integrations/_runtime_images/smoke_example_runtime_image.sh cuda-bnb

INVARLOCK_RUNTIME_IMAGE=invarlock-example-runtime:cuda-bnb \
uv run --extra hf --extra gpu \
  examples/integrations/hf_bnb/run_tiny_hf_bnb_8bit.sh \
  --allow-network \
  --force \
  --lane cuda
```

Strict container evidence should use the digest-pinned image reference recorded
in `runtime.manifest.json` when the artifact is being shared externally.

This strict lane is scoped to the configured tiny runtime-loaded bitsandbytes
subject and runtime image. Rerun the strict lane for the target runtime before
using the result as shared integration evidence.

### cpu-host-off lane

From the repository root:

```bash
uv run --extra hf --extra gpu \
  examples/integrations/hf_bnb/run_tiny_hf_bnb_8bit.sh \
  --allow-network \
  --force \
  --lane host \
  --device cpu
```

The default path uses `--execution-mode host --assurance off` because
bitsandbytes runtime support is platform-dependent. Use this lane for local
dependency setup; non-CUDA execution depends on the installed bitsandbytes
backend. It still runs the InvarLock evaluator, verifier, backend inventory,
and HTML renderer.

For `cuda-host-off` evaluation, use the same command with `--device cuda`.

## Outputs

The runner writes generated outputs under local output directories:

| Path | Role |
| --- | --- |
| `models/tiny-llama-bnb-baseline/` | Deterministic tiny HF checkpoint used as both dense baseline and bitsandbytes runtime-loaded subject. |
| `artifacts/tiny-hf-bnb-8bit/tiny_causal_text.jsonl` | Deterministic local text fixture for the small comparison. |
| `artifacts/tiny-hf-bnb-8bit/preset.yaml` | Generated preset pointing at the local fixture. |
| `artifacts/tiny-hf-bnb-8bit/fixture_summary.json` | Fixture parameters and file hashes. |
| `reports/tiny-hf-bnb-8bit/<artifact-lane>/evaluation.report.json` | Canonical verifier input. |
| `reports/tiny-hf-bnb-8bit/<artifact-lane>/verify.json` | Machine-readable verifier result. |
| `reports/tiny-hf-bnb-8bit/<artifact-lane>/evaluation.html` | Human-readable report. |
| `reports/tiny-hf-bnb-8bit/<artifact-lane>/backend_inventory.json` | bitsandbytes backend version, quantized module types, and smoke results. |
| `reports/tiny-hf-bnb-8bit/<artifact-lane>/lane_artifact.json` | Canonical artifact-lane label and effective runtime settings. |
| `reports/tiny-hf-bnb-8bit/<artifact-lane>/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-hf-bnb-8bit/<artifact-lane>/run_summary.txt` | Concise success or failure status, lane label, verifier status, runtime provenance status, and primary output paths. |

A successful run ends with the shared completion block documented in
`examples/integrations/_shared/README.md#expected-run-output`. If a run fails,
check the prerequisite message first, then inspect
`reports/tiny-hf-bnb-8bit/<artifact-lane>/run_command.txt`.

The generated preset uses local JSONL data so the evaluation data path is
offline after fixture creation. `--allow-network` is only needed for the HF
model files when they are not already cached.

The shell runner relies on InvarLock report persistence to emit
`backend_inventory.json` when adapter provenance is available.
