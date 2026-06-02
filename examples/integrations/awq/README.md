# AWQ Integration Example

Status: `runnable` on CUDA hosts; strict container evidence is verified on
CUDA for this tiny AWQ example with the example-only GPTQModel/AWQ image.

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

### Lane Support

| Artifact lane label | Command shape | Notes |
| --- | --- | --- |
| `cuda-container-strict` | `--lane cuda` | Primary evidence path with the example-specific GPTQModel/AWQ image. |
| `cuda-host-off` | `--lane host` | Secondary local CUDA dependency bring-up without strict container evidence. |

Host lanes run prerequisite preflight before model materialization and
evaluation. This AWQ example is CUDA-only because AWQ materialization requires
CUDA regardless of the final evaluator device.

### cuda-container-strict lane

Build and check the example-specific GPTQModel/AWQ image, then run this lane on a
CUDA host with that image configured:

```bash
examples/integrations/_runtime_images/build_example_runtime_image.sh cuda-gptqmodel
examples/integrations/_runtime_images/smoke_example_runtime_image.sh cuda-gptqmodel

INVARLOCK_RUNTIME_IMAGE=invarlock-example-runtime:cuda-gptqmodel \
uv run --extra awq \
  examples/integrations/awq/run_tiny_awq.sh \
  --allow-network \
  --force \
  --lane cuda
```

Use the digest-pinned image reference recorded in `runtime.manifest.json` when
the artifact is being shared externally.

This strict lane is scoped to the configured tiny AWQ checkpoint and runtime
image. Rerun the strict lane for the target runtime before using the result as
shared integration evidence.

### cuda-host-off lane

Use this lane on a CUDA host for local dependency setup:

```bash
uv run --extra awq \
  examples/integrations/awq/run_tiny_awq.sh \
  --allow-network \
  --force \
  --lane host
```

The host path uses `--execution-mode host --assurance off` because the AWQ
runtime depends on the selected CUDA host and installed GPTQModel wheel. To
exercise a different AWQ backend, pass `--awq-backend VALUE`.

## Evidence Boundary

The subject checkpoint is materialized before the InvarLock comparison. The
strict lane covers the configured baseline-vs-subject evaluation, `hf_awq`
adapter load, guard evidence, runtime manifest, and verifier result for that
produced subject. The AWQ materialization step is represented by
`external_edit_summary.json` and checkpoint hashes.

## Generated Artifacts

The default run writes generated artifacts under this directory:

| Path | Purpose |
| --- | --- |
| `models/tiny-llama-baseline/` | Deterministic tiny HF baseline checkpoint. |
| `models/tiny-llama-awq-4bit/` | GPTQModel AWQ-quantized subject checkpoint. |
| `artifacts/tiny-awq-fixture/tiny_causal_text.jsonl` | Deterministic local text fixture for evaluation and calibration. |
| `artifacts/tiny-awq-fixture/preset.yaml` | Generated preset pointing at the local fixture. |
| `artifacts/tiny-awq-fixture/fixture_summary.json` | Fixture parameters and file hashes. |
| `reports/tiny-awq/<artifact-lane>/evaluation.report.json` | Canonical verifier input. |
| `reports/tiny-awq/<artifact-lane>/verify.json` | Machine-readable verifier result. |
| `reports/tiny-awq/<artifact-lane>/evaluation.html` | Human-readable report. |
| `reports/tiny-awq/<artifact-lane>/backend_inventory.json` | GPTQModel backend version and AWQ module inventory when exposed. |
| `reports/tiny-awq/<artifact-lane>/lane_artifact.json` | Canonical artifact-lane label and effective runtime settings. |
| `reports/tiny-awq/<artifact-lane>/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-awq/<artifact-lane>/run_summary.txt` | Concise success or failure status, lane label, verifier status, runtime provenance status, and primary output paths. |
| `reports/tiny-awq/<artifact-lane>/checkpoint_refs.json` | Baseline and subject checkpoint references. |
| `reports/tiny-awq/<artifact-lane>/external_edit_summary.json` | AWQ quantization metadata and checkpoint file hashes. |

A successful run ends with the shared completion block documented in
`examples/integrations/_shared/README.md#expected-run-output`. If a run fails,
check the prerequisite message first, then inspect
`reports/tiny-awq/<artifact-lane>/run_command.txt`.

The helper fails if CUDA is unavailable, if GPTQModel does not expose a
quantized checkpoint configuration, or if the subject cannot be loaded back
through the Transformers AWQ loader with the selected backend.

`backend_inventory.json` is emitted by InvarLock report persistence when adapter
provenance is available; the shell runner does not write that sidecar directly.
