# GPTQModel Integration Example

Status: `runnable`. A `cuda-container-strict` result requires independent
acceptance inputs and the successful current run described below.

This example shows how to attach InvarLock regression evidence to a checkpoint
quantized by GPTQModel. It creates a deterministic tiny Llama-style Hugging Face
baseline, quantizes that checkpoint with GPTQModel, and compares the quantized
subject through InvarLock's `hf_gptq` adapter.

The example keeps GPTQModel in the example environment rather than the core
InvarLock install.

## Prerequisites

Install InvarLock with the GPTQ optional stack in the same example environment:

```bash
python -m pip install "invarlock[gptq]"
```

From a repository checkout, `uv` can provide the optional stack for the run:

```bash
uv run --extra gptq python - <<'PY'
from invarlock.gptqmodel_runtime import require_gptqmodel_runtime

require_gptqmodel_runtime()
PY
```

## Run

### Lane Support

| Artifact lane label | Command shape | Notes |
| --- | --- | --- |
| `cuda-container-strict` | `--lane cuda` | Primary evidence path with the example-specific GPTQModel/AWQ image. |
| `cuda-host-off` | `--lane host --device cuda` | Secondary host CUDA comparison path; requires host CUDA plus the same GPTQModel/Triton prerequisites. |
| `cpu-host-off` | `--lane host --device cpu` | Secondary local bring-up path when the host GPTQModel/Triton prerequisites support CPU. |

Host lanes run prerequisite preflight before model materialization and
evaluation. CUDA host lanes require GPTQModel's JIT prerequisites: a C++
compiler, `ninja`, matching Python development headers, and a CUDA toolkit with
`nvcc`. CPU host bring-up performs only the named GPTQModel import check.

### cuda-container-strict lane

Build and check the example-specific GPTQModel/AWQ image, then run this lane on a
CUDA host with that image configured:

```bash
examples/integrations/_runtime_images/build_example_runtime_image.sh cuda-gptqmodel
examples/integrations/_runtime_images/smoke_example_runtime_image.sh cuda-gptqmodel

TRUSTED_RUNTIME_IMAGE_DIGEST='sha256:REPLACE_WITH_REVIEWED_64_HEX_DIGEST'
INVARLOCK_ACCEPTANCE_BASELINE_REPORT=/path/to/raw-baseline-report.json \
INVARLOCK_ACCEPTANCE_POLICY_PACK=/path/to/acceptance-policy-pack.json \
INVARLOCK_RUNTIME_IMAGE=invarlock-example-runtime:cuda-gptqmodel \
INVARLOCK_EXPECTED_RUNTIME_IMAGE_DIGEST="$TRUSTED_RUNTIME_IMAGE_DIGEST" \
uv run --extra gptq \
  examples/integrations/gptqmodel/run_tiny_gptqmodel.sh \
  --allow-network \
  --force \
  --lane cuda
```

Obtain the trusted digest independently from reviewed build/release policy.
The matching digest in `runtime.manifest.json` is a manifest claim, not the source of
the verifier pin.

This strict lane is scoped to the configured tiny GPTQ checkpoint and runtime
image. Rerun the strict lane for the target runtime before using the result as
shared integration evidence.

### cpu-host-off lane

From the repository root:

```bash
uv run --extra gptq \
  examples/integrations/gptqmodel/run_tiny_gptqmodel.sh \
  --allow-network \
  --force \
  --lane host \
  --device cpu
```

The default path uses `--execution-mode host --assurance off` because GPTQModel
runtime loading is platform-dependent. Use this lane for local dependency
setup; non-CUDA execution depends on the installed GPTQModel backend. In
host mode the runner sets `TORCHDYNAMO_DISABLE=1` unless you already set it,
which avoids platform-local Torch compile failures during tiny compatibility runs. It
still runs the InvarLock evaluator, verifier, backend inventory, and HTML
renderer, but its output is diagnostic only and cannot serve as strict or
release evidence.

For `cuda-host-off` evaluation, use the same command with `--device cuda`.

## Evidence Boundary

The subject checkpoint is materialized before the InvarLock comparison. The
strict lane covers the configured baseline-vs-subject evaluation, `hf_gptq`
adapter load, guard evidence, runtime manifest, and verifier result for that
produced subject. The GPTQModel materialization step is represented by
`external_edit_summary.json` and checkpoint hashes.

## Outputs

The runner writes generated outputs under local output directories:

| Path | Role |
| --- | --- |
| `models/tiny-llama-baseline/` | Deterministic tiny HF baseline checkpoint. |
| `models/tiny-llama-gptq-4bit/` | GPTQModel-quantized subject checkpoint. |
| `artifacts/tiny-gptqmodel-fixture/tiny_causal_text.jsonl` | Deterministic local text fixture for evaluation and calibration. |
| `artifacts/tiny-gptqmodel-fixture/preset.yaml` | Generated preset pointing at the local fixture. |
| `artifacts/tiny-gptqmodel-fixture/fixture_summary.json` | Fixture parameters and file hashes. |
| `reports/tiny-gptqmodel/<artifact-lane>/evaluation.report.json` | Canonical verifier input. |
| `reports/tiny-gptqmodel/<artifact-lane>/verify.json` | Machine-readable verifier result. |
| `reports/tiny-gptqmodel/<artifact-lane>/evaluation.html` | Human-readable report. |
| `reports/tiny-gptqmodel/<artifact-lane>/backend_inventory.json` | GPTQModel backend version and quantized module inventory when exposed. |
| `reports/tiny-gptqmodel/<artifact-lane>/runtime_quantization_proof.json` | Strict-lane v1 process receipt listing recognized GPTQ runtime types; wrapper-side schema checks are not an independent runtime observation or checkpoint-artifact proof. |
| `reports/tiny-gptqmodel/<artifact-lane>/lane_artifact.json` | Canonical artifact-lane label and effective runtime settings. |
| `reports/tiny-gptqmodel/<artifact-lane>/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-gptqmodel/<artifact-lane>/run_summary.txt` | Concise success or failure status, lane label, verifier status, runtime provenance status, and primary output paths. |
| `reports/tiny-gptqmodel/<artifact-lane>/checkpoint_refs.json` | Baseline and subject checkpoint references. |
| `reports/tiny-gptqmodel/<artifact-lane>/external_edit_summary.json` | GPTQModel quantization metadata and checkpoint file hashes. |

A successful run ends with the shared completion block documented in
`examples/integrations/_shared/README.md#expected-run-output`. If a run fails,
check the prerequisite message first, then inspect
`reports/tiny-gptqmodel/<artifact-lane>/run_command.txt`.

The helper fails if GPTQModel does not produce a quantized checkpoint
configuration or if the subject cannot be loaded back through GPTQModel.

The shell runner relies on InvarLock report persistence to emit
`backend_inventory.json` when adapter provenance is available and, for
`cuda-container-strict`, requires `runtime_quantization_proof.json`. The shared
wrapper validates the receipt's v1 schema, selected adapter/backend binding,
allowed type-name surface, and matching backend inventory before `verify`.
Those checks validate sidecars written by the evaluated process; they are not
an independent runtime observation or checkpoint-artifact proof.
