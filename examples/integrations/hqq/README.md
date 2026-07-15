# HQQ Runtime Integration Example

Status: `runnable`. A `cuda-container-strict` result requires independent
acceptance inputs and the successful current run described below.
`cuda-host-off` and `cpu-host-off`
lanes are supported.

This example shows how to attach InvarLock regression evidence to a Hugging
Face causal checkpoint loaded through InvarLock's `hf_hqq` adapter. It creates a
tiny local Llama-style HF checkpoint, uses that checkpoint as the baseline, then
uses the same checkpoint as the subject loaded through `hf_hqq`, where HQQ
quantization is applied at adapter load time.

The example keeps HQQ in the example environment rather than the core
InvarLock install.

## Prerequisites

Install InvarLock with the HQQ optional stack in the same example environment:

```bash
python -m pip install "invarlock[hqq]"
```

From a repository checkout, `uv` can provide the optional stack for the run:

```bash
uv run --extra hqq python -c "import hqq"
```

## Run

### Lane Support

| Artifact lane label | Command shape | Notes |
| --- | --- | --- |
| `cuda-container-strict` | `--lane cuda` | Primary evidence path with the example-specific HQQ image. |
| `cuda-host-off` | `--lane host --device cuda` | Secondary local CUDA comparison path without strict container evidence. |
| `cpu-host-off` | `--lane host --device cpu` | Secondary local non-CUDA bring-up when the installed HQQ backend supports CPU. |

Host lanes run prerequisite preflight before fixture preparation and
evaluation. The `cuda-host-off` lane checks `torch.cuda.is_available()` before
the backend run.

### cuda-container-strict lane

Build and check the example-specific HQQ image, then run this lane on a CUDA host
with that image configured:

```bash
examples/integrations/_runtime_images/build_example_runtime_image.sh cuda-hqq
examples/integrations/_runtime_images/smoke_example_runtime_image.sh cuda-hqq

TRUSTED_RUNTIME_IMAGE_DIGEST='sha256:REPLACE_WITH_REVIEWED_64_HEX_DIGEST'
INVARLOCK_ACCEPTANCE_BASELINE_REPORT=/path/to/raw-baseline-report.json \
INVARLOCK_ACCEPTANCE_POLICY_PACK=/path/to/acceptance-policy-pack.json \
INVARLOCK_RUNTIME_IMAGE=invarlock-example-runtime:cuda-hqq \
INVARLOCK_EXPECTED_RUNTIME_IMAGE_DIGEST="$TRUSTED_RUNTIME_IMAGE_DIGEST" \
uv run --extra hqq \
  examples/integrations/hqq/run_tiny_hf_hqq.sh \
  --allow-network \
  --force \
  --lane cuda
```

Obtain the trusted digest independently from reviewed build/release policy.
The matching digest in `runtime.manifest.json` is a manifest claim, not the source of
the verifier pin.
This strict lane is scoped to the configured tiny `hf_hqq` runtime-load subject
and image. Rerun the strict lane for the target runtime before using the
artifact as shared integration evidence.

### cpu-host-off lane

From the repository root:

```bash
uv run --extra hqq \
  examples/integrations/hqq/run_tiny_hf_hqq.sh \
  --allow-network \
  --force \
  --lane host \
  --device cpu
```

Use this lane for local dependency setup and non-CUDA compatibility runs when the
installed HQQ backend supports the selected host.

For `cuda-host-off` evaluation, use the same command with `--device cuda`.

## Outputs

The runner writes generated outputs under local output directories:

| Path | Role |
| --- | --- |
| `models/tiny-llama-hf-hqq-baseline/` | Deterministic tiny HF checkpoint used by both sides of the comparison. |
| `artifacts/tiny-hf-hqq/tiny_causal_text.jsonl` | Deterministic local text fixture for evaluation. |
| `artifacts/tiny-hf-hqq/preset.yaml` | Generated preset pointing at the local fixture. |
| `artifacts/tiny-hf-hqq/fixture_summary.json` | Fixture parameters and file hashes. |
| `reports/tiny-hf-hqq/<artifact-lane>/evaluation.report.json` | Canonical verifier input. |
| `reports/tiny-hf-hqq/<artifact-lane>/verify.json` | Machine-readable verifier result. |
| `reports/tiny-hf-hqq/<artifact-lane>/evaluation.html` | Human-readable report. |
| `reports/tiny-hf-hqq/<artifact-lane>/backend_inventory.json` | HQQ backend version and quantized module inventory when exposed. |
| `reports/tiny-hf-hqq/<artifact-lane>/runtime_quantization_proof.json` | Strict-lane v1 process receipt listing recognized HQQ runtime types; wrapper-side schema checks are not an independent runtime observation or checkpoint-artifact proof. |
| `reports/tiny-hf-hqq/<artifact-lane>/lane_artifact.json` | Canonical artifact-lane label and effective runtime settings. |
| `reports/tiny-hf-hqq/<artifact-lane>/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-hf-hqq/<artifact-lane>/run_summary.txt` | Concise success or failure status, lane label, verifier status, runtime provenance status, and primary output paths. |
| `reports/tiny-hf-hqq/<artifact-lane>/checkpoint_refs.json` | Baseline and subject checkpoint references. |
| `reports/tiny-hf-hqq/<artifact-lane>/adapter_runtime_summary.json` | `hf_hqq` runtime adapter metadata, quantization settings, and file hashes. |

A successful run ends with the shared completion block documented in
`examples/integrations/_shared/README.md#expected-run-output`. If a run fails,
check the prerequisite message first, then inspect
`reports/tiny-hf-hqq/<artifact-lane>/run_command.txt`.

The example uses native HQQ runtime quantization after loading the HF checkpoint,
so the subject remains an HF-loadable checkpoint plus adapter runtime
configuration rather than an HQQ-lib-only checkpoint format.
The shell runner relies on InvarLock report persistence to emit
`backend_inventory.json` when adapter provenance is available and, for
`cuda-container-strict`, requires `runtime_quantization_proof.json`. The shared
wrapper validates the receipt's v1 schema, selected adapter/backend binding,
allowed type-name surface, and matching backend inventory before `verify`.
Those checks validate sidecars written by the evaluated process; they are not
an independent runtime observation or checkpoint-artifact proof.
