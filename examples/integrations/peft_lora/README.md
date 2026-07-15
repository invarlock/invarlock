# PEFT LoRA-Merge Integration Example

Status: `runnable`. A `cuda-container-strict` result requires independent
acceptance inputs and the successful current run described below.

This example executes the repository's immutable tiny PEFT LoRA training
profile, serializes and reloads the trained adapter, merges it into a
HF-loadable subject, independently recomputes the artifact evidence, and then
compares that subject against its pinned baseline revision.

The example keeps PEFT in the example environment rather than the core
InvarLock install.

## Prerequisites

Install InvarLock with the training dependencies:

```bash
python -m pip install "invarlock[training]"
```

From a repository checkout, an existing `.venv` with `invarlock[hf]` is also
fine:

```bash
uv sync --extra training
```

From a source checkout, keep the optional dependency scoped to the command:

```bash
uv run --extra training \
  examples/integrations/peft_lora/run_tiny_peft_lora.sh --help
```

## Run

### Lane Support

| Artifact lane label | Command shape | Notes |
| --- | --- | --- |
| `cuda-container-strict` | `--lane cuda` | Primary evidence path with the standard InvarLock CUDA runtime image. |
| `cuda-host-off` | `--lane host --device cuda` | Secondary local CUDA comparison path without strict container evidence. |
| `cpu-host-off` | `--lane host --device cpu` | Secondary local non-CUDA bring-up for the merged dense checkpoint. |

The selected training profile runs in the host Python environment before
container or host evaluation. CUDA training profiles fail before training when
that environment does not expose `torch.cuda`.

### cuda-container-strict lane

Build the standard InvarLock CUDA runtime image, then run this lane on a CUDA host with
that image configured. This example evaluates a merged dense checkpoint, so it
does not need the quant example images.

```bash
make runtime-image-cuda

TRUSTED_RUNTIME_IMAGE_DIGEST='sha256:REPLACE_WITH_REVIEWED_64_HEX_DIGEST'
INVARLOCK_ACCEPTANCE_BASELINE_REPORT=/path/to/raw-baseline-report.json \
INVARLOCK_ACCEPTANCE_POLICY_PACK=/path/to/acceptance-policy-pack.json \
INVARLOCK_RUNTIME_IMAGE=invarlock-runtime:cuda-local \
INVARLOCK_EXPECTED_RUNTIME_IMAGE_DIGEST="$TRUSTED_RUNTIME_IMAGE_DIGEST" \
uv run --extra training \
examples/integrations/peft_lora/run_tiny_peft_lora.sh \
  --allow-network \
  --force \
  --lane cuda
```

The runner defaults to the `release` profile so the strict verification path has
enough evaluation tokens for a stable primary-metric verdict.
Obtain the trusted digest independently from reviewed build/release policy.
The matching digest in `runtime.manifest.json` is a manifest claim, not the source of
the verifier pin.
This strict lane is scoped to the configured tiny merged dense checkpoint and
runtime image. Rerun the strict lane for the target runtime before using the
artifact as shared integration evidence.

### cpu-host-off lane

From the repository root:

```bash
uv run --extra training \
examples/integrations/peft_lora/run_tiny_peft_lora.sh \
  --allow-network \
  --force \
  --lane host \
  --device cpu
```

Use this lane for local dependency setup and non-CUDA compatibility runs.

For `cuda-host-off` evaluation, use the same command with `--device cuda`.

## Evidence Boundary

The runner invokes its immutable profile before the InvarLock comparison. The
strict lane covers the configured baseline-vs-subject evaluation, `hf_causal`
adapter load, guard evidence, runtime manifest, and verifier result for that
produced subject. `training_receipt.json` binds the immutable profile, pinned
training data, baseline state, serialized adapter, merged subject, and measured
deltas. `training_evidence_proof.json` verifies the receipt-bound saved state
and reload behavior for that subject; its backend label is a profile-specific
constrained runtime declaration, not independent proof of training execution. The
profile pins the exact Python version, exact Torch build string
(including the CUDA local-version suffix for CUDA profiles), and exact
Transformers and PEFT versions. CUDA driver and host-OS identity remain observed
provenance rather than independent trust anchors. Artifact recomputation does
not independently prove optimizer-history execution; trusted execution or an
independent rerun is still required.

## Outputs

The runner writes generated outputs under local output directories:

| Path | Role |
| --- | --- |
| `models/tiny-gpt2-peft-lora-merged/` | HF-loadable merged subject checkpoint. |
| `models/tiny-gpt2-peft-lora-merged/training_receipt.json` | Immutable profile and recomputable training-artifact evidence. |
| `artifacts/tiny-peft-lora-fixture/tiny_causal_text.jsonl` | Deterministic local text fixture for evaluation. |
| `artifacts/tiny-peft-lora-fixture/preset.yaml` | Generated preset pointing at the local fixture. |
| `artifacts/tiny-peft-lora-fixture/fixture_summary.json` | Fixture parameters and file hashes. |
| `reports/tiny-peft-lora/<artifact-lane>/evaluation.report.json` | Canonical verifier input. |
| `reports/tiny-peft-lora/<artifact-lane>/verify.json` | Machine-readable verifier result. |
| `reports/tiny-peft-lora/<artifact-lane>/evaluation.html` | Human-readable report. |
| `reports/tiny-peft-lora/<artifact-lane>/lane_artifact.json` | Canonical artifact-lane label and effective runtime settings. |
| `reports/tiny-peft-lora/<artifact-lane>/run_command.txt` | Wrapper, evaluate, verify, and render commands. |
| `reports/tiny-peft-lora/<artifact-lane>/run_summary.txt` | Concise success or failure status, lane label, verifier status, runtime provenance status, and primary output paths. |
| `reports/tiny-peft-lora/<artifact-lane>/training_receipt.json` | Copy of the verified training receipt used for the lane. |
| `reports/tiny-peft-lora/<artifact-lane>/training_binding.json` | Post-evaluation binding of the subject tree to the copied training receipt. |
| `reports/tiny-peft-lora/<artifact-lane>/training_evidence_proof.json` | Receipt-bound artifact replay and reload proof for the evaluated subject. |
| `reports/tiny-peft-lora/<artifact-lane>/training_profile_snapshot.json` | Immutable reviewed training profile and explicit `attn` validation scope. |

A successful run ends with the shared completion block documented in
`examples/integrations/_shared/README.md#expected-run-output`. If a run fails,
check the prerequisite message first, then inspect
`reports/tiny-peft-lora/<artifact-lane>/run_command.txt`.

The local profile requires forward/backward/AdamW steps, LoRA-only
trainable parameters, a changed serialized adapter, a pristine frozen base,
and a merged checkpoint whose recomputed delta matches the receipt. The
artifact-replay verifier confirms the resulting state and reload behavior, not
the recorded optimizer execution itself.
