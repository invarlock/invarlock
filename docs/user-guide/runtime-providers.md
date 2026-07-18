# Runtime providers

The runtime-provider ABI connects artifact-specific identity and execution to
one common evidence transaction. A provider authenticates an artifact,
prepares a session from caller-owned resources, scores the ordered schedule,
and emits typed records. InvarLock owns pairing, metric arithmetic, policy
evaluation, bundle publication, and independent verification.

!!! tip "User guide"

    **Outcome:** Select, inspect, and configure a provider so both artifacts
    emit contract-complete records for one independently verifiable comparison.

    **Audience:** Runtime integration engineers and evaluation operators working
    with Hugging Face text, Hugging Face vision-text, GGUF/`llama.cpp`,
    TensorRT-LLM, or complete imported provider material.

    **Prerequisites:** Materialized artifacts and support files, a canonical
    schedule and caller-approved policy, a provider-compatible offline environment,
    and caller-owned runtime-image and device configuration.

## Choose a provider path

| Artifact you need to evaluate | Provider | Packaging | Start with |
| --- | --- | --- | --- |
| Local Hugging Face safetensors checkpoint | `hf_transformers` | Built into `invarlock`; execution dependencies in `invarlock[hf]` | Run mode |
| Local Hugging Face vision-text checkpoint | `hf_vision_text` | `invarlock-runtime-hf-vision-text` add-in | Run mode with authenticated image content store |
| GGUF file used by `llama.cpp` | `llama_cpp` | `invarlock-runtime-gguf` add-in | Run mode after runtime inspection |
| TensorRT-LLM engine bundle | `tensorrt_llm` | `invarlock-runtime-tensorrt-llm` add-in | Run mode after GPU-bound inspection |
| Complete provider sidecars produced elsewhere | Provider named by the sidecars | Matching provider package must be installed | Import mode |

The text integrations declare `text_causal`; the vision-text add-in declares
`vision_text_generation`. All shipped integrations declare `exact_match`. The
built-in `hf_transformers` integration also declares
`normalized_nll_per_utf8_byte`, using teacher-forced expected-continuation
likelihoods over the authenticated local checkpoint. When authenticated
tokenizer contracts and paired token counts are comparable, the verifier may
render a token-weighted perplexity ratio as interpretation only. The GGUF,
TensorRT-LLM, and vision-text add-ins remain exact-match only.

| Metric | Hugging Face text | Vision-text | GGUF | TensorRT-LLM | Use when |
| --- | --- | --- | --- | --- | --- |
| `exact_match` | Yes | Yes | Yes | Yes | Exact generated text is the release criterion |
| `normalized_nll_per_utf8_byte` | Yes | No | No | No | Expected-continuation likelihood regression is the release criterion; tokenizers may differ |

## Common provider workflow

Every run-mode provider follows the same operator sequence:

1. Materialize the artifact and every support file before evaluation.
2. Pin the provider package, backend, and outer OCI image.
3. Inspect the exact artifact and backend to derive request settings; do not
   transcribe digests by hand.
4. Put artifact and support resources beneath caller-owned, non-symbolic roots.
5. Build one request whose baseline and subject use the same schedule and a
   provider-supported built-in or collection metric.
6. Run in the digest-pinned, network-disabled container.
7. Verify the resulting bundle with runtime digests obtained from independently maintained
   configuration, not copied from the bundle.

Provider conformance and evidence verification answer different questions:

- a conformance command checks package discovery, ABI compatibility, and
  lightweight provider behavior;
- an evaluation authenticates one concrete artifact and execution record; and
- `invarlock verify` replays one concrete bundle under independent anchors.

A conformance pass is therefore a prerequisite, not evidence that a model or
engine ran successfully.

### Qualification host environment

The maintained qualification targets use two Python variables with different
roles. `QUALIFICATION_DRIVER_PYTHON` launches the orchestration driver;
`PYTHON` runs provider discovery, preflight, `evaluate`, `verify`, and
`report`. The environment selected by `PYTHON` must contain the
matching-version InvarLock core wheel and the exact wheel for every optional
provider used by the request. Install those wheels into that interpreter, for
example with `"$PYTHON" -m pip install /path/to/wheel.whl`.

```bash
export PYTHON=/path/to/qualification-venv/bin/python
export QUALIFICATION_DRIVER_PYTHON="$PYTHON"
```

Adding package source directories to `PYTHONPATH` is not an installation and
is not sufficient. First-party provider authorization checks installed
distribution and `invarlock.runtime_providers` entry-point metadata in addition
to importing provider code. Using one isolated environment for `PYTHON` and
`QUALIFICATION_DRIVER_PYTHON` avoids interpreter drift.

Use the wrapper maintained beside the selected provider. Optional-provider
wrappers add the installed entry-point check and required provider resources;
do not substitute the root wrapper for them.

Ordinary `invarlock evaluate` receives its immutable image reference and digest
directly through CLI options or the embedding API. The first-party optional
provider Makefile wrappers expose a mutable `IMAGE` build/smoke handle, but
their signed `qualify-*` targets pass `QUALIFICATION_IMAGE`. That value defaults
to `IMAGE_DIGEST`; override it with an exact `repository@sha256:...` reference
when the qualification must remain fetchable across hosts. A mutable tag is
never a strict qualification identity.

| Provider | Signed canary | Readiness | Evidence |
| --- | --- | --- | --- |
| `hf_transformers` | `make runtime-qualification-canary` | `make runtime-qualification-readiness` | `make runtime-qualification-evidence` |
| `hf_vision_text` | `make -C addins/multimodal qualify-canary` | `make -C addins/multimodal qualify-preflight` | `make -C addins/multimodal qualify-evidence` |
| `llama_cpp` | `make -C addins/gguf qualify-canary` | `make -C addins/gguf qualify-preflight` | `make -C addins/gguf qualify-evidence` |
| `tensorrt_llm` | `make -C addins/tensorrt_llm qualify-canary` | `make -C addins/tensorrt_llm qualify-preflight` | `make -C addins/tensorrt_llm qualify-evidence` |

All four paths accept the same resource controls:
`QUALIFICATION_DEVICE`, `QUALIFICATION_CPUS`,
`QUALIFICATION_MEMORY_MIB`, and `QUALIFICATION_USER`. Set them once and keep
them unchanged across canary, readiness, and evidence. The root wrapper
requires an explicit device. The add-in wrappers have provider-appropriate
device defaults, but an explicit `cpu` or `cuda:<index>` keeps the execution
contract reviewable. Vision-text additionally requires `RESOURCE_ROOT` and
`CONTENT_STORE`; GGUF and TensorRT-LLM use the provider resource variables
documented in their sections below.

## Hugging Face Transformers

`hf_transformers` is the canonical built-in provider. Install its runtime
dependencies with:

```bash
python -m pip install "invarlock[hf]"
```

Strict run mode expects locally materialized safetensors and tokenizer files.
It binds the immutable revision or checkpoint-tree digest, tokenizer metadata
digest, deterministic scalar settings, backend and device facts, and the outer
runtime image digest.

### Prepare local material

Resolve any Hub revision to an immutable commit before the transaction, then
make the snapshot available locally. Hugging Face documents both
[`local_files_only=True` and offline mode](https://huggingface.co/docs/transformers/installation#offline-mode).
InvarLock additionally disables remote model code in its strict provider path.

A run-mode request side includes a request-relative artifact path:

```yaml
artifact:
  path: artifacts/baseline
  model_id: org/baseline
  locator: hf://org/baseline@0123456789abcdef0123456789abcdef01234567
runtime:
  provider: hf_transformers
  settings:
    batch_size: 1
    checkpoint_tree_sha256: PINNED_64_HEX_DIGEST
    context_length: 2048
    immutable_revision: 0123456789abcdef0123456789abcdef01234567
    max_output_tokens: 64
    offline: true
    seed: 0
    timeout_seconds: 300
    tokenizer_metadata_sha256: PINNED_64_HEX_DIGEST
```

Derive both fields from the exact local checkpoint with the stable engine
helpers. Run this in the same pinned HF environment used for evaluation:

```python
import json
from pathlib import Path

from transformers import AutoTokenizer

from invarlock.engine import (
    checkpoint_tree_sha256,
    hf_tokenizer_contract_sha256,
)

checkpoint = Path("artifacts/baseline").resolve()
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    local_files_only=True,
    trust_remote_code=False,
)
print(
    json.dumps(
        {
            "checkpoint_tree_sha256": checkpoint_tree_sha256(
                checkpoint
            ).removeprefix("sha256:"),
            "tokenizer_metadata_sha256": hf_tokenizer_contract_sha256(
                tokenizer
            ),
        },
        indent=2,
        sort_keys=True,
    )
)
```

Repeat for the subject and copy the printed bare lowercase digests into the
matching request side. `checkpoint_tree_sha256` rejects links, special files,
tree changes during the scan, and unsafe path transitions. The tokenizer digest
binds vocabulary, added vocabulary, special tokens, tokenizer backend contract,
chat template, padding/truncation sides, and stable tokenizer settings. A digest
mismatch is a reason to stop and inspect the material, not to update the
request until it passes.

The strict loader accepts canonical single-file or indexed SafeTensors layouts.
For an indexed checkpoint, every declared shard must be present, every storage
key must map consistently, no undeclared shard may be substituted, and the
loaded tensor/config binding is checked again before and after scoring. Pickled
model weights, remote code, symbolic links, and a tokenizer that cannot expose
a stable contract fail closed.

### Build or obtain the runtime image

The repository maintains separate CPU and x86_64 CUDA images. Both include the
core wheel plus a provider-matched pinned Hugging Face runtime closure:

```bash
SOURCE_COMMIT="$(git rev-parse HEAD)"
mkdir -p artifacts/source
SOURCE_BUNDLE="$PWD/artifacts/source/invarlock-$SOURCE_COMMIT.tar"
python scripts/qualification_source.py create \
  --commit "$SOURCE_COMMIT" --output "$SOURCE_BUNDLE" \
  > "$SOURCE_BUNDLE.json"
SOURCE_BUNDLE_SHA256="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_bundle_sha256"])' \
    "$SOURCE_BUNDLE.json"
)"
mkdir -p artifacts/runtime-build

RUNTIME_SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)" \
  make runtime-image \
    RUNTIME_SOURCE_COMMIT="$SOURCE_COMMIT" \
    RUNTIME_SOURCE_BUNDLE="$SOURCE_BUNDLE" \
    RUNTIME_SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
    RUNTIME_BUILD_STATEMENT="$PWD/artifacts/runtime-build/cpu.json"
make runtime-smoke

RUNTIME_SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)" \
  make runtime-image-cuda \
    RUNTIME_SOURCE_COMMIT="$SOURCE_COMMIT" \
    RUNTIME_SOURCE_BUNDLE="$SOURCE_BUNDLE" \
    RUNTIME_SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
    RUNTIME_BUILD_STATEMENT="$PWD/artifacts/runtime-build/cuda.json"
make runtime-smoke-cuda
```

The CPU target uses `runtime/Dockerfile`; the NVIDIA CUDA 12.8 target uses
`runtime/Dockerfile.cuda`. A CUDA device flag only exposes a GPU to a
CUDA-capable image.

Each build statement records the authenticated source identity, Dockerfile
digest, normalized build arguments, requested platform and base image, and
final immutable image ID. Retain it beside the image digest used by
qualification.

The final image ID is a local OCI config identity. It can be used directly for
local inspection, smoke, and qualification, but it is not a valid Dockerfile
`FROM` reference. A layered runtime such as the vision-text add-in requires a
named `repository@sha256:...` manifest reference for its base; publish or obtain
that exact manifest through the operator's registry process before building the
layer.

The default local tag is a mutable build handle. Strict execution accepts an
exact content-addressed local image ID or `repository@sha256:...` reference and
requires the same digest through `INVARLOCK_RUNTIME_IMAGE_DIGEST`. For portable
public evidence, publish or obtain the pinned image through the operator's
registry process and record its registry manifest reference; a local image ID
identifies local bytes but does not give another verifier a fetchable runtime.

Select exactly one image and resolve it before qualification. For a local CPU
image:

```bash
IMAGE_TAG=invarlock-runtime:local
export QUALIFICATION_DEVICE=cpu
export QUALIFICATION_CPUS=8
export QUALIFICATION_MEMORY_MIB=65536
export QUALIFICATION_USER=65532:65532
IMAGE="$(docker image inspect --format '{{.Id}}' "$IMAGE_TAG")"
DIGEST="$IMAGE"
```

For the local CUDA image, use
`IMAGE_TAG=invarlock-runtime:hf-cuda-local` and an explicit device such as
`export QUALIFICATION_DEVICE=cuda:0`. For a portable registry image, set
`IMAGE=registry.example/invarlock-runtime@sha256:PINNED_DIGEST` and
`DIGEST="${IMAGE##*@}"`. Keep the selected `IMAGE`, `DIGEST`, and
`QUALIFICATION_DEVICE` unchanged through all three qualification stages.

Invoke `evaluate` on the host with the pinned image and device. The CLI creates
the closed Docker or Podman boundary and re-enters the transaction inside it:

```bash
invarlock evaluate request.yaml \
  --signing-key ../private/evidence-signer.pem \
  --baseline-runtime-image 'registry.example/invarlock-runtime-cuda@sha256:BASELINE_DIGEST' \
  --baseline-runtime-image-digest 'sha256:BASELINE_DIGEST' \
  --subject-runtime-image 'registry.example/invarlock-runtime-cuda@sha256:SUBJECT_DIGEST' \
  --subject-runtime-image-digest 'sha256:SUBJECT_DIGEST' \
  --baseline-runtime-device cuda:0 \
  --subject-runtime-device cuda:1
```

Each image reference and digest must agree. Shared image, digest, device, and
entrypoint options remain convenient defaults when both sides use the same
configuration. The OCI project defines digests as content-addressed descriptor identities in the
[OCI Image Format](https://github.com/opencontainers/image-spec). Digest
agreement identifies the declared image; it does not attest that a trusted host
executed it.

The host prepares the authenticated schedule, then launches one local worker
per side with `--pull=never`, disabled networking, a read-only container root,
dropped Linux capabilities, side-specific read-only inputs, and an isolated
writable output. The host validates the outputs, publishes the no-clobber
evidence destination, and signs it with a key never exposed to model workers.
Workers sharing a generic or identical CUDA selector run sequentially;
explicitly different CUDA indexes can run in parallel. Strict mode also
observes container markers and rejects network, remote-code, or
arbitrary-provider enablement. Embedding applications that call the Python
transaction directly remain responsible for establishing an equivalent runtime
boundary.

Before running a group of qualifications, bootstrap one small, representative
signed canary through the exact runtime image. This is a real
`evaluate -> verify` transaction, not an execution-free check:

```bash
make dist-check
install -d -m 700 "$PWD/qualification"
python scripts/qualification_candidate_wheels.py \
  --wheel dist/invarlock-*.whl \
  --output "$PWD/qualification/candidate-wheels.json"
CANDIDATE_WHEEL_MANIFEST="$PWD/qualification/candidate-wheels.json"

CANARY_EVIDENCE="$PWD/canary/evidence"
CANARY_RECEIPT="$PWD/canary/verification-receipt.json"
CANARY_TRUST_PROFILE="$PWD/canary/trust-inputs.json"
mkdir -p "$PWD/canary" "$PWD/output"

make runtime-qualification-canary \
  REQUEST="$PWD/canary/request.yaml" \
  SIGNING_KEY="$PWD/private/evidence.key" \
  IMAGE="$IMAGE" IMAGE_DIGEST="$DIGEST" \
  EVIDENCE="$CANARY_EVIDENCE" \
  TRUST_PROFILE="$CANARY_TRUST_PROFILE" \
  RECEIPT="$CANARY_RECEIPT" \
  SUMMARY="$PWD/canary/qualification-summary.json" \
  QUALIFICATION_DEVICE="$QUALIFICATION_DEVICE" \
  SOURCE_COMMIT="$SOURCE_COMMIT" SOURCE_BUNDLE="$SOURCE_BUNDLE" \
  SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
  CANDIDATE_WHEEL_MANIFEST="$CANDIDATE_WHEEL_MANIFEST"
```

Retain the canary evidence, its strictly verified signed receipt, and the
verifier-owned trust profile together. They may be reused by later readiness
and evidence targets only while `IMAGE_DIGEST` is unchanged. A new image digest
requires a new signed canary.

Before destroying the verifier signing key, export its Ed25519 public key into
the retained trust unit. The maintained checker can then replay the signed
receipt after transfer without private material:

```bash
python scripts/qualification_receipt_check.py \
  --receipt "$CANARY_RECEIPT" \
  --evidence "$CANARY_EVIDENCE" \
  --trust-profile "$CANARY_TRUST_PROFILE" \
  --verifier-public-key "$PWD/canary/verifier-public.pem"
```

The supplied public key is authenticated by the verifier fingerprint in the
signed receipt; a substituted key, changed profile, policy, anchor, receipt, or
evidence manifest fails closed. The trust-profile digest remains the digest of
the profile used when the receipt was created.

Next, run readiness for each planned request with the same source binding,
image, verifier inputs, fresh destinations, and retained canary inputs:

```bash
make runtime-qualification-readiness \
  REQUEST="$PWD/request.yaml" SIGNING_KEY="$PWD/private/evidence.key" \
  IMAGE="$IMAGE" IMAGE_DIGEST="$DIGEST" \
  EVIDENCE="$PWD/output/evidence" \
  TRUST_PROFILE="$PWD/trust/trust-inputs.json" \
  RECEIPT="$PWD/output/verification-receipt.json" \
  CANARY_EVIDENCE="$CANARY_EVIDENCE" \
  CANARY_RECEIPT="$CANARY_RECEIPT" \
  CANARY_TRUST_PROFILE="$CANARY_TRUST_PROFILE" \
  QUALIFICATION_DEVICE="$QUALIFICATION_DEVICE" \
  SOURCE_COMMIT="$SOURCE_COMMIT" SOURCE_BUNDLE="$SOURCE_BUNDLE" \
  SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
  CANDIDATE_WHEEL_MANIFEST="$CANDIDATE_WHEEL_MANIFEST"
```

`REQUEST`, `SIGNING_KEY`, `SOURCE_BUNDLE`, and `TRUST_PROFILE` may be in
read-only directories. The evidence, receipt, optional report, and summary
must be distinct absent paths with existing real parent directories. They may
not use symlinked parents or be placed inside the evidence destination. The
source archive is limited to 512 MiB and 50,000 members; an execution-source
file is limited to 32 MiB. Readiness starts no model worker and publishes no
result. It also strictly reverifies the retained canary and its exact image
binding. Use the same variables with `make runtime-qualification-evidence`
plus `SUMMARY` and optional `REPORT` only after readiness succeeds. Carry the
same `QUALIFICATION_DEVICE`, and any CPU, memory, or user controls, into that
command.

`CANDIDATE_WHEEL_MANIFEST` is a strict
`invarlock/qualification-candidate-wheels-v1` JSON object whose `wheels` array
contains an absolute or manifest-relative `path` and expected `sha256` for the
core wheel and any maintained add-in wheels used by the request. Qualification
matches their package sources to the authenticated Git archive, extracts them
privately, and executes them with an isolated interpreter bootstrap. Dependencies
may be installed in the selected interpreter, but an older installed InvarLock
core or maintained add-in is not used as the qualification candidate.

The signed canary prevents image-wide fan-out after a basic image, provider, or
end-to-end transaction failure. It does not establish that every model fits
memory, loads under that provider, supports its tokenizer or processor, or
completes successfully. Readiness remains request-specific configuration
qualification, and each model run can still fail closed.

## Hugging Face vision-text

Install the optional provider into the interpreter used by the host
qualification transaction. Its base wheel includes the Pillow dependency
needed for execution-free media preflight:

```bash
"$PYTHON" -m pip install \
  /wheelhouse/Pillow-EXACT.whl \
  /wheelhouse/invarlock_runtime_hf_vision_text-EXACT.whl
"$PYTHON" -m invarlock_addins.multimodal.conformance
```

The heavier Torch and Transformers inference dependencies belong in the
digest-pinned runtime image. The `[runtime]` extra is for development that
executes the model outside that maintained image.

`hf_vision_text` evaluates `vision_text_generation` schedules containing one
`prompt` text part and one `image` content part per record. Schedule content
parts carry a canonical content ID, media type, positive byte length, and
SHA-256 rather than a host path. The provider resolves each ID as the exact
filename beneath the caller-authorized content store, opens it without following
links, and verifies file identity, length, digest, image format, frame count,
dimensions, and aggregate media limits before local inference.

Set the content resources outside `request.yaml` because they authorize host
material rather than express comparison intent:

```bash
export INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT=/pinned/vision-inputs
export INVARLOCK_HF_VISION_TEXT_CONTENT_STORE=images
```

Both bindings are required for host preflight and execution, and every selected
content object must already exist. Preflight uses the execution resolver and
Pillow to authenticate and safely decode the bytes without importing Torch or
Transformers, loading a model, or initializing CUDA. The side worker repeats
validation before model preparation, while the scorer reopens the content at
use time to detect later replacement.

Both request sides use `provider: hf_vision_text` and declare
`checkpoint_tree_sha256`, `tokenizer_metadata_sha256`,
`processor_metadata_sha256`, `batch_size: 1`, context and output limits, seed,
timeout, and `offline: true`. Derive the processor digest from the exact locally
loaded processor with `processor_contract_sha256`; the provider loads the
checkpoint and processor with local-files-only behavior, disabled remote code,
and safetensors.

Build the add-in image from the exact digest of the canonical CUDA image and
qualify it through `evaluate`, `verify`, and `report` on the target GPU. The
[add-in guide](https://github.com/invarlock/invarlock/blob/main/addins/multimodal/README.md)
contains the build, content-contract, and qualification commands.

## GGUF and `llama.cpp`

Install and check the optional provider:

```bash
"$PYTHON" -m pip install /wheelhouse/invarlock_runtime_gguf-EXACT.whl
"$PYTHON" -m invarlock_addins.gguf.conformance
```

The provider authenticates the GGUF file, its structured metadata and tensor
inventory, tokenizer metadata, a pinned `llama.cpp` executable, and a source
archive for the backend. The upstream
[GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
defines the file structure, while
[`llama.cpp`](https://github.com/ggml-org/llama.cpp) is the native executor.

Derive the request settings inside the exact runtime image:

```python
from pathlib import Path

from invarlock_addins.gguf.provider import LlamaCppProvider
from invarlock_addins.gguf.session import LlamaCppRuntimeBindings

bindings = LlamaCppRuntimeBindings(
    gguf_path=Path("model.gguf"),
    executable_path=Path("llama-cli"),
    source_archive_path=Path("llama.cpp-source.tar"),
)
spec = LlamaCppProvider().inspect_runtime_spec(
    bindings,
    seed=0,
    context_length=2048,
    batch_size=1,
    max_output_tokens=64,
    timeout_seconds=300,
)
print(spec.model_id)
print(spec.settings)
```

Use those exact values for the request side with `provider: llama_cpp`. At
evaluation time, map the same pinned files through caller-owned resources:

```bash
export INVARLOCK_GGUF_RESOURCE_ROOT=/pinned/gguf-runtime
export INVARLOCK_GGUF_BACKEND_EXECUTABLE=bin/llama-cli
export INVARLOCK_GGUF_BACKEND_SOURCE=source/llama.cpp-source.tar
export INVARLOCK_RUNTIME_IMAGE='registry.example/invarlock-gguf@sha256:PINNED_DIGEST'
export INVARLOCK_RUNTIME_IMAGE_DIGEST='sha256:PINNED_DIGEST'
```

The resource root is an absolute local directory. The executable and source
values are portable relative names beneath it. That root must enclose the
request-resolved GGUF artifact as well as both support files. The GGUF artifact
itself is the request-side `artifact.path`; backend paths are not published as
host paths in the canonical bundle.

## TensorRT-LLM

Install and check the optional provider in a compatible CUDA environment:

```bash
"$PYTHON" -m pip install /wheelhouse/invarlock_runtime_tensorrt_llm-EXACT.whl
"$PYTHON" -m invarlock_addins.tensorrt_llm.conformance
```

The provider binds the engine tree and inventory, builder configuration,
tokenizer contract, engine metadata, target compute capability, runner, and
outer image. TensorRT-LLM engines are runtime artifacts created by a build
workflow; NVIDIA documents that separation in the official
[TensorRT-LLM build workflow](https://nvidia.github.io/TensorRT-LLM/architecture/workflow.html)
and [architecture overview](https://nvidia.github.io/TensorRT-LLM/architecture/overview.html).

Inspect the exact engine on the target GPU inside the pinned image:

```python
from pathlib import Path

from invarlock_addins.tensorrt_llm.provider import TensorRTLLMProvider
from invarlock_addins.tensorrt_llm.session import TensorRTLLMRuntimeBindings

bindings = TensorRTLLMRuntimeBindings(
    engine_bundle_path=Path("engine"),
    tokenizer_contract_path=Path("tokenizer-contract.json"),
    runner_executable_path=Path("/opt/invarlock/bin/tensorrt-llm-runner"),
)
spec = TensorRTLLMProvider().inspect_runtime_spec(
    bindings,
    seed=0,
    context_length=2048,
    batch_size=1,
    max_output_tokens=64,
    timeout_seconds=300,
)
print(spec.model_id)
print(spec.settings)
```

Use those values for a request side with `provider: tensorrt_llm`, then supply
the same support resources during evaluation:

```bash
export INVARLOCK_TENSORRT_LLM_RESOURCE_ROOT=/pinned/tensorrt-runtime
export INVARLOCK_TENSORRT_LLM_TOKENIZER_CONTRACT=tokenizer/tokenizer-contract.json
export INVARLOCK_RUNTIME_IMAGE='registry.example/invarlock-tensorrt@sha256:PINNED_DIGEST'
export INVARLOCK_RUNTIME_IMAGE_DIGEST='sha256:PINNED_DIGEST'
export INVARLOCK_RUNTIME_DEVICE=cuda
```

The TensorRT-LLM resource root must enclose the request-resolved engine bundle
and tokenizer contract. Both are re-opened through root-confined, no-follow
resource paths. The runner is image-owned at the fixed authenticated path.

Reinspect and re-evaluate when the engine, tokenizer contract, installed
runner, outer image, GPU compute capability, or builder configuration changes.
A serialized engine is not assumed portable across incompatible TensorRT,
CUDA, or accelerator targets.

## Import provider evidence

Import mode is appropriate when execution occurs outside the core transaction
but the evaluation environment can supply the complete provider contract. It requires, for
both sides, the artifact identity, provider receipt, scoring observation,
runtime-side report, runtime manifest, runtime configuration, schedule, and
paired-record file.

Import is not a generic endpoint adapter and does not accept a score summary.
The installed provider must reproduce the artifact identity from request
settings, and InvarLock re-derives the pairs and report. See
[Evaluation request](evaluation-request.md#import-mode) for the exact shape and
the [offline example](https://github.com/invarlock/invarlock/blob/main/examples/README.md)
for runnable fixtures.

### Adapt per-record output without hand-assembling evidence

The stable Python facade provides a neutral record adapter for integrations
that already emit one result for every frozen schedule entry. A JSONL line uses
the runtime scoring-record fields, for example:

```json
{"input_sha256":"<64 lowercase hex>","output_sha256":"<64 lowercase hex>","output_text":"A","record_id":"sample-001","status":"ok"}
```

Log-probability integrations may instead include `logprob_sum`, `token_count`,
and `utf8_byte_count`. Every line is converted to a typed record. Blank lines,
unknown fields, aggregate metrics, missing schedule IDs, reordered IDs, input
digest mismatches, failed records, and output digest mismatches are rejected.

The following is the core of a reference adapter. The identity and provenance
values must come from inspection of the exact artifact and execution, not from
the comparison request or from a later evidence bundle:

```python
import hashlib
from pathlib import Path

from invarlock.engine import (
    HFSnapshotArtifactIdentity,
    RuntimeBackendIdentity,
    RuntimeDeviceFacts,
    RuntimeExecutionSettings,
    RuntimeProviderCapabilities,
    RuntimeProviderPluginIdentity,
    load_external_scoring_records_jsonl,
    load_runtime_behavioral_schedule,
    write_runtime_import_side,
)

schedule = load_runtime_behavioral_schedule(Path("schedule.json"))
records = load_external_scoring_records_jsonl(
    "baseline-records.jsonl",
    schedule=schedule,
)
policy_digest = "sha256:" + hashlib.sha256(
    Path("policy.json").read_bytes()
).hexdigest()

baseline = write_runtime_import_side(
    "imports/baseline",
    role="baseline",
    schedule=schedule,
    policy_digest=policy_digest,
    artifact_identity=HFSnapshotArtifactIdentity(
        model_id="org/baseline",
        immutable_revision="<pinned 40-64 character revision>",
        checkpoint_tree_sha256="<pinned 64 character digest>",
        tokenizer_metadata_sha256="<pinned 64 character digest>",
    ),
    records=records,
    plugin=RuntimeProviderPluginIdentity(
        name="hf_transformers",
        distribution="invarlock",
        distribution_version="<installed version>",
    ),
    backend=RuntimeBackendIdentity(
        name="transformers",
        version="<installed version>",
        source_sha256="<pinned 64 character digest>",
        binary_sha256=None,
        build_sha256=None,
    ),
    capabilities=RuntimeProviderCapabilities(
        provider_name="hf_transformers",
        artifact_formats=("hf_snapshot",),
        tasks=("text_causal",),
        metrics=("exact_match",),
        execution_modes=("in_process",),
        required_extra="hf",
        required_image=None,
    ),
    execution_settings=RuntimeExecutionSettings(
        seed=0,
        context_length=2048,
        batch_size=1,
        max_output_tokens=64,
        timeout_seconds=300,
        allow_network=False,
    ),
    device=RuntimeDeviceFacts(
        device_kind="cuda",
        device_name="<observed device name>",
        compute_capability="<observed major.minor>",
    ),
    runtime_image_ref=(
        "registry.example/pinned-evaluator@sha256:<pinned image digest>"
    ),
    runtime_image_digest="sha256:<pinned image digest>",
    generated_at_utc="<observed ISO 8601 UTC timestamp>",
)
```

Repeat the call for the subject, then invoke
`write_runtime_import_paired_records` with both returned side objects. That
function derives scores from the authenticated records and expected schedule
outputs; it has no parameter for caller-supplied aggregate values. Put its
output and the two complete side directories into the import request tree.

The authoring API makes the format reproducible and removes hand-built Docker
command sequences. A separate verifier establishes acceptance with
independently managed artifact identities, canonical schedule, policy,
runtime-image identities, evidence signer, and verifier signer.

## Validate the provider outcome and recover failures

| Symptom | First check |
| --- | --- |
| Provider is not discovered | Install the matching add-in and run its conformance command in the same environment as `invarlock`. |
| Artifact identity mismatch | Re-run inspection over the exact artifact; look for changed files, metadata, tokenizer material, or request settings. |
| Backend digest mismatch | Confirm the executable, source archive, or runner came from the pinned runtime root. |
| Runtime image mismatch | Resolve the image reference to an immutable digest and make the reference and separate digest agree. |
| Device or compute-capability mismatch | Rebuild or select an engine for the actual accelerator and repeat controlled inspection. |
| Records are missing or reordered | Fix the provider execution; do not patch paired records or the comparison report. |

A usable provider outcome has a complete record for every schedule ID in the
original order, authenticated artifact and runtime identities, and no
provider-side substitution of aggregate results for typed observations. Run
independent bundle verification after evaluation; provider success alone is
not an acceptance result.

If validation fails, preserve the original error and any unpublished
workspace, correct the artifact or runtime boundary, reinspect the exact
material, and evaluate into a new destination. Never repair provider records
inside a published evidence bundle.

For environment-variable details, see
[Environment variables](../reference/environment.md). For ABI and receipt
obligations, see [Runtime providers reference](../reference/runtime-providers.md).
