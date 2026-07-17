# Hugging Face vision-text add-in

This optional first-party package evaluates image-and-text generation models
through InvarLock's runtime-provider ABI. It supports one bounded transaction:

- a `vision_text_generation` schedule record containing exactly one `prompt`
  text part and one `image` content part;
- a local Hugging Face checkpoint authenticated by its complete tree digest;
- a locally loaded processor and tokenizer authenticated by their declared
  contracts;
- deterministic greedy generation; and
- `exact_match` evidence through the ordinary `evaluate`, `verify`, and
  `report` path.

The base add-in declares Pillow for execution-free host preflight; the
`[runtime]` extra adds the model inference stack. The core supplies the
ordered, content-addressed input contract. This add-in
supplies image decoding, Hugging Face processor behavior, and vision-text
inference, keeping the integration independently installable and ready for
separate qualification.

Install the base wheel in the host environment used for preflight and provider
discovery. Install the inference extra only in an environment that executes the
model; the maintained runtime image supplies those heavier dependencies:

```console
PYTHON=/path/to/venv/bin/python
"$PYTHON" -m pip install invarlock-runtime-hf-vision-text
# Development outside the maintained image only:
"$PYTHON" -m pip install 'invarlock-runtime-hf-vision-text[runtime]'
```

The install above is the standalone conformance path. Maintained qualification
does not treat first-party packages already installed in that environment as
the candidate. `CANDIDATE_WHEEL_MANIFEST` must authenticate the exact core and
vision-text wheels built from the qualified source commit. The driver extracts
those wheels into a private candidate site and loads them before any
first-party code visible to `PYTHON`. `PYTHON` supplies their third-party
dependencies, including Pillow for host preflight; the heavier inference stack
remains in the runtime image. `QUALIFICATION_DRIVER_PYTHON` launches the
standard-library qualification driver. Using one isolated dependency
environment for both variables is the simplest configuration.

After `make dist-check`, create the manifest with the maintained no-clobber
helper. Its destination parent must already exist, and the destination must be
new:

```console
install -d -m 700 qualification
python scripts/qualification_candidate_wheels.py \
  --wheel dist/invarlock-*.whl \
  --wheel dist/addins/invarlock_runtime_hf_vision_text-*.whl \
  --output "$PWD/qualification/vision-text-candidate-wheels.json"
export CANDIDATE_WHEEL_MANIFEST="$PWD/qualification/vision-text-candidate-wheels.json"
```

The helper records absolute wheel paths and SHA-256 digests. The qualification
driver additionally requires unique, version-aligned maintained distributions
whose contents match the authenticated source archive. Recreate the manifest
after any wheel changes, and pass the same manifest to canary, preflight, and
evidence qualification.

## Content store

Schedules contain no host path or URI. Each image part declares a canonical
`content_id`, media type, byte length, and SHA-256. The add-in resolves the ID as
one exact filename beneath a caller-selected content store, opens it without following
links, and rechecks its file identity, length, digest, media format, frame count,
dimensions, and bounded aggregate media budget before inference. The filename
must equal `content_id`; a differently named file with identical bytes does not
satisfy the schedule binding.

One schedule may bind at most 2,048 unique images, 512 MiB of declared image
bytes, and 200 million decoded pixels; each image remains limited to 64 MiB and
50 million pixels.

For OCI evaluation, place both local model snapshots beneath the resource root
and expose the image directory as `content_store`:

```console
export INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT="$PWD/vision-inputs"
export INVARLOCK_HF_VISION_TEXT_CONTENT_STORE=images
```

Both variables are required by host preflight and execution. Every object
selected by the prepared schedule must already exist and be readable. Host
preflight safely decodes and closes the media without importing Torch or
Transformers, loading a model, or initializing CUDA. The worker repeats the
same validation before model preparation, and scoring reopens the bytes to
detect replacement after preflight.

An image entry such as `vqa_0001` is then addressed by a schedule part like:

```json
{
  "kind": "content",
  "role": "image",
  "content_id": "vqa_0001",
  "media_type": "image/png",
  "byte_length": 2841,
  "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}
```

The companion text part uses role `prompt`. The record's `input_sha256` is the
core-defined digest of the ordered parts, so changing either the prompt or image
binding changes the schedule identity.

## Model contract

Both comparison sides declare these settings:

- `checkpoint_tree_sha256`
- `tokenizer_metadata_sha256`
- `processor_metadata_sha256`
- `batch_size` (currently `1`)
- `context_length`
- `max_output_tokens`
- `seed`
- `timeout_seconds`
- `offline: true`

`processor_contract_sha256(processor)` calculates the processor digest after an
`AutoProcessor` has been loaded locally. Runtime loading always uses
`local_files_only=True`, `trust_remote_code=False`, and safetensors.

The add-in owns its reproducible image layer. Build it on the exact digest of a
canonical InvarLock CUDA image, then run the offline CUDA/import smoke:

`BASE_IMAGE` is a Dockerfile `FROM` input, so it must be a named manifest
reference such as `repository@sha256:...` (an optional tag may appear before
`@`). A raw local `sha256:...` config ID is valid for `docker run` and local
qualification, but it is not a Dockerfile-compatible base reference. If the
CUDA image exists only under a local config ID, tag and publish that exact image
through the operator's registry, then select the matching `RepoDigests` entry
for that repository before building this layer. Do not prepend a repository
name to a config ID; config and manifest digests identify different objects.

```console
make -C addins/multimodal build \
  BASE_IMAGE=registry.example/invarlock-cuda@sha256:... \
  IMAGE=invarlock-hf-vision-text:local \
  SOURCE_COMMIT=0123456789abcdef0123456789abcdef01234567 \
  SOURCE_BUNDLE=/absolute/path/to/invarlock-source.tar \
  SOURCE_BUNDLE_SHA256=sha256:... \
  BUILD_STATEMENT=/absolute/path/to/vision-text-build.json
make -C addins/multimodal smoke IMAGE=invarlock-hf-vision-text:local
```

The Dockerfile builds the add-in wheel from this checkout and installs Pillow
12.3.0 from a generated, hash-pinned Linux lock. Torch, Transformers,
safetensors, and the core CLI are inherited from the exact base-image digest.

The qualification command uses the resulting digest-pinned image:

```console
CANARY_EVIDENCE="$PWD/canary/evidence"
CANARY_RECEIPT="$PWD/canary/verification-receipt.json"
CANARY_TRUST_PROFILE="$PWD/canary/trust-inputs.json"
export QUALIFICATION_DEVICE=cuda:0
export QUALIFICATION_CPUS=12
export QUALIFICATION_MEMORY_MIB=98304
export QUALIFICATION_USER=65532:65532

# A local qualification may use the exact config ID for both values.
export IMAGE_DIGEST="$(docker image inspect \
  --format '{{.Id}}' invarlock-hf-vision-text:local)"
export QUALIFICATION_IMAGE="$IMAGE_DIGEST"

make -C addins/multimodal qualify-canary \
  QUALIFICATION_IMAGE="$QUALIFICATION_IMAGE" \
  IMAGE_DIGEST="$IMAGE_DIGEST" \
  RESOURCE_ROOT="$PWD/vision-inputs" CONTENT_STORE=images \
  REQUEST=canary-request.yaml SIGNING_KEY=evidence.key \
  EVIDENCE="$CANARY_EVIDENCE" TRUST_PROFILE="$CANARY_TRUST_PROFILE" \
  RECEIPT="$CANARY_RECEIPT" SUMMARY=canary-qualification.json \
  CANDIDATE_WHEEL_MANIFEST="$CANDIDATE_WHEEL_MANIFEST" \
  SOURCE_COMMIT=0123456789abcdef0123456789abcdef01234567 \
  SOURCE_BUNDLE="$PWD/invarlock-source.tar" \
  SOURCE_BUNDLE_SHA256=sha256:...

make -C addins/multimodal qualify-preflight \
  QUALIFICATION_IMAGE="$QUALIFICATION_IMAGE" \
  IMAGE_DIGEST="$IMAGE_DIGEST" \
  RESOURCE_ROOT="$PWD/vision-inputs" CONTENT_STORE=images \
  REQUEST=request.yaml SIGNING_KEY=evidence.key EVIDENCE=evidence/ \
  TRUST_PROFILE=trust/trust-inputs.json \
  RECEIPT=verification-receipt.json \
  CANARY_EVIDENCE="$CANARY_EVIDENCE" \
  CANARY_RECEIPT="$CANARY_RECEIPT" \
  CANARY_TRUST_PROFILE="$CANARY_TRUST_PROFILE" \
  CANDIDATE_WHEEL_MANIFEST="$CANDIDATE_WHEEL_MANIFEST" \
  SOURCE_COMMIT=0123456789abcdef0123456789abcdef01234567 \
  SOURCE_BUNDLE="$PWD/invarlock-source.tar" \
  SOURCE_BUNDLE_SHA256=sha256:...

make -C addins/multimodal qualify-evidence \
  QUALIFICATION_IMAGE="$QUALIFICATION_IMAGE" \
  IMAGE_DIGEST="$IMAGE_DIGEST" \
  RESOURCE_ROOT="$PWD/vision-inputs" CONTENT_STORE=images \
  REQUEST=request.yaml SIGNING_KEY=evidence.key EVIDENCE=evidence/ \
  TRUST_PROFILE=trust/trust-inputs.json \
  RECEIPT=verification-receipt.json \
  CANARY_EVIDENCE="$CANARY_EVIDENCE" \
  CANARY_RECEIPT="$CANARY_RECEIPT" \
  CANARY_TRUST_PROFILE="$CANARY_TRUST_PROFILE" \
  CANDIDATE_WHEEL_MANIFEST="$CANDIDATE_WHEEL_MANIFEST" \
  REPORT=vision-text-evidence.html \
  SUMMARY=vision-text-qualification.json \
  SOURCE_COMMIT=0123456789abcdef0123456789abcdef01234567 \
  SOURCE_BUNDLE="$PWD/invarlock-source.tar" \
  SOURCE_BUNDLE_SHA256=sha256:...
```

`qualify-canary` runs one real, strictly verified transaction for the exact
image. Retain its evidence, receipt, and verifier-owned trust profile; a new
image digest requires a new signed canary. `qualify-preflight` reverifies that
canary, then runs execution-free checks before starting either target model
worker.
`RESOURCE_ROOT` and `CONTENT_STORE` are the Make equivalents of the two
environment bindings above; readiness authenticates the schedule-selected
objects before a GPU is allocated.
`TRUST_PROFILE` is a verifier-owned `invarlock/trust-inputs-v1` document that
binds the independently sourced policy, artifact, schedule, runtime, signer,
verifier, and scorer-authorization inputs used after evaluation.
Qualification authenticates the Git source archive and matching image labels,
then verifies the exact evidence path, normalized request, signed receipt, and
rendered report against one pack identity. The private `SUMMARY` is written
only after every stage succeeds.
`REPORT` is optional. Receipt, report, and summary destinations must be fresh,
distinct, outside the evidence path, and beneath existing non-symlinked
directories.

Qualification always uses the exact supplied image digest and runs without
network access. A target model can still require a different declared CUDA
base; that choice is explicit in `BASE_IMAGE` and becomes part of the resulting
runtime digest. The signed canary prevents image-level fan-out after a broken
representative transaction; it does not prove that another checkpoint loads,
fits memory, supports its processor, or completes successfully.

## Qualification status

Package tests cover ABI conformance, configuration closure, content tampering,
link rejection, structured-record requirements, and a deterministic inference
path with test doubles. A release claim still requires a real local checkpoint,
OCI image digest, representative image schedule, and strict
`evaluate`-then-`verify`-then-`report` journey on its target CPU or GPU
runtime.
