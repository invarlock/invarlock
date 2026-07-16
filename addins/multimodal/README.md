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

The core supplies the ordered, content-addressed input contract. This add-in
supplies image decoding, Hugging Face processor behavior, and vision-text
inference, keeping the integration independently installable and ready for
separate qualification.

Install the inference dependencies explicitly:

```console
python -m pip install 'invarlock-runtime-hf-vision-text[runtime]'
```

## Content store

Schedules contain no host path or URI. Each image part declares a canonical
`content_id`, media type, byte length, and SHA-256. The add-in resolves the ID as
one basename beneath a caller-selected content store, opens it without following
links, and rechecks its file identity, length, digest, media format, frame count,
and dimensions before inference.

For OCI evaluation, place both local model snapshots beneath the resource root
and expose the image directory as `content_store`:

```console
export INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT="$PWD/vision-inputs"
export INVARLOCK_HF_VISION_TEXT_CONTENT_STORE=images
```

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

```console
make -C addins/multimodal build \
  BASE_IMAGE=registry.example/invarlock-cuda@sha256:... \
  IMAGE=invarlock-hf-vision-text:local
make -C addins/multimodal smoke IMAGE=invarlock-hf-vision-text:local
```

The Dockerfile builds the add-in wheel from this checkout and installs Pillow
12.3.0 from a generated, hash-pinned Linux lock. Torch, Transformers,
safetensors, and the core CLI are inherited from the exact base-image digest.

The qualification command uses the resulting digest-pinned image:

```console
make -C addins/multimodal qualify-evidence \
  IMAGE=registry.example/invarlock-vision@sha256:... \
  IMAGE_DIGEST=sha256:... \
  RESOURCE_ROOT="$PWD/vision-inputs" CONTENT_STORE=images \
  REQUEST=request.yaml SIGNING_KEY=evidence.key EVIDENCE=evidence/ \
  POLICY=policy.json BASELINE_ARTIFACT=baseline.identity.json \
  SUBJECT_ARTIFACT=subject.identity.json SCHEDULE=schedule.json \
  SIGNER=signer.pub VERIFIER_KEY=verifier.key \
  VERIFIER_IDENTITY=independent-check RECEIPT=verification-receipt.json \
  REPORT=vision-text-evidence.html
```

Qualification always uses the exact supplied image digest and runs without
network access. A target model can still require a different declared CUDA
base; that choice is explicit in `BASE_IMAGE` and becomes part of the resulting
runtime digest.

## Qualification status

Package tests cover ABI conformance, configuration closure, content tampering,
link rejection, structured-record requirements, and a deterministic inference
path with test doubles. A release claim still requires a real local checkpoint,
OCI image digest, representative image schedule, and strict
`evaluate`-then-`verify`-then-`report` journey on its target CPU or GPU
runtime.
