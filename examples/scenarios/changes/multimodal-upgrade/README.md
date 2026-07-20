# Evaluate a multimodal model upgrade

## When to use this example

Use this recipe when releasing or replacing a vision-language checkpoint. The
optional vision-text add-in supplies image decoding, processor behavior, model
loading, and generation through the same evidence and verification transaction
used for text evaluation.

## Inputs you bring

- Immutable baseline and subject vision-language checkpoints.
- A local JSONL dataset with stable record IDs, prompts, expected answers, and
  content metadata.
- A content store whose filenames, media types, byte lengths, and SHA-256
  digests match the schedule declarations.
- An exact-match policy and digest-pinned vision-text runtime image.
- Independent artifact, schedule, runtime, policy, and signer anchors.

## InvarLock transaction

Set task `vision_text_generation`, metric `exact_match`, and provider
`hf_vision_text`. Preflight authenticates and safely decodes every selected
image without loading the model or allocating a GPU. Each worker reopens and
checks the same content before inference, and the signed schedule contains no
host path or mutable URI.

## What the result establishes

A passing receipt establishes that the two authenticated checkpoints were
compared on the same content-bound vision-text records and that the subject
satisfied the selected exact-match policy.

## Interpretation boundary

Exact match is appropriate for closed-answer VQA-style tasks. Open-ended
caption quality, semantic similarity, safety, or judge-based assessments need
a separately authenticated scorer or observation with a clearly stated claim.

## Run it

Prepare the content-addressed dataset and provider settings described in the
[vision-text add-in guide](../../../../addins/multimodal/README.md). Build and
smoke-test the exact runtime image, then run preflight before qualification:

```console
invarlock evaluate request.yaml --signing-key evidence-signer.pem \
  --runtime-image "$VISION_IMAGE" --runtime-image-digest "$VISION_DIGEST" \
  --runtime-device cuda:0 --preflight
invarlock evaluate request.yaml --signing-key evidence-signer.pem \
  --runtime-image "$VISION_IMAGE" --runtime-image-digest "$VISION_DIGEST" \
  --runtime-device cuda:0
```

The current 400-record vision-text comparison is indexed under
[`public_evidence/`](../../../../public_evidence/README.md). Complete the
independent handoff using the [common transaction](../../README.md#common-transaction).
