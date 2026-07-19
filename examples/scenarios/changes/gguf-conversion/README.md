# Evaluate a GGUF conversion

## When to use this example

Use this recipe when a Hugging Face or other source checkpoint has been
converted or quantized into GGUF for a llama.cpp-compatible deployment. The
conversion tool creates the candidate; the optional GGUF add-in authenticates
the complete file and executes it through the pinned backend.

## Inputs you bring

- The immutable source checkpoint used as the baseline.
- The exact GGUF file proposed for deployment.
- A digest-pinned llama.cpp runner and InvarLock GGUF runtime image.
- Stable closed-answer paired records and an exact-match policy.
- Independent artifact, schedule, policy, runtime, and signer anchors.

Use a baseline provider capable of scoring the source checkpoint and the
`llama_cpp` provider for the subject. If both sides are GGUF variants, bind each
file and runtime independently.

## InvarLock transaction

The two workers receive the same content-addressed schedule but load different
artifact and runtime identities. Exact match is the portable built-in metric
for unlike tokenizer/runtime paths. Host preflight authenticates the GGUF file,
runner, runtime configuration, output destination, and image identities before
starting either worker.

## What the result establishes

A passing receipt establishes that the exact GGUF artifact, through the pinned
llama.cpp runtime, satisfied the selected paired output policy relative to the
authenticated baseline.

## Interpretation boundary

This result does not claim token-level likelihood parity across different
tokenizers. Quantization level, throughput, memory use, and hardware behavior
remain separately measured properties.

## Run it

Install and smoke-test the optional package using the commands in the
[GGUF add-in guide](../../../../addins/gguf/README.md). Author a run-mode request
with `hf_transformers` or `llama_cpp` for the baseline and `llama_cpp` for the
subject, then use per-side digest-pinned images:

```console
invarlock evaluate request.yaml --signing-key evidence-signer.pem \
  --baseline-runtime-image "$BASELINE_IMAGE" \
  --baseline-runtime-image-digest "$BASELINE_DIGEST" \
  --subject-runtime-image "$GGUF_IMAGE" \
  --subject-runtime-image-digest "$GGUF_DIGEST" --preflight
invarlock evaluate request.yaml --signing-key evidence-signer.pem \
  --baseline-runtime-image "$BASELINE_IMAGE" \
  --baseline-runtime-image-digest "$BASELINE_DIGEST" \
  --subject-runtime-image "$GGUF_IMAGE" \
  --subject-runtime-image-digest "$GGUF_DIGEST"
```

Complete the independent handoff in the
[scenario catalog](../../README.md#common-transaction).
