# BF16-to-GGUF deployment comparison

This journey starts from one revision- and byte-pinned post-trained
`Qwen/Qwen3.5-9B` checkpoint. It executes that source checkpoint in BF16 through
the built-in Hugging Face Transformers provider, converts the same checkpoint
with a pinned llama.cpp source revision, quantizes the resulting BF16 GGUF to
Q5_K_M, and executes the derived artifact through the optional llama.cpp
provider.

Both sides use the ordered 400-record balanced MMLU-Pro Qwen schedule already
used by the current-model evaluator transactions. The policy is fixed before
execution: at least 20% accuracy on each side, a paired interval no wider than
10 percentage points, and a subject-minus-baseline delta of at least −2
percentage points.

## Compute and storage

The maintained run uses Linux, Docker, one CUDA-capable GPU with at least 24 GB
of memory for the BF16 baseline, 64 GB of system memory, and about 70 GB of free
disk space while conversion is active. The llama.cpp subject uses CPU because
that provider's qualified execution profile is CPU-bound. On a recent server,
allow roughly one to three hours for image construction, conversion,
quantization, and the 400-record transaction. Cached image layers reduce later
runs.

Network access is used only while building the pinned images and downloading
the exact checkpoint files. Conversion, quantization, model execution, and
verification run with networking disabled.

## Run the signed journey

Use distinct caller-owned evidence and verifier Ed25519 keys plus a new
trust-root directory:

```bash
make example-gguf-deployment EXAMPLE_ARGS="\
  --evidence-signing-key /secure/keys/evidence.pem \
  --verifier-signing-key /secure/keys/verifier.pem \
  --trust-root /secure/trust/gguf-deployment"
```

The command builds source-bound baseline and subject images, stages the exact
checkpoint, records the intermediate and final GGUF identities, writes one
mixed-provider request, runs `evaluate`, independently runs `verify`, renders
the report, and removes the temporary image tags it created. The workspace
retains the large source and subject artifacts so its evidence can be audited;
remove that workspace after preserving the signed evidence and receipt you
need.

Already inspected local images can be supplied by immutable config ID:

```bash
make example-gguf-deployment EXAMPLE_ARGS="\
  --baseline-runtime-image sha256:BASELINE_IMAGE_ID \
  --subject-runtime-image sha256:SUBJECT_IMAGE_ID \
  --evidence-signing-key /secure/keys/evidence.pem \
  --verifier-signing-key /secure/keys/verifier.pem \
  --trust-root /secure/trust/gguf-deployment"
```

This transaction demonstrates a current model's concrete deployment-format
and runtime change. The retained Gemma 4 transaction provides the separate
model-family portability check.
