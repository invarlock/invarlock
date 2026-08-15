# BF16-to-GGUF deployment comparison

This journey selects one closed, revision- and byte-pinned deployment profile.
It executes the source checkpoint in BF16 through the built-in Hugging Face
Transformers provider, converts the same checkpoint with a pinned llama.cpp
source revision, quantizes the resulting BF16 GGUF to Q5_K_M, and executes the
derived artifact through the optional llama.cpp provider.

| Profile | Source | Purpose |
| --- | --- | --- |
| `qwen35-9b` (default) | `Qwen/Qwen3.5-9B` | Maintained deployment non-regression result for the Qwen3.5 text-causal projection |
| `ministral3-8b` | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | Independent-family, text-only canary over a multimodal-backed checkpoint |

The execution scope is the checkpoint's text-causal component. For Qwen3.5,
Transformers loads the exact native causal projection from the authenticated
multimodal checkpoint while the stored vision and MTP tensors remain bound to
the source identity and verified outside the live model state. For Ministral
3, Transformers authenticates and loads the complete checkpoint but receives
text-only inputs, while llama.cpp converts and executes its language-model
component. Each profile gives its BF16 and GGUF sides the same text behavior
boundary.

Both sides use the same ordered 400-record balanced MMLU-Pro semantic
selection, rendered through the selected model's pinned chat format. The policy
is fixed before execution: at least 20% accuracy on each side, a paired interval
no wider than 10 percentage points, and a subject-minus-baseline paired
interval whose lower bound is at least −2 percentage points. The signed subject
request separately pins record batching, llama.cpp prompt and micro-batch
sizes, and CPU thread count; the journey fixes the corresponding worker CPU
limit.

## Retained result

The [published signed transaction](../../../public_evidence/evidence/qwen3.5-9b-bf16-to-q5-k-m-gguf/)
records 212 of 400 exact matches for BF16 and 219 of 400 for Q5_K_M. The paired
subject-minus-baseline effect is +1.75 percentage points, with a 95% interval
from -0.83 to 4.32 percentage points. The interval width is 5.15 percentage
points, and both side accuracies exceed the 20% floor, so the frozen policy
passes. This is a finite-schedule deployment non-regression result, not a claim
of general output equivalence or broader model quality.

## Compute and storage

The maintained run uses Linux, Docker, one CUDA-capable GPU with at least 24 GB
of memory for the BF16 baseline, 64 GB of system memory, and about 70 GB of free
disk space while conversion is active. The llama.cpp subject uses CPU because
that provider's qualified execution profile is CPU-bound. Budget roughly eight
to twelve hours for image construction, conversion, quantization, and the
400-record transaction on a recent server; the CPU subject dominates elapsed
time. Cached image layers reduce later runs.

Network access is used only while building the pinned images and downloading
the exact checkpoint files. The conversion, quantization, and model-execution
containers run with networking disabled. Verification is a local operation and
does not require network access.

## Run the signed journey

Use distinct caller-owned evidence and verifier Ed25519 keys plus a new
trust-root directory:

```bash
make example-gguf-deployment EXAMPLE_ARGS="\
  --evidence-signing-key /secure/keys/evidence.pem \
  --verifier-signing-key /secure/keys/verifier.pem \
  --trust-root /secure/trust/gguf-deployment"
```

Select the independent-family canary with `--profile ministral3-8b`:

```bash
make example-gguf-deployment EXAMPLE_ARGS="\
  --profile ministral3-8b \
  --evidence-signing-key /secure/keys/evidence.pem \
  --verifier-signing-key /secure/keys/verifier.pem \
  --trust-root /secure/trust/gguf-deployment-ministral3"
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

These profiles demonstrate concrete deployment-format and runtime changes
across independent model families. Other retained transactions exercise Gemma
and additional evaluator and runtime paths.
