# Evaluate a pruned checkpoint

## When to use this example

Use this recipe after a pruning system has materialized a sparse checkpoint.
It applies to structured or unstructured pruning when both sides can be scored
through an authenticated runtime. InvarLock evaluates the release candidate;
it does not choose masks, apply pruning, or claim a compression ratio.

## Inputs you bring

- The immutable unpruned checkpoint and materialized pruned checkpoint.
- A runtime able to load both artifacts without silently densifying or
  replacing either one.
- Stable paired records representative of the release question.
- A likelihood-regression policy with minimum-count and precision controls.
- Independent artifact, schedule, runtime, policy, and signer anchors.

Record sparsity and storage measurements separately if they matter to the
release. They can be attached as authenticated observations but do not replace
the paired acceptance metric.

## InvarLock transaction

For portable Hugging Face `safetensors`, use `hf_transformers`,
`text_causal`, `normalized_nll_per_utf8_byte`, and run mode. If the sparse
layout needs a specialized loader, implement the runtime-provider ABI or
import its closed per-record side results; do not make the core reinterpret an
unsupported checkpoint.

## What the result establishes

A passing receipt establishes that the bound pruned artifact stayed within the
selected normalized-NLL regression policy on the paired schedule.

## Interpretation boundary

The evidence does not infer sparsity, throughput, memory use, or hardware
support from the checkpoint name. Authenticate those measurements separately.
The likelihood result remains specific to the selected records and tokenizer
contract.

## Run it

Start from the run-mode request structure in the
[Hugging Face CPU example](../../../run/README.md). Replace its generated
artifact identities with identities derived from the complete baseline and
pruned checkpoint trees, then preflight and execute:

```console
invarlock evaluate request.yaml --signing-key evidence-signer.pem \
  --runtime-image "$RUNTIME_IMAGE" --runtime-image-digest "$RUNTIME_DIGEST" \
  --runtime-device cuda:0 --preflight
invarlock evaluate request.yaml --signing-key evidence-signer.pem \
  --runtime-image "$RUNTIME_IMAGE" --runtime-image-digest "$RUNTIME_DIGEST" \
  --runtime-device cuda:0
```

Complete the independent `verify` and `report` handoff described in the
[scenario catalog](../../README.md#common-transaction).
