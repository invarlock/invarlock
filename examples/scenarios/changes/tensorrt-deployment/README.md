# Evaluate a TensorRT-LLM deployment

## When to use this example

Use this recipe after a deployment pipeline compiles a checkpoint into a
TensorRT-LLM engine. The add-in authenticates the closed engine bundle,
tokenizer contract, runner, runtime image, and execution settings before the
candidate is compared with the source checkpoint or an earlier engine.

## Inputs you bring

- The immutable source checkpoint or previously deployed baseline engine.
- The complete single-rank TensorRT-LLM engine bundle proposed for release.
- The closed tokenizer contract and pinned TensorRT-LLM runner.
- A digest-pinned runtime image qualified on the intended GPU class.
- Stable paired records, an exact-match policy, and independent trust inputs.

## InvarLock transaction

Use the `tensorrt_llm` provider for the candidate and either
`hf_transformers` or `tensorrt_llm` for the baseline. The required canary
qualifies the exact runtime path. Target preflight then verifies the canary,
candidate wheel set, source archive, engine inputs, image identity, request,
policy, schedule, and output locations before full execution.

## What the result establishes

A passing receipt establishes that the authenticated engine bundle, tokenizer,
runner, image, schedule, and policy formed one consistent transaction and that
the candidate satisfied its paired output policy.

## Interpretation boundary

The result is a finite-schedule release decision. It is not a claim about model
quality, throughput, latency, numerical parity on every prompt, or portability
to a different TensorRT, CUDA, driver, or GPU configuration.

## Run it

Follow the add-in's maintained sequence for image construction, smoke testing,
signed canary creation, target preflight, and evidence qualification in the
[TensorRT-LLM guide](../../../../addins/tensorrt_llm/README.md). The final target
request still uses the ordinary public transaction:

```console
invarlock evaluate request.yaml --signing-key evidence-signer.pem \
  --baseline-runtime-image "$BASELINE_IMAGE" \
  --baseline-runtime-image-digest "$BASELINE_DIGEST" \
  --subject-runtime-image "$TENSORRT_IMAGE" \
  --subject-runtime-image-digest "$TENSORRT_DIGEST" \
  --subject-runtime-device cuda:0 --preflight
```

Run the signed qualification target only after preflight succeeds. A complete
400-record TensorRT-LLM evidence entry is indexed under
[`public_evidence/`](../../../../public_evidence/README.md).
