# Evaluate a quantized Hugging Face checkpoint

## When to use this example

Use this recipe after AWQ, GPTQModel, HQQ, bitsandbytes, Quanto, TorchAO, or
another quantization stack has created the candidate artifact. The relevant
question is whether that exact artifact, under its exact loader and runtime,
preserves the paired outcomes required for release.

## Inputs you bring

- The immutable full-precision baseline and quantized candidate.
- A provider implementation that authenticates and loads the quantized format,
  or closed side results exported by that runtime.
- Stable paired records and a policy suited to the task.
- Receipts binding quantization configuration, model identity, loader, and
  runtime when those facts are not contained in the artifact itself.
- Independent verification anchors and signing identities.

The built-in Hugging Face provider is appropriate only when the candidate is a
portable checkpoint that the maintained runtime explicitly supports. A format
that needs an additional loader belongs behind the provider ABI or import
boundary.

## InvarLock transaction

Use `exact_match` for closed-answer behavior across unlike tokenization or
runtime paths. Use normalized NLL only when both sides expose comparable,
authenticated tokenizer and target-token contracts. Import mode validates the
canonical schedule, paired records, side identities, observations, manifests,
configuration, and receipts before constructing evidence.

## What the result establishes

A passing receipt establishes that the authenticated quantized candidate
satisfied the selected paired release policy through its bound runtime.

## Interpretation boundary

Quantization method, bit width, file size, memory use, latency, and throughput
are deployment facts, not conclusions derived from the task score. Attach such
facts as authenticated observations when they are relevant.

## Run it

Normalize the runtime's stable per-record output into the six closed side files
shown by the [offline import example](../../../README.md). Bind the same
canonical schedule to both sides, then run:

```console
invarlock evaluate request.yaml --signing-key evidence-signer.pem --preflight
invarlock evaluate request.yaml --signing-key evidence-signer.pem
```

Complete the independent `verify` and `report` handoff described in the
[scenario catalog](../../README.md#common-transaction). Keep aggregate runtime
scores outside acceptance unless a verifier-replayable scorer extension binds
and recomputes them.
