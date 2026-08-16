# Hugging Face Transformers integration

This example downloads the official Apache-2.0 `Qwen/Qwen3.5-0.8B` checkpoint at
immutable revision `2fc06364715b967f1860aea9cf38778875588b17`. It saves that
checkpoint as the baseline and creates a distinct subject by fitting one causal
output row to favor the expected continuation across 50 paired records. The
source model, revision, baseline checkpoint identity, transformation, and
subject identity are authenticated in the resulting evidence.

A passing normalized-NLL decision comes from executing and scoring both Qwen3.5
checkpoints. The report also derives a token-weighted perplexity ratio when the
authenticated tokenizer and target-token counts are comparable; that derived
value does not control acceptance.

The closed evaluation request remains in the disposable workspace, while the
caller-owned evidence key and independent trust root stay outside it. The
script refuses to reuse an existing workspace or evidence output.

Run the complete journey from the repository root:

```bash
make example-hf-transformers \
  EXAMPLE_ARGS="--evidence-signing-key /secure/keys/evidence.pem \
  --verifier-signing-key /secure/keys/verifier.pem \
  --trust-root /secure/trust/hf-transformers"
```

That one command creates an exact Git source bundle, builds the source-bound
runtime image, creates two distinct local checkpoints, invokes the public CLI
commands, and prints the disposable output directory. Each model runs in its
own network-disabled, read-only worker; CUDA is selected when available and CPU
remains supported. The host retains the
evidence-signing key, publishes the evidence pack, verifies it against the
separate artifact, schedule, runtime, policy, and signer inputs, and writes an
HTML report.

To inspect the generated request and trust inputs before spending container
time, use a new workspace with `--prepare-only`:

```bash
make example-hf-transformers \
  EXAMPLE_ARGS="--prepare-only --workspace /tmp/invarlock-hf-inputs \
  --evidence-signing-key /secure/keys/evidence.pem \
  --verifier-signing-key /secure/keys/verifier.pem \
  --trust-root /secure/trust/hf-transformers"
```

Preparation uses an explicit placeholder runtime digest because no image is
executed. The complete journey derives the real image digest and passes it in
the generated `invarlock/trust-inputs-v1` profile to `invarlock verify`.

The first run downloads the pinned checkpoint and needs several gigabytes of
cache and workspace capacity. The 50 records and fitted subject intentionally
form a compact integration demonstration. They demonstrate real Qwen3.5 scoring,
isolation, and trust bindings; they are not a benchmark or a model-quality
claim.
