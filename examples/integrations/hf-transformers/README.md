# Hugging Face Transformers integration

This example downloads the official Apache-2.0 `Qwen/Qwen3-0.6B` checkpoint at
immutable revision `c1899de289a04d12100db370d81485cdf75e47ca`. It saves that
checkpoint as the baseline and creates a distinct subject by fitting one causal
output row to favor the expected continuation across 50 paired records. The
source model, revision, baseline checkpoint identity, transformation, and
subject identity are authenticated in the resulting evidence.

A passing normalized-NLL decision comes from executing and scoring both Qwen3
checkpoints. The report also derives a token-weighted perplexity ratio when the
authenticated tokenizer and target-token counts are comparable; that derived
value does not control acceptance.

The generated workspace keeps the closed evaluation request in `evaluation/`,
separate verifier inputs and its signing key in `verifier/`, and the evidence
signing key in `keys/`. The script refuses to reuse an existing workspace or
evidence output. For a real acceptance decision, the verifier owner should
choose and retain its own trust anchors and signing key.

Run the complete journey from the repository root:

```bash
make example-hf-transformers
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
  EXAMPLE_ARGS="--prepare-only --workspace /tmp/invarlock-hf-inputs"
```

Preparation uses an explicit placeholder runtime digest because no image is
executed. The complete journey derives the real image digest and passes it in
the generated `invarlock/trust-inputs-v1` profile to `invarlock verify`.

The first run downloads the pinned checkpoint and needs several gigabytes of
cache and workspace capacity. The 50 records and fitted subject intentionally
form a compact integration demonstration. They demonstrate real Qwen3 scoring,
isolation, and trust bindings; they are not a benchmark or a model-quality
claim.
