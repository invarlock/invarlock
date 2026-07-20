# Hugging Face Transformers integration

This example creates two tiny, distinct GPT-2 `safetensors` checkpoints and
evaluates them on 50 deterministic paired records. The baseline checkpoint
suppresses the expected token; the subject checkpoint favors it. A passing
normalized-NLL decision therefore comes from real model scoring. The report
also derives a token-weighted perplexity ratio as an interpretation when the
authenticated tokenizer and target-token counts are comparable; that derived
value does not control acceptance.

The generated workspace keeps the closed evaluation request in `evaluation/`,
independent verifier inputs and its signing key in `verifier/`, and the evidence
signing key in `keys/`. The script refuses to reuse an existing workspace or
evidence output.

Run the complete journey from the repository root:

```bash
make example-hf-transformers
```

That one command creates an exact Git source bundle, builds the source-bound CPU
runtime image, creates two distinct local checkpoints, invokes the public CLI
commands, and prints the disposable output directory. Each model runs in its
own network-disabled, read-only CPU worker; the host retains the
evidence-signing key, publishes the evidence pack, verifies it against
independent artifact, schedule, runtime, policy, and signer inputs, and writes
an HTML report.

To inspect the generated request and trust inputs before spending container
time, use a new workspace with `--prepare-only`:

```bash
make example-hf-transformers \
  EXAMPLE_ARGS="--prepare-only --workspace /tmp/invarlock-hf-inputs"
```

Preparation uses an explicit placeholder runtime digest because no image is
executed. The complete journey derives the real image digest and passes it in
the generated `invarlock/trust-inputs-v1` profile to `invarlock verify`.

The 50 records intentionally repeat one deterministic prompt. This journey
demonstrates real scoring, isolation, and trust bindings; it is not a benchmark
or a model-quality claim.
