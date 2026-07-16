# Real Hugging Face run-mode example

This example creates two tiny, distinct GPT-2 `safetensors` checkpoints and
evaluates them on one deterministic paired record. The baseline checkpoint
suppresses the expected token; the subject checkpoint favors it. A passing
normalized-NLL decision therefore comes from real model scoring. The report
also derives a token-weighted perplexity ratio as an interpretation when the
authenticated tokenizer and target-token counts are comparable; that derived
value does not control acceptance.

The generated workspace keeps the closed evaluation request in `evaluation/`,
independent verifier inputs in `verifier/`, and both private keys in `keys/`.
The script refuses to reuse an existing workspace or evidence output.

Build and inspect the CPU runtime image first:

```bash
make runtime-image
docker image inspect --format '{{.Id}}' invarlock-runtime:local
```

Then run the complete `evaluate` → `verify` → `report` journey with the
inspected digest:

```bash
python examples/run/hf_cpu_decision.py \
  --workspace /tmp/invarlock-hf-cpu-example \
  --container-engine docker \
  --runtime-image invarlock-runtime:local \
  --runtime-image-digest sha256:<64-lowercase-hex>
```

The script invokes the public CLI commands directly. Each model runs in its own
network-disabled, read-only CPU worker; the host retains the evidence-signing
key, publishes the evidence pack, verifies it against independent artifact,
schedule, runtime, policy, and signer inputs, and writes an HTML report.

To inspect the generated request and trust inputs before spending container
time, use a new workspace with `--prepare-only`:

```bash
python examples/run/hf_cpu_decision.py \
  --workspace /tmp/invarlock-hf-cpu-inputs \
  --prepare-only
```

The example is deliberately small. It demonstrates the release-regression
transaction and trust bindings, not model quality or benchmark significance.
