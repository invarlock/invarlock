# PEFT LoRA integration

This journey invokes Hugging Face PEFT directly. It creates a tiny local GPT-2
baseline, trains LoRA parameters on a fixed target token, saves and reloads the
adapter, merges it into a standalone `safetensors` checkpoint, and passes the
baseline and merged subject to InvarLock's built-in Hugging Face runtime.

From a clean checkout with `uv` and Docker or Podman installed:

```bash
make example-peft-lora
```

The command installs the locked PEFT example dependency in an isolated
environment, builds the source-bound runtime image, then completes
`evaluate`, independent `verify`, and `report`. It prints the disposable
workspace containing:

- the serialized PEFT adapter and an upstream-operation summary;
- the merged subject checkpoint;
- the canonical evidence pack;
- the signed independent verification receipt; and
- the HTML comparison report.

To prepare and inspect every input without building an image or starting a
worker:

```bash
make example-peft-lora \
  EXAMPLE_ARGS="--prepare-only --workspace /tmp/invarlock-peft-inputs"
```

The 50 deterministic records exercise the integration and transaction. They
do not measure general fine-tuning quality. A release decision should use a
representative, digest-pinned schedule and a policy with an appropriate sample
and precision requirement.
