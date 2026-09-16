# PEFT LoRA integration

Use this example to compare the behavior of a base checkpoint and a merged
LoRA adapter. PEFT creates the model change; InvarLock evaluates the two saved
checkpoints through its native Hugging Face provider. The evidence therefore
applies to the merged checkpoint, not an unmerged adapter loaded by another
serving stack.

This journey invokes Hugging Face PEFT directly. It downloads the official
Apache-2.0 `Qwen/Qwen3.5-0.8B` checkpoint at immutable revision
`2fc06364715b967f1860aea9cf38778875588b17`, trains LoRA parameters on fixed
continuations, saves and reloads the adapter, and merges it into a standalone
`safetensors` checkpoint. The baseline and merged subject then pass through
InvarLock's built-in Hugging Face runtime.

From a clean committed checkout with `uv` and Docker or Podman installed, follow
the [shared key and path setup](../README.md#before-running-a-model-example), then run:

```bash
make example-peft-lora \
  EXAMPLE_ARGS="--evidence-signing-key /secure/keys/evidence.pem \
  --verifier-signing-key /secure/keys/verifier.pem \
  --trust-root /secure/trust/peft-lora"
```

The command installs the locked PEFT example dependency, builds the
source-bound runtime image, then completes
`evaluate`, separately signed `verify`, and `report`. It prints the disposable
workspace containing:

- the serialized PEFT adapter and an authenticated transformation summary;
- the merged subject checkpoint;
- the canonical evidence pack;
- the separately signed verification receipt; and
- the HTML comparison report.

To prepare and inspect every input without building an image or starting a
worker:

```bash
make example-peft-lora \
  EXAMPLE_ARGS="--prepare-only --workspace /new/path/invarlock-peft-inputs \
  --evidence-signing-key /secure/keys/evidence.pem \
  --verifier-signing-key /secure/keys/verifier.pem \
  --trust-root /secure/trust/peft-lora"
```

Replace `/new/path` with an existing parent directory without symlinks.
Preparation lets you inspect inputs but does not produce an independently
verified result. After a complete run, read the report alongside the separate
verification receipt to distinguish the measured change from policy acceptance.
Preparation still downloads the model and trains, saves, reloads and merges the
adapter. It skips container construction and evaluation, not LoRA training.

CUDA is selected when available and CPU remains supported. The first run needs
several gigabytes of download, cache, and workspace capacity. The 50 distinct
deterministic contexts exercise real Qwen3.5 adapter training and the complete
transaction. The authenticated transformation records the source model and
revision, target modules, training loss, saved adapter, and merged subject
identity. This compact journey does not measure general fine-tuning quality; a
release decision should use a representative, digest-pinned schedule and a
policy with an appropriate sample and precision requirement.
