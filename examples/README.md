# Runnable examples

These examples execute InvarLock's public transaction instead of describing a
hypothetical integration. Each maintained command produces a signed evidence
pack, verifies it against independently supplied trust inputs, and renders a
human-readable report.

| Journey | Command | What actually runs |
| --- | --- | --- |
| Hugging Face Transformers | `make example-hf-transformers` | Two distinct local `safetensors` checkpoints scored by the built-in provider |
| PEFT LoRA merge | `make example-peft-lora` | Real adapter training, save, reload, merge, model scoring, verification, and reporting |
| Evidence handoff | `make example-evidence-handoff` | Imported paired records, separate evidence and verifier keys, policy rejection, and byte-tamper rejection |

The first two journeys live in [`integrations/`](integrations/). They create a
fresh workspace, build the exact checked-out source into a source-bound runtime
image, invoke `invarlock evaluate`, `invarlock verify`, and `invarlock report`,
then print the disposable output directory.

The evidence-handoff journey uses the committed fixtures at this directory
root. It deliberately includes an accepted comparison, a valid policy failure,
and an integrity failure so the trust boundary can be inspected without a
model download or GPU.

The optional GGUF, multimodal, and TensorRT-LLM packages keep their runtime
qualification commands beside their implementations under `addins/`. Those
commands operate on real model or engine fixtures and remain qualification
workflows rather than pretending to be zero-input tutorials.

## Inspect inputs without starting a runtime

Both model journeys support preparation-only mode:

```bash
make example-hf-transformers \
  EXAMPLE_ARGS="--prepare-only --workspace /tmp/invarlock-hf-inputs"

make example-peft-lora \
  EXAMPLE_ARGS="--prepare-only --workspace /tmp/invarlock-peft-inputs"
```

Preparation writes the request, policy, schedule, checkpoints, keys, and trust
profile. Complete execution requires a clean committed checkout because the
runtime image is authenticated against that exact Git source tree.

These small schedules demonstrate integration behavior. Public model evidence
uses representative digest-pinned schedules and explicit precision controls;
the tutorials do not make general model-quality claims.

For the underlying contracts, see the
[getting-started guide](../docs/user-guide/getting-started.md),
[model-change workflow guide](../docs/user-guide/change-scenarios.md), and
[evidence and verification guide](../docs/user-guide/evidence-and-verification.md).
