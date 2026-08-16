# Runnable examples

These examples execute InvarLock's public transaction instead of describing a
hypothetical integration. Each maintained command produces a signed evidence
pack, verifies it against separately generated trust inputs, and renders a
human-readable report. The self-contained examples create both sets of keys;
an acceptance workflow should have the verifier owner choose and hold its own
trust anchors and signing key.

| Journey | Command | What actually runs |
| --- | --- | --- |
| Hugging Face Transformers | `make example-hf-transformers` | A pinned Qwen3.5-0.8B checkpoint and an explicit behavioral derivative scored by the built-in provider |
| Hugging Face vision-text | `make example-hf-vision-text` | Pinned Qwen2-VL 2B and 7B checkpoints scored on a four-record authenticated image fixture through the optional vision-text provider |
| PEFT LoRA merge | `make example-peft-lora` | Real adapter training, save, reload, merge, model scoring, verification, and reporting |
| TorchAO INT8 | `make example-torchao-int8` | Real weight-only quantization, exact dense-state materialization, authenticated live-kernel observations, and checkpoint comparison |
| GGUF with llama.cpp | `make example-gguf-llama-cpp` | An official Qwen3.5-0.8B Q8 GGUF and its authenticated Q5 derivative executed through a source-bound llama.cpp image |
| BF16-to-GGUF deployment | `make example-gguf-deployment` | Closed Qwen3.5 9B and Ministral 3 8B profiles executed as BF16 with Transformers and as source-derived Q5_K_M GGUFs with llama.cpp |
| LM Evaluation Harness | `make example-lm-evaluation-harness` | Real upstream per-record runs imported through a configuration- and sample-bound adapter; aggregate scores are ignored |
| TensorRT-LLM | `make example-tensorrt-llm` | A pinned-runtime Qwen3-0.6B compatibility fixture that builds BF16 and calibrated FP8 engines on two H100 GPUs |
| Evidence handoff | `make example-evidence-handoff` | Imported paired records, separate evidence and verifier keys, policy rejection, and byte-tamper rejection |

The model and harness journeys live in [`integrations/`](integrations/). They create a
fresh workspace, build the exact checked-out source into a source-bound runtime
image, invoke `invarlock evaluate`, `invarlock verify`, and `invarlock report`,
then print the disposable output directory.

The evidence-handoff journey uses the committed fixtures at this directory
root. It deliberately includes an accepted comparison, a valid policy failure,
and an integrity failure so the trust boundary can be inspected without a
model download or GPU.

The TensorRT-LLM journey builds the immutable runtime image and both engines on
the target H100 system. Its [README](integrations/tensorrt-llm/README.md) also
documents a lower-level command for qualified, caller-prepared engines. The
optional vision-text package keeps its real-model qualification command beside
its implementation under `addins/multimodal`.

## Prerequisites

| Journeys | Requirements |
| --- | --- |
| Hugging Face, PEFT, TorchAO | `uv`, Git, Docker or Podman, enough memory for Qwen3.5-0.8B, and network access for the first locked dependency and image build; CUDA is used when available |
| Hugging Face vision-text | Linux, Docker with the NVIDIA container runtime, one GPU with at least 24 GB of memory, `uv`, Git, first-run network access, and roughly 35 GB of temporary disk |
| GGUF | The common requirements plus roughly 3 GB of temporary disk for the compact GGUF comparison; the BF16 deployment profiles require Linux, a CUDA GPU with at least 24 GB of memory, 64 GB of system memory, and about 70 GB of free disk |
| LM Evaluation Harness | The common requirements plus the pinned Harness dependency; model execution is offline after the image build |
| TensorRT-LLM | Linux, Docker with two visible H100 GPUs, and roughly 20 GB of temporary disk |

Exact time and disk use depend on the local image and dependency caches. Every
complete journey requires a checkout with no tracked source changes because
its runtime is authenticated against the committed tree.

## Inspect inputs without starting a runtime

The Hugging Face, vision-text, PEFT, and TorchAO journeys support
preparation-only mode:

```bash
make example-hf-transformers \
  EXAMPLE_ARGS="--prepare-only --workspace /tmp/invarlock-hf-inputs"

make example-hf-vision-text \
  EXAMPLE_ARGS="--prepare-only --workspace /tmp/invarlock-vision-inputs"

make example-peft-lora \
  EXAMPLE_ARGS="--prepare-only --workspace /tmp/invarlock-peft-inputs"

make example-torchao-int8 \
  EXAMPLE_ARGS="--prepare-only --workspace /tmp/invarlock-torchao-inputs"
```

Preparation writes the request, policy, schedule, keys, trust profile, and the
authenticated vision content store. The vision-text preparation records the
immutable model coordinates without downloading either checkpoint. Complete
execution requires a clean committed checkout because the runtime image is
authenticated against that exact Git source tree.

These small schedules demonstrate integration behavior. Public model evidence
uses representative digest-pinned schedules and explicit precision controls;
the tutorials do not make general model-quality claims.

For the underlying contracts, see the
[getting-started guide](../docs/user-guide/getting-started.md),
[model-change workflow guide](../docs/user-guide/change-scenarios.md), and
[evidence and verification guide](../docs/user-guide/evidence-and-verification.md).
