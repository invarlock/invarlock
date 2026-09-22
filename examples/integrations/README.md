# Runnable integrations

Use these examples to compare an actual model change, connect a runtime, or
bring an evaluator's per-record results into a signed InvarLock comparison.
Start with the table below to choose the operation and hardware you need.
The model examples download or create artifacts and execute real inference;
SPDX checks a small metadata fixture, and ModelKit verifies an existing handoff.

The launchers live in this checkout. They are not additional installed CLI
commands. Native model examples use InvarLock runtime providers; the Harness,
Inspect and OpenAI Evals examples execute upstream evaluators in containers and
import their complete records through the native evidence contract. LM
Evaluation Harness and Inspect AI also have retained signed transactions you
can verify without running a model.

The evaluator transaction contracts, native adapters, and bounded result
transfer helper are example-owned support under
`examples/integrations/evaluator_transaction/`. They are not evaluator plugins
or part of InvarLock's installed evaluator-neutral API. Each signed evaluator
launcher also removes the exact temporary base and child image tags it created
after the journey, including when a retained workspace is requested.

For an evaluator workflow that already runs elsewhere, save its original per-case
SDK results as JSON and use `adapter: evaluator-native-json` in the
[captured-results journey](../captured-results/README.md). Dedicated mappings
cover all 19 maintained ecosystems through the same evaluation, verification and
reporting interface. Keep your evaluator environment separate; the recipient
needs only core InvarLock, with no evaluator SDK or account. The
[native shape recipes](../evaluator-qualification/maintained/CAPTURE.md#dedicated-native-shapes)
show the required SDK fields, result tables and per-case wrappers.

When dependencies permit co-installation, `invarlock.engine.export_evaluator_result`
is an optional convenience for SDK objects and writes an envelope for
`adapter: evaluator-json`. Both routes preserve the same native facts. Preserve
independently planned IDs, actual model/source identities and metadata slices;
optional numeric metrics require explicit recorded-score provenance.
`capture_evaluator_run` remains the public route for explicitly mapped canonical
records.

Captured comparisons can select InvarLock exact match, normalized NLL or judge
scoring when the required facts are available. Explicit projections preserve
structured task inputs, typed likelihoods bind actual reference measurements,
and judge evidence retains complete calls under the declared recipe. An aggregate
score cannot supply missing cases or establish any of these measurements.
The [Mistral 7B sentinel](evaluator-live/references/mistral-7b-sentinel/README.md)
retains real two-model execution across all 19 profiles and independently
replayable results for all three scorers. The smaller
[Harness likelihood control](../captured-results/references/harness-likelihood/README.md)
retains six same-model CPU pairs. Both are separate from the native signed OCI
profiles listed below.

The [priority workflow reference](evaluator-live/references/priority-workflows/README.md)
adds 64-case local comparisons, controlled HTTP-service comparisons and live judge
stop/resume and budget checks for Inspect, Harness, Promptfoo and Langfuse. It
preserves policy rejections and insufficient-evidence outcomes alongside the
independently verified results.

| Integration | Command | Execution |
| --- | --- | --- |
| [Hugging Face Transformers](hf-transformers/) | `make example-hf-transformers` | Qwen3.5-0.8B checkpoint and an explicit behavioral derivative |
| [Hugging Face vision-text](hf-vision-text/) | `make example-hf-vision-text` | Qwen2-VL 2B and 7B checkpoints on an authenticated four-color image fixture |
| [Hugging Face PEFT](peft-lora/) | `make example-peft-lora` | Qwen3.5-0.8B LoRA training, save/reload, and merge |
| [TorchAO](torchao-int8/) | `make example-torchao-int8` | Qwen3.5-0.8B INT8 weight-only quantization and a materialized checkpoint |
| [GGUF with llama.cpp](gguf-llama-cpp/) | `make example-gguf-llama-cpp` | Official Qwen3.5-0.8B Q8 GGUF and an authenticated Q5 derivative |
| [BF16-to-GGUF deployment](gguf-deployment/) | `make example-gguf-deployment` | Closed Qwen3.5 9B, Qwen3.8 27B, and Ministral 3 8B profiles executed through Transformers/CUDA and derived Q5_K_M GGUFs through llama.cpp/CPU |
| [SPDX 3.0.1 AI observation](spdx-ai-observation/) | `make example-spdx-ai-observation` | CPU-only mapping of one canonical AI document into the existing authenticated observation boundary; no SPDX conformance or acceptance claim |
| [ModelKit recipient handoff](modelkit-handoff/) | `python examples/integrations/modelkit_handoff.py --request recipient.json` | Offline package/content binding, evidence replay, and current recipient acceptance against actual model directories |
| [LM Evaluation Harness](lm-evaluation-harness/) | `make example-lm-evaluation-harness` | Real upstream per-record output across compact CPU and retained CUDA profiles |
| [Inspect AI](inspect-ai/) | `make example-inspect-ai` | Native Inspect Task/scorer execution across compact CPU and retained CUDA profiles |
| [OpenAI Evals](openai-evals/) | `make example-openai-evals` | CPU, example-owned OpenAI Evals Match adapter; no retained signed transaction |
| [TensorRT-LLM](tensorrt-llm/) | `make example-tensorrt-llm` | Linux, Docker, two compatible CUDA GPUs, and a Qwen3-0.6B compatibility fixture for the pinned runtime |

## Before running a model example

Run commands from a clean, committed repository root with Git, Make, Python 3.12
or newer, and `uv` available. The Make targets select the repository's locked
dependencies. You also need the container engine and hardware specified by the
chosen page, network access for initial downloads/builds, and space for model
snapshots and image layers. A successful package installation alone does not
establish that the selected model fits your hardware.

Paths under `/secure` in the commands are placeholders. Replace them with real
paths to Ed25519 PEM keys and a new trust-root directory outside the workspace.
The evidence key signs the evaluation pack; the separate verifier key signs the
recipient's replay result. Evaluator bridges also use a builder key to sign the
runtime image identity and its matching public key to check that signature.
Choose a new trust root and workspace for each invocation.

The maintained evaluator commands require caller-owned Ed25519 evidence,
verifier, and builder key material plus a new trust-root directory. Keep the
builder private key for image construction and provide only its public key to
completion. Keep all keys and the trust root
outside the transaction workspace, for example:

```bash
make example-inspect-ai EXAMPLE_ARGS="--evidence-signing-key /secure/keys/evidence.pem --verifier-signing-key /secure/keys/verifier.pem --builder-signing-key /secure/keys/builder.pem --builder-public-key /secure/keys/builder-public.pem --trust-root /secure/trust/inspect-ai"
```

Use a path whose directory components are not symlinks. On macOS, use the
resolved system temporary directory rather than the `/tmp` alias, which is
rejected because it is a symlink. Production keys and trust roots belong in
the operator's protected storage.

Apply the evidence, verifier, and trust-root options to the Hugging Face, PEFT,
TorchAO, vision-text, GGUF, and prepared TensorRT-LLM commands. The three
signed evaluator commands additionally require the builder signing/public-key
pair shown above. `--ephemeral-trust-root` is an
explicit disposable-demo escape hatch for legacy workers; it is not an
acceptance workflow and is never used by the signed evaluator bridge.

Completion reruns each evaluator inside the inspected, source-bound image and
retains the upstream per-record outputs in signed provenance. Prepared worker
outputs are never authoritative.

All maintained commands obtain or create their artifacts and complete the transaction
from a clean committed checkout. The TensorRT-LLM showcase builds its engines
on the selected compatible GPUs and authenticates the resulting engine identities; it does
not assume that independently compiled engine bytes will be identical. The
first-party runtime providers also expose conformance and real-model
qualification commands beside their implementations in the core package.

The GPU-backed checkpoint examples accept `--runtime-device cuda:1` when several
accelerators are available. The three evaluator bridges use `--device cuda:1`
instead. Append optional flags to the full `EXAMPLE_ARGS` value shown on each
page; replacing it with only a workspace or device flag drops the required key
and trust-root arguments.

## Check the result

Successful model journeys print the workspace and the evidence, verification
receipt, and HTML report paths. Keep the complete evidence directory immutable
and retain the separate receipt and recipient trust inputs. The HTML report
explains the measured comparison; the independent verification result tells you
whether that evidence met the selected policy under those trust inputs.

An authentic comparison can fail its policy. Where an evaluator page documents
`--allow-policy-fail`, that flag retains the rejection for inspection; it does
not change the verdict or permit an integrity failure. If preparation or a
worker fails, fix the reported prerequisite and use new output paths. None of
these small fixtures establishes quality for your deployment workload.

The root `make example-evidence-handoff` command runs accepted, policy-rejected,
and tampered evidence through separately signed verification.
