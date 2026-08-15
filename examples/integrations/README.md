# Runnable integrations

Every entry in this directory has one maintained command for the named
upstream operation or runtime call. LM Evaluation Harness and Inspect AI also
have retained model-running signed `evaluate → verify → report` OCI journeys.
The remaining entries demonstrate their named model-change, runtime, or
evaluator compatibility paths.

The evaluator transaction contracts, native adapters, and bounded result
transfer helper are example-owned support under
`examples/integrations/evaluator_transaction/`. They are not evaluator plugins
or part of InvarLock's installed evaluator-neutral API. Each signed evaluator
launcher also removes the exact temporary base and child image tags it created
after the journey, including when a retained workspace is requested.

| Integration | Command | Execution |
| --- | --- | --- |
| [Hugging Face Transformers](hf-transformers/) | `make example-hf-transformers` | Qwen3.5-0.8B checkpoint and an explicit behavioral derivative |
| [Hugging Face vision-text](hf-vision-text/) | `make example-hf-vision-text` | Qwen2-VL 2B and 7B checkpoints on an authenticated four-color image fixture |
| [Hugging Face PEFT](peft-lora/) | `make example-peft-lora` | Qwen3.5-0.8B LoRA training, save/reload, and merge |
| [TorchAO](torchao-int8/) | `make example-torchao-int8` | Qwen3.5-0.8B INT8 weight-only quantization and a materialized checkpoint |
| [GGUF with llama.cpp](gguf-llama-cpp/) | `make example-gguf-llama-cpp` | Official Qwen3.5-0.8B Q8 GGUF and an authenticated Q5 derivative |
| [BF16-to-GGUF deployment](gguf-deployment/) | `make example-gguf-deployment` | Closed Qwen3.5 9B and Ministral 3 8B profiles executed through Transformers/CUDA and derived Q5_K_M GGUFs through llama.cpp/CPU |
| [LM Evaluation Harness](lm-evaluation-harness/) | `make example-lm-evaluation-harness` | Real upstream per-record output across compact CPU and retained CUDA profiles |
| [Inspect AI](inspect-ai/) | `make example-inspect-ai` | Native Inspect Task/scorer execution across compact CPU and retained CUDA profiles |
| [OpenAI Evals](openai-evals/) | `make example-openai-evals` | CPU, maintained native OpenAI Evals Match adapter; signed journey not yet retained |
| [TensorRT-LLM](tensorrt-llm/) | `make example-tensorrt-llm` | Linux, Docker, two H100 GPUs, and a Qwen3-0.6B compatibility fixture for the pinned runtime |

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
on the target H100s and authenticates the resulting engine identities; it does
not assume that independently compiled engine bytes will be identical. The
first-party runtime packages also expose conformance and real-model
qualification commands beside their implementations under `addins/`.

The GPU-backed checkpoint examples accept an explicit device when several
accelerators are available, for example
`EXAMPLE_ARGS="--runtime-device cuda:1"`.

The root `make example-evidence-handoff` command runs accepted, policy-rejected,
and tampered evidence through separately signed verification.
