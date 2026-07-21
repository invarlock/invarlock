# Runnable integrations

Every entry in this directory has one maintained command that performs the
named upstream operation or runtime call and completes InvarLock's
`evaluate → verify → report` transaction.

| Integration | Command | Execution |
| --- | --- | --- |
| [Hugging Face Transformers](hf-transformers/) | `make example-hf-transformers` | Qwen3-0.6B checkpoint and an explicit behavioral derivative |
| [Hugging Face vision-text](hf-vision-text/) | `make example-hf-vision-text` | Qwen2-VL 2B and 7B checkpoints on an authenticated four-color image fixture |
| [Hugging Face PEFT](peft-lora/) | `make example-peft-lora` | Qwen3-0.6B LoRA training, save/reload, and merge |
| [TorchAO](torchao-int8/) | `make example-torchao-int8` | Qwen3-0.6B INT8 weight-only quantization and a materialized checkpoint |
| [GGUF with llama.cpp](gguf-llama-cpp/) | `make example-gguf-llama-cpp` | Official Qwen3-0.6B Q8 GGUF and an authenticated Q5 derivative |
| [LM Evaluation Harness](lm-evaluation-harness/) | `make example-lm-evaluation-harness` | CPU, real upstream per-record output imported into the evidence transaction |
| [TensorRT-LLM](tensorrt-llm/) | `make example-tensorrt-llm` | Linux, Docker, two H100 GPUs, and Qwen3-0.6B BF16-to-FP8 conversion |

All seven commands obtain or create their artifacts and complete the transaction
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
