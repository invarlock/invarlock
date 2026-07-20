# Runnable integrations

Every entry in this directory has one maintained command that performs the
named upstream operation or runtime call and completes InvarLock's
`evaluate → verify → report` transaction.

| Integration | Command | Execution |
| --- | --- | --- |
| [Hugging Face Transformers](hf-transformers/) | `make example-hf-transformers` | CPU, local deterministic checkpoints |
| [Hugging Face PEFT](peft-lora/) | `make example-peft-lora` | CPU, real LoRA save/reload/merge |

Optional runtime distributions keep their executable journeys beside the code
they exercise:

- GGUF and llama.cpp: `addins/gguf`
- Hugging Face vision-text: `addins/multimodal`
- TensorRT-LLM: `addins/tensorrt_llm`

Those packages expose conformance, image smoke, canary, readiness, and
evidence-qualification targets for real model or engine fixtures. Their
runbooks define the exact environment and preflight contract; they are not
presented as zero-input tutorials.

The root `make example-evidence-handoff` command runs accepted, policy-rejected,
and tampered evidence through independently signed verification.
