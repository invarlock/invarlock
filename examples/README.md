# Examples: choose a workflow

These examples are for developers and ML engineers who want to compare a model
change, use results from an existing evaluator, or check evidence supplied by
another team. You should be comfortable running Python commands in a terminal;
you do not need to know InvarLock's file formats before starting.

**New to InvarLock? Start with the [CPU quickstart](quickstart/README.md).** It
verifies an included comparison and creates a report without running a model.
Use the installed package and example files from the same release or source
revision, as explained in [getting started](../docs/user-guide/getting-started.md#matching-wheels-and-examples).

## Choose by what you want to do

| Your task | Example | What it runs |
| --- | --- | --- |
| Check someone else's result | [CPU quickstart](quickstart/README.md) | Offline verification of included evidence, followed by a receipt and HTML report |
| Evaluate results you already have | [Captured results](captured-results/README.md) | An offline starter, then paths for exact match, normalized NLL and judge scoring |
| Collect answers from your pipeline | [Answer capture](answer-capture/README.md) | An offline demonstration you can adapt to your own pipeline executable or Python module |
| Run a model and judge its answers | [Native judge](native-judge/README.md) | Setup for model execution and bounded judge calls; requires your runtime and judge configuration |
| Evaluate already collected judge ratings | [Judge import](judge-measurements/README.md) | Offline replay of a small included fixture with an expected insufficient-evidence result |
| Require deterministic and judge checks together | [Combined checks](judge-with-deterministic/README.md) | An offline example with a single recipient decision over both results |
| Compare two HTTP service runs | [Hosted service](hosted-service/README.md) | New calls to an endpoint you supply, followed by offline evaluation and verification |
| Inspect actual model comparison results | [Retained comparisons](captured-results/references/README.md) | Offline replay of routing, likelihood and HTTP results, including policy rejections |
| Inspect a larger judge study | [Held-out judge reference](judge-measurements/references/k2-32b-luna-xhigh-heldout/README.md) | Offline replay of 10,260 retained ratings across fixed QA and extraction tasks |
| Check a delivered model package | [ModelKit handoff](integrations/modelkit-handoff/README.md) | Checks that the package matches approved evidence and recipient policy |
| Add verification to CI or a policy engine | [CI examples](ci/README.md) and [OPA/CUE example](policy-engine-interop/README.md) | Automation patterns with their supported receipt and policy boundaries |

## Terms used in the examples

- **Baseline** is the model or configuration you compare against. **Subject** is
  the proposed model, prompt or configuration change.
- A **case** is one task input with any required reference answer. A **pair**
  contains the baseline and subject measurements for the same case.
- A **policy** contains the acceptance requirements you choose before evaluating
  results, such as an allowed score change or minimum number of cases.
- An **evidence pack** retains the comparison's inputs, measurements and result.
  A **verification receipt** records the separate verifier's checks.
- A **fixture** is included example data. Synthetic fixtures test that the
  workflow works; retained model references contain actual measurements. Neither
  automatically establishes that a model meets your own workload's requirements.

`evaluate` creates the comparison result, `verify` checks evidence against the
recipient's independently supplied expectations, and `report` presents the
recorded result. Rendering a report does not perform recipient verification.
Some examples intentionally fail their policy: reproducing that rejection is
successful verification of the example, not acceptance of the subject.

## Run a model-backed integration

These examples are for readers ready to run models or connect a specific
framework. Start with the linked guide for hardware, dependencies and a complete
command. The Make targets below are entry points, not complete invocations.

| Integration guide | Entry point | Comparison |
| --- | --- | --- |
| [Hugging Face Transformers](integrations/hf-transformers/README.md) | `make example-hf-transformers` | A pinned Qwen3.5-0.8B checkpoint and a defined behavioral change |
| [Hugging Face vision-text](integrations/hf-vision-text/README.md) | `make example-hf-vision-text` | Qwen2-VL 2B and 7B on a four-record image fixture |
| [PEFT LoRA merge](integrations/peft-lora/README.md) | `make example-peft-lora` | Adapter training, reload, merge and model comparison |
| [TorchAO INT8](integrations/torchao-int8/README.md) | `make example-torchao-int8` | Weight-only quantization, live-kernel observations and checkpoint comparison |
| [GGUF with llama.cpp](integrations/gguf-llama-cpp/README.md) | `make example-gguf-llama-cpp` | Qwen3.5-0.8B Q8 and a Q5 derivative |
| [BF16-to-GGUF deployment](integrations/gguf-deployment/README.md) | `make example-gguf-deployment` | Selected 8B, 9B and 27B models across Transformers BF16 and llama.cpp Q5_K_M |
| [LM Evaluation Harness](integrations/lm-evaluation-harness/README.md) | `make example-lm-evaluation-harness` | Imports per-case model results; does not rely on the evaluator's aggregate score |
| [TensorRT-LLM](integrations/tensorrt-llm/README.md) | `make example-tensorrt-llm` | Qwen3-0.6B BF16 and calibrated FP8 engines on two H100 GPUs |
| [Offline evidence handoff](integrations/README.md) | `make example-evidence-handoff` | Included paired records, policy rejection and file-tampering rejection; no model or GPU |

Completed transactions write a signed evidence pack, a verifier receipt and an
evidence report. Model-backed runs require caller-owned evidence and verifier
keys, a new directory for recipient trust inputs, and a clean committed checkout
so the runtime can be tied to its source. Some evaluator integrations also
require builder keys. Follow [integration setup](integrations/README.md) for the
key roles and environment requirements. The offline handoff uses disposable
example keys.

The Transformers, vision-text, PEFT and TorchAO guides also show
`--prepare-only`. It skips the full evaluation, but Transformers, PEFT and
TorchAO preparation still download or prepare model artifacts. Vision-text
preparation writes inputs without downloading checkpoints. Their `--ephemeral-trust-root` option creates throwaway demonstration
keys and does not establish recipient acceptance.

These compact schedules demonstrate integration behavior. Use an appropriate
case set and policy for an actual model decision. The separate
[K2 Horizon qualification protocol](qualification/k2-horizon/README.md) is for
maintainers validating runtime support; CPU preparation does not qualify its
five model configurations.

## Adapt an example to your own comparison

Choose cases and requirements that match your task before collecting results.
For native model execution, see the [model-change guide](../docs/user-guide/change-scenarios.md).
For existing evaluator records, follow the
[captured-results guide](../docs/user-guide/captured-results.md). To understand
which inputs the reviewer must supply independently, use the
[evidence and verification guide](../docs/user-guide/evidence-and-verification.md).
