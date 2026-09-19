# Model-change workflows

InvarLock compares an authenticated baseline with an authenticated subject on
the same paired schedule, applies a named policy, and creates evidence another
party can verify independently. The system that trained, pruned, quantized,
converted, or compiled the subject remains responsible for creating that
artifact.

> **User guide**
>
> **In plain language:** Create the candidate with the tool you already use,
> then give InvarLock the immutable before-and-after artifacts or complete
> per-record results.
>
> **Outcome:** A signed comparison bundle, an independently signed
> verification receipt, and an evidence report.
>
> **Audience:** Model adaptation, runtime, evaluation, and release teams.
>
> **Prerequisites:** Stable record IDs, a representative paired schedule, a
> policy, and independently managed artifact, runtime, signer, and policy
> identities.

## Choose the execution boundary

| Candidate form | InvarLock path | Maintained implementation |
| --- | --- | --- |
| Hugging Face causal checkpoint | Run mode | Built-in `hf_transformers` provider |
| PEFT adapter merged into a checkpoint | Run mode | Built-in provider; [runnable PEFT journey](https://github.com/invarlock/invarlock/tree/main/examples/integrations/peft-lora) |
| TorchAO weight-only quantization materialized as a checkpoint | Run mode | Built-in provider; [runnable TorchAO journey](https://github.com/invarlock/invarlock/tree/main/examples/integrations/torchao-int8) |
| GGUF artifact | Run mode | Built-in `llama_cpp` provider; [runnable llama.cpp journey](https://github.com/invarlock/invarlock/tree/main/examples/integrations/gguf-llama-cpp) |
| Vision-text checkpoint | Run mode | Built-in `hf_vision_text` provider |
| TensorRT-LLM engine | Run mode | Built-in `tensorrt_llm` provider |
| Complete InvarLock provider sidecars produced by a harness | Native import mode | Runtime-import authoring API and closed request contract; [runnable LM Evaluation Harness journey](https://github.com/invarlock/invarlock/tree/main/examples/integrations/lm-evaluation-harness) |
| Evaluator exports or hosted endpoint captures | Captured mode | [Captured results](captured-results.md), with complete-run pins and captured assurance |

The runtime must match the artifact that will be released. A quantized model
loaded by llama.cpp should be evaluated through the GGUF provider rather than a
different Hugging Face representation. A compiled TensorRT-LLM engine remains
the subject; InvarLock does not rebuild it from its source checkpoint.

## Keep the change boundary explicit

```text
training · pruning · quantization · conversion · compilation
                             |
                             v
              immutable candidate or paired records
                             |
                             v
                invarlock evaluate request.yaml
                             |
                             v
                 signed canonical evidence bundle
                             |
                             v
                independent verify → report
```

Native requests can attach configuration, lineage, throughput, memory, sparsity,
and similar facts as authenticated observations. Observation payloads have no
acceptance authority; the selected scorer and its policy use their own
authenticated inputs.

## Select a metric that matches the task

- Use exact match for closed-answer tasks. Native pack-v1 reports include paired
  regressions, paired improvements, effect size, an interval, and McNemar's exact
  test. Captured exact-match comparisons report the effect and interval without
  those discordance counts or McNemar probability.
- Use normalized NLL for expected-continuation likelihood. It does not measure
  general model quality. When tokenizers and target-token accounting are
  comparable, native pack-v1 reports render perplexity ratio as a derived
  interpretation, not a second acceptance metric. Captured NLL reports retain
  the mean ratio and its interval without that perplexity interpretation.
- Use a verifier-replayable native scorer extension for task-specific F1,
  structured extraction, VQA normalization, or another deterministic text score
  computed only from authenticated record facts.
- Use the built-in `judge` scorer and [native judge workflow](evaluation-request.md#judge)
  when fixed answers, a declared
  rubric, a supported text judge, repeated ratings, and independent-unit
  analysis fit the decision. Keep other model-judge results as authenticated
  observations.

## Prepare a meaningful paired schedule

Every conclusion is limited by its schedule. Native baseline and subject
observations must use the same stable IDs in schedule order. Captured
deterministic comparisons require identical ID sets and matching input,
reference and metadata facts; they pair records in sorted ID order. Select
records from the real task distribution, record the source revision and selection method, and include
important subgroups.

A small tutorial can prove that integration code works; it cannot support a
release conclusion. The maintained public qualification suites use 400
balanced records. Production policies should set both a minimum record count
and a maximum interval width, then increase the sample when the observed
precision or subgroup coverage is inadequate.

## Execute, verify, and report

Run execution-free preflight before allocating model compute:

```console
invarlock evaluate request.yaml --signing-key evidence-signer.pem \
  --runtime-image "$RUNTIME_IMAGE" --runtime-image-digest "$RUNTIME_DIGEST" \
  --preflight
```

Resolve each artifact, provider, schedule, policy, runtime, and destination
error. Preflight validates deterministic prerequisites; it cannot predict model
behavior.

Run the same request without `--preflight`, deliver the immutable evidence pack
to the verifier, and provide expected identities through a separate channel:

```console
invarlock evaluate request.yaml --signing-key evidence-signer.pem \
  --runtime-image "$RUNTIME_IMAGE" --runtime-image-digest "$RUNTIME_DIGEST"
invarlock verify evidence/ --trust-profile verifier/trust-inputs.json \
  --receipt verifier/verification.receipt.json
invarlock report evidence/ --html verifier/report.html --explain
```

Import mode uses the same public transaction without runtime-image arguments.
It requires complete provider sidecars and paired records, not aggregate scores.
For evaluator exports and hosted captures, use captured mode and its independent
run/request pins. Deterministic scorers replay retained facts; judge verification
replays admitted ratings and analysis without calling the judge again.

## Interpret the outcome

A passing receipt says that the named subject met the named policy relative to
the named baseline on the authenticated paired schedule under the recorded
runtime identities. A policy failure can still be intact evidence. An integrity
or trust mismatch means the submitted comparison is not independently
acceptable.

Read every result together with the schedule composition, paired effect and
interval, metric semantics, artifact and runtime identities, policy thresholds,
and separately authenticated operational observations.

## Run the maintained journeys

The repository examples exercise real public commands:

```console
make example-hf-transformers
make example-hf-vision-text
make example-peft-lora
make example-torchao-int8
make example-gguf-llama-cpp
make example-lm-evaluation-harness
make example-tensorrt-llm
make example-evidence-handoff
```

The TensorRT-LLM showcase downloads one pinned Qwen3-0.6B revision and builds
BF16 and ModelOpt-calibrated FP8 engines on separate compatible CUDA GPUs before
completing the signed transaction.
It remains a backend-compatibility fixture because the pinned TensorRT-LLM
1.2.1 runtime supports Qwen3 but does not provide a Qwen3.5 adapter; the other
compact examples use Qwen3.5 0.8B.
Its README also documents the lower-level command for qualified,
caller-prepared engines. Both paths bind the observed engine identities rather
than assuming independently compiled engine bytes will be identical. Optional
runtime providers also expose conformance and evidence-qualification targets
beside their implementations.
