# Model-change workflows

InvarLock compares an authenticated baseline with an authenticated subject on
the same paired schedule, applies a named policy, and creates evidence another
party can verify independently. The system that trained, pruned, quantized,
converted, or compiled the subject remains responsible for creating that
artifact.

!!! tip "User guide"

    **In plain language:** Create the candidate with the tool you already use,
    then give InvarLock the immutable before-and-after artifacts or complete
    per-record results.

    **Outcome:** A signed comparison bundle, an independently signed
    verification receipt, and a human-readable report.

    **Audience:** Model adaptation, runtime, evaluation, and release teams.

    **Prerequisites:** Stable record IDs, a representative paired schedule, a
    policy, and independently managed artifact, runtime, signer, and policy
    identities.

## Choose the execution boundary

| Candidate form | InvarLock path | Maintained implementation |
| --- | --- | --- |
| Hugging Face causal checkpoint | Run mode | Built-in `hf_transformers` provider |
| PEFT adapter merged into a checkpoint | Run mode | Built-in provider; [runnable PEFT journey](https://github.com/invarlock/invarlock/tree/main/examples/integrations/peft-lora) |
| GGUF artifact | Run mode | Optional [`invarlock-runtime-gguf`](https://github.com/invarlock/invarlock/tree/main/addins/gguf) package |
| Vision-text checkpoint | Run mode | Optional [`invarlock-runtime-hf-vision-text`](https://github.com/invarlock/invarlock/tree/main/addins/multimodal) package |
| TensorRT-LLM engine | Run mode | Optional [`invarlock-runtime-tensorrt-llm`](https://github.com/invarlock/invarlock/tree/main/addins/tensorrt_llm) package |
| Complete per-record results from a harness or endpoint | Import mode | Runtime-import authoring API and closed request contract |

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

Configuration, lineage, throughput, memory, sparsity, and similar facts can be
attached as authenticated observations. They do not influence acceptance
unless a versioned policy and replayable scorer explicitly authorize them.

## Select a metric that matches the task

- Use exact match for closed-answer tasks. InvarLock reports paired regressions,
  paired improvements, effect size, an interval, and McNemar's exact test.
- Use normalized NLL for expected-continuation likelihood. It does not measure
  general model quality. When tokenizers and target-token accounting are
  comparable, the report renders perplexity ratio as a derived interpretation,
  not a second acceptance metric.
- Use a verifier-replayable scorer extension for task-specific F1, structured
  extraction, code execution, SQL execution, semantic similarity, VQA
  normalization, or another deterministic score.
- Keep model-judge results as authenticated observations until the judge,
  prompt, references, and calibration have an independently verifiable
  acceptance contract.

## Prepare a meaningful paired schedule

Every conclusion is limited by its schedule. Baseline and subject must use the
same stable IDs in the same order. Select records from the real task
distribution, record the source revision and selection method, and include
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
It accepts complete per-record results with stable IDs, not aggregate scores.
The verifier recomputes the named built-in or extension score from those
records.

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
make example-peft-lora
make example-evidence-handoff
```

The optional runtime packages expose conformance, image smoke, canary,
readiness, and evidence-qualification targets in their package-owned
Makefile. Their runbooks document the required real model or engine fixtures.
