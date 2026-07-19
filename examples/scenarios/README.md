# Change and deployment scenarios

These examples connect common model-development and deployment changes to the
distilled InvarLock transaction. The external tool or team creates the
candidate artifact or evaluation records. InvarLock authenticates the inputs,
compares the baseline and subject on one paired schedule, produces a canonical
evidence pack, verifies that pack against independently supplied trust inputs,
and renders the result.

The scenario catalog deliberately contains no fine-tuning, pruning,
quantization, conversion, or serving implementation. Each directory contains
only a closed scenario manifest and a runbook. That keeps integrations useful
to their respective ecosystems without moving artifact creation back into the
core package.

## Choose a scenario

| Scenario | Relevant users | Execution path | Availability |
| --- | --- | --- | --- |
| [`fine-tuned-checkpoint`](changes/fine-tuned-checkpoint/) | PEFT, TRL, and training-pipeline users | Built-in Hugging Face run mode | Operator recipe |
| [`pruned-checkpoint`](changes/pruned-checkpoint/) | Structured and unstructured pruning users | Hugging Face run mode or authenticated import | Operator recipe |
| [`hf-quantized-checkpoint`](changes/hf-quantized-checkpoint/) | AWQ, GPTQModel, HQQ, bitsandbytes, Quanto, and TorchAO users | Compatible provider or authenticated import | Operator recipe |
| [`gguf-conversion`](changes/gguf-conversion/) | llama.cpp and GGUF publishing users | Optional GGUF add-in | Operator recipe |
| [`tensorrt-deployment`](changes/tensorrt-deployment/) | TensorRT-LLM deployment users | Optional TensorRT-LLM add-in | Operator recipe |
| [`model-upgrade`](changes/model-upgrade/) | Model and application release teams | Built-in Hugging Face run mode | Operator recipe |
| [`multimodal-upgrade`](changes/multimodal-upgrade/) | Vision-language model teams | Optional vision-text add-in | Operator recipe |
| [`external-harness`](imports/external-harness/) | Evaluation-harness maintainers and users | Authenticated import | Operator recipe |
| [`serving-endpoint`](imports/serving-endpoint/) | Hosted inference and serving teams | Provider ABI plus authenticated import | Operator recipe |
| [`evidence-handoff`](journeys/evidence-handoff/) | Release reviewers and independent verifiers | Verify and report | Runnable |

## Common transaction

For an executed comparison, author a request using the relevant provider,
schedule, metric, and policy, then run an execution-free preflight before
allocating model compute:

```console
invarlock evaluate request.yaml --signing-key evidence-signer.pem \
  --runtime-image runtime.example/model-evaluator@sha256:... \
  --runtime-image-digest sha256:... --preflight
invarlock evaluate request.yaml --signing-key evidence-signer.pem \
  --runtime-image runtime.example/model-evaluator@sha256:... \
  --runtime-image-digest sha256:...
```

For imported records, omit the runtime-image options. Import mode authenticates
the paired records and both closed side-result sets instead of starting model
workers.

The verifier receives the immutable evidence pack and separately provisioned
trust profile, then signs its own receipt:

```console
invarlock verify evidence/ --trust-profile verifier/trust-inputs.json \
  --receipt verifier/verification.receipt.json \
  --verifier-signing-key verifier/verifier.pem \
  --verifier-identity release-verifier
invarlock report evidence/ --html verifier/report.html --explain
```

The ellipses above are placeholders and are not valid digests. Use exact
artifact, dataset, schedule, policy, runtime-image, signer, and verifier
identities in a real transaction.

## Maintainer check

The schema closes the scenario metadata, ensures each related repository path
exists, requires a consistent runbook structure, and rejects artifact-creation
scripts beneath these recipes:

```console
python scripts/checks/check_example_scenarios.py
```

The real CPU Hugging Face journey remains in [`examples/run/`](../run/). The
offline accepted, rejected, and tamper-detection transaction remains in the
parent [`examples/`](../) directory. Hardware-dependent add-ins retain their
runtime construction and qualification details beside their own packages.
