# Change scenarios

InvarLock applies one evidence transaction to several model and runtime
changes. A training, compression, conversion, deployment, evaluation, or
serving system creates the candidate artifact or per-record results. InvarLock
starts at that boundary and authenticates what is compared, how it is
evaluated, which policy applies, and what an independent verifier accepted.

!!! tip "User guide"

    **In plain language:** Keep using the tool that fine-tunes, prunes,
    quantizes, converts, or serves the model. Give InvarLock the immutable
    before-and-after artifacts or their replayable paired results.

    **Outcome:** Select a scenario, prepare its baseline, subject, schedule,
    provider, metric, and policy, then produce one independently verifiable
    release decision.

    **Audience:** Model adaptation teams, runtime and inference engineers,
    evaluation-system maintainers, application release teams, and independent
    verifiers.

    **Prerequisites:** A materialized candidate or closed per-record result set,
    stable evaluation IDs, an appropriate policy, and independently managed
    trust inputs and signing identities.

## Select the release question

Choose the scenario from the decision being made, not from a tool name alone:

| Release question | Scenario | Typical path | Built-in metric |
| --- | --- | --- | --- |
| Did adaptation preserve expected-continuation likelihood? | [Fine-tuned checkpoint](https://github.com/invarlock/invarlock/tree/main/examples/scenarios/changes/fine-tuned-checkpoint) | Hugging Face run mode | Normalized NLL |
| Did removing weights remain within the likelihood policy? | [Pruned checkpoint](https://github.com/invarlock/invarlock/tree/main/examples/scenarios/changes/pruned-checkpoint) | Hugging Face run mode or import | Normalized NLL |
| Did a lower-precision artifact preserve paired task outcomes? | [Quantized checkpoint](https://github.com/invarlock/invarlock/tree/main/examples/scenarios/changes/hf-quantized-checkpoint) | Compatible provider or import | Exact match |
| Did GGUF conversion preserve closed-answer behavior? | [GGUF conversion](https://github.com/invarlock/invarlock/tree/main/examples/scenarios/changes/gguf-conversion) | Optional GGUF add-in | Exact match |
| Did a compiled engine preserve closed-answer behavior? | [TensorRT-LLM deployment](https://github.com/invarlock/invarlock/tree/main/examples/scenarios/changes/tensorrt-deployment) | Optional TensorRT-LLM add-in | Exact match |
| Is a replacement checkpoint acceptable relative to production? | [Model upgrade](https://github.com/invarlock/invarlock/tree/main/examples/scenarios/changes/model-upgrade) | Per-side providers | Exact match or normalized NLL |
| Did a vision-language change meet the VQA-style policy? | [Multimodal upgrade](https://github.com/invarlock/invarlock/tree/main/examples/scenarios/changes/multimodal-upgrade) | Optional vision-text add-in | Exact match |
| Can existing harness records support a replayed decision? | [External harness](https://github.com/invarlock/invarlock/tree/main/examples/scenarios/imports/external-harness) | Authenticated import | Built-in or scorer extension |
| Did an endpoint configuration preserve recorded outcomes? | [Serving endpoint](https://github.com/invarlock/invarlock/tree/main/examples/scenarios/imports/serving-endpoint) | Provider bridge plus import | Exact match or scorer extension |
| Does submitted evidence match independently maintained anchors? | [Evidence handoff](https://github.com/invarlock/invarlock/tree/main/examples/scenarios/journeys/evidence-handoff) | Verify and report | Recorded evidence metric |

The same ecosystem may use more than one scenario. For example, an AWQ or GPTQ
workflow may produce a Hugging Face checkpoint, a GGUF file, or a runtime-bound
engine. Select the scenario matching the artifact and loader that will actually
be released.

## Keep the external boundary explicit

The scenario begins after the external change has completed:

```text
training · pruning · quantization · conversion · compilation · serving job
                                  |
                                  v
                immutable candidate or paired records
                                  |
                                  v
                    invarlock evaluate request.yaml
                                  |
                                  v
                  canonical signed evidence bundle
                                  |
                                  v
               independent verify → human-readable report
```

This separation has practical consequences:

1. The selected training or compression framework creates the candidate
   artifact; InvarLock evaluates its authenticated output.
2. The external system records configuration and lineage needed to identify the
   candidate; InvarLock authenticates those facts as artifact, runtime, receipt,
   or observation inputs.
3. An artifact requiring a special loader uses the runtime-provider ABI or an
   authenticated import bridge rather than an implicit fallback.
4. Acceptance depends only on the selected paired metric or scorer and policy.
   Size, throughput, memory, sparsity, latency, and similar properties remain
   separately authenticated observations unless a future policy contract gives
   them explicit authority.

## Prepare a meaningful paired schedule

Every release conclusion is limited by its schedule. Stable IDs and identical
ordering across baseline and subject are mandatory. Select records from the
actual task distribution, record the source revision and selection method, and
exclude records that cannot be scored deterministically under the chosen
metric.

A tiny schedule can prove that the execution or import path works. It cannot
support a meaningful release conclusion. The maintained public qualification
suites use 400 balanced records. A scenario should normally start with
at least 400 eligible paired records, then increase that count when the
observed interval is too wide or the represented task has important subgroups.
Encode both a minimum record count and maximum interval width in policy; record
count alone does not guarantee precision or representativeness.

## Choose execution and scoring

Use run mode when an installed provider can authenticate and execute both
artifacts now. Use import mode when a controlled external job already produced
complete provider side files and canonical paired records. Import mode is not
an aggregate-score upload.

Use exact match for closed-answer tasks and unlike tokenizer/runtime paths. Use
normalized NLL for expected-continuation likelihood when both sides expose the
required authenticated likelihood facts. Use a scorer extension only when the
verifier can replay the exact descriptor, configuration, per-record inputs, and
results. Keep model judges and network-dependent scoring in authenticated
observations until their acceptance contract and calibration are independently
established.

## Execute, verify, and report

Run execution-free preflight before allocating model compute:

```console
invarlock evaluate request.yaml --signing-key evidence-signer.pem \
  --runtime-image "$RUNTIME_IMAGE" --runtime-image-digest "$RUNTIME_DIGEST" \
  --preflight
```

Resolve every reported error at the artifact, provider, schedule, policy,
runtime, or output boundary. A successful preflight checks deterministic
inputs; it does not predict model behavior or replace execution.

Run the same request without `--preflight`, deliver only the immutable evidence
pack to the verifier, and keep expected identities on an independent channel:

```console
invarlock evaluate request.yaml --signing-key evidence-signer.pem \
  --runtime-image "$RUNTIME_IMAGE" --runtime-image-digest "$RUNTIME_DIGEST"
invarlock verify evidence/ --trust-profile verifier/trust-inputs.json \
  --receipt verifier/verification.receipt.json \
  --verifier-signing-key verifier/verifier.pem \
  --verifier-identity release-verifier
invarlock report evidence/ --html verifier/report.html --explain
```

Import scenarios use the same commands without runtime-image options. Optional
providers may add qualification steps before the public transaction; their
runbooks link to the package-owned commands.

## Read the outcome at the correct level

A passing receipt establishes one bounded statement: the named subject met the
named policy relative to the named baseline on the authenticated paired
schedule under the recorded runtime identities. A policy failure can still be
intact, useful evidence. An integrity or trust mismatch means the verifier
cannot rely on the submitted comparison.

Interpret every result together with:

- schedule composition and selected record identities;
- point effect, paired regressions and improvements, and interval width;
- metric semantics and any scorer-extension boundary;
- artifact and runtime identities;
- policy thresholds and sample-qualification controls; and
- separately authenticated operational observations.

## Maintainer contract

Each repository scenario contains one `scenario.yaml` and one `README.md`.
[`scenario.schema.json`](https://github.com/invarlock/invarlock/blob/main/examples/scenarios/scenario.schema.json)
closes the metadata. The repository checker requires consistent runbook
sections, validates related paths, rejects duplicate IDs, and prevents
transformation scripts from growing inside the scenario directories:

```console
make example-scenarios-check
```

The scenario catalog provides adoption guidance. Runtime conformance,
real-model execution, strict pack verification, and public evidence publication
are reported independently for each named ecosystem.
