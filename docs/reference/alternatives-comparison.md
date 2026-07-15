# Alternatives Comparison

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Position InvarLock among existing evaluation, model-validation, numerical-debugging, and provenance tools. |
| **Audience** | Prospective users, evaluators, and maintainers designing an assurance workflow for edited checkpoints. |
| **Scope** | Current project positioning and handoff guidance for adjacent toolchains. |
| **Source of truth** | `docs/assurance/14-trust-model.md`, `docs/reference/reports.md`, `docs/reference/guards.md`. |

InvarLock is not the owner of model evaluation, regression testing, model
validation, or software provenance as problem categories. Mature tools already
cover each of those areas. Its narrower role is to integrate three concerns for
an edited checkpoint: paired baseline-versus-subject regression measurement,
weight/activation guard evidence, and a report bundle whose internal fields are
bound to a standalone verifier contract.

That combination can reduce custom glue in checkpoint-edit review, but it does
not replace broad benchmark suites, deployment-runtime debugging, independent
artifact provenance, or production monitoring.

## Are These Problems Already Real?

Yes, although they are normally addressed by several controls rather than one
product:

- Candidate-versus-baseline metric gating is available in mainstream MLOps
  tooling. [MLflow model validation](https://mlflow.org/docs/latest/ml/evaluation/#model-validation)
  supports fixed scalar thresholds; with a baseline result,
  [`MetricThreshold`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.models.html#mlflow.models.MetricThreshold)
  can require a positive minimum absolute or relative improvement. A
  tolerated-degradation rule requires a derived candidate-minus-baseline
  metric or custom policy code.
- Reduced precision and engine conversion can change numerical results.
  NVIDIA's [TensorRT accuracy guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)
  recommends checking layer outputs for NaNs or infinite values and comparing
  them with golden outputs; its PTQ guidance also calls for representative
  calibration data and accuracy evaluation.
- Evaluation setup is itself a reproducibility input. The
  [lm-evaluation-harness task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)
  treats task YAML plus the code commit as shareable inputs for reproducing a
  run.
- Artifact metadata is not trustworthy merely because it is present. The
  [`SLSA` verification procedure](https://slsa.dev/spec/v1.2/verifying-artifacts)
  requires a verifier to check the artifact digest, provenance signature,
  trusted builder identity, and expected build parameters.

These controls are most valuable when a checkpoint edit could silently degrade
quality, introduce non-finite or unstable internal values, or be reviewed by a
independent party. They are less valuable for exploratory work
where the checkpoint is disposable, no acceptance decision is being made, and
a normal benchmark rerun already supplies enough evidence.

## Existing Solutions And Where They Fit

| Existing tool or standard | What it already solves | Relationship to InvarLock |
| --- | --- | --- |
| [MLflow model evaluation](https://mlflow.org/docs/latest/ml/evaluation) | Computes evaluation metrics and artifacts and validates fixed thresholds. With a baseline result it can require minimum absolute or relative improvement; its built-in change thresholds are not a bounded-regression rule. | MLflow may be sufficient when aggregate metrics plus a derived delta or custom validation rule meet the acceptance need. InvarLock adds a checkpoint-oriented paired-window contract, internal guard evidence, and a bound report/verifier format; it is not a replacement for MLflow tracking or its broader evaluator ecosystem. |
| [NVIDIA NeMo Evaluator quality gates](https://docs.nvidia.com/nemo/evaluator/nightly/tutorials/quality-gate) | `nel gate` compares baseline and candidate bundles on paired per-item results, computes a 95% confidence interval, applies a caller-supplied per-benchmark absolute/relative regression policy, distinguishes breach from insufficient evidence, requires configured benchmarks, aggregates a release verdict, and exposes CI exit codes when `--strict` is used. NVIDIA documents quantization and perplexity release-gate patterns. | NeMo Evaluator already provides paired, multi-benchmark baseline/candidate release gating, which is broader than InvarLock's current single-comparison contract. InvarLock is relevant when the same checkpoint review also needs its supported internal guard measurements and verifier/evidence-pack format; otherwise use NeMo Evaluator or another established evaluation gate. |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Runs broad language-model benchmarks. Its [YAML task configuration](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md) captures datasets, prompts, few-shot settings, generation settings, metrics, filters, and task metadata; sharing it with the code revision supports reproducible task setup. | Use the harness for breadth and standardized task scores. InvarLock's distinct scope is a fixed paired baseline/subject comparison and guard/report evidence. A strong workflow can run the same pinned harness tasks on both checkpoints and then use an explicit regression gate. |
| [Deepchecks](https://docs.deepchecks.com/stable/getting-started/welcome.html) | Validates data integrity, dataset drift/leakage, model performance, calibration, weak segments, and related conditions. Its [CI/CD guide](https://docs.deepchecks.com/stable/general/usage/ci_cd.html) shows suites and conditions failing a pipeline before deployment. | Deepchecks is broader for tabular/data-centric validation and CI. Use it when data quality, slice behavior, or drift is the primary risk. InvarLock does not supersede those checks; its guard surface is specific to the supported checkpoint-edit workflow. |
| [`Giskard` Checks](https://docs.giskard.ai/oss/checks/reference/checks) and [vulnerability scans](https://docs.giskard.ai/oss/solutions/scan-vulnerabilities) | Builds reusable behavioral tests for AI applications and scans agents for safety/security failure scenarios. Saved suites can be rerun in CI and exported as JUnit; custom checks and LLM judges cover application-specific behavior. | Use `Giskard` for agent, RAG, safety, or application-behavior testing. Those risks are outside a weight/activation guard. InvarLock can be an earlier checkpoint acceptance gate, followed by `Giskard` tests against the deployed application. |
| [TensorRT `Polygraphy` / accuracy debugging](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html) | Compares backend and layer outputs with golden values, dumps intermediate tensors, and checks for NaNs or infinite values with `--validate`; it is designed to localize conversion or reduced-precision discrepancies. | Use `Polygraphy` for TensorRT/ONNX numerical diagnosis and per-layer backend comparison. InvarLock does not replace it; InvarLock records supported checkpoint-level regression and guard results in one acceptance artifact. |
| [NVIDIA Model Optimizer PTQ](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/llm_ptq/README.md) | Produces quantized checkpoints and includes an explicit accuracy-evaluation step. The documented PTQ loop calibrates, evaluates the optimized model against task metrics, adjusts sensitive layers or quantization choices, and exports for deployment. | Use ModelOpt to perform the optimization and its task-specific evaluation to select a quantization recipe. InvarLock can consume the resulting checkpoint as the subject for an additional paired and guard-based acceptance check; it does not claim to replace PTQ calibration or optimization. |
| [Sigstore Cosign](https://docs.sigstore.dev/cosign/verifying/verify/) and [`SLSA`](https://slsa.dev/spec/v1.2/) | Verify signatures/attestations and establish software-supply-chain provenance. `SLSA` verification checks the subject digest, trusted signer/builder, build type, and expected external parameters rather than trusting report-supplied metadata alone. | These are stronger and more general controls for artifact authenticity and build provenance. InvarLock's report/manifest binding is not a `SLSA` claim. Sign the evidence bundle and checkpoint, and verify their supply-chain provenance separately when authenticity matters. |
| [Hugging Face model cards](https://huggingface.co/docs/hub/en/model-cards) and [evaluation results](https://huggingface.co/docs/hub/eval-results) | Document intended use, limitations, datasets, and metrics; structured evaluation-result files can display scores on model pages and benchmark leaderboards, with source links and optional verification tokens. | Use model cards and Hub evaluation results for disclosure, discoverability, and result publication. They are presentation and exchange surfaces, not automatically proof of a paired run. InvarLock can export a concise result into a model card while the detailed report and independent anchors remain separate evidence. |

This is an overlap map, not a claim that the tools are interchangeable. For
example, MLflow can host a baseline-regression gate by logging a derived
candidate-versus-baseline delta or using custom validation code; its built-in
baseline-change thresholds express required improvement, not permitted
regression. `Polygraphy` can provide a much deeper numerical comparison for a
TensorRT engine. Conversely, neither one by itself represents InvarLock's
complete paired-report-plus-guard contract.

## How Teams Commonly Handle The Risk Today

A defensible industry workflow is layered:

1. Produce the edited artifact with the optimization or training tool. For PTQ,
   that may be ModelOpt, Optimum, or another compression tool.
2. Run task-level evaluation with lm-evaluation-harness or a domain-specific
   suite. Pin the dataset revision, task configuration, prompt/template,
   dependencies, seed, and model revision.
3. Compare candidate metrics with an approved baseline and independently supplied
   thresholds in NeMo Evaluator, MLflow, or ordinary CI code. Preserve
   sample-level or window-level records when paired inference is important.
4. Run specialized diagnostics for the deployment path. For TensorRT, use
   `Polygraphy` and task-level accuracy checks; for data and model quality, use
   Deepchecks; for agent or RAG behavior, use `Giskard` or equivalent tests.
5. Sign artifacts and verify provenance with Sigstore/`SLSA`-compatible
   controls. Do not treat a manifest written by the evaluation environment as an
   independent trust anchor.
6. Publish the model's intended use, limitations, datasets, and evaluation
   results in a model card or registry, linking to detailed evidence where
   possible.

InvarLock can occupy part of steps 2–4 for its supported checkpoint-edit lanes
and can package the result for step 6. It still needs independent signer,
runtime-image, checkpoint, policy, and sampling anchors when those facts are
material to the review.

## When To Use InvarLock

- You need one configured workflow that evaluates a baseline and edited subject
  on paired records, applies supported weight/activation guards, and emits a
  verifier-bound report.
- Review policy requires both task-regression evidence and internal model
  measurements, not only a leaderboard score or aggregate metric.
- A downstream verifier needs to re-check report schema, arithmetic, pairing,
  hashes, and declared guard gates offline. Claims about execution,
  representative sampling, checkpoint origin, and artifact authenticity still
  require independent trust anchors or reruns.

## When Another Tool Is Enough Or Better

| Need | Prefer |
| --- | --- |
| Broad or standardized LLM benchmark coverage | lm-evaluation-harness or LightEval |
| Multi-benchmark baseline/candidate release gating with paired per-item evidence and an external policy | NVIDIA NeMo Evaluator `nel gate` |
| Fixed metric thresholds or required improvement versus a baseline inside an experiment registry | MLflow validation; use a derived delta or custom rule for tolerated degradation |
| Data integrity, drift, leakage, slice, calibration, or model-quality CI checks | Deepchecks |
| Agent/RAG regression tests, adversarial scenarios, or safety scans | `Giskard` or another application-evaluation framework |
| TensorRT/ONNX output mismatch, NaN or infinite value, or per-layer golden comparison | `Polygraphy` |
| Producing and tuning a quantized artifact | ModelOpt, Optimum, Intel Neural Compressor, GPTQModel, or another compression tool |
| Cryptographic signatures and trusted build provenance | Sigstore/Cosign and `SLSA`-aware verification |
| Publishing capabilities, limitations, datasets, and benchmark results | Hugging Face model cards and structured evaluation results |
| Production telemetry, live drift, dashboards, or incident response | An observability/MLOps platform |

## Recommended Combined Workflow

1. Use the edit or compression tool to produce the subject checkpoint.
2. Use lm-evaluation-harness, LightEval, NeMo Evaluator, or a domain suite for
   broad downstream benchmark confidence and independently supplied release gates.
3. Use MLflow, Deepchecks, `Giskard`, and deployment-specific diagnostics where
   their validation surfaces match the risk.
4. Use InvarLock for the supported paired checkpoint-regression and guard
   workflow, producing a machine-checkable report for that configured
   comparison.
5. Verify signatures and provenance independently with Sigstore/`SLSA`-aligned
   controls, and supply the policy, runtime-image, signer, checkpoint, and
   sampling anchors required by the review.
6. Export the accepted result into existing workflow surfaces with
   `invarlock report export --format mlflow-tags`,
   `--format model-card-md`, or `--format release-review-md`.

## Related Documentation

- [Trust Model](../assurance/14-trust-model.md) — Strict pass scope
- [Strict Assurance Checklist](../assurance/15-strict-assurance-checklist.md) — Evidence acceptance criteria
- [Reports Reference](reports.md) — Schema, telemetry, and HTML export
- [Guards Reference](guards.md) — Configuration and evidence
- [Model Family Catalog](model-family-catalog.md) — Support tiers and backlog
- [Compare & evaluate (BYOE)](../user-guide/compare-and-evaluate.md) — Primary BYOE workflow
