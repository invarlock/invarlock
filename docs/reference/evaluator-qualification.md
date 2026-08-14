# Evaluator qualification

InvarLock qualifies evaluator evidence through one closed, evaluator-neutral
boundary. The core does not contain evaluator-name dispatch, evaluator SDK
imports, or native-output parsers. Example-owned runners execute representative
upstream tools and normalize their results into the same four contracts.

!!! info "Reference"

    - **Surface:** Four canonical JSON contracts, the
      `invarlock-qualify-evaluator` companion CLI, and
      `invarlock.engine.qualify_evaluator_export`
    - **Stability:** Closed v1 qualification formats
    - **Use this page when:** Integrating an open or proprietary evaluator, or
      reviewing the maintained upstream-execution matrix

A matrix row demonstrates the named version and entry point; it does not make
that evaluator a built-in InvarLock plugin.

## Three independent status axes

The repository does not assign one cumulative “integration level.” Support,
replay authority, and retained transaction evidence answer different questions
and cannot safely substitute for one another.

| Axis | Values | What it proves |
| --- | --- | --- |
| Adapter support | `maintained_adapter` or an external adapter | Whether this repository maintains the source-specific runner, dependency lock, and upstream entry point; support grants no replay authority |
| Replay authority | `deterministic_per_record` or `observation_only` | Whether complete ordered facts can be independently recomputed and imported, or only retained as authenticated context |
| Signed-journey maturity | Retained with a named corpus and record count, or `—` | Whether a model-running, signed `evaluate` → `verify` → `report` OCI transaction has completed and been retained as release evidence |

The stable qualification-result contract continues to emit
`outcome: qualified_for_import` with `authority: verdict_authority` for an
independently replayable result. It emits `observation_only` for both fields
when replay is unavailable. Those wire values do not claim that InvarLock
maintains the adapter or that a signed OCI journey has been demonstrated.

## Qualification matrix

The small conformance corpus uses two local records: one exact match and one
mismatch. It proves every maintained upstream entry point and both authority
modes without downloading a model or calling a hosted evaluator.

The independently replayable rows also execute against a retained 102-record
evaluation produced by the immutable `Qwen/Qwen3.5-0.8B` revision recorded in
the corpus. The model produced 61 exact matches and 41 mismatches. Each evaluator
scores all 102 model outputs through its real upstream entry point. InvarLock
then recomputes every score and replays all 102 normalized records through the
runtime-import authoring boundary.

Each row identifies a maintained evaluator entry point with retained upstream
execution. The rows cover distinct evaluator ecosystems and workflows;
`matrix.json` records their source, version, authority, and status metadata.

LM Evaluation Harness and Inspect AI additionally have independently
replayable 102-record shared-output imports. Their retained current-model OCI
journeys execute Qwen3.5 9B over the same ordered 400-record MMLU-Pro schedule.
LM Evaluation Harness also retains a Gemma 4 12B instruction-to-QAT portability
transaction. The other rows retain qualification and compatibility evidence
without claiming a native signed journey.

The compact evidence packs, signed verifier receipts, independent policies,
and builder-signed OCI attestations are retained under
`examples/evaluator-qualification/signed-transactions/` and replayed by
`make evaluator-qualification`.

The 400-item semantic suite is selected deterministically from a revision- and
hash-pinned
[TIGER-Lab MMLU-Pro dataset](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro).
It balances all 14 subject categories and answer labels A–J. Qwen and Gemma
prompts derive from those same semantic records using separately pinned
model-native no-thinking renderings. The 400-record size supports the maximum
10-percentage-point paired interval width declared in advance: Qwen achieved a
7.85-point width and Gemma achieved a 4.68-point width.

The retained flagship comparison confirms an identical ordered schedule and
reports native agreement without creating another acceptance verdict. LM
Evaluation Harness and Inspect AI agree on every baseline and subject output
and score across all 800 side records. The shared-output replay and native
signed journeys answer complementary questions: the first isolates evaluator
scoring across the wider matrix, while the second exercises both flagship
frameworks' real model paths.

The matrix represents the Microsoft PromptFlow lineage with Azure AI Evaluation
rather than preserving the deprecated `promptflow-evals` package as a second
legacy row. The OpenAI Evals qualification profile is installed from an
immutable source revision, while its signed transaction integration runs the
upstream `basic.Match` evaluator from the hash-pinned `evals==3.0.1.post1` wheel
in its isolated image.

The generated matrix below describes the retained generic qualification
profiles. The signed bridges execute LM Evaluation Harness and the native
Inspect task (`inspect_ai.eval` plus `inspect_ai.scorer.match`) in clean OCI
transactions. Their build attestations, worker protocol, and native adapters
remain example-owned; the installed core receives evaluator-neutral
runtime-import and signed transaction contracts.

The signed bridges retain native evaluator facts but keep the transaction
metric independent. Inspect's pinned causal HF decoder has an explicit
authenticated boundary recovery, and InvarLock replays byte-exact equality for
the signed acceptance decision after checking each evaluator's native event
semantics.

<!-- evaluator-matrix:start -->
<!-- Generated by examples/evaluator-qualification/render_docs_matrix.py; do not edit by hand. -->

### Application evaluation SDKs

| Upstream evaluator | Pinned version | Executed upstream entry point | Adapter support | Replay authority | Retained signed transactions |
| --- | --- | --- | --- | --- | --- |
| Promptfoo | `promptfoo@0.121.19` | `promptfoo eval` | Maintained | Independently replayable (102 shared outputs) | — |
| DeepEval | `deepeval==4.1.3` | `deepeval.metrics.ExactMatchMetric.measure` | Maintained | Independently replayable (102 shared outputs) | — |
| Ragas | `ragas==0.4.3` | `ragas.metrics.collections.ExactMatch.ascore` | Maintained | Independently replayable (102 shared outputs) | — |
| Pydantic Evals | `pydantic-evals==2.18.0` | `pydantic_evals.Dataset.evaluate_sync/EqualsExpected` | Maintained | Independently replayable (102 shared outputs) | — |
| Braintrust AutoEvals | `autoevals==0.3.0` | `autoevals.ExactMatch.__call__` | Maintained | Independently replayable (102 shared outputs) | — |
| OpenEvals | `openevals==0.2.0` | `openevals.exact.exact_match` | Maintained | Independently replayable (102 shared outputs) | — |
| Azure AI Evaluation | `azure-ai-evaluation==1.18.1` | `azure.ai.evaluation.evaluate` | Maintained | Independently replayable (102 shared outputs) | — |

### Benchmark harnesses

| Upstream evaluator | Pinned version | Executed upstream entry point | Adapter support | Replay authority | Retained signed transactions |
| --- | --- | --- | --- | --- | --- |
| LM Evaluation Harness | `lm-eval==0.4.12` | `lm_eval.api.metrics.exact_match_hf_evaluate` | Maintained | Independently replayable (102 shared outputs) | Retained (2 signed transactions, 400 records each) |
| Inspect AI | `inspect-ai==0.3.254` | `inspect_ai.scorer.match` | Maintained | Independently replayable (102 shared outputs) | Retained (2 signed transactions, 400 records each) |
| LightEval | `lighteval==0.13.0` | `lighteval.metrics.metrics_sample.ExactMatches.compute` | Maintained | Independently replayable (102 shared outputs) | — |
| OpenAI Evals | source revision `8eac7a7` (`3.0.1.post1`) | `evals.elsuite.modelgraded.classify_utils.MATCH_FNS['exact']` | Maintained | Independently replayable (102 shared outputs) | — |

### Evaluation and observability platforms

| Upstream evaluator | Pinned version | Executed upstream entry point | Adapter support | Replay authority | Retained signed transactions |
| --- | --- | --- | --- | --- | --- |
| MLflow Model Evaluation | `mlflow==3.14.0` | `mlflow.models.evaluate` | Maintained | Observation-only: aggregate only | — |
| Arize Phoenix Evals | `arize-phoenix-evals==3.3.0` | `phoenix.evals.metrics.exact_match` | Maintained | Independently replayable (102 shared outputs) | — |
| Langfuse | `langfuse==4.14.1` | `langfuse.Langfuse.run_experiment` | Maintained | Independently replayable (102 shared outputs) | — |
| Opik | `opik==2.2.7` | `opik.evaluation.metrics.Equals.score` | Maintained | Independently replayable (102 shared outputs) | — |
| Evidently | `evidently==0.7.21` | `evidently.Dataset.from_pandas/Evidently ExactMatch` | Maintained | Independently replayable (102 shared outputs) | — |
| TruLens | `trulens==2.9.0` | `trulens.core.Metric.__call__` | Maintained | Independently replayable (102 shared outputs) | — |

### General metric libraries

| Upstream evaluator | Pinned version | Executed upstream entry point | Adapter support | Replay authority | Retained signed transactions |
| --- | --- | --- | --- | --- | --- |
| Hugging Face Evaluate | `evaluate==0.4.6` | `evaluate.load('exact_match').compute` | Maintained | Independently replayable (102 shared outputs) | — |

### Security and red-team evaluators

| Upstream evaluator | Pinned version | Executed upstream entry point | Adapter support | Replay authority | Retained signed transactions |
| --- | --- | --- | --- | --- | --- |
| Garak | `garak==0.15.1` | `python -m garak` | Maintained | Observation-only: unsupported replay semantics | — |
<!-- evaluator-matrix:end -->

Each row under
[`examples/evaluator-qualification/artifacts/`](https://github.com/invarlock/invarlock/tree/main/examples/evaluator-qualification/artifacts)
retains:

- a profile binding the upstream package, project, runner bundle, dependency
  declaration, and authority classification;
- the normalized upstream execution output, including the resolved Python
  package inventory where applicable;
- an export binding the profile, independent schedule, raw output, runner, and
  dependency declaration by SHA-256; and
- the independently recomputed qualification result.

Native outputs may contain timestamps, temporary paths, run IDs, or large
payloads. The example runner retains a deterministic projection of the native
result rather than claiming byte-for-byte preservation of those volatile
fields. The reviewed runner bundle and its digest define that projection.

Run the retained, network-free verification:

```bash
make evaluator-qualification
```

This verifies every maintained adapter profile and replays every retained
independently replayable import. To re-execute all pinned upstream tools over both corpora
and refresh the retained artifacts:

```bash
make evaluator-upstream-qualification
```

The upstream command requires `uv`, Node.js with `npx`, and network access on a
cold cache. The retained model outputs do not require a model during evaluator
execution.

To independently regenerate and compare the 102 model outputs, make the pinned
model snapshot available locally and run the repository's locked Hugging Face
environment:

```bash
make evaluator-replayable-corpus
```

The model corpus binds the immutable model revision, curated snapshot-tree
digest, exact generation settings, dataset digest, generation package versions,
and every output. This command performs model execution; ordinary offline
qualification does not.

## Claim boundary

The independently replayable demonstrations prove full per-record exact-match
imports for one real, pinned model evaluation. They do not demonstrate every
metric, task type, hosted mode, or model-judge feature offered by those
evaluators. They also do not turn example runners into engine plugins.

Profiles marked `Retained` in the matrix separately carry model execution and
signed release-assurance evidence with an explicit record count. A replayable
row without a retained package begins with the already authenticated
model-output corpus and demonstrates evaluator execution, normalization,
qualification, and strict runtime-import replay.

## Authority rules

`deterministic_per_record` is currently limited to exact match. The export must
cover the independent schedule in exact order, bind each input and output, and
carry successful record status. InvarLock ignores the evaluator's aggregate
claim and recomputes `output_sha256 == reference_output_sha256` for every
record. A mismatch between the upstream-reported score and this replay rejects
the export.

`observation_only` requires one explicit reason:

- `aggregate_only`
- `human_judgment`
- `nondeterministic_judge`
- `unsupported_replay_semantics`

An observation-only result contains no import records and cannot be promoted
with a CLI flag. `--require-verdict-authority` instead rejects it.

## Proprietary evaluator path

A private SDK, CLI, or HTTP-backed evaluator uses the same boundary:

1. Execute the proprietary evaluator outside InvarLock.
2. Retain its response bytes or a reviewed deterministic projection as
   `upstream-output.json`.
3. Create a canonical profile with package ecosystem `private`, its immutable
   version, project URI, the runner-bundle digest, and dependency-declaration
   digest.
4. Normalize complete ordered records into
   `invarlock/evaluator-qualification-export-v1`, or declare the result
   observation-only when replay is unavailable.
5. Qualify the four inputs through the CLI or Python SDK.

```bash
invarlock-qualify-evaluator qualify \
  profile.json schedule.json export.json upstream-output.json \
  --output qualification-result.json \
  --require-verdict-authority --json
```

The runner can call a Python SDK, spawn a proprietary CLI, or call an API. That
transport is outside the core contract. Do not put credentials, endpoint URLs
with secrets, request headers, or private paths in retained public artifacts.
The profile and export must bind the immutable evaluator identity and the exact
runner/dependency inputs used to produce the normalized evidence.

The JSON schemas are
`evaluator_qualification_profile.schema.json`,
`evaluator_qualification_schedule.schema.json`,
`evaluator_qualification_export.schema.json`, and
`evaluator_qualification_result.schema.json`. Package-owned copies ship with
InvarLock and are the validation authority.
