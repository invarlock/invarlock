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

## Demonstration levels

The repository keeps three cumulative claims separate. They describe retained
evidence, not permanent evaluator classes or an architectural support ceiling.
Any profile can advance to a deeper level by adding the corresponding retained
evidence.

| Level | What it proves | Coverage |
| --- | --- | --- |
| Qualification profile | The named upstream entry point executes and its output crosses the generic qualification contract | All maintained profiles |
| Authoritative import adapter | A complete real evaluation is retained per record, independently recomputed, and replayed through the strict runtime-import loader | Every deterministic per-record profile |
| End-to-end release-assurance journey | The evaluator participates in a model-running, signed `evaluate` → `verify` → `report` transaction | Profiles marked `Demonstrated` in the matrix |

An authoritative import adapter is not described as an end-to-end journey.
Profiles can advance to that deeper level through the existing evaluator-neutral
boundary by adding maintained model-running and signed-transaction fixtures,
not a new evaluator-specific engine plugin.

## Qualification matrix

The small conformance corpus uses two local records: one exact match and one
mismatch. It proves every maintained upstream entry point and both authority
modes without downloading a model or calling a hosted evaluator.

The authoritative rows also execute against a retained 102-record
evaluation produced by the immutable `Qwen/Qwen3-0.6B` revision recorded in the
corpus. The model produced 52 exact matches and 50 mismatches. Each evaluator
scores all 102 model outputs through its real upstream entry point. InvarLock
then recomputes every score and replays all 102 normalized records through the
runtime-import authoring boundary.

The catalog is reviewed rather than quota-driven. A row must represent a
recognizable current evaluator or a maintained successor, add a distinct
ecosystem or workflow, and support retained real upstream execution. Review
timing and activity-window metadata live in `matrix.json`; the resulting catalog
is reviewed coverage, not a hard cap.

The matrix represents the Microsoft PromptFlow lineage with Azure AI Evaluation
rather than preserving the deprecated `promptflow-evals` package as a second
legacy row. OpenAI Evals is installed from the immutable source revision in its
dependency declaration, so qualification binds the executed source rather than
only a distribution label.

### Benchmark harnesses

| Upstream evaluator | Pinned version | Executed upstream entry point | Qualification profile | Authoritative import | End-to-end transaction |
| --- | --- | --- | --- | --- | --- |
| LM Evaluation Harness | `lm-eval==0.4.12` | `exact_match_hf_evaluate` | Per-record | 102 records | Demonstrated |
| Inspect AI | `inspect-ai==0.3.249` | `scorer.match` | Per-record | 102 records | Not yet demonstrated |
| LightEval | `lighteval==0.13.0` | `ExactMatches.compute` | Per-record | 102 records | Not yet demonstrated |
| OpenAI Evals | source revision `8eac7a7` (`3.0.1.post1`) | `MATCH_FNS['exact']` | Per-record | 102 records | Not yet demonstrated |

### Application evaluation SDKs

| Upstream evaluator | Pinned version | Executed upstream entry point | Qualification profile | Authoritative import | End-to-end transaction |
| --- | --- | --- | --- | --- | --- |
| Promptfoo | `promptfoo@0.121.19` | `promptfoo eval` with the local echo provider | Per-record | 102 records | Not yet demonstrated |
| DeepEval | `deepeval==4.1.3` | `ExactMatchMetric.measure` | Per-record | 102 records | Not yet demonstrated |
| Ragas | `ragas==0.4.3` | `ExactMatch.ascore` | Per-record | 102 records | Not yet demonstrated |
| Pydantic Evals | `pydantic-evals==2.18.0` | `Dataset.evaluate_sync` with `EqualsExpected` | Per-record | 102 records | Not yet demonstrated |
| Braintrust AutoEvals | `autoevals==0.3.0` | `ExactMatch` | Per-record | 102 records | Not yet demonstrated |
| OpenEvals | `openevals==0.2.0` | `exact_match` | Per-record | 102 records | Not yet demonstrated |
| Azure AI Evaluation | `azure-ai-evaluation==1.18.1` | `azure.ai.evaluation.evaluate` | Per-record | 102 records | Not yet demonstrated |

### Evaluation and observability platforms

| Upstream evaluator | Pinned version | Executed upstream entry point | Qualification profile | Authoritative import | End-to-end transaction |
| --- | --- | --- | --- | --- | --- |
| Arize Phoenix Evals | `arize-phoenix-evals==3.3.0` | `phoenix.evals.metrics.exact_match` | Per-record | 102 records | Not yet demonstrated |
| Langfuse | `langfuse==4.14.1` | `Langfuse.run_experiment` | Per-record | 102 records | Not yet demonstrated |
| Opik | `opik==2.2.7` | `Equals.score` | Per-record | 102 records | Not yet demonstrated |
| MLflow Model Evaluation | `mlflow==3.14.0` | `mlflow.models.evaluate` | Observation-only: aggregate result | No | Not yet demonstrated |
| Evidently | `evidently==0.7.21` | `Dataset.from_pandas` with `ExactMatch` | Per-record | 102 records | Not yet demonstrated |
| TruLens | `trulens==2.9.0` | `Metric.__call__` | Per-record | 102 records | Not yet demonstrated |

### General metric libraries

| Upstream evaluator | Pinned version | Executed upstream entry point | Qualification profile | Authoritative import | End-to-end transaction |
| --- | --- | --- | --- | --- | --- |
| Hugging Face Evaluate | `evaluate==0.4.6` | `exact_match.compute` | Per-record | 102 records | Not yet demonstrated |

### Security and red-team evaluators

| Upstream evaluator | Pinned version | Executed upstream entry point | Qualification profile | Authoritative import | End-to-end transaction |
| --- | --- | --- | --- | --- | --- |
| Garak | `garak==0.15.1` | Garak CLI with its offline repeat generator | Observation-only: unsupported replay semantics | No | Not yet demonstrated |

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

This verifies every maintained qualification profile and replays every retained
authoritative import. To re-execute all pinned upstream tools over both corpora
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
make evaluator-authoritative-corpus
```

The model corpus binds the immutable model revision, curated snapshot-tree
digest, exact generation settings, dataset digest, producer package versions,
and every output. This command performs model execution; ordinary offline
qualification does not.

## Claim boundary

The authoritative import demonstrations prove full per-record exact-match
imports for one real, pinned model evaluation. They do not demonstrate every
metric, task type, hosted mode, or model-judge feature offered by those
evaluators. They also do not turn example runners into engine plugins.

Profiles marked `Demonstrated` in the matrix add the deeper model execution and
signed release-assurance transaction. An authoritative row without that mark
begins with the already authenticated model-output corpus and demonstrates
evaluator execution, normalization, qualification, and strict runtime-import
replay.

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
