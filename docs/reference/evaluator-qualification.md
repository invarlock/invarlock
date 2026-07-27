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
    - **Non-claim:** A matrix row demonstrates the named version and entry
      point; it does not make that evaluator a built-in InvarLock plugin

## Qualification matrix

The retained matrix uses two local records: one exact match and one mismatch.
No model, dataset, API key, or hosted evaluator service is used.

| Upstream evaluator | Pinned version | Executed upstream entry point | Qualification |
| --- | --- | --- | --- |
| LM Evaluation Harness | `lm-eval==0.4.12` | `exact_match_hf_evaluate` | Per-record verdict authority |
| Inspect AI | `inspect-ai==0.3.249` | `scorer.match` | Per-record verdict authority |
| Promptfoo | `promptfoo@0.121.19` | `promptfoo eval` with the local echo provider | Per-record verdict authority |
| DeepEval | `deepeval==4.1.3` | `ExactMatchMetric.measure` | Per-record verdict authority |
| Ragas | `ragas==0.4.3` | `ExactMatch.ascore` | Per-record verdict authority |
| LightEval | `lighteval==0.13.0` | `ExactMatches.compute` | Per-record verdict authority |
| Hugging Face Evaluate | `evaluate==0.4.6` | `exact_match.compute` | Per-record verdict authority |
| Pydantic Evals | `pydantic-evals==2.18.0` | `Dataset.evaluate_sync` with `EqualsExpected` | Per-record verdict authority |
| Braintrust AutoEvals | `autoevals==0.3.0` | `ExactMatch` | Per-record verdict authority |
| OpenEvals | `openevals==0.2.0` | `exact_match` | Per-record verdict authority |
| MLflow Model Evaluation | `mlflow==3.14.0` | `mlflow.models.evaluate` | Observation-only: aggregate result |
| Garak | `garak==0.15.1` | Garak CLI with its offline repeat generator | Observation-only: unsupported replay semantics |

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

Re-execute all twelve pinned upstream tools and then requalify the resulting
artifacts:

```bash
make evaluator-upstream-qualification
```

The upstream command requires `uv`, Node.js with `npx`, and network access on a
cold cache. It downloads evaluator packages only; it does not download or call
a model.

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
