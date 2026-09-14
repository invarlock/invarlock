# Capture existing evaluator cases for InvarLock scoring

> **Outcome:** Preserve cases from an existing evaluator workflow and make their
> available facts usable by InvarLock's exact-match, normalized-NLL or judge scorer.
> **Audience:** Evaluator users who already retain model inputs and outputs.
> **Prerequisites:** An installed InvarLock wheel, its matching example checkout,
> original per-case data and the evaluated model's artifact digest.

`capture.py` uses the public `capture_evaluator_run` SDK, also exported by
`invarlock.engine`. It captures
existing case facts; InvarLock owns subsequent scoring. Naming an evaluator does
not execute that package or transfer its scoring authority. Capability output
reports which records have the required facts, rather than claiming a completed
judge execution or model-likelihood measurement.

## Shortlist coverage

The qualification path joins original case text to a retained per-case export and
its independent schedule. It checks profile, schedule, export and raw-output
bindings, source identities, record order, input/reference digests and literal
outputs. It retains the original profile identity and source digests in each
record's context. Historical runners, profiles and qualification results remain
separate from this new capture.

| Ecosystem | Retained qualification capture | Existing workflow capture |
| --- | --- | --- |
| LM Evaluation Harness | Per-case export plus original cases | Canonical records |
| Inspect AI | Per-case export plus original cases | Canonical records |
| Promptfoo | Per-case export plus original cases | Canonical records |
| DeepEval | Per-case export plus original cases | Canonical records |
| Ragas | Per-case export plus original cases | Canonical records |
| LightEval | Per-case export plus original cases | Canonical records |
| Hugging Face Evaluate | Per-case export plus original cases | Canonical records |
| Pydantic Evals | Per-case export plus original cases | Canonical records |
| Braintrust AutoEvals | Per-case export plus original cases | Canonical records |
| OpenEvals | Per-case export plus original cases | Canonical records |
| MLflow Model Evaluation | Aggregate observation only | Original per-case prediction table |
| Garak | Detector observation only | Explicit reviewed per-attempt case capture |
| OpenAI Evals | Per-case export plus original cases | Canonical records |
| Arize Phoenix Evals | Per-case export plus original cases | Canonical records |
| Langfuse | Per-case export plus original cases | Canonical records |
| Opik | Per-case export plus original cases | Canonical records |
| Azure AI Evaluation | Per-case export plus original cases | Canonical records |
| Evidently | Per-case export plus original cases | Canonical records |
| TruLens | Per-case export plus original cases | Canonical records |

Every canonical path has the same fact requirements. Exact match requires a
string output and reference. Judge input requires original input/output text and
an optional string reference; a configured rubric and actual retained judge calls
are additional requirements. `prompt.reference_mode: per_case` requires a string
reference for every judged case and adds it as a separate judge request field.
Omitted or `none` keeps references out of that request; neither mode changes the
evaluated model input. Normalized NLL requires typed reference-continuation
likelihood facts. The retained qualification exports contain no such likelihoods.
MLflow aggregate accuracy and Garak detector counts cannot supply missing rows,
references or log probabilities.

Print the machine-readable matrix with:

```bash
python examples/evaluator-qualification/maintained/capture.py matrix
```

## Use an installed export parser

The installed `load_run` API and captured request sources support `invarlock`,
`jsonl`, `inspect-json`, `lm-eval-samples` and `promptfoo-jsonl`. These parsers
normalize their declared per-case export shapes; they do not run an evaluator or
qualify an arbitrary source. The `capture.py` paths below additionally support
explicit record mapping and retained qualification joins for the full shortlist.

For a captured request, set `comparison.metric` to `exact_match`,
`normalized_nll_per_utf8_byte` or `judge`. Exact match and NLL must agree with the
single metric in the comparison policy. Judge uses its own recipe and full
retained-call contract, with an optional configured collector. Omitting the
selector keeps ordinary multi-metric comparison behavior.

## Capture a retained per-case export

Set `MODEL_ARTIFACT_DIGEST` to the digest of the artifact that produced the model
outputs. Run from the matching example checkout. Substitute another per-case
profile from the matrix to exercise its retained export:

```bash
python examples/evaluator-qualification/maintained/capture.py qualification \
  --ecosystem inspect-ai \
  --cases examples/evaluator-qualification/cases.json \
  --schedule examples/evaluator-qualification/schedule.json \
  --profile examples/evaluator-qualification/artifacts/inspect-ai/profile.json \
  --export examples/evaluator-qualification/artifacts/inspect-ai/export.json \
  --raw-output examples/evaluator-qualification/artifacts/inspect-ai/upstream-output.json \
  --run-id existing-baseline \
  --artifact-digest "$MODEL_ARTIFACT_DIGEST" \
  --output captured-baseline.json
```

These retained runs executed upstream exact-match scorers on supplied cases.
Capturing them does not establish a fresh upstream run or a native judge/NLL
execution. Fresh maintained qualification outputs can use the same command with
their own cases, schedule, profile, export and raw-output paths.

## Capture cases inside an existing workflow

Keep model execution and framework evaluation where they already run. At the
point where the original case is available, map its fields explicitly to a JSON
array of SDK records:

```python
from invarlock.engine import capture_evaluator_run

records = [
    {
        "id": case["record_id"],
        "input": case["input"],
        "expected": case["reference"],
        "output": case["output"],
    }
    for case in original_cases
]
run = capture_evaluator_run(
    records,
    source={"name": evaluator_package_name, "version": evaluator_package_version},
    run_id=existing_run_id,
    artifact_digest=model_artifact_digest,
)
```

For MLflow, map the original prediction table's stable case ID, original prompt,
`prediction` and `target` columns to `id`, `input`, `output` and `expected`.
Capture these original records separately from the retained aggregate report;
the report contains no individual predictions.

For Garak, prepare explicit canonical cases from reviewed actual attempts.
Preserve a distinct ID for every attempt/output pair and its exact original
prompt and generated output. Supply a reviewed reference where one exists, or
`expected: null` for a judge task without a reference. The retained detector
summaries do not contain these attempts, so they cannot be used as this input.
This helper does not guess native report layouts or convert detector scores into
answer quality.

Save the mapped JSON array as `original-records.json`, then run:

```bash
python examples/evaluator-qualification/maintained/capture.py records \
  --ecosystem mlflow \
  --source-version "$EVALUATOR_VERSION" \
  --records original-records.json \
  --run-id existing-baseline \
  --artifact-digest "$MODEL_ARTIFACT_DIGEST" \
  --output captured-baseline.json
```

Use the appropriate `--ecosystem` for all other workflows. Structured inputs can
select one text field with `--input-pointer /input/question`; the SDK retains and
binds the original structured source and the exact projection. Pointers may
select existing strings beneath `/input/` or `/context/`. It never silently
stringifies an object or array. Canonical `adapter: invarlock` sources already
carry these bindings and reject a request-level projection override.

To enable normalized NLL, retain the SDK's typed `likelihood` object with the
actual reference-continuation log-probability sum, positive token count, reference
UTF-8 byte count, original input/reference digests, model artifact digest,
configuration and tokenizer digests, and source identity. These must come from
the model measurement. A generated-answer confidence, reward, or aggregate score
cannot substitute for reference likelihood.

Capture baseline and subject runs with the same paired IDs, inputs, references
and case metadata. Use the resulting files as `adapter: invarlock` sources in an
`invarlock/evaluation-request-v2` captured comparison. Select the policy and scorer
for the available facts, then evaluate and independently verify the published
evidence. The capture command refuses an existing output destination; choose a
new filename when correcting source data.

See the [captured-results guide](../../../docs/user-guide/captured-results.md)
for policy, request, evaluation and verification steps. The production tests in
`tests/examples/test_evaluator_scorer_capture.py` exercise the retained joins,
exact scoring, typed-likelihood scoring and missing-fact failures across the
shortlist. Their explicit canonical test rows demonstrate adapter behavior;
they are not additional upstream executions.

## Acceptance and qualification

Keep three conclusions separate:

| Milestone | Required evidence |
| --- | --- |
| Implemented integration | Installed `evaluate`, independent `verify` and `report` pass for the declared input profile, including meaningful failure cases |
| Real-workflow qualification | Retained inputs come from the actual evaluator/model measurement and preserve its exact task and settings |
| Supported quality claim | The qualified result and its reviewed scope support the specific published claim |

A capability refusal does not complete a promised combination. Distinguish
missing capture facts, an import that has not been implemented, and a task or
runtime that cannot supply the required measurement. When the facts are
available for a promised profile, implement and test its positive path.

The canonical NLL contract tests use synthetic likelihood facts. The separate
[Harness likelihood reference](../../captured-results/references/harness-likelihood/README.md)
retains a real six-pair CPU measurement and installed signed journey, including
the exact context and reference continuation, log-probability sum, token/byte
counts, tokenizer, configuration and model pins. Its same-model conformance
result establishes that declared integration profile, not model quality or
likelihood qualification across the whole matrix. A generation-only export is
insufficient; that does not mean its evaluator cannot measure likelihoods.

The judge tests likewise establish replay and integration behavior rather than
a new hosted quality result. Existing retained qualification observations and
judge pilot outcomes keep their original scope and decisions.
