# Capture existing evaluator cases for InvarLock scoring

> **Outcome:** Preserve cases from an existing evaluator workflow and make their
> available facts usable by InvarLock's exact-match, normalized-NLL or judge scorer.
>
> **Audience:** Evaluator users who already retain model inputs and outputs.
>
> **Prerequisites:** Core InvarLock in the recipient environment, the matching
> example checkout, original per-case data and the evaluated model's identity.

Keep your existing evaluator environment and save its original cases or results
as JSON using the source-specific profiles below. Set
`adapter: evaluator-native-json` in the captured request, alongside the actual
source name/version, run ID, and artifact or hosted-service identity. All 19
profiles share the same captured `evaluate`, `verify` and `report` flow. The
evaluator environment needs no InvarLock installation or common export envelope; the recipient
needs only core InvarLock, with no evaluator SDK or account.

Implementation tests exercise pinned SDK exports and actual task callbacks.
The separate [live campaign](../../integrations/evaluator-live/README.md) tracks
fresh model execution, likelihood measurement and plan-bound judge collection.
Its pending qualification must not be inferred from a passing serialization test.

For example, serialize the original DeepEval test cases with the SDK's own
explicit serializer:

```python
import json

rows = [{"id": case_id, "test_case": test_case.model_dump(mode="json"),
         "metadata": test_case.metadata or {}}
        for case_id, test_case in captured_test_cases]
if [row["id"] for row in rows] != planned_ids:
    raise ValueError("capture differs from the complete planned schedule")
with open("subject-native.json", "x", encoding="utf-8") as stream:
    json.dump(rows, stream, ensure_ascii=False, allow_nan=False)
```

`captured_test_cases` contains original `LLMTestCase` objects and stable IDs;
`planned_ids` is the complete schedule frozen before execution, including
failures. The outer `metadata` preserves case slices, explicit numeric scores
and likelihood facts; metadata inside `test_case` remains native context. For a
failed task, retain its planned row with `actual_output=None` and set the outer
`error` to the captured failure message. Do not omit the row or invent an answer.
Capture the baseline with the same IDs, inputs, references and case metadata. Pin the independently reviewed `expected_case_set_digest` in policy,
and approved complete-run digests when required. See the
[raw source declaration](../../../docs/reference/evaluation-records.md#dedicated-evaluator-exports)
and [captured workflow](../../../docs/user-guide/captured-results.md).

When evaluator and InvarLock dependency requirements are compatible, the
convenience API accepts the SDK objects directly:

```python
from importlib.metadata import version
from invarlock.engine import export_evaluator_result, evaluator_input_capabilities

run = export_evaluator_result(
    "deepeval",
    [{"id": case_id, "test_case": test_case, "metadata": test_case.metadata or {}}
     for case_id, test_case in captured_test_cases],
    "subject-export.json",
    expected_ids=planned_ids,
    source_version=version("deepeval"),
    run_id="subject-campaign",
    artifact_digest=model_artifact_digest,
)
print(evaluator_input_capabilities(run))
```

This API checks complete membership, refuses existing output paths, and writes
an `invarlock/evaluator-export-v1` envelope for `adapter: evaluator-json`.
Use the same source/run/model identities. Hosted calls use
`artifact_digest=None, service_identity=identity` with a complete descriptor.
Both routes reconstruct the same native facts and use the same downstream
scorers. SDK serializer smoke tests execute from source with minimal test
dependencies; they do not prove SDK/core co-installation.

MLflow 3.14.0 requires `cryptography<49`, which conflicts with core InvarLock's
`cryptography>=50`. Keep that pinned evaluator environment separate, export its original
prediction table as JSON, and import with `evaluator-native-json`. There is no
need to change either environment's dependencies for this handoff.

## Dedicated native shapes

The raw JSON file follows the named profile below. Convert listed SDK objects
with their explicit field serializers in the evaluator environment. With
compatible co-installation, those SDK objects can instead be passed as the
Python exporter's `result` argument. Scalar entries carry a caller-owned `id`; their `metric_result` is
optional. Capture original outputs without running an upstream metric when
InvarLock will score them. If supplied, the native metric result is preserved and
validated. Arbitrary SDK objects and ambiguous multiple responses require an
explicit mapping or selection by the capture process. Reference fields may be omitted
for reference-free judge tasks where the SDK profile permits them. LightEval
retains its explicit choices/gold-index profile. Missing references make exact
match and NLL unavailable; they do not require a fabricated gold answer. JSON
outputs remain structured for compatible policies; text-only scorers report
those cases as unavailable rather than implicitly converting them to text.

| Evaluator key | Native `result` profile | Pairing and reference fields |
| --- | --- | --- |
| `lm-evaluation-harness` | List of `--log_samples` rows: `doc`, `arguments`, `target`, `filtered_resps` | `doc_id`, or explicit `metadata.invarlock_id` in this dedicated profile; target is the original reference; select one completion |
| `inspect-ai` | EvalLog SDK object or JSON log, `version: 1` or `2`, `status: success`, `samples` | Sample `id`, `input`, `target`, `output.choices`; one epoch and completion |
| `promptfoo` | List of full result rows containing `testCase`, rendered `prompt`, `response` | `testCase.metadata.invarlock_id` and `invarlock_expected`; runtime failures remain errors |
| `deepeval` | List of `{id, test_case: LLMTestCase}` | `input`, `actual_output`, `expected_output`; optional measured metric or MetricsData |
| `ragas` | List of `{id, sample: SingleTurnSample}` | `user_input`, `response`, `reference`; optional MetricResult |
| `lighteval` | List of `{id, doc: Doc, model_response: ModelResponse}` | `query`, `choices`, `gold_index`, one raw `text` response; optional sample metric |
| `hugging-face-evaluate` | List of `{id, input, predictions: [output], references: [reference]}` | Singleton per-case batches; optional `compute()` metric dictionary |
| `pydantic-evals` | EvaluationReport or `{cases: [...], failures: [...]}` | Case `name`, `inputs`, `output`, `expected_output`; preserve failure `error_message` |
| `autoevals` | List of `{id, input, output, expected}` | Original arguments; optional native Score in `metric_result` |
| `openevals` | List of `{id, inputs, outputs, reference_outputs}` | Original arguments; optional EvaluatorResult or list in `metric_result` |
| `mlflow` | EvaluationResult with `tables.eval_results_table`, or `{prediction_table: rows, metrics: {...}}` | `record_id`, `input`, `prediction`, `target`; explicit `columns` can rename fields |
| `garak` | `{attempts: [Attempt, ...], source_cases: [...]}` or report `entries` | Native `uuid:generation_index`; independently join planned IDs and references with `source_cases` |
| `openai-evals` | `{events: [...]}` from the recorder | Join events by `sample_id`; actual prompt/sampled output; optional match expected value |
| `arize-phoenix-evals` | List of `{id, record: {input, output, expected}}` | Original evaluation arguments; optional Score or list in `metric_result` |
| `langfuse` | Bare ExperimentResult JSON fields: `name`, `run_name`, `item_results`, `run_evaluations`, and captured experiment/dataset fields | Hosted dataset item ID or local `metadata.invarlock_id`; see the dedicated handoff example |
| `opik` | List of `{id, dataset_item: {input, output, reference}}` | Original dataset fields; optional ScoreResult or list in `metric_result` |
| `azure-ai-evaluation` | Native `evaluate()` result containing `rows` | `inputs.record_id`, `inputs.query`, `inputs.response`, `inputs.ground_truth`; `inputs.id`, `inputs.input`, `outputs.response` aliases supported |
| `evidently` | Dataset/DataFrame or `{rows: [...], score_columns: [...]}` | `record_id`, `input`, `output`, `reference`; explicit `columns` can rename fields |
| `trulens` | `(records_dataframe, feedback_columns)` or `{records: [...], feedback_results: [...]}` | `record_id`, `main_input`, `main_output`, `ground_truth`; feedback joined by record ID |

For table profiles, `columns` maps the canonical roles `id`, `input`, `output`
and `expected` to your actual column names. MLflow also recognizes
`predictions`/`targets`. Retain source tables and per-case errors alongside
aggregate results; a summary table cannot reconstruct omitted predictions.
For TruLens model records, `meta` carries case metadata and native FeedbackResult
objects can be supplied separately. For Azure rows, `inputs.metadata` carries case metadata; native numeric
`outputs.<evaluator>.<metric>` fields remain attributed per-case observations.

Garak source cases explicitly join each generation to the planned task:

```python
source_cases = [{
    "native_id": f"{attempt_uuid}:0",
    "id": planned_case_id,
    "input": original_prompt,
    "expected": reviewed_reference,
    "metadata": {"category": "reviewed-task"},
}]
result = {"attempts": actual_attempts, "source_cases": source_cases}
```

Provide every generation, including errors, and preserve the exact original
prompt. Attack targets and detector scores are not answer references. Use a
reviewed `expected: null` for a reference-free judge task. Attempt history and
its completion state remain in context.

## Capture a Langfuse experiment as JSON

Langfuse 4.14.1 uses ordinary result classes. Copy their public fields explicitly;
this recipe needs only the SDK and Python's standard library. Supply the actual
ExperimentResult as `result` and the complete independently planned `planned_ids`.
It accepts local dictionary items and hosted DatasetItem objects, preserving
metadata, evaluations, trace IDs and dataset fields.

```python
import json
from langfuse.api import DatasetItem


def capture_evaluation(value):
    return {key: getattr(value, key) for key in (
        "name", "value", "comment", "metadata", "data_type", "config_id"
    )}


def capture_item(value):
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, DatasetItem):
        raise TypeError("expected a local item dictionary or DatasetItem")
    item = {key: getattr(value, key) for key in (
        "id", "input", "expected_output", "metadata", "source_trace_id",
        "source_observation_id", "dataset_id", "dataset_name"
    )}
    item["status"] = value.status.value
    for key in ("created_at", "updated_at"):
        item[key] = getattr(value, key).isoformat()
    item["media_references"] = [
        media.model_dump(mode="json") for media in value.media_references
    ]
    return item


native = {key: getattr(result, key) for key in (
    "name", "run_name", "description", "experiment_id",
    "dataset_run_id", "dataset_run_url"
)}
native["item_results"] = [{
    "item": capture_item(row.item),
    "output": row.output,
    "evaluations": [capture_evaluation(value) for value in row.evaluations],
    "trace_id": row.trace_id,
    "dataset_run_id": row.dataset_run_id,
} for row in result.item_results]
native["run_evaluations"] = [
    capture_evaluation(value) for value in result.run_evaluations
]
ids = []
for row in native["item_results"]:
    item = row["item"]
    local_id = (item.get("metadata") or {}).get("invarlock_id")
    hosted_id = item.get("id")
    if local_id is not None and hosted_id is not None and local_id != hosted_id:
        raise ValueError("conflicting local and hosted item IDs")
    identity = hosted_id if hosted_id is not None else local_id
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("every item requires a stable ID")
    ids.append(identity)
if (not planned_ids or len(planned_ids) != len(set(planned_ids))
        or len(ids) != len(set(ids)) or set(ids) != set(planned_ids)):
    raise ValueError("capture differs from the complete planned schedule")
with open("langfuse-native.json", "x", encoding="utf-8") as stream:
    json.dump(native, stream, ensure_ascii=False, allow_nan=False)
```

Use `adapter: evaluator-native-json`, `source: {name: langfuse, version: "4.14.1"}`
and the actual `result.run_name` as the request's `run_id`. Keep the whole
planned schedule: Langfuse may omit failed tasks from `item_results`. Capture
failures in your task wrapper with null output and
`item.metadata.invarlock_error`, or complete the missing capture before export.
The ID check rejects omitted results. Hosted item dates use ISO text and Python
field names, avoiding the API aliases used by generic SDK dumps. With compatible
co-installation, the common Python exporter accepts ExperimentResult directly.
The [Langfuse example](../../integrations/langfuse/README.md) explains identities
and failure capture in the complete handoff.

## Metrics, slices and scorer facts

Per-case string metadata supplies slices. Scalar wrappers carry `metadata` beside
`id`; batch profiles retain row metadata, and TruLens maps `meta`. The native
profiles above retain their declared per-case numeric metrics. Additional explicit
`metadata.invarlock_scores` can select named finite numeric observations;
conflicting values are rejected. For table metrics, use `score_columns`, or
TruLens `feedback_columns`, to select the original columns. Keep aggregate scores
in their native summary context.

To compare numeric observations, pass the reviewed `score_provenance` to the
exporter and select a `recorded` policy with matching `accepted_provenance`.
This verifies attribution and paired arithmetic; the evaluator name does not
authenticate the original scoring process. Exact match, NLL and judge scoring
keep their own fact requirements and do not reinterpret numeric scores.

Pass actual typed reference-continuation facts in
`metadata.invarlock_likelihood`. A likelihood-only case may retain a null output;
never invent generated text. Structured inputs can use the exporter's
`input_projection={"kind": "json-pointer", "pointer": "/input/question"}`.
This retains the original source and the exact text selection for replay.

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

## Historical qualification capture

The checkout's `capture.py` helper remains available for explicit canonical
records and retained qualification joins. Its `qualification` path checks the
original cases, independent schedule, profile, raw output and export bindings.
The historical shortlist contains 17 per-case deterministic exports, plus
MLflow aggregate observations and Garak detector summaries. Those two historical
summaries cannot supply native prediction rows or attempts. The new dedicated
profiles above capture the original per-case facts from current workflows.

```bash
python examples/evaluator-qualification/maintained/capture.py matrix
```

This helper writes canonical runs and prints input capabilities; it is a
checkout script, not an installed CLI or an evidence acceptance decision.
Use `adapter: invarlock` for its canonical output. The common native exporter
instead writes files consumed with `adapter: evaluator-json`.

For a captured request, set `comparison.metric` to `exact_match`,
`normalized_nll_per_utf8_byte` or `judge`. Exact match and NLL must agree with the
single metric in the comparison policy. Judge uses its own recipe and complete
retained-call contract. Omitting the selector keeps ordinary multi-metric
comparison behavior.

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
The paths above select the small two-case conformance fixture. To capture the
real 102-record model corpus, use the matching cases, schedule and evaluator
artifacts under `examples/evaluator-qualification/authoritative/` together;
do not mix files from the two corpora. The supplied artifact digest attributes
the run and must come from the source evaluation, not from hashing this export.

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

Check the printed capability counts before writing the comparison request. An
unavailable ID means its retained facts cannot support that scorer. Fix the
source capture or choose a scorer supported by those facts; do not drop the
case to make the counts look complete. A successful capture still needs the
policy, publication and recipient verification steps below.

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

When capture fails, distinguish missing per-case facts from an unsupported
native export layout or a runtime that cannot supply the requested measurement.
For an unsupported layout, explicitly map the original records with the SDK;
mapping cannot recover outputs or likelihoods absent from the source data.

Native-shaped retained replay exercises all 19 export/import paths with the
shared scorers and recipient verification/reporting. It adapts retained model
answers and likelihoods with explicit source bindings, and uses synthetic
complete judge-call fixtures. It does not claim that each SDK produced new
model or judge measurements. Separate pinned SDK smoke tests use actual native
objects and local evaluator calls to check the capture interface. Neither kind
of test is a new model-quality campaign or a replacement for the historical
qualification matrix.

The installed gate separately re-evaluates complete real Luna grounded-QA
measurements: the 422-case held-out study passes and the 40-case corrected pilot
remains insufficient evidence. Both use the unchanged original runs, plan,
policy and ratings through preflight, signed evaluation, independent verification
and reporting, with changed-measurement rejection. These authentic journeys
complement the adapter fixtures; old measurements cannot be rebound to newly
exported run identities. See the [real judge journey](../../integrations/evaluator-parity/README.md).

The [Mistral 7B likelihood reference](../../captured-results/references/mistral-7b-likelihood/README.md)
retains actual baseline-to-Instruct measurements for 400 paired narrative
continuations and a signed captured handoff. The earlier
[Harness likelihood control](../../captured-results/references/harness-likelihood/README.md)
retains six same-model CPU pairs. Both preserve the context, continuation,
log-probability sum, token/byte counts, tokenizer, configuration and model pins.
Their measured scope remains specific to those runs. Additional synthetic NLL
contract tests cover edge cases; none of these replays qualifies likelihood
measurement across all 19 SDKs. A generation-only export remains insufficient
for NLL, even if the upstream evaluator can measure likelihoods separately.

The judge tests likewise establish replay and integration behavior rather than
a new hosted quality result. Existing retained qualification observations and
judge pilot outcomes keep their original scope and decisions.
