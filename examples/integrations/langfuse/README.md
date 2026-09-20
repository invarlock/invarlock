# Use Langfuse experiments with InvarLock

> **Outcome:** Keep your Langfuse experiment workflow and turn its complete
> records into an independently verifiable InvarLock comparison.
>
> **Audience:** Teams evaluating model or prompt changes with Langfuse.
>
> **Prerequisites:** The matching InvarLock example checkout and an installed
> InvarLock package. Producing exports requires `langfuse==4.14.1`.

## Capture an experiment

Run your existing task with the Langfuse experiment SDK. Give every local dataset
item a stable `metadata.invarlock_id` shared by baseline and subject. Hosted
dataset items use their dataset item ID. Preserve the expected ID list before
execution, including cases whose task might fail.

```python
from examples.integrations.langfuse_export import write_experiment_export

expected_ids = [item["metadata"]["invarlock_id"] for item in data]
result = langfuse.run_experiment(
    name="candidate-comparison",
    data=data,
    task=your_existing_task,
)
write_experiment_export(result, "subject.json", expected_ids=expected_ids)
```

Here `langfuse`, `data` and `your_existing_task` belong to your existing workflow.
Repeat for the baseline with the same case IDs, task inputs and references.
Review the captured fields before distributing them; the exporter preserves
inputs, outputs and metadata. It does not need or serialize a client credential.

The exporter retains the SDK's public experiment and item fields inside an
explicit `invarlock/langfuse-export-v1` envelope. It refuses existing output
paths, duplicate IDs and missing results. Langfuse may omit failed tasks from
its result list; handle those failures in your task wrapper and retain a null
output with `metadata.invarlock_error`, or complete the capture before exporting.
Do not manufacture an answer or drop the expected case to make export succeed.

## Select the scorer

| Scorer | Required captured facts |
| --- | --- |
| Exact match | Original task input, answer and independent expected output |
| Normalized NLL | Explicit reference-continuation log probabilities, byte/token counts, and model/tokenizer/configuration identities |
| Judge | Task and answer text, declared rubric and reference mode, plus complete judge-call measurements |

For NLL, carry the canonical likelihood object in
`item.metadata.invarlock_likelihood`. It must identify the actual measurement
and capture source. Token usage, cost and generic Langfuse scores are not
likelihood measurements. For judge scoring, use an explicit input projection
when the task input is structured; per-case references remain separate from
the model's input. A Langfuse judge score is retained context and cannot replace
InvarLock's complete judge-measurement contract.

## Evaluate, verify and report

Configure each captured request source with `adapter: langfuse-json`, the export
path, `source: {name: langfuse, version: "4.14.1"}`, and `run_id` equal to that
export's `result.run_name`. Supply the actual model artifact identity, or a
complete hosted-service identity if the model weights are unavailable. Preserve
the original source bytes; InvarLock binds their digest into the normalized run.

```bash
invarlock evaluate request.yaml
invarlock verify evidence/ --trust-profile recipient.json
invarlock report evidence/ --html report.html
```

Use the [captured request guide](../../../docs/user-guide/captured-results.md)
to supply the selected policy, signing key and independently approved recipient
inputs. Exporting a run does not choose those requirements for the recipient.
Import, verification and reporting run offline without a Langfuse SDK or account.

## What the reference establishes

The [retained SDK reference](../../captured-results/references/langfuse/README.md)
executes the actual Langfuse experiment runner over existing Mistral 7B answers
and likelihood facts. This checks the handoff; it does not rerun those models
or claim new model-quality evidence. Judge contract tests use explicitly
synthetic retained measurements and do not qualify a new live judge campaign.

The historical Langfuse 102-record qualification keeps its original identity
and scope. This adapter adds a convenient installed import route; it does not
turn every Langfuse trace, modality or score into supported evaluation evidence.
