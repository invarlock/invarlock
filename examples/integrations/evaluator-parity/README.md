# Evaluator export parity

This example checks each of the 19 dedicated native export profiles through the
same installed InvarLock recipient. By default it constructs native field
dictionaries. With `--native-captures`, it consumes hash-bound JSON files created
separately by actual SDK serializers. The recipient itself calls no evaluator
SDK, model, judge service or container. Neither route claims fresh model execution.

The separate [live campaign](../evaluator-live/README.md) executes framework
callbacks against a persistent model worker and checks the actual captures.
Its qualification remains separate from these retained contract replays.

Every scorer runs with both direct native JSON (`evaluator-native-json`) and the
optional InvarLock export envelope (`evaluator-json`): 114 installed journeys.
Direct JSON lets the evaluator write its result in its own environment and the
recipient import it separately. The capture process does not need InvarLock or its
dependencies installed. The recipient supplies the source name and version,
run identity, model or service identity, and independently prepared trust inputs.
Both paths bind the exact source bytes and preserve the same recorded outcomes.

The three scorer journeys have different evidence sources:

| Scorer | Inputs and expected result |
| --- | --- |
| Exact match | Original 400-case retained Mistral 7B HTTP answers and policy; pass |
| Normalized NLL | Original 400-case retained Mistral 7B continuation measurements and policy; regression |
| Bounded judge | Explicitly synthetic two-case answers and complete constructed measurement records; pass |

The gate also runs two complete real judge studies through the installed
recipient, using their original canonical inputs and unchanged Luna ratings:

| Retained grounded-QA study | Cases | Real ratings | Expected outcome |
| --- | ---: | ---: | --- |
| [Held-out study](../../judge-measurements/references/k2-32b-luna-xhigh-heldout/README.md) | 422 | 2,532 | Pass |
| [Corrected pilot](../../judge-measurements/references/k2-32b-luna-xhigh-pilot/README.md) | 40 | 240 | Insufficient evidence |

These journeys authenticate the pinned archives, reconstruct the exact original
plan and policy, then run preflight, signed evaluation, independent verification
and reporting. Changed measurements must be rejected. They make no new model or
judge calls. The pilot's inconclusive result remains unchanged.
The replay subprocesses discard any inherited network permission and block
outbound connections, including during preflight and evaluation.

To retain these real journeys locally:

```bash
python examples/integrations/evaluator-parity/real_judge.py \
  --recipient-python /path/to/recipient/bin/python \
  --output /path/to/new-real-judge-output
```

The output includes local demonstration signing keys; share selected public
reports and evidence rather than the entire directory. `--reference heldout`
or `--reference pilot` selects one study; the default runs both.

Real ratings bind the original runs and plan. An evaluator-specific re-export
changes those identities, even when the answer text stays identical. The adapter
matrix therefore retains synthetic judge fixtures for those new bindings. The
real journeys prove the complete retained-measurement judge workflow, without
claiming fresh judge collection through all 19 evaluators.

The exact-match and NLL input origins and file hashes come from the
[retained Langfuse reference](../../captured-results/references/langfuse/README.md).
Their original policies and outcomes remain unchanged. Each profile preserves
record IDs, prompts, references, outputs, slice tags and numerical likelihood
facts. The copied likelihood source field identifies the new export source. For
Harness and Promptfoo, its input digest binds the native structured input before
text projection. Original unchanged likelihood facts remain in native metadata;
original source files and hashes remain in `origin.json`. These are transformed
contract replays, not the original signed evidence. Point measurements and policy
outcomes must match; the recipient independently recomputes each interval, whose
resampling representation may differ. The original judge
campaign archives are not rebound to different runs or edited to support a new
integration claim. `synthetic-judge.json` is a contract fixture, not an observed
judge response.

`native_shapes.py` builds the supported fields for Inspect, Harness, Promptfoo,
Langfuse, DeepEval, Ragas, LightEval, Hugging Face Evaluate, AutoEvals, OpenEvals,
Phoenix Evals, Opik, Pydantic Evals, Azure AI Evaluation, Evidently, MLflow,
Garak, TruLens and OpenAI Evals. The profile names and pinned source versions
come from the [qualification matrix](../../evaluator-qualification/matrix.json).
SDK smoke evidence, where available, is a separate check from this contract
replay. An aggregate score never substitutes for missing cases, reference
continuation likelihood measurements or a complete bounded judge record.

Build the candidate wheel, then run the complete installed gate:

```bash
python -m build --wheel
PYTHON=python3.12 bash scripts/evaluator_parity_gate.sh
```

The gate requires exactly one wheel in `dist/` and pytest with pytest-xdist in
the launching Python environment. It installs the core dependency
lock in a temporary environment, verifies that InvarLock loads from that
environment's installed package directory, and checks that Python evaluator SDKs
are absent. Four test workers run all 19 profiles with all three scorers and both
input formats. Each export, evaluation, verification and report runs through the
installed recipient. Promptfoo is a
JavaScript evaluator; this program neither invokes Node.js nor calls Promptfoo.
Installing dependencies may use the package index; the parity journeys are
offline. Temporary keys and generated evidence are removed when the gate exits.

To keep one journey for inspection, supply a separate installed-wheel Python:

```bash
python examples/integrations/evaluator-parity/run.py \
  --evaluator inspect-ai --scorer normalized_nll \
  --input-format native-json \
  --recipient-python /path/to/recipient/bin/python \
  --output /path/to/new-parity-output
```

The default `--input-format envelope` exercises the optional export helper.
`--input-format native-json` writes the native payload directly and imports it
with the same profile. The output contains the selected source JSON format,
independently prepared trust inputs,
signed evidence, a recipient verification result, an HTML report and the compact
`result.json`. It also contains temporary demonstration private keys: keep the
output local and share selected public results rather than the entire directory.
The expected NLL policy rejection remains exit 7 for evaluation and verification;
the example exits zero when the declared outcome and tamper rejection both match.
No failed comparison is converted to acceptance.

Run the source contract checks with:

```bash
python -m pytest tests/integration/test_evaluator_parity.py -k 'not installed'
```

Set `INVARLOCK_EVALUATOR_PARITY_PYTHON` to the isolated candidate-wheel interpreter
to include all installed-recipient integration tests. Those tests reject an
interpreter importing evaluator SDKs or an editable checkout. Every journey
runs preflight without creating evidence, verifies independently prepared run
and policy anchors, confirms exact recorded outcomes, then rejects changed evidence. The exporter also rejects omitted and
duplicate cases against a separately supplied expected-ID list and refuses to
replace an existing export.


## Connect actual SDK exports to the recipient

```bash
make evaluator-sdk-test EVALUATOR=ragas
```

This gate builds a candidate wheel and installs it with hash-locked dependencies
in a core-only recipient environment. A separate environment installs the pinned
evaluator SDK. The SDK serializes both sides of the retained 400-case exact-match
and NLL comparisons, and the two-case synthetic judge comparison. Those exact
files then pass through the installed recipient with both import routes: six
journeys per evaluator. The gate includes preflight, signed evaluation,
independent verification, reporting, unchanged case facts and tamper rejection.

SDKs with case or report objects use those public objects. SDKs exposing only
metric results retain the original case arguments beside the SDK result. Required
wrapper fields and unused metric context are serialization fixtures. OpenEvals
uses its offline exact-match function on nullable JSON fields because its result
type is a plain mapping; that auxiliary grade is identified separately from the
retained measurements, and original task and answer text remain unchanged.
No new model answers or likelihood values are computed, and synthetic judge ratings remain
identified as synthetic. This proves the handoff, not a fresh model campaign.

To inspect externally captured files, add `--native-captures /path/to/captures`
to the single-journey command. That directory contains `baseline.json`,
`subject.json` and `origin.json`. The closed manifest declares the evaluator,
source version, scorer and SHA-256 of each fixed filename, plus capture provenance.
An optional `provenance.input_projection` explicitly supplies the task-text
mapping; the same mapping is retained in both request sources. The example
rejects changed bytes, missing cases, wrong identities and changed measurements.
It preserves exact capture files and the manifest beside the comparison.
