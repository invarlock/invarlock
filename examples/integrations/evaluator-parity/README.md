# Evaluator export parity

This example checks each of the 19 dedicated native export profiles through the
same installed InvarLock recipient. It does not call an evaluator SDK, model,
judge service or container. Native field dictionaries exercise the documented
export contracts; they are not evidence of a new upstream SDK execution.

Every scorer runs with both direct native JSON (`evaluator-native-json`) and the
optional InvarLock export envelope (`evaluator-json`): 114 installed journeys.
Direct JSON lets the evaluator write its result in its own environment and the
recipient import it separately. The producer does not need InvarLock or its
dependencies installed. The recipient supplies the source name and version,
run identity, model or service identity, and independently prepared trust inputs.
Both paths bind the exact source bytes and preserve the same recorded outcomes.

The three scorer journeys have different evidence sources:

| Scorer | Inputs and expected result |
| --- | --- |
| Exact match | Original 400-case retained Mistral 7B HTTP answers and policy; pass |
| Normalized NLL | Original 400-case retained Mistral 7B continuation measurements and policy; regression |
| Bounded judge | Explicitly synthetic two-case answers and complete constructed measurement records; pass |

The real input origins and file hashes come from the
[retained Langfuse reference](../../captured-results/references/langfuse/README.md).
Their original policies and outcomes remain unchanged. Each profile preserves
record IDs, prompts, references, outputs, slice tags and numerical likelihood
facts. The copied likelihood source field identifies the new export producer. For
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
verifies independently prepared run and policy anchors, confirms exact recorded
outcomes, then rejects changed evidence. The exporter also rejects omitted and
duplicate cases against a separately supplied expected-ID list and refuses to
replace an existing export.
