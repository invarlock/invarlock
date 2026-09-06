# Existing pipeline example

Install the unreleased source checkout or its candidate wheel, then run:

```bash
invarlock-pipeline init release-check --example extraction
invarlock-pipeline compare release-check/pipeline.json --output release-check/result
```

The installed command creates a complete 40-case synthetic example, a reusable
project and policy, and quality plus latency checks on the full dataset and a
selected category. Also try `--example classification` or `--example judge`.
The examples are illustrative; they do not establish deployment quality.

Follow the [integration guide](../../docs/user-guide/pipeline-integration.md) to
replace these records with your evaluator's outputs, set approved thresholds,
use the SDK or native parsers and add the gate to CI.

The distribution smoke test runs outside the source checkout, clears Python's
source-path override and exercises all examples, signing, independent
verification, reports, repeated destinations and every decision exit code:

```bash
python examples/pipeline/wheel_smoke.py --cli invarlock-pipeline
```

It requires an installed candidate wheel. All generated examples, keys and
reports stay in a temporary directory that is removed on completion.

## Real native evaluator rehearsal

`native_rehearsal.py` runs a small local model through Inspect, LM Evaluation
Harness or Promptfoo and retains each framework's original sample export.
`native_handoff.py` imports those exports through the installed public SDK,
checks them against an independently selected protocol and capture digest,
and adds explicitly attributed model-call latency. This is a maintainer
integration check; ordinary pipeline users can import their existing exports
directly using the integration guide above.

The fixed protocol contains four sentiment classifications, four JSON field
extractions and four arithmetic questions. Each runs with both a baseline prompt
and a more explicit candidate prompt, giving 24 actual completions per evaluator.
Before inference, the protocol records model file hashes, generation settings,
expected answers and a 75% candidate quality floor. Incorrect output remains a
quality observation; malformed generated JSON or numbers score zero.

Use Python 3.12 on Linux or macOS and an isolated evaluator environment with:

| Evaluator | Required package or command |
| --- | --- |
| Inspect | `inspect-ai==0.3.254`; [maintained installation recipe](../integrations/inspect-ai/Dockerfile) |
| LM Evaluation Harness | `lm-eval==0.4.12+invarlock.exactmatch.1`; [authenticated derivation and installation](../integrations/lm-evaluation-harness/Dockerfile) |
| Promptfoo | `promptfoo@0.121.19`, with Node.js satisfying `^20.20.0` or `>=22.22.0` |

Each model environment also needs Torch 2.13.0 and Transformers 5.14.1.
Torch's CPU wheel suffix is accepted. Use the maintained image recipes for the
Linux runtime dependency closures; the capture records the complete installed
Python package versions. A local-version LM wheel must come from the authenticated
repository derivation, not an unrelated package with the same version string.
The capture script does not install dependencies or download model weights.

Select a small local model directory containing ordinary files, without symbolic
links. For example, a copied cache of `sshleifer/tiny-gpt2` at revision
`5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be` is enough to exercise the interface.
This model is intentionally unsuitable for demonstrating useful task quality.
Set the paths below to the selected model and evaluator environment:

```bash
MODEL_DIR=/absolute/path/to/copied-model
CAPTURE_PYTHON=/absolute/path/to/evaluator-env/bin/python
EVALUATOR=inspect
PROMPTFOO_BIN=/absolute/path/to/promptfoo
mkdir -p rehearsal
PROTOCOL_SHA256=$(python examples/pipeline/native_rehearsal.py prepare \
  --model "$MODEL_DIR" \
  --model-id sshleifer/tiny-gpt2 \
  --revision 5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be \
  --output rehearsal/protocol.json)
```

Record `PROTOCOL_SHA256` before inference. Do not edit the protocol after seeing
results. Use a fresh capture directory for every attempt; failed attempts retain
diagnostics and cannot publish a completed capture manifest.

Run the opt-in integration check from the development environment. The model
executes in `CAPTURE_PYTHON`, while pytest orchestrates and checks the capture:

```bash
INVARLOCK_RUN_NATIVE_PIPELINE=1 \
INVARLOCK_NATIVE_EVALUATOR="$EVALUATOR" \
INVARLOCK_NATIVE_PYTHON="$CAPTURE_PYTHON" \
INVARLOCK_NATIVE_MODEL="$MODEL_DIR" \
INVARLOCK_NATIVE_PROTOCOL="$PWD/rehearsal/protocol.json" \
INVARLOCK_NATIVE_PROTOCOL_SHA256="$PROTOCOL_SHA256" \
INVARLOCK_NATIVE_OUTPUT="$PWD/rehearsal/$EVALUATOR-capture" \
INVARLOCK_NATIVE_PROMPTFOO="$PROMPTFOO_BIN" \
python -m pytest -q tests/integration/test_native_pipeline_rehearsal.py --no-cov
```

Repeat with `EVALUATOR=lm-eval` and `EVALUATOR=promptfoo`, selecting the matching
Python environment. `PROMPTFOO_BIN` is used only for Promptfoo. Its HTTP provider
requires a loopback listener; model inference stays local. Inspect uses pytest's
normal suppression of its desktop View notification. Trace files, dataset
caches and Promptfoo state are placed inside the capture directory.

The manifest includes package versions, the capture script, raw file hashes,
first-result and total elapsed times, process CPU time and peak resident memory.
Each sample has its raw completion, error status and measured model-call latency.
LM timing includes its HF wrapper; these measurements do not support a
cross-framework performance ranking.

Obtain and independently approve the completed capture digest:

```bash
CAPTURE_SHA256=$(python -c \
  'import hashlib,sys; print("sha256:" + hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
  "rehearsal/$EVALUATOR-capture/capture.json")
python examples/pipeline/native_handoff.py \
  --capture "rehearsal/$EVALUATOR-capture" \
  --protocol rehearsal/protocol.json \
  --expected-protocol "$PROTOCOL_SHA256" \
  --expected-capture "$CAPTURE_SHA256" \
  --output "rehearsal/$EVALUATOR-project"
```

The projection creates a separate project and policy for `classification`,
`extraction` and `numeric`. Sign a comparison using the installed public CLI:

```bash
invarlock-pipeline keygen rehearsal/keys
invarlock-pipeline compare "rehearsal/$EVALUATOR-project/classification/pipeline.json" \
  --output "rehearsal/$EVALUATOR-classification-result" \
  --signing-key rehearsal/keys/private.pem
```

Exit 1 means a measured quality regression or failed absolute floor; exit 3 means
insufficient evidence. Both can be correct outcomes for a tiny model. Exit 2 is
an integration error. Repeat the comparison for extraction and numeric projects.

For independent verification, install the candidate core wheel in a fresh
environment outside the checkout. Copy `native_handoff.py`, the original native
capture, protocol, evidence and public key to the recipient. Keep the signing
key with the evaluation operator. Supply the approved protocol digest, capture
digest and public key through the recipient's trust process, then rerun the
projection into a fresh directory. Use its regenerated policy and run digests:

```bash
python native_handoff.py \
  --capture native-capture \
  --protocol protocol.json \
  --expected-protocol "$PROTOCOL_SHA256" \
  --expected-capture "$CAPTURE_SHA256" \
  --output recipient-project
BASELINE_SHA256=$(invarlock-pipeline digest recipient-project/classification/baseline.json --run)
CANDIDATE_SHA256=$(invarlock-pipeline digest recipient-project/classification/candidate.json --run)
invarlock-pipeline verify classification-result/evidence.json \
  --public-key public.pem \
  --policy recipient-project/classification/policy.json \
  --expected-baseline "$BASELINE_SHA256" \
  --expected-candidate "$CANDIDATE_SHA256"
```

The recipient needs neither model weights nor evaluator packages. Projection
rejects source/configuration/reference/output drift, missing or reordered records,
ambiguous completions, invalid latency and native scores that disagree with
strict equality. Inspect's native `match` scorer can normalize whitespace;
normalization-dependent results need a different explicit mapping. This profile
also excludes tools, non-text answers, remote model code and multiple epochs.
A signature authenticates the retained records and supports arithmetic replay;
it does not establish that a dishonest evaluation operator ran the model.

Use the ordinary example tests for deterministic orchestration and failure
coverage; they replace optional SDK transports and are not inference evidence:

```bash
python -m pytest -q \
  tests/examples/test_native_pipeline_handoff.py \
  tests/examples/test_native_rehearsal_execution.py
make coverage-examples
```

These authored cases establish integration behavior only. They do not qualify
K2 models, demonstrate production task quality, or measure customer demand.
