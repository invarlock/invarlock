# Evaluate results from an existing pipeline

Use this example when your evaluator has already produced answers or likelihood
measurements and you want InvarLock to compare them, verify the evidence and
create a report. You can keep your existing model execution pipeline. The
[supported capture profiles](../../docs/user-guide/captured-results.md) define
which records each scorer needs; an aggregate score alone is not enough.

The **baseline** is the model or configuration you compare against. The
**subject** is the proposed change. Both runs must cover the same declared cases.

## Start with an offline example

You need Python, an installed InvarLock package and this example checkout from
the same release or source revision. No model, GPU, provider account or evaluator
SDK is needed for this starter. Run from the checkout in a directory where
`release-check`, `report.html`, `summary.md` and `results.xml` do not already exist:

```bash
invarlock evaluate --init release-check --example extraction
invarlock evaluate release-check/request.yaml --unsigned
invarlock report release-check/artifacts/evidence --html report.html \
  --markdown summary.md --junit results.xml --explain
```

The first command writes sample baseline and subject records, a policy and a
request linking them. The second evaluates those records. The third writes the
reports and prints an explanation. These are synthetic extraction results, so a
pass demonstrates the workflow rather than useful model quality.

This starter is unsigned. For a result another team can verify, follow the
[complete signed handoff](../../docs/user-guide/captured-results.md#signed-handoff):
the operator signs the evidence, and the recipient supplies its own expected
runs, request, policy and signing identity before issuing a separate receipt.
A passing receipt does not authorize deployment.

## Choose the data for your scorer

| Scorer | What your pipeline must retain | Next step |
| --- | --- | --- |
| Exact match | Case inputs, baseline and subject answers, and references | Adapt the starter using the [captured-results guide](../../docs/user-guide/captured-results.md) |
| Normalized NLL | The specified reference continuation, its measured log probabilities and required identity/token metadata | Use the [Harness likelihood example](references/harness-likelihood/README.md) |
| Judge | Frozen task inputs and answers, an explicit judging plan and either new or retained judge measurements | Follow [judge captured answers](../../docs/user-guide/captured-results.md#judge-captured-answers) |

Normalized NLL measures how likely the reference text was, not whether a generated
answer matched it. Token usage alone cannot supply that score. The judge route
can import retained ratings without calling a judge again.

## Inspect real retained comparisons

The [reference index](references/README.md) links to complete results you can
replay without inference:

- [K2 Horizon 32B routing](references/k2-32b-routing/README.md): 4,000 paired
  records compare two prompts for the same model. The recorded policy rejects
  the subject; the reference retains all scores.
- [Harness likelihood](references/harness-likelihood/README.md): six pairs from
  separately loaded copies of one small model check the likelihood integration.
  They do not demonstrate task quality.
- [Mistral 7B likelihood](references/mistral-7b-likelihood/README.md): 400 public
  narrative passages compare two distinct full checkpoints. The subject fails
  the declared policy, and verification reproduces that rejection.

These are captured records, not native runtime qualification. Their pages explain
what was measured, which inputs are retained and what the results can support.

## Check an installed package

These scripts are for integration developers and maintainers checking an
installed core wheel. They exercise the commands in temporary directories, print
check summaries and remove their generated outputs on completion:

```bash
python examples/captured-results/wheel_smoke.py --cli invarlock
python examples/captured-results/scorer_wheel_smoke.py --cli invarlock \
  --fixture examples/judge-measurements
```

The first checks sample classification, extraction and recorded-score workflows,
signed handoffs, reports and expected failure exits. The second exercises exact
match, normalized NLL and retained judge import. Its NLL inputs are synthetic;
its one-case judge fixture must remain `insufficient_evidence`. Neither script
runs a model or contacts a judge. Successful scripts exit zero after checking
both passing and deliberately rejected examples. The three-scorer script gives
each CLI command a 60-second timeout.

Use a core-only environment with no Inspect or provider SDKs. To check a separate
recipient installation, copy the scripts and judge fixture outside the checkout
and run them with that installation's Python and `invarlock` executable.

For automation, add `--json` to individual CLI commands. Captured report results
use `requested_outputs`, `written_outputs`, `failed_output` and `errors`, rather
than a top-level `html` path. Use `--fail-on-policy` with evaluation when CI should
fail on an adverse result: exit `0` means pass, `7` means an adverse policy
decision, and `2` means invalid inputs or a local budget failure. Render only the
new evidence pack from a completed evaluation.

## Hosted-service campaigns

The [hosted requalification guide](../../docs/user-guide/hosted-service-requalification.md)
uses the same captured request for periodic or incident-triggered campaigns.
A hosted run has `artifact_digest: null` and explicit service identity,
configuration, harness and observation window. Fresh calls belong to the capture
harness; the core evaluates retained facts and verifies them offline. Local
fixtures establish contract behavior only and are not externally hosted
qualification evidence.

## Run a fresh comparison through an external evaluator

This advanced section is for integration developers who want to exercise the
capture itself. It runs inference and needs a separate evaluator environment
and local model files. For an offline demonstration, use the sections above.

`native_rehearsal.py` runs a small local model through Inspect, LM Evaluation
Harness or Promptfoo and retains each framework's original sample export.
`native_handoff.py` projects those exports with the installed engine SDK under
independently approved protocol/capture digests. Projection is a captured result,
not an InvarLock runtime-provider execution certificate.

The fixed protocol includes four sentiment classifications, four JSON field
extractions and four arithmetic questions, each with a baseline and a more
explicit candidate prompt: 24 actual completions per evaluator. Before inference
it records model file hashes, generation settings, expected answers and a 75%
subject quality floor. Incorrect output is retained; malformed generated JSON or
numbers score zero. Historical protocol/capture fields named `candidate` retain
their original bytes; the new evaluation requests and normalized runs use
`subject` fields.

Use Python 3.12 on Linux or macOS in a separate evaluator environment:

| Evaluator | Required package or command |
| --- | --- |
| Inspect | `inspect-ai==0.3.254`; [image recipe](../integrations/inspect-ai/Dockerfile) |
| LM Evaluation Harness | `lm-eval==0.4.12+invarlock.exactmatch.1`; [authenticated derivation](../integrations/lm-evaluation-harness/Dockerfile) |
| Promptfoo | `promptfoo@0.121.19`; Node.js `^20.20.0` or `>=22.22.0` |

The model environment requires Torch 2.13.0 and Transformers 5.14.1. Torch's CPU
wheel suffix is accepted. These dependency pins are the protocol's runtime
requirements, not InvarLock publication pins. The LM local-version wheel must
come from the authenticated derivation, not an unrelated same-version package.
The capture script does not install packages or download weights.

Select ordinary model files without symlinks. A copied `sshleifer/tiny-gpt2`
snapshot at `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be` exercises the interface but
is intentionally unsuitable for demonstrating useful task quality.

```bash
MODEL_DIR=/absolute/path/to/copied-model
CAPTURE_PYTHON=/absolute/path/to/evaluator-env/bin/python
EVALUATOR=inspect
PROMPTFOO_BIN=/absolute/path/to/promptfoo
mkdir -p rehearsal
PROTOCOL_SHA256=$(python examples/captured-results/native_rehearsal.py prepare \
  --model "$MODEL_DIR" --model-id sshleifer/tiny-gpt2 \
  --revision 5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be \
  --output rehearsal/protocol.json)

INVARLOCK_RUN_NATIVE_CAPTURE=1 \
INVARLOCK_NATIVE_EVALUATOR="$EVALUATOR" \
INVARLOCK_NATIVE_PYTHON="$CAPTURE_PYTHON" \
INVARLOCK_NATIVE_MODEL="$MODEL_DIR" \
INVARLOCK_NATIVE_PROTOCOL="$PWD/rehearsal/protocol.json" \
INVARLOCK_NATIVE_PROTOCOL_SHA256="$PROTOCOL_SHA256" \
INVARLOCK_NATIVE_OUTPUT="$PWD/rehearsal/$EVALUATOR-capture" \
INVARLOCK_NATIVE_PROMPTFOO="$PROMPTFOO_BIN" \
python -m pytest -q tests/integration/test_native_capture_rehearsal.py --no-cov
```

Record the protocol digest before inference and never edit it after seeing
results. Repeat with `lm-eval` or `promptfoo` in the matching environment. Use a
fresh capture directory for each attempt; failure retains diagnostics and cannot
publish a completed capture manifest. Promptfoo needs a loopback listener;
inference stays local. Captures retain raw hashes, installed package versions,
script identity, raw completions, error status, model-call latency, first-result
and total elapsed time, process CPU time and peak resident memory. LM timing
includes its HF wrapper; these observations do not rank framework performance.

Independently approve the completed `capture.json` SHA-256 and transfer the
original capture/protocol bytes with `native_handoff.py` to the recipient:

```bash
python examples/captured-results/native_handoff.py \
  --capture "rehearsal/$EVALUATOR-capture" --protocol rehearsal/protocol.json \
  --expected-protocol "$PROTOCOL_SHA256" --expected-capture "$CAPTURE_SHA256" \
  --output recipient-project
invarlock evaluate recipient-project/classification/request.yaml \
  --signing-key evaluation-signer.pem
```

Repeat for `extraction` and `numeric`. The tiny model may correctly produce
regression or insufficient evidence. Publication alone exits `0`; opt into the
policy gate with `--fail-on-policy`. For independent verification, use the exact
wheel's checkout examples (or its matching released tag) in a fresh environment
outside the checkout. Keep the evaluation signing key with the operator. Rerun
projection using independently approved protocol and capture digests, then derive
the trust profile from the regenerated policy, baseline/subject runs and request
using the [captured handoff](../../docs/user-guide/captured-results.md#signed-handoff).
The recipient needs neither evaluator packages nor model weights.

Projection rejects configuration/reference/output drift, missing or reordered
records, ambiguous completions, invalid latency and native scores inconsistent
with strict equality. Inspect `match` can normalize whitespace; results depending
on that normalization need another explicit mapping. This profile excludes
tools, non-text answers, remote model code and multiple epochs. A signature does
not prove that an untrusted evaluation operator ran a model.

The ordinary tests substitute SDK transports to check orchestration/failure
handling; they are not inference evidence:

```bash
python -m pytest -q tests/examples/test_native_capture_handoff.py \
  tests/examples/test_native_rehearsal_execution.py
```

These authored integration cases do not qualify K2 models or demonstrate
production task quality. The [K2 campaign](../qualification/k2-horizon/README.md)
retains its blocked/unqualified observations; no tiny or synthetic run replaces
those observations or the original upstream fixture bytes.
