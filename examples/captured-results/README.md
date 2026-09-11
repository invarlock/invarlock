# Captured Results Example

The [retained routing reference](references/k2-32b-routing/README.md) contains
4,000 paired records from real K2 Horizon 32B model captures. It provides a
complete signed pack and offline recipient replay of the expected regression,
with recorded-score assurance and no native runtime qualification claim.

This directory demonstrates the neutral captured-evaluation workflow. The
wheel smoke invokes the installed core `invarlock` command and checks signed
`evaluate`, independent `verify`, and non-mutating `report` before add-ins are
installed.

```bash
python examples/captured-results/wheel_smoke.py --cli invarlock
```

The native rehearsal and handoff helpers produce captured records for external
evaluators. They do not add a second CLI or SDK namespace. Runtime pipelines
remain ordinary upstream integrations; their exported records enter through
the `invarlock/evaluation-request-v2` request contract.

The smoke covers classification, extraction and recorded-judge starters, signed
handoffs, unsigned local reports, repeated destinations, all policy gate exits,
and v2 report output maps. Commands default to readable text; use `--json` for
automation. Captured report v2 has `requested_outputs`, `written_outputs`,
`failed_output`, and `errors`, not a top-level `html` path.

```bash
invarlock evaluate --init release-check --example extraction
invarlock evaluate release-check/request.yaml --unsigned --json
invarlock report release-check/artifacts/evidence --html report.html \
  --markdown summary.md --junit results.xml --explain
```

The starter is illustrative, not deployment evidence. A signed handoff uses
`evaluate --signing-key`, recipient-owned `--trust-profile` v2 run/request/policy/
signer pins and `verify --receipt` outside the pack. A passing captured receipt
cannot authorize native acceptance or deployment. `--fail-on-policy` gates
evaluation at `0` (pass), `7` (adverse decision), or `2` (input/local-budget
failure); render only newly published packs. See the
[complete signed/local guide](../../docs/user-guide/captured-results.md).

## Real native capture

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
