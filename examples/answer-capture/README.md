# Capture frozen answers from your pipeline

## Purpose

This optional POSIX example calls a user-owned pipeline adapter once per case
for the baseline and once for the subject. It writes the two
`invarlock/evaluation-run-v1` files consumed by captured-result evaluation and
bounded judge preparation. It makes no quality decision and does not score the
answers. The same frozen answers can support deterministic metrics and judge
measurements without generating them again.

The included adapter is an offline wiring fixture. Its toy answers, tokenizer,
artifact digests and prices are not model qualification or useful evaluation
results. No provider package is installed or contacted by the example itself.

## Run the offline example

From a repository checkout with InvarLock installed, prepare a new configuration
with a fixed deadline. The committed configuration has an expired deadline so
it cannot accidentally start a capture.

```bash
python - <<'PY'
import json
import time
from pathlib import Path

source = Path("examples/answer-capture/config.json")
config = json.loads(source.read_text())
config["limits"]["deadline_unix_seconds"] = int(time.time()) + 300
with Path("answer-capture-config.json").open("x") as output:
    json.dump(config, output)
PY
python -m examples.answer_capture \
  --config answer-capture-config.json \
  --cases examples/answer-capture/cases.json \
  --adapter examples.answer_capture_offline \
  --directory answer-capture-output \
  --execute
```

The output directory contains the frozen manifest, per-call attempts and results,
and `baseline_run.json` and `subject_run.json`. The command prints the run and
case-set digests. Keep the directory to retain provenance and support resume.
Rerunning the exact command reads completed results without generating again.

For judge collection, prepare a judge plan bound to these exact run digests and
case-set digest. Follow the
[judge measurements example](../judge-measurements/README.md); its committed
one-case fixture plan must be replaced with a plan for your new answers. Answer
capture ends before judge collection, so capture budgets do not authorize judge
calls.

## Connect a real pipeline

Create an importable Python module exposing these two functions:

```python
def count_input_tokens(request: dict) -> int:
    # Render the exact request using the identified tokenizer. No network calls.
    ...

async def generate(request: dict) -> dict:
    # Execute exactly one attempt with retries, fallbacks and caches disabled.
    # Enforce request["max_output_tokens"] in the actual provider request.
    return {
        "output": "the captured answer",
        "input_tokens": 123,
        "output_tokens": 45,
    }
```

Each request contains `side`, `case_id`, `input`, `model` and
`max_output_tokens`. `model` contains the explicit provider, model, revision,
tokenizer, deployment artifact digest and generation configuration. Render any
system prompt from that configuration. Reference answers and case metadata are
retained in the frozen case set but are never passed to the adapter. Include
necessary task context in `input`.

Use immutable deployment identities where the provider exposes them, and retain
all generation settings. The supplied artifact digest identifies the deployment
being compared; it must not be an invented model-weight hash. A hosted alias
alone does not prove which weights served a call. The manifest binds the adapter
module's source file; pin and retain its dependencies separately. These are
caller-supplied capture facts, not authenticated provider attestations.

The adapter is trusted executable code. It must use the intended endpoint,
perform no hidden retries or best-of selection, honor the output cap, report all
billed token usage, and support cancellation. Use environment credentials;
configuration and outputs are retained and must not contain secrets. Test and
qualify the adapter against its actual SDK, tokenizer and provider before using
its captures as reference evidence. The offline test does not establish that
qualification.

## Bounds and interruption

The example admits the complete case set only when two calls per case fit the
call cap, and the full per-call token reservations fit the total token and cost
caps. Costs use integer millionths of a US dollar: `1000000` means one dollar.
Set input
and output rates to conservative upper bounds across both deployments, including
any billed reasoning tokens or provider minimums. These are caller-supplied
prices, not a pricing lookup.

The concurrency limit controls simultaneous adapter calls. Each call has a
local timeout and shares a fixed campaign deadline; resuming cannot extend that
deadline or change membership, models, adapter source, prices or limits. The
provider must enforce its token cap. Local cancellation cannot guarantee that a
remote server stops computing or billing, and reported usage is checked only
after a response. Do not describe this helper as a provider-side spending limit.

A durable attempt is written before dispatch. A completed answer, including a
poor or empty answer, is retained once. An interrupted or invalid response leaves
an unresolved attempt, which blocks automatic resume before any further calls.
Inspect the provider logs and retain the incident; this example has no command
to delete an attempt, select a better answer, or retry an ambiguous call. A new
capture is a new study and requires an explicit decision about its validity and
budget. Never silently combine replacement answers with the old capture.

Use a private output directory on a local filesystem you control. Symlink paths,
unrelated files, changed retained inputs, and simultaneous capture processes are
rejected. Files are published without overwriting existing content. This
example uses POSIX file locks and is not a distributed scheduler.

The configurable output byte cap supplements token accounting because tokens
and UTF-8 bytes are different units. This small example also bounds the manifest
to 16 MiB, output to at most 1 MiB per call, and its conservative serialized-data
reservation to 128 MiB. It checks those reservations before dispatch. These are
helper memory and storage safeguards, not new InvarLock evaluation limits; use
your pipeline's own capture implementation for larger exports.
