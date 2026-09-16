# Save baseline and subject answers from your pipeline

Use this helper if you already have a model pipeline and want to save its answers
for an InvarLock comparison. It runs each task once against the baseline and once
against the subject, then keeps those answers unchanged for later scoring.
Here, **frozen** means retained without regenerating or selecting a better answer.

This example is for developers who can supply a Python adapter or executable
for their pipeline. The included demonstration needs Python, InvarLock and a
POSIX system such as Linux or macOS. It runs offline without a model or account.
Start with the offline run below; the adapter protocol and limits are reference
material for connecting your own pipeline afterward.

## What it produces

The helper writes `baseline_run.json` and `subject_run.json`, two
`invarlock/evaluation-run-v1` files that captured-result evaluation and judge
preparation can read. It makes no quality decision and does not score the
answers. The same frozen answers can support exact-match scoring and judge
measurements without generating them again. This helper does not capture
reference-continuation likelihoods, so its token usage cannot supply normalized
NLL. That scorer requires separately retained, bound likelihood facts.

The included adapter returns fixed example answers to demonstrate the connection. Its toy answers, tokenizer,
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

## Use an existing pipeline executable

The included subprocess transport avoids writing an importable Python adapter.
Configure an existing trusted executable that supports the JSON protocol below.
This offline fixture shows the complete configuration:

```bash
python - <<'PYCODE'
import json
import sys
from pathlib import Path

script = Path("examples/answer-capture/pipeline.py").resolve()
transport = {
    "argv": [str(Path(sys.executable).resolve()), "-I", str(script)],
    "assets": [str(script)],
    "environment": [],
}
with Path("answer-transport.json").open("x") as output:
    json.dump(transport, output)
PYCODE
python -m examples.answer_capture \
  --config answer-capture-config.json \
  --cases examples/answer-capture/cases.json \
  --transport answer-transport.json \
  --directory answer-process-output \
  --execute
```

Replace `argv` with your pipeline executable and arguments. The executable must
be an absolute non-symlink path. List every script and local configuration file
in `assets` using absolute paths. Their content hashes, the executable hash,
transport implementation hash and full argument list are retained. Keep
secrets out of arguments and files listed as configuration. `environment` names
only the environment variables explicitly passed to the process; values are not
retained. The default environment is empty, and the working directory is `/`.
Use absolute file paths. Pin and retain any other runtime dependencies yourself.

The transport launches one process per operation, without a shell. Standard
input contains one JSON object and a newline:

```json
{"operation":"generate","request":{"side":"baseline","case_id":"case-1","input":"Capital of France?","model":{},"max_output_tokens":32}}
```

`model` is the complete model object from the capture configuration; it is
abbreviated here. The `count_input_tokens` operation must tokenize locally,
without inference or a billed request, and return exactly `{"input_tokens": 3}`.
The `generate` operation must perform exactly one inference attempt and return
exactly these fields:

```json
{"output":"Paris","input_tokens":3,"output_tokens":1}
```

Both operations emit only JSON on stdout. Token usage must be accurate; input
usage must equal the prepared count. Stderr is bounded and discarded to avoid
printing credentials. A failed process produces an error with its exit status.
There are no transport retries, response selection or caching. The executable
itself is trusted and must disable those behaviors internally. This is a local
integration boundary, not a sandbox or a verified model provider adapter.

Input is capped at 1 MiB per operation. Stdout is capped at six times the
configured answer byte cap plus 4 KiB for JSON escaping and response fields;
stderr is capped at 64 KiB. Every operation shares the fixed capture deadline and
per-call timeout. Cancellation terminates the process group, including ordinary
child processes. Trusted executables must not detach children. Neither this
transport nor local cancellation can stop already dispatched remote billing.
Retained completed answers are never regenerated; an expired process transport
requires offline inspection rather than invoking its tokenizer after deadline.

## Continue from captured answers to judge preparation

For the installed captured-result route, use these two canonical runs as sources
in `invarlock/evaluation-request-v2`, select `comparison.metric: judge`, and
supply a reviewed `invarlock/native-judge-policy-v1` recipe. The
[captured-results guide](../../docs/user-guide/captured-results.md#judge-captured-answers)
shows collection and offline measurement import. The public
`prepare_evaluator_judge` helper binds the recipe to the frozen records and
supports optional `plan.prompt.reference_mode: per_case` for string references.
References are never sent to the answer-generation adapter.

The following example helper prepares a v3 request from a finalized plan
template. Set `prompt.reference_mode: per_case` in that template to include each
string reference in judging; omitted or `none` mode leaves it out. The helper
binds the exact rendered requests for the selected mode.

Prepare a judge plan and request without generating answers again or calling a
judge. Supply a reviewed plan template, analysis policy, collection limits, and
an explicit mapping from each case to its statistical unit. Cases from the same
source or conversation should share a unit; do not invent independence by
assigning each related row a different unit.

This command uses the committed offline templates for a wiring check:

```bash
python -m examples.answer_capture_judge \
  --capture answer-process-output \
  --plan-template examples/judge-measurements/plan.json \
  --policy examples/judge-measurements/analysis_policy.json \
  --units examples/answer-capture/units.json \
  --collection examples/judge-measurements/collection.json \
  --directory captured-judge-inputs
invarlock evaluate captured-judge-inputs/request.json --preflight
```

The helper checks retained attempts/results against the frozen answers, binds
both exact run digests, case membership, answer hashes and rendered judge
requests, and writes a new directory. It preserves the template's rubric,
model, repetitions, sampling basis and policy thresholds. It does not lower
minimum sample requirements to make a small capture pass. The one-case fixture
is intentionally too small to meet its policy. Preparation and preflight can
succeed, but they do not establish a passing comparison.

For actual work, review those templates and units before preparation. Then use
the [judge collection example](../judge-measurements/README.md) with the generated
plan, runs, policy and collection configuration. Its optional collector script
accepts these paths. The offline example does not qualify a provider, model,
tokenizer or rubric. Capture budgets authorize answer capture only; review and
authorize judge collection separately.

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
Set input and output rates to conservative upper bounds across both deployments, including
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
