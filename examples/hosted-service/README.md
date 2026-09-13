# Hosted-service capture and handoff

These helpers collect bounded text completions from an OpenAI-compatible HTTP
endpoint, import the retained observations, and run a signed handoff between two
installed core wheels. They are example integrations, not native runtime
providers or a scheduling service. No provider SDK is required.

[example-protocol.json](example-protocol.json) contains four explicitly synthetic
smoke cases. It is a starting configuration, not measured service evidence. Its
policy requires 20 cases, so this four-case fixture yields insufficient evidence
even when every answer matches. Do not interpret the fixture as model or agent
quality qualification. For an actual campaign, approve a relevant schedule and
policy before capture using the [requalification guide](../../docs/user-guide/hosted-service-requalification.md).

## Prepare the environment and protocol

Use Python 3.12 or newer with the core wheel installed. Run these commands from
the matching checkout: for a released wheel, use its exact release tag archive;
for a local wheel, use the source checkout that built it. Follow the
[matching wheel convention](../../docs/user-guide/getting-started.md#matching-wheels-and-examples).
Use ordinary files and directories without symlink parents. Output paths must
be new; the helpers reject overwrites.

```bash
mkdir campaign
cp examples/hosted-service/example-protocol.json campaign/protocol.json
```

Before the next command, edit `campaign/protocol.json` to name an endpoint you
control or are authorized to call, the requested and expected returned model
identities, and the deployment labels. The example assumes an already running
loopback test endpoint; it does not start a model server. Keep its cases and
policy when exercising only the synthetic smoke. Replace them with an approved
campaign design before making a qualification claim.

The closed protocol requires these fields:

| Field | Meaning |
| --- | --- |
| `services.baseline`, `services.subject` | Provider/service/deployment labels, exact endpoint, requested `model`, `expected_observed_model` and expected `exposed_revision` |
| `configuration` | Shared `temperature: 0`, literal `system_prompt` and `max_tokens` from 1 through 4096 |
| `limits` | Per-side call, token, response-byte, request-time and whole-window limits described below |
| `cases` | Ordered unique IDs of at most 128 characters, with literal input and expected-output strings; references are not sent to the endpoint |
| `policy` | A captured comparison policy using only `exact_match` metrics, fixed before results are inspected and validated before service calls |
| `environment` | Bounded JSON object describing relevant non-secret environment facts |
| `harness` | Source name/version of at most 128 characters each, reused as the canonical run source, and physical source digest of the approved capture harness |
| `collector_source_digest`, `journey_source_digest` | SHA-256 of the exact `capture.py` and `journey.py` file bytes |

Admission validates the complete policy before making service calls. If the
policy declares `expected_case_set_digest`, it must match the declared cases
with empty metadata, derived using the public `freeze_case_set` and
`case_set_digest` helpers. This text-only collector supports `exact_match`
metrics; it rejects NLL, recorded-score and other metric kinds. Use another
capture path with the required facts for those core scorers.

The endpoint must have path `/v1/chat/completions`. HTTPS is required except for
literal loopback HTTP hosts `127.0.0.1` and `::1`. URLs reject embedded credentials,
query strings and fragments. There is no proxy discovery, redirect following,
automatic retry, streaming or tool-call handling. Each request contains one
system message and one user message, with `n: 1` and `stream: false`.

A successful response needs a model string, one assistant text choice at index
zero, `finish_reason` of `stop` or `length`, and nonnegative integer
`usage.completion_tokens` within `max_tokens`. Non-null
`expected_observed_model` requires an exact returned-model match. Protocol
`exposed_revision` is an expectation for the response's optional top-level
`revision` field: a non-null expectation requires an actual matching value.
Set it to `null` when no revision expectation is available. The canonical
observed revision comes from retained responses and remains `null` if unavailable;
it is never copied into evidence as an observed fact merely because the protocol
names it. Successful responses must agree on their observed identities within
one capture window. These labels do not establish immutable weights.

Limits apply separately to each side: at most `max_calls` sequential calls,
`max_total_output_tokens` reserved output tokens, `max_response_bytes` per response,
`timeout_seconds` per request and `max_wall_seconds` for the whole window. The
schedule must fit the call and token reservations before collection. Each request
uses the smaller of its timeout and the remaining window. A failed
`deadline_exceeded` attempt alone may include up to one second for worker
termination and reaping; successful answers receive no such grace. A completed
capture must still fit its whole-window and cumulative-duration bounds. Replay
checks retained timing claims and does not independently establish timing accuracy
or execution. The fixture reserves four calls and 32 output tokens per side, for eight calls and 64 output tokens
across both sides. It caps each response at 4096 bytes, each request at 10 seconds
and each side's window at 60 seconds.

Admission also reserves encoded output space before calls. The protocol and
declared raw-response budget are each limited to 16 MiB. For each side, the
helper conservatively estimates the exported canonical run, including repeated
inputs, references, system/user requests, Base64 response expansion and normalized
answer text, and rejects a reservation above 64 MiB. The limit concerns encoded
bytes, not only character counts or the expected size of a successful response.

The next command replaces the three illustrative source pins with hashes of the
actual approved helper bytes and prints the physical protocol digest. Here the
collector itself is the harness. If an enclosing harness prepares the experiment,
record that harness's own source digest instead. Review the resulting protocol
and retain its pin through a recipient-controlled channel before collection.

```bash
PROTOCOL_SHA256=$(python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("examples/hosted-service")
protocol_path = Path("campaign/protocol.json")
protocol = json.loads(protocol_path.read_text())
def physical_digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
protocol["collector_source_digest"] = physical_digest(root / "capture.py")
protocol["journey_source_digest"] = physical_digest(root / "journey.py")
protocol["harness"]["source_digest"] = protocol["collector_source_digest"]
raw = (json.dumps(protocol, sort_keys=True, separators=(",", ":")) + "\n").encode()
protocol_path.write_bytes(raw)
print(physical_digest(protocol_path))
PY
)
```

These are hashes of physical source/protocol bytes. They are distinct from the
core's canonical complete-run, service-identity and normalized-request digests.
Any source or protocol edit requires new approved pins and a new campaign;
do not change historical inputs to make an old capture pass.

## Collect baseline and subject

For an unauthenticated loopback endpoint, run:

```bash
BASELINE_SHA256=$(python examples/hosted-service/capture.py collect \
  --protocol campaign/protocol.json --expected-protocol-sha256 "$PROTOCOL_SHA256" \
  --role baseline --output campaign/baseline)
SUBJECT_SHA256=$(python examples/hosted-service/capture.py collect \
  --protocol campaign/protocol.json --expected-protocol-sha256 "$PROTOCOL_SHA256" \
  --role subject --output campaign/subject)
```

For an endpoint requiring authorization, add `--token-env SERVICE_TOKEN` to each
`collect` command after securely populating that environment variable. The option
names a variable, never a literal token. Do not put tokens in the protocol,
command arguments or retained files. The collector sends the token as a Bearer
header and suppresses response bodies containing detected credential echoes,
including JSON-escaped echoes. That guard does not replace reviewing other
sensitive request and response content before sharing it.

Each successful command prints the physical SHA-256 of its new `capture.json`.
Retain the original protocol and full capture directories, including the numbered
attempt and result files. The handoff requires the baseline observation window
to end before the subject window starts. It compares these observed windows;
it does not measure behavior between them.

A failed request is retained with an explicit error instead of being retried or
dropped. Some failures can finish a complete schedule with unusable records, which
remain visible to the scorer. Missing or excessive token accounting, exhaustion
of the whole window, and interrupted collection leave partial attempt/result
files without a completed `capture.json`. Preserve diagnostics, resolve the cause
and start a new destination under the declared retry rules. Do not export an
incomplete capture or silently fill gaps with later calls.

## Export offline or run the signed handoff

A recipient independently approves the original protocol and capture pins. A hash
printed by the operator is a value to authenticate, not automatic recipient
approval. Transfer the original files and approved helper sources; keep independent
expectations outside any submitted evidence pack.

To export one side without service access or credentials:

```bash
python examples/hosted-service/capture.py export \
  --protocol campaign/protocol.json --expected-protocol-sha256 "$PROTOCOL_SHA256" \
  --role subject --capture campaign/subject/capture.json \
  --expected-capture-sha256 "$SUBJECT_SHA256" --output campaign/subject-run.json
```

Repeat with the baseline role and approved baseline capture pin if preparing the
[core captured request](../../docs/user-guide/captured-results.md) yourself. Export
checks physical pins, request order, timing, response facts and service identity,
then writes a canonical run with `artifact_digest: null`. It makes no HTTP calls.

For the complete wheel handoff, create separate operator and recipient environments
with the same core version. The following installs a released version matching
the core already selected in your current Python environment. For a local build,
replace both `invarlock==...` package arguments with the exact same built wheel.
Neither environment should contain provider SDKs or an editable InvarLock install.

```bash
INVARLOCK_VERSION=$(python -c 'from importlib.metadata import version; print(version("invarlock"))')
python -m venv operator-env
python -m venv recipient-env
operator-env/bin/python -m pip install "invarlock==$INVARLOCK_VERSION"
recipient-env/bin/python -m pip install "invarlock==$INVARLOCK_VERSION"
python examples/hosted-service/journey.py \
  --operator-cli operator-env/bin/invarlock --recipient-cli recipient-env/bin/invarlock \
  --protocol campaign/protocol.json --protocol-sha256 "$PROTOCOL_SHA256" \
  --baseline campaign/baseline/capture.json --baseline-sha256 "$BASELINE_SHA256" \
  --subject campaign/subject/capture.json --subject-sha256 "$SUBJECT_SHA256" \
  --output campaign/handoff
```

The helper independently imports the approved captures in each wheel environment,
checks that their derived run/request expectations agree, generates separate
example signer/verifier keys, publishes signed evidence, verifies it and renders
HTML, Markdown and JUnit. It rejects a shared environment or editable installs.

This is a local handoff rehearsal: the launcher supplies the pins and transfers
the freshly generated signer fingerprint. For operational use, the recipient
must independently authorize the signer and control its policy and run/request
expectations; two environments alone do not establish that independence. Keep
private keys with their respective roles and exclude them from shared evidence.
The generated keys are example keys, not an authorization scheme.

Inspect `campaign/handoff/result.json`, the immutable
`campaign/handoff/recipient/evidence/`, its separate
`verification.receipt.json`, and `report.html`, `report.md` and `report.xml` in
the recipient directory. The synthetic four-case example should report
`insufficient_evidence` under its 20-case minimum. Successful helper completion
means authenticated replay finished, even when the policy decision is adverse;
inspect the recorded decision separately. Verification and report rendering make
no new model calls and do not attest execution or authorize deployment.

For a bounded illustration of evaluating resulting files instead of an agent's
completion message, see [agent_outcomes.py](agent_outcomes.py) and the
[agent-outcome explanation](../../docs/user-guide/hosted-service-requalification.md#evaluate-an-agents-resulting-work).
It executes only fixed synthetic candidate files and tests, without an agent,
model or judge call.
