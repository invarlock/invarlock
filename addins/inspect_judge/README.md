# Inspect judge collection and import

This optional package collects bounded text judgments through the installed
`evaluate` command or a caller-supplied Inspect model, and imports its
expanded-event projection into
`judge-measurements-v1`. It does not import arbitrary Inspect `.eval` archives.

Installed live collection requires exactly Inspect `0.3.263`, OpenAI `3.13.0`
and `httpx==0.28.1`; retained exports from Inspect `0.3.254` also remain replayable
offline. The projection requires one epoch and an explicit grader, and retains all scheduled slots. Missing attempts remain
incomplete. SDK retries, cache reuse, tools, unrecorded generation settings,
extra request headers and arbitrary request bodies are rejected.

Live collection uses the retained Chat Completions projection. The caller must
construct the Inspect model with Chat Completions selected and both Inspect and
provider-client retries set to zero. The model may carry only the explicit
`responses_api=false`, `max_retries=0` and optional `service_tier=default`
construction arguments. Other inherited model, provider or generation settings are
rejected before a call is admitted.

The installed native and captured `metric: judge` workflows and frozen-answer
`judge_collect` request use `collect_configured`. It validates the pinned SDK environment,
constructs the supported model explicitly, and closes its client on success,
failure or cancellation. Keep `OPENAI_API_KEY` in the process environment.
Remove `OPENAI_BASE_URL` and `OPENAI_API_BASE` entirely; even empty overrides in
the process or explicitly supplied environment are rejected. The configured
client uses `https://api.openai.com/v1`, Chat Completions, explicit
`service_tier=default` and disabled model memoization. This selects standard
processing instead of inheriting the project's tier. Remove
`OPENAI_SAFETY_IDENTIFIER` as well; an inherited identifier would add an
unsupported request control and is rejected before collection. Missing or
mismatched dependencies and credentials fail preflight.

Offline replay accepts historical requests that omitted the tier without
rewriting them or asserting their billing tier. An explicit tier must be
`default`, with the same returned tier on completed calls. Inspect's disabled
response cache does not disable provider prompt caching; declared cost
reservations must include applicable cache-write charges.

```bash
python -m pip install .
python -m pip install 'addins/inspect_judge[inspect]'
invarlock evaluate judge-request.yaml --preflight --json
invarlock evaluate judge-request.yaml --signing-key signer-private.pem --json
```

Preflight checks without calling a provider. Execution uses the declared cost,
call, token and time ceilings, the official OpenAI endpoint and a private
checkpoint. It never puts the API key in the request, plan, checkpoint or retained
output. `validate_collection_environment` exposes the same execution-free checks
to Python callers. Native requests automatically freeze runtime answers before
judging; captured requests freeze an explicit recipe against normalized evaluator
records, and frozen-answer requests supply their approved plan and runs directly.
A captured v2 request with `comparison.judge.measurements`, or a v3
`judge_import` request, replays retained measurements without this package or
credentials. Caller-owned collectors can use the core `prepare_evaluator_judge`
and `import_judge_sources` APIs for the generic retained-call format; arbitrary
upstream scalar scores cannot replace those calls.

For `openai/gpt-5.6-sol` and `openai/gpt-5.6-luna`, the pinned SDK converts
system messages to developer messages, uses `max_completion_tokens`, and omits
temperature from the provider request. The approved plan must therefore declare
temperature `1`, the provider default, and an explicit non-null
`reasoning_effort`. The adapter passes that effort into Inspect and verifies it
in both the retained generation event and provider request. Collection rejects
a missing or changed effort before the evidence can be accepted. Offline replay
accepts exactly this version-bound projection for those two models; other models
keep their existing message and sampling-control checks. This SDK projection
support does not qualify either model's judging quality. Normalized requests
retain the approved system message and configuration alongside the actual
provider call.

`bind_requests` creates exact request digests for a plan before independent
approval. `render_request` uses the core renderer to separate rubric, input and answer
values in JSON fields. With `prompt.reference_mode: per_case`, the core renderer
adds each case's string reference as a separate bounded field; it never adds it
to the evaluated model input. Missing or non-string references fail validation.
Omitted or `none` reference mode preserves the original requests and retains
references only in the frozen evidence. Templates remain literal instructions. This separation
does not prove resistance to prompt injection. The original answers remain
digest-bound. Import and checkpoint replay require both frozen
`evaluation-run-v1` objects and validate their approved digests.

`prepare_collection` returns a deterministic bounded next batch and reservations
for calls, tokens, cost and retained storage. `collect` executes those batches
through an explicit Inspect model, applies a per-invocation deadline and request
pacing, and takes an exclusive one-writer lock on a caller-owned checkpoint
directory with no group or other access. It writes an immutable admission
shard before dispatching each provider call and a separate immutable result shard
after validating the retained event. An admission without a result is an
ambiguous timeout that consumes the reserved call and cannot be retried. A
validated checkpoint preserves completed responses, parse failures, refusals
and ambiguous timeouts. The live collector currently requires one attempt per
trial; imported evidence may retain explicitly declared transport retries.
Checkpoint reads and locking use a retained directory descriptor, and directory
ancestry is checked through dispatch and publication. Provider failure details
are reduced to a stable status/code and generic public message; raw exception
messages and error-response bodies are not retained.

Retained sources are deterministically divided at whole-trial boundaries. Each
source is at most 16 MiB, with at most 1,000 sources and 384 MiB of canonical
measurements in total. Each expanded event is at most 2 MiB. Admission reserves
20 MiB of possible retained growth and one possible additional source per active
call, plus mapping overhead for every slot; storage may reduce concurrency below
the requested value and stops further calls before the reserve is exhausted.
The maximum supported trial count therefore depends on retained content, not
only the plan's slot limit. The maintained capacity fixture retains 7,728 short
completed trials across multiple sources. This does not promise that 7,728
maximum-size responses fit the aggregate allowance.

`prepare_inspect_config` optionally constructs the pinned SDK's generation
configuration. Install the package's `inspect` extra to use that helper. Ordinary
import and planning do not import Inspect or a provider SDK.

## Export boundary

The `invarlock/inspect-judge-export-v1` projection has a closed envelope containing
collection options and scheduled samples. Samples bind case, side, repetition,
plan and answer digests. Each expanded model event includes an explicit grader,
generation settings, normalized model input, the bounded provider request and
response, accessible completion, model identity and attempt outcome. The
synthetic fixture in `tests/fixtures/export.json` describes this projection and
makes no external execution claim.

The maintained `examples/judge-measurements/import_inspect.py` command imports
that projection without the SDK extra or provider credentials. Install matching
core and add-in packages, then run it from the copied example directory:

```bash
python import_inspect.py --export inspect-export.json --collection collection-inspect.json --output measurements-inspect.json
invarlock evaluate request-inspect.yaml --unsigned --json
invarlock report evidence-inspect --html report-inspect.html --json
```

The included export is synthetic. Both frozen runs, the plan and the explicit
collection settings are checked before writing a new measurements file. Import
rejects unsupported profiles and does not repair missing judgments. The example
README documents path and size limits and continuation into signed publication.

Inspect's native `ModelCall` contains provider-specific request and response
objects. The retained `retained-inspect-model-events-v1` source keeps the bounded
Chat Completions request and response while the deterministic rating parser reads
the accessible model completion. Offline verification checks the provider
messages, model, generation controls, completion, resolved model, request ID,
finish reason and token usage against the normalized event and trial table. It
does so without loading Inspect. As with any retained API log, these bytes
establish what the producer signed and retained; they do not independently prove
that a provider performed the call.

The pinned SDK interfaces used for configuration and event-field inspection are
[`GenerateConfig`](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#generateconfig),
[`ModelEvent`](https://inspect.aisi.org.uk/reference/inspect_ai.event.html#modelevent)
and [`ModelCall`](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#modelcall).

The release gate `make inspect-judge-sdk-test` installs built core and add-in
wheels with the real `inspect` extra against the dedicated hashed dependency
locks, runs `pip check`, and executes real SDK request/event conversion using an
offline HTTP transport. Missing or mismatched SDK dependencies fail that gate.
The separate evaluator-qualification runtime retains its own historical version
and dependency locks.
