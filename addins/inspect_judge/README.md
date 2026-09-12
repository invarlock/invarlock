# Inspect judge collection and import

This optional package collects bounded text judgments through a caller-supplied
Inspect model and imports its expanded-event projection into
`judge-measurements-v1`. It does not import arbitrary Inspect `.eval` archives.

Live collection pins Inspect `0.3.263` and OpenAI `3.13.0`; retained exports from
Inspect `0.3.254` also remain replayable offline. The projection requires one epoch and an
explicit grader, and retains all scheduled slots. Missing attempts remain
incomplete. SDK retries, cache reuse, tools, unrecorded generation settings,
extra request headers and arbitrary request bodies are rejected.

Live collection uses the retained Chat Completions projection. The caller must
construct the Inspect model with Chat Completions selected and both Inspect and
provider-client retries set to zero. The model may carry only the explicit
`responses_api=false` and `max_retries=0` construction arguments needed for
those choices. Other inherited model, provider or generation settings are
rejected before a call is admitted.

For `openai/gpt-5.6-sol`, the pinned SDK converts system messages to developer
messages, uses `max_completion_tokens`, and omits temperature from the provider
request. The approved plan must therefore declare temperature `1`, the provider
default. Collection rejects other temperatures before admission. Offline replay
accepts exactly this version-bound projection for that model; other models keep
their existing message and sampling-control checks. Normalized requests retain
the approved system message and configuration alongside the actual provider call.

`bind_requests` creates exact request digests for a plan before independent
approval. `render_request` uses the core renderer to separate rubric, input and answer
values in JSON fields. Templates remain literal instructions. This separation
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
