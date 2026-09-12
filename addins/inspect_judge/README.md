# Inspect judge collection and import

This optional package collects bounded text judgments through a caller-supplied
Inspect model and imports its expanded-event projection into
`judge-measurements-v1`. It does not import arbitrary Inspect `.eval` archives.

The supported projection pins Inspect `0.3.254`, requires one epoch and an
explicit grader, and retains all scheduled slots. Missing attempts remain
incomplete. SDK retries, cache reuse, tools, unrecorded generation settings,
extra request headers and arbitrary request bodies are rejected.

Live collection uses the retained Chat Completions projection. The caller must
construct the Inspect model with Chat Completions selected and both Inspect and
provider-client retries set to zero. The model may carry only the explicit
`responses_api=false` and `max_retries=0` construction arguments needed for
those choices. Other inherited model, provider or generation settings are
rejected before a call is admitted.

`bind_requests` creates exact request digests for a plan before independent
approval. `render_request` uses the core renderer to separate rubric, input and answer
values in JSON fields. Templates remain literal instructions. This separation
does not prove resistance to prompt injection. The original answers remain
digest-bound. Import and checkpoint replay require both frozen
`evaluation-run-v1` objects and validate their approved digests.

`prepare_collection` returns a deterministic bounded next batch and reservations
for calls, tokens and cost. `collect` executes those batches through an explicit
Inspect model, applies a per-invocation deadline and request pacing, and takes an
exclusive one-writer lock on the checkpoint. It writes an immutable admission
shard before dispatching each provider call and a separate immutable result shard
after validating the retained event. An admission without a result is an
ambiguous timeout that consumes the reserved call and cannot be retried. A
validated checkpoint preserves completed responses, parse failures, refusals
and ambiguous timeouts. The live collector currently requires one attempt per
trial; imported evidence may retain explicitly declared transport retries.

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
