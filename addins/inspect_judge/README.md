# Inspect judge collection and import

This optional package collects bounded text judgments through a caller-supplied
Inspect model and imports its expanded-event projection into
`judge-measurements-v1`. It does not import arbitrary Inspect `.eval` archives.

The supported projection pins Inspect `0.3.254`, requires one epoch and an
explicit grader, and retains all scheduled slots. Missing attempts remain
incomplete. SDK retries, cache reuse, tools, unrecorded generation settings,
extra request headers and arbitrary request bodies are rejected.

`bind_requests` creates exact request digests for a plan before independent
approval. `render_request` uses the core renderer to separate rubric, input and answer
values in JSON fields. Templates remain literal instructions. This separation
does not prove resistance to prompt injection. The original answers remain
digest-bound. Import and checkpoint replay require both frozen
`evaluation-run-v1` objects and validate their approved digests.

`prepare_collection` returns a deterministic bounded next batch and reservations
for calls, tokens and cost. `collect` executes those batches through an explicit
Inspect model, applies a wall deadline and request pacing, and writes each retained
attempt as an immutable checkpoint shard before advancing. A validated checkpoint
preserves completed responses, parse failures, refusals and ambiguous timeouts.
Only a declared transport failure can produce another attempt.

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
objects. The retained `retained-inspect-model-events-v1` source keeps those
bounded objects while the deterministic rating parser reads the accessible model
completion. Offline verification replays the event-to-attempt mapping, input,
generation configuration, usage, completion and normalized trial table without
loading Inspect. As with any retained API log, these bytes establish what the
producer signed and retained; they do not independently prove that a provider
performed the call.

The pinned SDK interfaces used for configuration and event-field inspection are
[`GenerateConfig`](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#generateconfig),
[`ModelEvent`](https://inspect.aisi.org.uk/reference/inspect_ai.event.html#modelevent)
and [`ModelCall`](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#modelcall).
