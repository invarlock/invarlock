# Inspect judge preparation

This optional package prepares bounded text judging requests and imports an
explicit expanded-event projection into `judge-measurements-v1`. It does not
execute models or import arbitrary Inspect `.eval` archives.

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
for calls, tokens and cost. A validated checkpoint preserves completed responses,
parse failures, refusals and ambiguous timeouts. Only a declared transport failure
can produce another attempt. This function does not schedule calls, track calls
in flight, persist checkpoints, enforce billing or sleep for rate limits.

`prepare_inspect_config` optionally constructs the pinned SDK's generation
configuration. Install the package's `inspect` extra to use that helper. Ordinary
import and planning do not import Inspect or a provider SDK.

## Export boundary

The `invarlock/inspect-judge-export-v1` projection has a closed envelope containing
collection options and scheduled samples. Samples bind case, side, repetition,
plan and answer digests. Each expanded model event includes an explicit grader,
generation settings, request, accessible rating response, model identity and
attempt outcome. The synthetic fixture in `tests/fixtures/export.json` describes
this projection and makes no external execution claim.

Inspect's native `ModelCall` contains provider-specific request and response
objects. The current core parser expects accessible JSON with exactly one
`rating` field. Consequently this projection does not preserve or replay every
provider-native wrapper. A native provider trace must not be relabelled as this
projection by dropping material. Full provider response retention and an actual
Inspect collection/export qualification are required before claiming a maintained
live collector or an independently replayable native Inspect import.

The current output retains canonical normalized trials and authenticates their
source mappings. It does not preserve the original projection bytes or establish
that a model was called. Independent verification remains limited to the retained
measurement contract.

The pinned SDK interfaces used for configuration and event-field inspection are
[`GenerateConfig`](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#generateconfig),
[`ModelEvent`](https://inspect.aisi.org.uk/reference/inspect_ai.event.html#modelevent)
and [`ModelCall`](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#modelcall).
