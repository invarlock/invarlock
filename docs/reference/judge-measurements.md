# Bounded judge measurements

Judge evidence compares repeated ratings of frozen baseline and subject answers
under one declared rubric. The `text-frozen-answer-v1` profile retains accessible
requests, responses, sanitized error outcomes, attempts and source mappings for
offline replay. Failed calls omit private raw error/response details; their
admission and full budget reservation remain recorded.

> **Reference**
>
> **Surface:** Judge request, measurement, analysis and evidence contracts
>
> **Stability:** Additive versioned formats; native metric selection and separately scoped judge evidence
>
> **Use this page when:** Running a native judge scorer, importing ratings or reviewing judge evidence

Use the core wheel and example files from the same source revision. For released packages, use their matching release documentation;
see [matching wheels and examples](../user-guide/getting-started.md#matching-wheels-and-examples).

## Native scorer

Use `comparison.metric: judge` in `invarlock/evaluation-request-v1`, with the
same baseline, subject, dataset and `execution.mode: run` or `import` fields as
exact match and normalized NLL. Add:

```yaml
comparison:
  metric: judge
  policy: judge-policy.json
  judge:
    workspace: judge-work
    signer_identity: evaluation-signer
```

These fields belong inside the complete native request; the fragment does not
replace its baseline, subject, dataset or task. Generate a complete starter with:

```bash
invarlock evaluate --init my-judge --example native-judge
```

The [native starter](https://github.com/invarlock/invarlock/blob/main/examples/native-judge/README.md) contains a request,
local dataset and `invarlock/native-judge-policy-v1` policy. Replace the model and
runtime placeholders, choose your cases, and supply a native runtime profile and
signing key before preflight. Its two illustrative cases deliberately cannot
meet the policy's precision requirement.

The policy has five closed fields: `format`, `plan`, `analysis`, `collection` and
`runner`. The plan contains the existing judge plan's rubric, prompt, model,
parser, scale, sampling and schedule choices. Omit run/case/answer digests,
`rubric.sha256` and `schedule.expected_trials`: InvarLock derives them after native
capture. The analysis contains the judge analysis policy without `plan_sha256`,
which is also derived. `collection` declares the selected collector's bounds.
For hosted collection, `runner` declares `scorer_id` and
`invocation_timeout_seconds` (1–604800 seconds). For local collection it contains
only `scorer_id`; the model runtime settings own the execution timeout.

The exact original policy bytes are bound into the runtime configuration before
answers are generated. Native collection retains all six original provider files
for each side, the schedule, normalized request, policy and observations in
`native_capture.json`. Offline replay validates that provenance, regenerates the
frozen runs and finalized plan, and recomputes the judge analysis. Signed
bindings include `native_capture_sha256`; native capture cannot be stripped to
turn the result into ordinary imported-answer evidence.

Preflight validates the rubric and full trial reservation, native artifacts and
resources, and the selected collector without making calls. Native providers
keep their existing network restrictions. Hosted collection additionally checks
its SDK and API environment. Only the explicitly configured hosted collector
calls a supported endpoint. The grader prefix
selects `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` (or
`GEMINI_API_KEY`), or `OPENROUTER_API_KEY`; credentials never enter request
files or evidence.

`judge-work` is private and resumable. If the invocation ends with unattempted
trials, evaluation returns an incomplete result and leaves the final evidence
destination absent. Rerun the same request to continue without regenerating
answers or retrying admitted calls. Changed artifacts, runtime images, policy,
rubric or data require a new workspace. An interrupted native answer capture is
marked as unfinished and cannot silently execute again. If retained storage
capacity is exhausted, evaluation publishes terminal `insufficient_evidence`
with `collection.stop_reason: retained_capacity_exhausted`; it does not promise
that another identical invocation can continue.

The profile supports exactly one text part per scheduled input. It grades the
task input and answer under the declared rubric and optional global references.
By default, the per-case reference remains authenticated evidence but is not
sent to the judge. Set `plan.prompt.reference_mode: per_case` in the recipe to
include each case's string reference in a separate `reference` field of the
judge request. Missing references or references that are not strings are
rejected. This field counts toward the request bounds and changes the
authenticated request digest. It is never added to the evaluated model's input. Omitted or `none` reference mode
preserves existing request bytes.

## Existing evaluator workflows

The three built-in scorers also accept captured evaluator records through
`invarlock/evaluation-request-v2`. Choose `comparison.metric: judge`, the same
recipe, and a private `comparison.judge.workspace` with an explicit
`comparison.judge.signer_identity`. The
[captured-results guide](../user-guide/captured-results.md#judge-captured-answers)
shows source mapping, explicit text projection and offline measurement import.

For frozen answers from an existing evaluator, select the pinned Inspect SDK
collector for supported hosted providers, `openai-compatible-judge` for an
explicitly configured Chat Completions service, or `runtime-provider-judge` for
an authenticated local HF or GGUF artifact. Neither the service nor the local
route requires the upstream evaluator to use Inspect or change versions.
Existing evaluators supply frozen case facts; InvarLock applies its own scorer.
Core import, verification and reporting work without Inspect. A caller-owned
collector can use `invarlock.engine.prepare_evaluator_judge` and
`import_judge_sources` to supply complete retained calls under the same contract.
Neither route converts unrelated upstream scalar scores into replayable trials.

For v2, omit `comparison.judge.measurements` to collect new ratings, or set it
to a request-relative measurements file for offline import. Import requires
neither the optional SDKs nor credentials. Preflight and evaluation both
require a signing key or explicit `--unsigned`. Sources may be canonical frozen
runs or supported evaluator exports; an explicit `input_projection` selects one
string through a JSON pointer rooted at `/input` or `/context`. Structured
messages are never implicitly joined or converted to text. Projection settings,
the original input and context, and their hashes remain in the signed records;
offline replay checks them before pairing or judging.

The public Python preparation and import interfaces are:

```python
from invarlock.engine import import_judge_sources, prepare_evaluator_judge

# recipe, baseline and subject are already loaded, reviewed JSON objects.
plan, analysis_policy = prepare_evaluator_judge(recipe, baseline, subject)
measurements = import_judge_sources(
    {"source-1": retained_source_bytes},
    plan=plan,
    baseline_run=baseline,
    subject_run=subject,
)
```

`retained_source_bytes` is an unchanged UTF-8 JSON shard with exactly `format`
and `trials`, using `invarlock/retained-judge-json-v1`. Trial records must bind the
approved requests, answers, attempts, responses or errors, and source positions.
The importer validates the plan before reading shards and replays every retained
trial; incomplete slots remain incomplete. It accepts 1–1,000 shards of at most
16 MiB each within the 384 MiB measurement allowance. A source hash authenticates
retained bytes, not execution by the named evaluator or hosted provider.

## Frozen-answer requests and preflight

The self-contained example at `examples/judge-measurements/README.md` uses
`invarlock/evaluation-request-v3` with this closed request:

```yaml
format_version: invarlock/evaluation-request-v3
execution:
  mode: judge_import
  collection: null
comparison:
  baseline_run: baseline_run.json
  subject_run: subject_run.json
  plan: plan.json
  measurements: measurements.json
  policy: analysis_policy.json
output:
  evidence: evidence
  signer_identity: example-signer
```

Paths are relative to the request root. Traversal, symlinked inputs and output
that contains an input are rejected. Baseline and subject use frozen
`evaluation-run-v1` records. The plan binds both runs, case membership, answers,
rendered requests, judge identity/configuration, rubric, scale and trial schedule.
The analysis policy is a separate `judge-analysis-policy-v1` document.
The judge configuration always declares `reasoning_effort`. Use `null` only
when the selected model has no approved reasoning control; otherwise bind one
of the supported effort names explicitly. This field is part of the plan digest,
every rendered request digest and retained provider-call validation.

```bash
invarlock evaluate request.yaml --preflight --json
invarlock evaluate request.yaml --unsigned --fail-on-policy --json
```

Preflight makes no provider calls and publishes nothing. It displays cases,
independent units, expected trials, maximum attempts, coverage, judge settings,
policy thresholds and missing files. Evaluation requires either explicit
`--unsigned` or `--signing-key signer-private.pem`. Signed publication does not
establish independent recipient acceptance. `output.signer_identity` names the
identity placed in a signed envelope; a recipient still has to pin that identity
and its public-key fingerprint independently. Unsigned output retains no signer.

Hosted `judge_collect` mode instead declares
`execution.collection: {integration: inspect-judge, configuration: collector.json, workspace: judge-work}`
and `comparison.measurements: null`. The configuration is closed and contains no
credentials or provider URL:

```json
{
  "grader": "openai/example-judge",
  "inspect_version": "0.3.263",
  "profile": "inspect-text-frozen-answer-v1",
  "epochs": 1,
  "log_model_api": true,
  "log_samples": true,
  "sdk_max_retries": 0,
  "tools": false,
  "concurrency": 4,
  "requests_per_minute": 120,
  "request_timeout_seconds": 120,
  "max_calls": 2000,
  "max_input_tokens": 4000000,
  "max_output_tokens": 256000,
  "max_cost_microusd": 20000000,
  "input_tokens_per_call": 2000,
  "cost_microusd_per_call": 10000
}
```

Preflight validates the full Inspect profile and displays concurrency, rate,
timeout, call, token and cost reservations. It also computes the maximum calls
admitted by the tightest reservation and whether that covers the full plan. A
valid collection preflight exits successfully and makes no provider call.
The committed `examples/judge-measurements` fixture includes a separate
`request-collect.yaml` and bounded `collection.json` so this route can be
inspected without editing the import example.

The core `invarlock.judge_measurements` module exposes
`collect_configured` for installed execution and a lower-level `collect` API for
callers that construct their own Inspect model. Both use the same admitted-call
checkpoint. Installed live collection requires exactly Inspect `0.3.263`,
OpenAI `3.13.0`, Anthropic `1.6.0`, Google Gen AI `2.24.0`, and
`httpx==0.28.1` and `httpx2==2.12.0`. It supports pinned Inspect integrations for `openai/`,
`anthropic/`, `google/`, and `openrouter/` graders with one attempt per trial,
zero collector retries, no tools or Inspect result cache, and no inherited model
settings.
An admitted call without a retained result is an ambiguous timeout that cannot
be retried. `verify` and `report` make no provider calls.

Python collection callers can set `RunnerOptions.stop_after_batches` to a
positive integer to pause after that many completed batches in the current
invocation. Each batch's admitted calls finish and their results are durably
retained before the runner reports `requested`; no later batch is admitted.
Resume with the same plan, collection options and checkpoint directory, omitting
the stop limit when ready to finish. Earlier calls and budget reservations remain
in force. This execution control does not change plan or checkpoint identity and
is not a request-file field. It does not cancel a provider request already in
flight; the invocation deadline still applies. A complete schedule or exhausted
budget takes precedence over the requested pause.

Use `collect_configured` for Google's per-call clients. It fixes the endpoint to
`https://generativelanguage.googleapis.com`, allows one SDK attempt, disables
automatic function calling, and stops Inspect's internal malformed-function
retry before a second request. Configured direct providers reject alternate
service paths such as `google/vertex/...`; OpenRouter retains its routed model
names. The pinned Google and Anthropic adapters require a null `seed` because
they do not forward it. Anthropic requires `top_p=1`. Without extended
reasoning, its qualified request sends the approved temperature. With supported extended
reasoning, Inspect sends its mapped thinking budget instead of temperature,
and InvarLock checks that request against the approved effort and output limit.
The pinned projection supports nondefault effort for Claude 4.5 and Gemini 2.5
model families when the output limit exceeds the mapped thinking budget.
All live Claude 4.6-and-newer, Gemini 3-and-newer, and unrecognized model names
are rejected during preflight until their pinned SDK request shapes are
qualified, regardless of effort. Historical evidence for those models remains
subject to its original offline verification contract.
The pinned Google adapter sends `BLOCK_NONE` for five provider safety
categories; this fixed SDK setting is checked against the retained request.
The judge result is a quality measurement under the declared rubric, not a
content-safety assessment.
The configured Anthropic client also stops automatic `pause_turn` continuations
before another request. For all four providers, raw response content, model,
finish reason and token usage are checked against the SDK output before the
retained response is normalized.
During collection, the pinned SDK request is checked against the approved plan.
The retained evidence contains its normalized request and response; offline
verification checks those retained facts and the analysis, not the original
provider HTTP bytes.

Each available provider response ID must identify exactly one retained Inspect
call across trials, source segments and resumed collection. Reusing a response
under a new local event ID is rejected. Generic retained-record imports keep
their own declared source identity semantics.

`openai/gpt-5.6-sol` and `openai/gpt-5.6-luna` additionally require an explicit
non-null `reasoning_effort` in their approved plans. The Inspect adapter passes
that exact value to the SDK and requires the retained provider request to match
it. This bounded SDK support does not qualify either model's judging quality.
The [retained K2 pilot](https://github.com/invarlock/invarlock/tree/main/examples/judge-measurements/references/k2-32b-pilot)
binds Sol effort `none` and 480 completed trials. Both 40-unit analyses remain
`insufficient_evidence` because their interval widths exceed the frozen
maximum. The [Luna held-out reference](https://github.com/invarlock/invarlock/tree/main/examples/judge-measurements/references/k2-32b-luna-xhigh-heldout)
retains two executed final plans, 10,260 ratings and their reference-label
comparison. Each archive retains its own plan, source identity and outcome;
none qualifies a different judge model, effort or rubric.
Rubric development and reference-label review are study-design choices, not
additional requirements of the native judge workflow.

Install the core with its live-collection extra and use the installed command:

```bash
python -m pip install "invarlock[judge]"
# Supply the key selected by the grader prefix through your secret manager.
invarlock evaluate judge-request.yaml --preflight --json
INVARLOCK_ALLOW_JUDGE_NETWORK=1 invarlock evaluate judge-request.yaml --signing-key signer-private.pem --json
```

The process environment opt-in grants network access only within the configured
judge lifecycle, including SDK initialization and cleanup. Native capture and
other concurrent tasks retain the default network guard. Without that opt-in or
an already allowed caller policy, collection stops before model construction or
call admission. Use this scoped switch for native judge requests: the global
`INVARLOCK_ALLOW_NETWORK` switch is incompatible with strict native execution.
Preflight, retained-measurement import, verification and reporting remain offline.

The `judge` extra installs pinned Inspect, OpenAI/OpenRouter, Anthropic, Google,
and both HTTP client SDK dependencies directly. Collection, scoring,
retained-measurement import, verification and reporting are implemented in core.
For local source builds, use
`python -m pip install '.[judge]'` from the repository root. Offline import,
verification and reporting do not require the extra.

Review every call, token, cost and time cap first. The collector rejects custom
provider URLs and reads credentials only from its environment. Remove the
selected provider's base-URL variables and alternate-auth controls; their
presence is rejected even when empty. OpenAI calls explicitly select
`service_tier=default` for standard processing. Historical OpenAI requests
without the field replay unchanged without acquiring a standard-tier claim.
Disabled Inspect response caching does not disable provider prompt caching;
cost reservations must cover applicable cache-write charges. The Anthropic
adapter's prompt-cache marker is bound into each recorded provider request.
Missing credentials, missing or mismatched SDK dependencies, and dependency
import failures stop collection preflight. These requirements do not apply to
offline import. The optional `execution.collection.scorer_id` defaults to `judge` and
`invocation_timeout_seconds` defaults to 3600. An omitted workspace defaults to
`<output.evidence>.judge-work`; an explicit workspace is preferable for resuming
after changing only the final output destination. Runtime resource profiles,
installed deterministic scorer execution and bootstrap overrides do not apply to
frozen-answer v3 requests; they retain their normal meaning for native v1 requests.

Cost admission uses the declared per-call reservation. Retained token use and
available SDK cost fields support accounting, but do not verify a provider invoice.

## Native local judge collection

Select `runtime-provider-judge` to grade frozen text with a local
`hf_transformers` checkpoint or `llama_cpp` GGUF artifact. The judge is distinct
from the baseline and subject models whose answers it grades. This profile runs
the judge through InvarLock's authenticated native runtime, with local model
files, a pinned OCI image and network-disabled execution. No hosted SDK,
credential or judge network opt-in is required. A localhost HTTP endpoint is
still a service boundary and does not establish these native artifact bindings.

The judge model uses the existing artifact/runtime request shape. Native v1 and
captured v2 requests supply it as `comparison.judge.model`; frozen-answer v3 requests use
`execution.collection.model` and
`execution.collection.integration: runtime-provider-judge`. Keep the model path
and referenced support files beneath the request root. A captured request cannot
combine a local model with `comparison.judge.measurements` for offline import.
The separate collection configuration contains exactly:

```json
{
  "profile": "runtime-provider-text-frozen-answer-v1",
  "max_calls": 2,
  "max_output_tokens": 256
}
```

The two reservations above cover one case, two sides and one 128-token rating
per side. Both bounds must cover the complete planned schedule. Context size,
seed, batch size and timeout belong to the authenticated model runtime settings.
This profile rejects the hosted `invocation_timeout_seconds` option. The private
request workspace holds a separate admission and retained result for each trial.
Completed trials are reusable for unchanged inputs; an admission without its
result is ambiguous and cannot silently execute again. Retained measurements
are bounded to 384 MiB, including duplicated observations and outputs. Before
each new inference, the collector counts exact retained bytes and reserves a
48 MiB next source plus a 1 MiB envelope. Schedules exceeding the full-schedule
worst-case capacity require durable checkpoints, supplied by the CLI workspace.
This incremental reservation allows larger schedules when actual outputs are
small. If another source cannot fit, collection stops before the next admission,
preserves completed shards, and does not publish incomplete evidence.

The InvarLock process must already be inside the strict offline container for
local collection, with the judge artifact and backend resources mounted. For a
native v1 `mode: run` request with an explicit local judge model, the CLI runs
the baseline, subject and judge sequentially inside that one container. All
three roles must bind the container's declared `INVARLOCK_RUNTIME_IMAGE_DIGEST`;
the CLI does not start nested OCI workers or require an engine socket. Explicit
OCI engine, worker CPU/memory/user or entrypoint controls are rejected for this
inline path instead of being ignored. Other native run requests retain normal
host-orchestrated OCI execution. Normal host execution cannot use the local
profile by supplying a model path alone.
The boundary check confirms container intent plus a kernel-visible container
marker and rejects network, remote-code and third-party-plugin opt-ins. It does
not inspect the surrounding engine configuration or independently identify the
running image. The operator must launch the declared image with external network
isolation, a read-only root and reduced capabilities, and must verify its digest
before launch; the evidence binds that declared digest.
`INVARLOCK_JUDGE_RUNTIME_DEVICE` selects its device, falling back to
`INVARLOCK_RUNTIME_DEVICE`; baseline and subject device overrides do not select
the judge device. A signing key mounted into this container is available to the
trusted runtime process; native v1 run mode does not provide host-separated
signing or an unsigned switch. To keep the signing key on a separate host, use
the frozen-answer collection and existing `judge_import` route, transferring the
unchanged runs, plan, measurements and analysis policy for publication.
See the
[runnable local judge example](https://github.com/invarlock/invarlock/blob/main/examples/native-local-judge/README.md)
for both providers, container setup and generation of complete frozen-answer v3
request files. Its fixture pair is a setup demonstration, not real-model
qualification evidence.

The plan declares `judge.model_identity.kind: local_weights` and binds
`weights_sha256` to `artifact_identity_sha256` of the complete canonical artifact
identity. This binds the provider's model, immutable revision, checkpoint or GGUF
content, and tokenizer identity as applicable. It is not the hash of a display
name, a mutable model directory path, or just one weight shard. Retain the
runtime image and backend identity separately; a model digest cannot substitute
for execution provenance.

The plan's `prompt.runtime_format` selects the exact input sent to the direct
runtime. Omit it or use `canonical-json-v1` to send the complete canonical
normalized judge-request JSON. Set it to `chatml-v1` for a model that expects
ChatML: the declared system, user and assistant messages are rendered with
explicit role delimiters and a final assistant prefix. No tokenizer-selected
chat template is applied. Other chat formats are not supported by this profile.

The selected format is part of the approved plan. Replay reconstructs the exact
rendered bytes and checks their runtime input digest. ChatML rejects embedded
role delimiters in message content before inference. Its system prompt and
rubric must explicitly describe the allowed JSON ratings; the separate API
response-format object is not sent as a ChatML message. Changing formats requires
a new plan and workspace, not reuse of a previous collection checkpoint.

Select an artifact that can follow the declared input and return the plan's
JSON rating format. Generation uses temperature `0`, top-p `1`, an
explicit seed, `reasoning_effort: null`, no tools, no response cache, and one
attempt without retries.
Context and output bounds must fit the complete rendered request. A malformed
rating or interrupted admitted execution remains a retained failed or incomplete
outcome; it does not authorize a new unrecorded attempt.

Retained `retained-runtime-provider-judge-v1` sources include the exact
artifact identity, runtime observation, provider receipt, ordered schedule,
input and output observations. Offline verification checks their mutual
bindings and reconstructs measurements and analysis without loading the model
or contacting a service. Signature authentication and replay do not independently
attest the host or guarantee identical results from a future runtime. Ordinary
judge precision, coverage, independent-unit and recipient trust requirements
still apply; local execution does not turn an insufficient result into a pass.

## OpenAI-compatible judge services

Use `openai-compatible-judge` for a vLLM, Ollama, LM Studio or explicitly selected
compatible service exposing `/v1/chat/completions`. The v3 collection block
references a configuration file and workspace, with an optional `scorer_id`.
It has no local `model` artifact binding and no `invocation_timeout_seconds`.
The native and captured judge recipes use the same collection configuration
with a runner containing only `scorer_id`.

```json
{
  "profile": "openai-compatible-text-frozen-answer-v1",
  "service": "vllm",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "judge-model",
  "authentication": "none",
  "request_timeout_seconds": 300,
  "max_calls": 2,
  "max_input_bytes": 131072,
  "max_output_tokens": 256
}
```

`service` accepts `vllm`, `ollama`, `lm_studio` or `openai_compatible`. The base
URL explicitly selects HTTP or HTTPS and a `/v1` path, without credentials,
query parameters or fragments. Authentication is either `none` or `bearer_env`;
the latter reads only `INVARLOCK_OPENAI_COMPATIBLE_API_KEY`. Keep credentials out
of request files. Bearer authentication requires HTTPS except on an explicit
loopback endpoint. Ambient OpenAI endpoint variables do not redirect this route.
Input-byte limits count aggregate canonical transmitted JSON bodies, excluding
authorization headers; output-token limits reserve the planned calls.
When a service reports token usage, collection and replay reject a completion
count above the approved per-call output limit. If it omits usage, the retained
request proves the requested cap, but the actual token count cannot be checked
independently; response-byte and measurement-size limits still apply.

This collector retains one source shard per call, so the evidence format's
1,000-shard limit allows at most 1,000 planned calls: 500 paired cases with one
rating per side, or fewer cases with repetitions. Preflight rejects a larger
schedule. The 384 MiB measurement limit can bind earlier when responses are
large. Input-byte reservations are enforced while requests are prepared, before
any call is admitted.

The optional `response_format` is `json_object` by default or `json_schema` for
services that require an explicit strict schema. The endpoint example selects
`json_schema` for LM Studio and derives the rating enum from the frozen plan.
This choice is included in `judge.service_identity`, so changing it requires a
new plan and collection workspace.

The plan uses `judge.provider: openai_compatible`, the exact requested model,
an explicit approved response-model list, and
`model_identity: {kind: hosted_api, weights_sha256: null}`. It fixes one attempt,
no retry or response cache, and no reasoning effort. Its `service_identity`
binds the authored service family and the bare SHA-256 digest of the canonical
normalized base URL, including the trailing slash. The service may apply a chat
template; transmitted settings do not prove its internal implementation.

Preflight does not contact the endpoint. Authorize actual collection with the
process-scoped `INVARLOCK_ALLOW_JUDGE_NETWORK=1` switch, including for loopback
HTTP. Retained `retained-openai-compatible-judge-v1` sources bind the exact wire
request, response, endpoint and normalized ratings to the frozen plan. Completed
checkpoints can be reused unchanged; an admitted request without its retained
result is ambiguous and cannot be automatically repeated. Replay, verification
and reporting remain offline and need only core InvarLock.

The authored service family and returned model identifiers or fingerprints are
service observations. They do not establish the underlying weights, backend
binary or host identity. Use `runtime-provider-judge` for authenticated direct
artifact execution. The
[endpoint example](https://github.com/invarlock/invarlock/blob/main/examples/openai-compatible-judge/README.md)
covers preparation, bounded collection, signing and independent verification.
It does not establish real-model qualification for every server version.

### Tested local configurations

Real execution checks used Qwen2.5-7B-Instruct to judge eight frozen answer pairs
from two distinct Mistral 7B models. Each configuration used separate plans for
per-case references and reference-free judging, with 16 ratings per plan.
The counts below describe valid rating responses, not correct judgments.

| Route | Tested configuration | Valid ratings across both plans |
| --- | --- | --- |
| Direct HF | Qwen2.5-7B-Instruct, explicit ChatML | 32/32 |
| Direct GGUF | Qwen2.5-7B-Instruct Q4_K_M, llama.cpp b10015, explicit ChatML | 27/32 |
| Ollama | 0.34.2, JSON-object responses | 32/32 |
| vLLM | 0.30.0, JSON-object responses | 32/32 |
| LM Studio | llmster 0.0.25+1, JSON-schema responses | 32/32 |

The HF checkpoint revision was
`a09a35458c702b33eeacc393d103063234e8bc28`; the GGUF file SHA-256 was
`65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`.
The five invalid GGUF responses contained fenced JSON and were rejected by the
strict rating parser. An earlier direct canonical-JSON plan produced 0/16 valid
ratings. Earlier LM Studio configurations failed because of a missing runtime
library and an unsupported JSON-object response format. Those failures were
retained separately from the corrected configurations.

A separate native run executed baseline, subject and local judge in one offline
container, produced signed evidence and completed both scheduled ratings.
An independently installed recipient replayed the fresh evidence, including
the native signature and verification receipt. These small studies all returned
`insufficient_evidence`; they establish execution and replay for the stated
configurations, not statistical acceptance or general judge accuracy.

For real passing comparisons, the existing
[Luna held-out reference](https://github.com/invarlock/invarlock/blob/main/examples/judge-measurements/references/k2-32b-luna-xhigh-heldout/README.md)
contains 422 grounded-QA cases and 1,288 extraction cases, totaling 10,260 ratings.
Both recorded policies pass independent replay. That reference exercises the
shared analysis and verification workflow with a hosted judge; it does not
qualify a local judge's rating quality. Full local campaign records are retained
separately from the product source and are not distributed with these examples.

## Replay, authentication and acceptance

Evidence contains `plan.json`, `measurements.json`, `baseline_run.json`,
`subject_run.json`, `case_set.json`, `analysis_policy.json`, `analysis_result.json`
and `envelope.json`. Native requests additionally retain `native_capture.json`.
Captured v2 judge evidence retains the normalized evaluator runs and their
projection provenance, without `native_capture.json`. It uses the same judge
recipient policy and receipt contract as frozen-answer v3 evidence; a
deterministic captured-result trust profile cannot authorize it.
The additive signed envelope binds all artifact digests and
the intended subject under `bounded-judge-fixed-benchmark-v1`.

The recipient maintains its own `judge-measurement-recipient-policy-v1` outside
the submitted evidence. It pins the signer identity and public-key fingerprint,
intended subject, exact artifact digests, metric and bounded decision scope.
An embedded public key cannot authorize itself. For a local subject,
`intended_subject` equals its attributed artifact digest. For a hosted subject,
it equals `invarlock.engine.evaluated_subject_digest(subject_run)`, the digest of
the complete service descriptor. Neither identity replaces the complete-run
pin or establishes current hosted behavior.

A complete policy has this shape. The recipient obtains every value through its
own approval process: the signer identity and key fingerprint come from a trusted
channel, while the artifact pins identify the exact plan, frozen runs,
measurements, policy and recomputed result the recipient has chosen to accept.
Run and case-set digests include the `sha256:` prefix; the four canonical judge
object digests are bare lowercase SHA-256 values, as declared by their contracts.

```json
{
  "format": "invarlock/judge-measurement-recipient-policy-v1",
  "decision_scope": "bounded-judge-fixed-benchmark-v1",
  "intended_subject": "sha256:<64 lowercase hexadecimal characters>",
  "required_metric_name": "factual-correctness",
  "trusted_signer": {
    "identity": "release-evidence-signer",
    "public_key_sha256": "sha256:<64 lowercase hexadecimal characters>"
  },
  "bindings": {
    "baseline_run_sha256": "sha256:<64 lowercase hexadecimal characters>",
    "subject_run_sha256": "sha256:<64 lowercase hexadecimal characters>",
    "case_set_sha256": "sha256:<64 lowercase hexadecimal characters>",
    "plan_sha256": "<64 lowercase hexadecimal characters>",
    "measurements_sha256": "<64 lowercase hexadecimal characters>",
    "analysis_policy_sha256": "<64 lowercase hexadecimal characters>",
    "analysis_result_sha256": "<64 lowercase hexadecimal characters>"
  },
  "required_decision": "pass"
}
```

Do not generate this policy by copying the submitted envelope. That would make
the submitted claims authorize themselves. Store the completed policy outside
the evidence directory and review it before verification.

```bash
invarlock verify evidence --trust-profile recipient-policy.json \
  --receipt receipt.json \
  --verifier-signing-key verifier-private.pem \
  --verifier-identity release-verifier \
  --json
invarlock report evidence --html report.html --markdown report.md --junit report.xml --json
```

Verification authenticates the signer against recipient pins and reconstructs
the measurement and statistical result offline. JSON separates `authenticated`,
`replayed`, `verified`, `accepted` and `decision`. Verified evidence can remain
unaccepted because its result is adverse, inconclusive or advisory. Missing trust
inputs exit 2, verification failure exits 4, and verified but unaccepted evidence
exits 7. Evaluation with `--fail-on-policy` likewise exits 7 for a non-pass result.

Verification first creates an unsigned `invarlock/judge-verification-result-v1`
local result. When `--receipt` is present, both verifier options are required and
the command writes a separate Ed25519
`invarlock/judge-measurement-verification-receipt-v1` outside the evidence. Its
judge-specific signature binds the complete validated local result, verifier
identity and key fingerprint, recipient-policy digest and bounded decision scope.
The JSON command field `ok` equals `accepted`; authenticated and replayed adverse,
inconclusive or advisory evidence therefore remains command failure.

Applications can use `verify_signed_judge_verification_receipt` to authenticate a
received receipt from independently sourced verifier and policy anchors, or
`replay_signed_judge_verification_receipt` to authenticate it and require exact
fresh evidence replay. `verify_stored_judge_result` performs the corresponding
fresh check for an unsigned stored local result. Receipts, stored results,
recipient policies and verifier keys used with evidence replay must remain
outside submitted evidence. A valid receipt authenticates the recorded bounded
result; it does not establish benchmark representativeness, general model quality
or population-wide safety.

Reports replay retained artifacts but do not perform recipient authorization.
JSON, HTML and Markdown distinguish replay, signature presence and acceptance.
They show baseline/subject, requested and observed judge models, rubric, scale,
prompt summary, signer identity, artifact digests, coverage, interval and
threshold. Detail views expose at most 50 cases with 2,000-character text
excerpts; evidence retains the complete bounded records. Repeat
`report --case-id ID` to inspect any retained cases in HTML or with `--explain`;
selection changes presentation only and the complete measurement set is still
replayed.
JUnit records regression as failure and insufficient evidence as error. The
current request carries one metric; it uses the shared metric presentation.

The primary gate table shows the measured interval endpoints and required
directional threshold, along with completeness, independent-unit count and
precision requirements. It distinguishes evidence of a violation from bounds
that remain inconclusive. Method and family-confidence information apply to the
declared fixed benchmark; required and advisory roles remain explicit.

## Statistical scope

The shipped method is `fixed-benchmark-hoeffding-v1`. It uses a conservative,
distribution-free Hoeffding bound for independent bounded units; those units need
not be identically distributed. This simple guarantee is the reason this profile
ships Hoeffding rather than empirical Bernstein. Repetitions and cases within an
independent unit are averaged before inference and do not increase the sample
size. Baseline and subject remain paired within each unit.

The declared `comparison_family_size` applies a Bonferroni adjustment to the
family error budget `alpha`. It is at least two for the subject and paired-effect
intervals and must cover all enclosing claims; combining metrics requires the
same family alpha and sufficient family size for every published interval.
Direction, allowed degradation, optional subject bound, minimum units and maximum
interval width are declared before the decision. Missing or failed scheduled
trials cannot be silently discarded or replaced with extra successful samples.

Policy and other non-integer wire numbers are canonical decimal strings with at
most 15 fractional digits, no exponent and no unnecessary trailing zeros. Integer
counts remain integers. Replay uses the declared exact values, independent of
ambient floating-point settings; report plot geometry does not drive decisions.

A curated fixed benchmark supports a conclusion only for its declared benchmark
and independence assumptions. It does not establish traffic representativeness,
provider-hidden reasoning, general model quality or population-wide safety.

## Operational bounds and measured reference workload

The judge contracts use separate limits because a case, trial, attempt, request
and retained source are different objects. Repetitions multiply trials without
increasing the number of independent cases.

| Object | Contract ceiling |
| --- | ---: |
| Cases / independent-unit mappings | 10,000 |
| Repetitions per side | 10 |
| Scheduled trials | 200,000 |
| Attempts per imported trial | 3 |
| Live collector attempts per trial | 1 |
| Normalized request or retained response | 1 MiB |
| One retained expanded model event | 2 MiB |
| One retained source shard / source shards | 16 MiB / 1,000 |
| Canonical plan / measurements | 64 MiB / 384 MiB |
| Captured judge recipe / each captured run | 4 MiB / 128 MiB |
| Combined judge workflow inputs | 384 MiB |

These are safety ceilings, not recommended workload sizes or evidence-quality
targets. The 384 MiB aggregate limit normally binds before every field can reach
its individual maximum. Collection also reserves possible retained growth before
each active call, so large records can reduce effective concurrency or stop a
campaign before its call cap.

The K2 extraction final plan is the largest single planned reference pack: 1,288
cases, three repetitions per side and 7,728 calls. Its rendered grading requests
have a 3,628-byte median, 4,385-byte 95th percentile and 4,661-byte maximum. The
separate grounded-QA pack adds 2,532 calls, for 10,260 hosted calls across two
independently verified workflows. The maintained capacity test exercises 7,728
completed short trials through construction, source sharding and offline replay.

As an engineering reference, the 7,728-trial construction, sharding and offline
replay test completed in 8.6 seconds with a 468 MiB process RSS high-water mark
under CPython 3.12.13 on Darwin arm64. This one-machine observation is not a
latency or memory service level. It includes fixture construction in the same
process and excludes provider calls, network limits, checkpoint synchronization,
signing and report generation. Run the maintained capacity test on the intended
recipient host before selecting an operational allowance.

The retained 480-trial pilot completed in 265.454 seconds, with
260,271 input tokens, 3,868 output tokens and no incomplete trials. This one-run
observation used a 128-token output cap and does not predict performance for
another plan or host.
The final K2 plan caps output at 256 tokens, while the retained response contract
allows up to 1 MiB. Measure response-size tails, provider throughput, errors and
checkpoint overhead on the intended workload before estimating final collection
duration. Preflight call/token/cost ceilings remain admission bounds, not a
runtime forecast.
