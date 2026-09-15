# Bounded judge measurements

Judge evidence compares repeated ratings of frozen baseline and subject answers
under one declared rubric. The `text-frozen-answer-v1` profile retains accessible
requests, responses, sanitized error outcomes, attempts and source mappings for
offline replay. Failed calls omit private raw error/response details; their
admission and full budget reservation remain recorded.

!!! info "Reference"

    - **Surface:** Judge request, measurement, analysis and evidence contracts
    - **Stability:** Additive versioned formats; native metric selection and separately scoped judge evidence
    - **Use this page when:** Running a native judge scorer, importing ratings or reviewing judge evidence

Use the core wheel, optional collection package and example files from the same
source revision. For released packages, use their matching release documentation;
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
which is also derived. `collection` declares the budgets below;
`runner` declares `scorer_id` and `invocation_timeout_seconds` (1–604800 seconds).

The exact original policy bytes are bound into the runtime configuration before
answers are generated. Native collection retains all six original provider files
for each side, the schedule, normalized request, policy and observations in
`native_capture.json`. Offline replay validates that provenance, regenerates the
frozen runs and finalized plan, and recomputes the judge analysis. Signed
bindings include `native_capture_sha256`; native capture cannot be stripped to
turn the result into ordinary imported-answer evidence.

Preflight validates the rubric and full trial reservation, native artifacts and
resources, installed collector and API environment without making calls. Native
providers keep their existing network restrictions. Only the explicitly
configured judge collector calls the supported hosted endpoint; credentials stay
in `OPENAI_API_KEY`, never in request files or evidence.

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

The optional pinned Inspect package is the maintained live collection backend.
It does not require an upstream evaluator to use Inspect or change versions.
Existing evaluators supply frozen case facts; InvarLock applies its own scorer.
Core import, verification and reporting work without Inspect. A caller-owned
collector can use `invarlock.engine.prepare_evaluator_judge` and
`import_judge_sources` to supply complete retained calls under the same contract.
Neither route converts unrelated upstream scalar scores into replayable trials.

For v2, omit `comparison.judge.measurements` to collect new ratings, or set it
to a request-relative measurements file for offline import. Import requires
neither the optional package nor credentials. Preflight and evaluation both
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

The installed `judge_collect` mode instead declares
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

The optional `invarlock-inspect-judge[inspect]` package exposes
`collect_configured` for installed execution and a lower-level `collect` API for
callers that construct their own Inspect model. Both use the same admitted-call
checkpoint. Installed live collection requires exactly Inspect `0.3.263`,
OpenAI `3.13.0` and `httpx==0.28.1`. It supports the pinned Inspect Chat
Completions integration with one attempt per trial, zero Inspect and
provider-client retries, no tools or cache, and no inherited model settings.
An admitted call without a retained result is an ambiguous timeout that cannot
be retried. `verify` and `report` make no provider calls.

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

Install matching packages and use the installed command:

```bash
python -m pip install .
python -m pip install 'addins/inspect_judge[inspect]'
# Supply OPENAI_API_KEY through your secret manager.
invarlock evaluate judge-request.yaml --preflight --json
invarlock evaluate judge-request.yaml --signing-key signer-private.pem --json
```

Review every call, token, cost and time cap first. The collector rejects custom
provider URLs and reads credentials only from its environment. Remove both
`OPENAI_BASE_URL` and `OPENAI_API_BASE`; their presence is rejected even when
empty. Remove `OPENAI_SAFETY_IDENTIFIER` too; inherited identifier controls are
unsupported and rejected before collection. Configured calls explicitly select
`service_tier=default` for standard processing, and completed retained responses
must report that tier. Historical requests without the field replay unchanged
without acquiring a standard-tier claim. Disabled Inspect response caching does
not disable provider prompt caching; cost reservations must cover applicable
cache-write charges. Missing credentials, missing or mismatched SDK dependencies, and dependency
import failures stop collection preflight. These requirements do not apply to
offline import. The optional `execution.collection.scorer_id` defaults to `judge` and
`invocation_timeout_seconds` defaults to 3600. An omitted workspace defaults to
`<output.evidence>.judge-work`; an explicit workspace is preferable for resuming
after changing only the final output destination. Runtime resource profiles,
installed deterministic scorer execution and bootstrap overrides do not apply to
frozen-answer v3 requests; they retain their normal meaning for native v1 requests.

Cost admission uses the declared per-call reservation. Retained token use and
available SDK cost fields support accounting, but do not verify a provider invoice.

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
