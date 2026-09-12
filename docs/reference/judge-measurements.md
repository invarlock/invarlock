# Bounded judge measurements

Judge evidence compares repeated ratings of frozen baseline and subject answers
under one declared rubric. The `text-frozen-answer-v1` profile retains accessible
requests, responses, errors, attempts and source mappings for offline replay.

!!! info "Reference"

    - **Surface:** Judge request, measurement, analysis and evidence contracts
    - **Stability:** Additive versioned formats; existing native and captured formats remain unchanged
    - **Use this page when:** Importing frozen-answer ratings or reviewing bounded judge evidence

## Request and preflight

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

The reserved `judge_collect` mode instead declares
`execution.collection: {integration: inspect-judge, configuration: collector.json}`
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

The optional `invarlock-inspect-judge` package exposes an asynchronous `collect`
API for a trusted host that explicitly constructs the Inspect model. Live
collection currently supports Inspect 0.3.263 Chat Completions with one attempt
per trial, zero Inspect and provider-client retries, no tools or cache, and no
inherited model settings beyond `responses_api=false` and `max_retries=0`.
Collection uses an exclusive checkpoint lock, writes a durable admission before
each call and treats an admitted call without a retained result as an ambiguous
timeout that cannot be retried.

The maintained `examples/judge-measurements/collect.py` runner supplies the
complete supported wiring. It loads the plan, collection options and frozen runs,
constructs the explicit pinned Inspect model, uses a private resumable checkpoint
and writes a new measurement file:

```bash
# From the same source checkout as this documentation:
python -m pip install .
python -m pip install 'addins/inspect_judge[inspect]'
export OPENAI_API_KEY=your-key-from-a-secret-store
python collect.py --execute-collection
```

Until a release containing this workflow is published, install both packages
from the same checkout as shown above. The release gate builds and installs the
matching core and add-in wheels in a clean environment before running the pinned
SDK checks.

Run it only after reviewing preflight and the declared call, token, cost and time
caps. It rejects custom provider URLs. The collector never reads a key from a
request file or writes one to retained evidence.

The core CLI does not construct providers or accept credentials. Running
`evaluate` without `--preflight` for `judge_collect` therefore names the optional
API as the next action and exits 2. After collection, use `judge_import` to replay
and publish the retained measurements.
Runtime resource profiles, installed-scorer execution and bootstrap overrides do
not apply to this request format.

## Replay, authentication and acceptance

Evidence contains `plan.json`, `measurements.json`, `baseline_run.json`,
`subject_run.json`, `case_set.json`, `analysis_policy.json`, `analysis_result.json`
and `envelope.json`. The additive signed envelope binds all artifact digests and
the intended subject under `bounded-judge-fixed-benchmark-v1`.

The recipient maintains its own `judge-measurement-recipient-policy-v1` outside
the submitted evidence. It pins the signer identity and public-key fingerprint,
intended subject, exact artifact digests, metric and bounded decision scope.
An embedded public key cannot authorize itself.

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
the producer's claims authorize themselves. Store the completed policy outside
the evidence directory and review it before verification.

```bash
invarlock verify evidence --trust-profile recipient-policy.json --receipt receipt.json --json
invarlock report evidence --html report.html --markdown report.md --junit report.xml --json
```

Verification authenticates the signer against recipient pins and reconstructs
the measurement and statistical result offline. JSON separates `authenticated`,
`replayed`, `verified`, `accepted` and `decision`. Verified evidence can remain
unaccepted because its result is adverse, inconclusive or advisory. Missing trust
inputs exit 2, verification failure exits 4, and verified but unaccepted evidence
exits 7. Evaluation with `--fail-on-policy` likewise exits 7 for a non-pass result.

The locally recomputed judge receipt is not a signed native receipt. Reusing one
requires replay through `verify_stored_judge_receipt`; possession of a receipt
alone carries no authority. Native receipt-signing overrides are rejected.

Reports replay retained artifacts but do not perform recipient authorization.
JSON, HTML and Markdown distinguish replay, signature presence and acceptance.
They show baseline/subject, requested and observed judge models, rubric, scale,
prompt summary, signer identity, artifact digests, coverage, interval and
threshold. Detail views expose at most 50 cases with 2,000-character text
excerpts; evidence retains the complete bounded records.
JUnit records regression as failure and insufficient evidence as error. The
current request carries one metric; it uses the shared metric presentation.

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
