# Requalify a hosted service

!!! tip "User guide"
    **Outcome:** Compare a fresh hosted-service capture with an approved baseline,
    publish signed evidence, verify it offline, and record a scoped decision.
    **Audience:** Evaluation engineers and owners of periodic or incident-triggered
    service qualification.
    **Prerequisites:** A capture harness with service access, a reviewed baseline
    and case set, policy fixed before capture, the core wheel, and independent trust inputs.

Use a wheel and example files from the same source revision. For released
packages, use their matching documentation; see
[matching wheels and examples](getting-started.md#matching-wheels-and-examples).

Use this workflow when the evaluated subject is a hosted model or an application
using one. Your harness makes fresh service calls and retains the results;
InvarLock imports those results through `execution.mode: captured`. The core
provides comparison, signed evidence, offline verification and reports. It does
not supply a native hosted runtime, scheduling service or continuous monitoring
guarantee.

The claim concerns observed behavior within the recorded observation window,
configuration, environment and evaluated cases. A service name or revision
reported by an endpoint does not establish immutable model weights. A passing
comparison supports one qualification decision under its stated policy; it does
not establish that the service remained unchanged between campaigns.

## Fix the decision before capture

Choose the relevant boundary: a model endpoint, a deployment with a fixed system
prompt, or a complete application including retrieval and tools. Record the
configuration that can affect that boundary: model selection, decoding, prompts,
retrieval corpus, tool versions, permissions and environment as applicable. Keep
secrets out of evidence. Missing configuration limits what differences can be
attributed to the model or service.

Approve the baseline capture, intended subject, case membership, expected
answers, scoring method and thresholds before inspecting subject outcomes. Pin
the reviewed case set with `expected_case_set_digest` when using a deterministic
comparison policy. Define independent evaluation units, repeat executions,
exclusions, retries, stopping rules, minimum sample requirements and the handling
of failed calls. Declare any absolute subject-quality floor as well as allowable
regression: matching a weak baseline does not establish useful performance.

Preserve the historical baseline capture, original policy, signatures and
receipts. Each campaign writes new files and uses a fresh evidence directory.
Changing a policy or grader creates a new analysis; it does not rewrite the old
result. If the baseline endpoint is unavailable, frozen old outputs remain a
valid declared baseline for an output comparison, but do not represent a fresh
execution of that old service.

For periodic qualification, use an external scheduler to start each bounded
campaign. For an incident-triggered campaign, record the trigger and choose a
reviewed case set relevant to the suspected behavior. Define alert routing,
escalation, repeat-confirmation and multiple-alert policy before repeated looks
at results. Per-campaign intervals are not a joint guarantee across repeated
campaigns, and incidental alerts do not establish an incident detector.

## Capture fresh executions

The [HTTP capture and handoff example](https://github.com/invarlock/invarlock/blob/main/examples/hosted-service/README.md)
provides a portable protocol, bounded collection and separate installed-wheel
replay. Its four synthetic smoke cases exercise integration and do not qualify
a hosted service.

The [retained local HTTP reference](https://github.com/invarlock/invarlock/blob/main/examples/hosted-service/references/mistral-7b-http/README.md)
also provides 400 measured pairs from distinct full 7B checkpoints. It preserves
the original responses, signed comparison and separate offline verification.
Its low literal exact-match scores remain visible despite passing the declared
comparative policy; it does not qualify an external provider or production task.

Run the harness against the approved endpoint and retain complete per-case inputs,
outputs, errors and source facts. Include actual service requests and responses
where needed for attribution, after applying the approved secret-handling rules.
Record the harness source identity and the start and end of the observation
window. Preserve failed and superseded attempts according to the declared rules;
do not silently replace them with successful calls.

The HTTP example enforces request deadlines using elapsed time from a monotonic
clock. UTC timestamps describe observation windows and request ordering; replay
checks those bounds separately because the system clock can change during
capture. Recorded timings are source observations, not independent proof of how
long a remote service executed.

A hosted canonical run uses `artifact_digest: null` and the explicit
`service_identity` described in the [record reference](../reference/evaluation-records.md#hosted-service-identity).
It records the provider, service, deployment, requested model, observed model and
exposed revision when available, configuration digest, harness identity and UTC
observation window. Unavailable observed identities remain `null`. Do not hash a
model alias into an artifact digest or claim that an endpoint label identifies
weights. The complete-run digest binds the service identity together with all
retained records.

Use `capture_evaluator_run` in `invarlock.engine` to normalize the harness's
explicit per-case rows. The scorer needs the following facts:

| Scorer | Required capture | What offline replay establishes |
| --- | --- | --- |
| Exact match | Outputs and independently reviewed reference strings | String correctness and paired policy arithmetic over retained records |
| Normalized NLL | Actual reference-continuation log probabilities, byte/token counts, tokenizer and configuration bindings | Recomputed likelihood comparison from supplied measurements |
| Judge | Frozen task/answer text and bounded retained judge calls under the declared recipe | Admitted ratings, aggregation and policy arithmetic |

NLL is available only if the service exposes the required reference-continuation
measurements. Its policy declares one shared `configuration_digest`; both hosted
service descriptors and all likelihood rows must bind that same configuration.
Different requested models, observations and tokenizer identities may be retained,
but differing service configurations cannot use this NLL comparison profile. Generated-answer token probabilities, summary scores or guessed
likelihoods do not supply them. Use [captured results](captured-results.md) for
scorer selection and explicit input projections.

Repeated ratings of a frozen answer measure grading variability. They are not
fresh service executions and do not increase the number of independent tasks.
To study execution variability, capture the separately declared executions and
retain their identities. Avoid treating executions or ratings of the same task
as independent cases merely to increase the sample size; use an analysis whose
unit assumptions match the campaign.

When judging old and new outputs, freeze both first and grade both within one
bounded campaign using the same rubric, judge configuration, references,
repetition plan and declared independent units. Retain the rendered requests,
responses, attempts and source mappings. Comparing old grades with newly
collected grades confounds service change with grading change. Replay authenticates
the retained rating process and its arithmetic; it does not establish that a
rating is correct. Review rubric suitability and rating validity separately.

## Evaluate an agent's resulting work

For an agent task, define completion through the resulting artifacts and checks
relevant to the task. Preserve the starting state, changes, test definition,
execution settings and observed outcome. A convincing final message does not
establish that files changed correctly or that a check passed. Independent review
must establish that the checks express the intended task and cover its relevant
failure cases.

The [bounded outcome fixture](https://github.com/invarlock/invarlock/blob/main/examples/hosted-service/agent_outcomes.py)
is a synthetic integration example. It writes a small inclusive-range bug and
applies two fixed candidate replacements in separate temporary directories. Both
candidates claim the same successful completion. A fixed `unittest` check actually
runs against each resulting file through isolated Python; the correct replacement
passes and the incorrect replacement fails.

```bash
python examples/hosted-service/agent_outcomes.py --output outcome-fixture
```

The output retains starting files, candidate files, test source and their digests,
the exact command, timeout, bounded stdout/stderr, exit status and derived outcome
in `capture.json`. Canonical `baseline.json` and `subject.json` use the task as
input, `pass` as the expected reference, and the actual `pass`/`fail` outcome as
the exact-match output. The completion message remains context and is not graded.
The [fixture tests](https://github.com/invarlock/invarlock/blob/main/tests/examples/test_agent_outcomes.py)
exercise real test failures, separate directories, changed candidate and test
bindings, timeout handling, output bounds and offline projection without execution.

Use these canonical runs with the captured request below and a policy appropriate
to an integration fixture. One synthetic task does not establish agent quality.
The script executes only its fixed sources; it is not a general agent runner and
accepts no arbitrary code or user-supplied tests for execution. No model or judge
is called. Imported outcome provenance remains a source assertion: authenticating
a capture does not attest that the claimed execution happened.

A task-specific judge may instead consume an explicitly frozen deterministic text
representation of retained files, checks and outcomes under the existing bounded
judge recipe. Declare that evidence projection and rubric, preserve their pins,
and grade both sides in the same campaign. Grading the completion message alone
cannot replace the task checks; offline judge replay does not establish the
correctness of the check design or the rating.

## Evaluate the paired captures

Install the core wheel and use the matching checkout for example helpers, following
the [wheel and example convention](getting-started.md#matching-wheels-and-examples).
The capture environment may contain provider SDKs; the offline recipient needs
only the core package and retained inputs.

Prepare `baseline.json`, `subject.json` and `policy.json` beneath one campaign
directory and create this request there:

```yaml
format_version: invarlock/evaluation-request-v2
execution:
  mode: captured
comparison:
  baseline:
    path: baseline.json
    adapter: invarlock
  subject:
    path: subject.json
    adapter: invarlock
  metric: exact_match
  policy: policy.json
output:
  evidence: evidence
```

This request consumes canonical captures; it does not call the hosted service.
For NLL, select `normalized_nll_per_utf8_byte` and its matching policy. For judge
scoring, use the [captured judge request](captured-results.md#judge-captured-answers)
and its separate recipient policy. Import retained measurements for fully
offline grading replay, or authorize a bounded new grading campaign explicitly.

```bash
invarlock evaluate campaign/request.yaml --signing-key evidence-signer.pem \
  --preflight --json
invarlock evaluate campaign/request.yaml --signing-key evidence-signer.pem --json
```

Preflight checks inputs without inference, judging or publication. Successful
publication exits `0` even if the recorded decision is adverse. Add
`--fail-on-policy` for a CI policy gate; an adverse decision then exits `7` after
publishing evidence. An input or publication failure must not cause reporting of
a stale directory. Inspect the command result and retain diagnostics before
retrying with a new output destination.

## Verify offline using independent expectations

The recipient obtains the approved runs, request, policy and signer authorization
through a channel it controls, derives their canonical pins, and prepares
`trust/trust-inputs.json` using the [signed handoff](captured-results.md#signed-handoff).
The hosted identity is covered by complete-run pins; it is not a native artifact
anchor. Do not derive expected values solely from the submitted evidence pack.

```bash
invarlock verify campaign/evidence/ --trust-profile trust/trust-inputs.json \
  --receipt campaign/verification.receipt.json --json
invarlock report campaign/evidence/ --html campaign/report.html \
  --markdown campaign/report.md --json
```

Run verification in a recipient environment without service credentials or
provider SDKs. Verification makes no fresh service measurement. It checks signed
bytes, complete identities, policy and permitted deterministic replay; it is not
execution attestation. A captured receipt cannot authorize native acceptance.

Independent expectations are useful for ordinary handoff mistakes:

| Mistake | Required independent expectation | Check |
| --- | --- | --- |
| Wrong subject capture | Approved subject identity and complete-run pin | Reject a mismatched run binding |
| Old or unintended policy | Approved current policy bytes and request pin | Reject a mismatched policy or request |
| Altered results or incorrect arithmetic | Authenticated complete records and declared scoring policy | Reject integrity or replay inconsistency |

These checks rely on the expected values being correct. Independence means
separate control of expectations and verification inputs; it does not require
two organizations or two people. A copied expectation can agree perfectly with
the wrong submission. Consult the [verification failure lab](verification-failure-lab.md)
for executable recipient-failure examples under its native evidence contract.

## Read the result and decide what happens next

Keep four technical states separate from the organizational decision:

| State | Meaning | Next action |
| --- | --- | --- |
| Policy satisfied | The retained comparison meets the declared requirements | Review scope, freshness and remaining acceptance criteria |
| Regression or other policy failure | A comparison requirement was not met | Inspect affected cases and apply the declared escalation or remediation rule |
| Insufficient evidence | Missing facts, sample size or uncertainty requirements prevent the requested conclusion | Diagnose the missing requirement; collect a new campaign if appropriate |
| Integrity or verification failure | The submission cannot be authenticated or replayed against approved inputs | Resolve the binding or evidence failure before relying on the comparison |

A report renders evidence; it does not inherit independent verification from the
presence of a signature or an adjacent receipt. Retain the external verifier
result and signed receipt alongside the immutable pack. An authentic rejection
receipt is still a rejection. Inspect its verdict separately from authentication
of the receipt itself.

Record the campaign purpose, observation windows, configuration and environment,
baseline and subject run pins, policy approval, evidence and receipt identities,
technical outcome, known limitations, decision owner and organizational action.
An organization may defer or decline use even after technical policy satisfaction.
Keep the decision record outside the pack, with any remediation or follow-up
campaign linked to the original evidence.

A new execution, a changed policy, or a new judge campaign gets a new record.
Rerendering or reverifying old evidence preserves its historical meaning; it
does not refresh the service observation window. Use the
[report reference](../reference/reports.md) to interpret the selected metric's
interval and assurance scope.
