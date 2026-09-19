<div class="invarlock-hero" markdown>

<img
  class="invarlock-hero__mark"
  src="assets/invarlock-logo-dark.svg"
  alt="InvarLock"
/>

<p class="invarlock-hero__kicker">Model evaluation and independent verification</p>

# Evaluate model changes. Verify the evidence

<p class="invarlock-hero__lead">
Compare a candidate with a baseline under your chosen tests and policy.
Run a supported comparison or use records from an existing evaluator, then
produce signed evidence that a customer or internal reviewer can check offline.
</p>

</div>

## Start with your task

| You want to… | Start here | What you need |
| --- | --- | --- |
| See a verified result and HTML report | [CPU quickstart](https://github.com/invarlock/invarlock/tree/main/examples/quickstart) | Python 3.12+ and matching package/example files; no model or API key |
| Compare existing evaluator outputs | [Captured results](user-guide/captured-results.md) | Paired per-case observations, source identities and a comparison policy |
| Execute a model comparison | [Getting started](user-guide/getting-started.md) | Pinned local artifacts, evaluation data and an authorized runtime image |
| Judge baseline and subject answers | [Judge measurements](reference/judge-measurements.md) | Frozen answers or native capture inputs, a rubric, declared units and retained calls or bounded collection |
| Recheck a hosted service | [Hosted-service requalification](user-guide/hosted-service-requalification.md) | A harness that captures fresh service executions and their configuration |
| Verify a delivered result | [Evidence and verification](user-guide/evidence-and-verification.md) | The evidence and independently obtained trust inputs for its workflow |

Use the package, documentation and examples from the same source revision or
release. The [installation convention](user-guide/getting-started.md#matching-wheels-and-examples)
explains how to match them. Start with `invarlock --help` for installed commands.

## Evaluate, verify, report

The three commands share a workflow; the selected request determines which
evidence and verification contract applies. For a captured deterministic
comparison, prepare the request, keys and recipient-owned trust profile, then run:

```bash
invarlock evaluate request.yaml --signing-key signing-key.pem --preflight --json
invarlock evaluate request.yaml --signing-key signing-key.pem
invarlock verify evidence/ --trust-profile trust/trust-inputs.json --receipt verification.receipt.json
invarlock report evidence/ --html report.html
```

Use the evidence destination from your request. Native runs additionally need
runtime resources; judge receipts need the judge verifier key and identity.
The linked task guides provide those complete family-specific examples,
including trust-profile and signing-key preparation.

<div class="invarlock-transaction" markdown>

<div class="invarlock-transaction__step" markdown>

<span class="invarlock-transaction__number">Transaction 01</span>

### `evaluate`

Validate the request and its inputs, run or import the selected comparison,
apply the declared policy and publish canonical evidence. Preflight checks setup without
model or provider execution. Supported captured and frozen-answer workflows
also allow explicit unsigned local evaluation.

</div>

<div class="invarlock-transaction__step" markdown>

<span class="invarlock-transaction__number">Transaction 02</span>

### `verify`

Check the submitted package against independent expectations for the policy,
identities and signer, then reconstruct the supported analysis. Verification
can issue a separate signed receipt. It makes no model or judge calls.

</div>

<div class="invarlock-transaction__step" markdown>

<span class="invarlock-transaction__number">Transaction 03</span>

### `report`

Explain the comparison identities, measured changes, uncertainty and every
configured check. HTML, terminal and Markdown presentations share the result;
JSON and JUnit support automation where available. Directory-pack reporting
checks embedded signatures when signed. Judge reporting replays retained
measurements but leaves signature authentication to `verify`. Rendering does
not replace recipient verification. Explicit unsigned captured reports retain
their local, unauthenticated status.

</div>

</div>

Commands default to readable text; use `--json` for machine-readable status.
Publication, recorded policy outcome and independent verification are separate
results. Use the [CLI's exit-code contract](reference/cli.md) for automation,
rather than interpreting a rendered report's successful exit as policy approval.

## Select a scorer and its evidence

| Scorer | Required facts | What the result measures |
| --- | --- | --- |
| `exact_match` | Paired outputs and independent references | Change in literal-answer accuracy |
| `normalized_nll_per_utf8_byte` | Bound reference-continuation likelihoods, byte/token counts and tokenizer identities | Ratio of mean byte-normalized NLL |
| `judge` | Frozen task text and answers, rubric, units, repetitions and retained calls | Change in bounded ratings under the declared judge profile |

The native and captured entry points expose these three scorer choices.
Native judging first retains runtime-bound answers; captured judging binds
supplied answer records. Live collection uses the optional Inspect judge
package with explicit budgets. Offline import and replay need no provider SDK
or credentials. Per-case references are an explicit judge-profile choice and
remain separate from the evaluated model input.

Captured requests without an explicit `comparison.metric` can combine policy
metrics and slices under the captured bootstrap contract. Selecting a built-in
scorer dispatches to that scorer's evidence and statistical treatment. Native
exact match uses its paired Newcombe interval, normalized NLL uses paired
schedule resampling, and bounded judging uses its declared independent-unit
analysis. These methods do not share an interchangeable confidence claim.

Deterministic extensions cover normalized labels, numeric tolerances,
structured fields and token overlap. [Evidence sets](reference/evidence-sets.md)
combine independently verified components over the same frozen answers without
claiming a joint confidence bound. Consult [schedule and policy](user-guide/schedule-and-policy.md),
[captured records](reference/evaluation-records.md) and
[judge measurements](reference/judge-measurements.md) for exact requirements.

## Run here or integrate your workflow

**Native execution** prepares an ordered schedule from pinned local data and
runs baseline and subject in authorized Docker or Podman images. Workers receive
scoped artifact and support mounts; the evidence-signing key stays on the host.
A [runtime profile](reference/cli.md#reusable-runtime-profiles) supplies reusable
execution settings without choosing policy or signer trust. Authenticated
provider import uses complete existing sidecars instead of rerunning answers.

![Native comparisons, captured records and frozen answers feed evaluation, followed by independent verification and reporting](assets/evaluation-verification-flow.svg)

**Captured evaluation** accepts supported Inspect AI, Harness, Promptfoo and Langfuse
export profiles, canonical records or data prepared through the
[Python API](reference/api-guide.md). Original per-case observations and their
identities are required; aggregate scores cannot fill gaps. The
[qualification matrix](reference/evaluator-qualification.md) distinguishes
adapter support, replay authority and retained runtime demonstrations.

**Hosted-service requalification** uses fresh runs collected by your harness.
Service identity records customer-controlled configuration and observation
windows without asserting hidden model weights. Scheduling and service calls
belong to the collection workflow; verifying a historical package does not
measure the service again.

Hugging Face Transformers is the built-in runtime provider. Optional GGUF,
TensorRT-LLM and vision-text providers support their declared collection
profiles. Provider collection capabilities and the scorer applied to those
observations are different interfaces; see [runtime providers](user-guide/runtime-providers.md).
Optional [diagnostics](user-guide/diagnostics.md) are observation-only and do
not determine policy acceptance.

## Understand the result

A signed pack records the supported comparison and its policy result. A separate
recipient supplies its expected identities, policy and signer trust; a signed verification
receipt records that recipient's check. Imported observations and native
runtime bindings retain distinct provenance claims.

The result is bounded by the supplied cases, observations and assumptions.
Verification does not independently rerun model execution, establish production
representativeness or authorize deployment. A valid signature alone cannot
supply an independent trust decision, and a comparative pass can coexist with
poor absolute task performance. Read the [assurance case](assurance/assurance-case.md)
and [trust model](security/trust-model.md) for the complete boundaries.

## Find the detailed contract

| Documentation | Use it to… |
| --- | --- |
| [User guides](user-guide/getting-started.md) | Complete a task, validate its output and recover from errors |
| [Assurance notes](assurance/assurance-case.md) | Understand claims, statistical meaning, assumptions and limits |
| [Reference](reference/cli.md) | Look up exact fields, defaults, outputs and failure behavior |
| [Security guidance](security/best-practices.md) | Configure trust, keys, isolation and evidence handling |
| [Runnable examples](https://github.com/invarlock/invarlock/tree/main/examples) | Exercise a specific supported integration |
| [Public evidence](user-guide/public-evidence.md) | Inspect and publish scoped retained comparisons |
| [Contributing](https://github.com/invarlock/invarlock/blob/main/CONTRIBUTING.md) | Run development checks and prepare a pull request |

InvarLock is pre-1.0. Artifact formats carry explicit versions; the Python
embedding facade may evolve between minor releases. The
[compatibility covenant](reference/compatibility.md) preserves historical v0.13
evidence and receipts while leaving acceptance to current recipient policy.
