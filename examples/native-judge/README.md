# Native judge starter

This starter selects `metric: judge` in the same native request used for exact
match and normalized NLL. It generates baseline and subject answers through the
native runtime, freezes the complete runtime evidence, collects bounded judge
ratings, and publishes evidence for offline `verify` and `report`.

The two invented questions demonstrate wiring. They cannot establish benchmark
quality or satisfy the interval precision in the example policy. Replace them
with your own reviewed regression cases and independent-unit assignments before
using the result for a decision. Context needed to judge an answer belongs in
`prompt`. By default, the separate `expected` reference remains authenticated
but is not sent to the judge. To grade against it, set
`plan.prompt.reference_mode: per_case` in `judge-policy.json`. Every case must
then have a string reference, which is sent as a separate `reference` field and
counts toward request limits. It is never added to the evaluated model input.
Omitting the mode or setting it to `none` preserves the original request bytes.

## Prepare

Install matching core and collector packages from the repository root:

```bash
python -m pip install .
python -m pip install 'addins/inspect_judge[inspect]'
```

Installed collection requires exactly Inspect `0.3.263`, OpenAI `3.13.0` and
`httpx==0.28.1`, supplied by the extra. It uses the official OpenAI Chat Completions
endpoint. Remove `OPENAI_BASE_URL` and `OPENAI_API_BASE` from the environment;
even empty overrides are rejected. The starter explicitly selects temperature
`1` and reasoning effort `none` for `openai/gpt-5.6-sol`. Keep those controls
consistent with the approved plan and collection configuration.

Copy this directory to a private workspace. Replace the baseline and subject
artifact identities, immutable revisions, checkpoint and tokenizer digests in
`request.yaml` with your actual models. Supply their artifact files and native
runtime resources as described in the native evaluation guide. The placeholder
digests deliberately prevent this example from claiming real model execution.

Review `judge-policy.json`: the rubric, judge model and configuration, rating
scale, independent units, repetition count, decision thresholds, and call, token,
cost and timeout limits are explicit. Its cost reservations are safety ceilings,
not a price quote. Confirm the approved resolved model identity for your account.
Update the dataset SHA-256 in `request.yaml` whenever the exact case file changes.
Keep the API key in `OPENAI_API_KEY`, supplied through your secret manager.

## Evaluate and inspect

With your normal native runtime profile and signing key configured:

```bash
invarlock evaluate request.yaml --runtime-profile runtime-profile.json --signing-key signer-private.pem --preflight --json
invarlock evaluate request.yaml --runtime-profile runtime-profile.json --signing-key signer-private.pem --json
invarlock verify evidence --trust-profile recipient-policy.json --json
invarlock report evidence --html report.html --json
```

Preflight makes no provider calls. It checks the native resources, installed
collector, environment, rubric and complete call reservations. Evaluation makes
billable judge calls only within the declared limits. The recipient independently
chooses the trust policy, signer, subject and evidence bindings; do not derive
trust by blindly copying the submitted envelope.

If judging stops with pending trials, rerun the same request. `judge-work` retains
the original answers and call admissions; the final evidence destination stays
absent until all trial slots have an outcome. A failed answer capture is marked
and cannot silently regenerate answers. Inspect that failure before explicitly
starting a new capture in a new workspace. Changing the model, data, rubric,
policy or runtime identity requires a new workspace. Exhausted retained-storage
capacity publishes terminal insufficient evidence with an explicit stop reason.

Native `execution.mode: import` can use existing complete provider side files
instead of generating answers. Those files must bind the original judge policy
bytes, schedule and exact output observations. This mode still collects judge
ratings through the installed collector.

Existing evaluator exports can instead use `metric: judge` in a captured
`invarlock/evaluation-request-v2` request with this same policy recipe. That
route accepts explicit text projections and can import retained measurements
through `comparison.judge.measurements` without the collector or credentials.
Already finalized plans and frozen runs use the separate v3 `judge_import` or
`judge_collect` request. These captured-answer routes authenticate the retained
records and judge replay; they do not claim native runtime provenance. See the
[judge reference](https://invarlock.github.io/invarlock/reference/judge-measurements/) for the
request and recipient trust contracts.
