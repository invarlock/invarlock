# Native judge starter

This starter selects `metric: judge` in the same native request used for exact
match and normalized NLL. It generates baseline and subject answers through the
native runtime, freezes the complete runtime evidence, collects bounded judge
ratings, and publishes evidence for offline `verify` and `report`.

The two invented questions demonstrate wiring. They cannot establish benchmark
quality or satisfy the interval precision in the example policy. Replace them
with your own reviewed regression cases and independent-unit assignments before
using the result for a decision. Context needed to judge an answer belongs in
`prompt`; the current text profile does not send the separate `expected` field
to the judge.

## Prepare

Install matching core and collector packages from the repository root:

```bash
python -m pip install .
python -m pip install 'addins/inspect_judge[inspect]'
```

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
bytes, schedule and exact output observations. Arbitrary frozen runs use the
separate `judge_collect` request; they do not claim native runtime provenance.
