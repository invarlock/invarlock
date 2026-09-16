# CI deployment approval

This example turns a successful InvarLock evidence verification into a
separately authenticated deployment decision. It deliberately uses two trust
domains:

- `release-review` owns evidence-verification anchors and the verifier signing
  key;
- `production` owns deployment approval inputs and independently checks the
  signed receipt before running the deployment command.

> **Outcome:** Check a retained signed receipt locally, then understand how to
> place the same check between evidence verification and deployment in CI.
>
> **Audience:** ML platform engineers connecting evaluation results to a release
> pipeline.
>
> **Prerequisites:** The matching checkout and installed InvarLock package for the
> local example; a repository development environment for its tests. The GitHub
> workflow additionally needs protected environments and your deployment scripts.

## Run the example

Run the direct command below from the repository root. It uses committed public
fixtures, performs no inference, and does not deploy anything. Success prints a
JSON approval record with format `invarlock/deployment-approval-v1`; a mismatch
exits unsuccessfully instead of emitting an approval. The fixture's trusted
inputs demonstrate the shape of a real decision but are not production trust
configuration.

```bash
python examples/ci/standalone-consumer/review/verify_deployment_receipt.py \
  --approval-inputs examples/ci/standalone-consumer/review/inspect-ai-deployment-approval-inputs.json \
  --evidence examples/evaluator-qualification/signed-transactions/deployment-approval-inspect-ai/evidence \
  --policy examples/ci/standalone-consumer/review/policy/acceptance.json \
  --receipt examples/evaluator-qualification/signed-transactions/deployment-approval-inspect-ai/verification.receipt.json
```

For a consumer repository, start with the
[standalone layout](standalone-consumer/).

## How the CI boundary works

[`standalone-consumer/`](standalone-consumer/) is a copyable consumer-repository
fixture. Its workflow, policy, recipient anchors, and receipt checker do not
import helper code from the InvarLock source tree. Replace the demonstration
requirements, evidence, candidate resolver, deployment adapter, and trust
inputs with repository-owned equivalents. Keep the action and third-party
actions pinned to immutable commits.

The checked-in behavioral example consumes the retained Inspect AI evidence
pack and signed verification receipt. Its separate example recipient anchors
live in
[`inspect-ai-deployment-approval-inputs.json`](standalone-consumer/review/inspect-ai-deployment-approval-inputs.json),
outside the submitted evidence. The fixture makes a passing signed evaluator
approval concrete; production deployments must obtain the same classes of anchors
through recipient-controlled channels.

`standalone-consumer/review/verify_deployment_receipt.py` accepts an
`invarlock/deployment-approval-inputs-v1` object containing exactly:

- baseline and subject artifact digests;
- baseline and subject runtime digests;
- schedule and policy digests;
- the authorized evidence-signer fingerprint;
- the authorized verifier identity and fingerprint; and
- the verifier trust-profile digest recorded when the receipt was issued, or
  `null` when receipt issuance used explicit anchors without a trust profile.

The helper fails closed if any anchor, signature, policy, receipt verdict, or
evidence binding differs. On success it emits a canonical
`invarlock/deployment-approval-v1` record carrying the approved baseline and
subject artifact identities, runtime identities, and schedule digest. The
output is no-clobber when `--output` is used.

The deployment adapter must resolve or recompute the candidate's InvarLock
artifact identity and compare it with `artifact_digests.subject` in that record
before deployment. The workflow calls a repository-owned
`fetch-approved-candidate.sh` boundary for that purpose; replacing it with a
plain path or mutable tag would reopen a time-of-check/time-of-use gap.

## Test the integration

With the development environment active, run the behavioral tests:

```bash
python -m pytest -q tests/examples/test_ci_deployment_approval_example.py
```

The tests also change verifier and policy anchors to prove that artifact
transfer or job success alone cannot authorize deployment.

## Adapt it for your release pipeline

Replace the retained evidence and demonstration anchors together with your
independently approved policy and identities. Configure both protected
environments before enabling the workflow, and implement the candidate resolver
so it checks the artifact that will actually be deployed. A passing receipt
does not establish model safety, hardware execution, or suitability for your
application. The deployment owner remains responsible for those requirements.
