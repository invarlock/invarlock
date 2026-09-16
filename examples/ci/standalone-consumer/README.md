# Standalone protected deployment consumer

Use this layout when your application repository needs to verify a model
release produced elsewhere. The evaluation operator supplies evidence; your
repository supplies the policy, expected identities and deployment code.

> **Outcome:** Configure two CI jobs that verify evidence, authenticate the
> resulting receipt and deploy only the approved artifact identity.
>
> **Audience:** Developers adapting the CI example into their own repository.
>
> **Prerequisites:** A GitHub repository with protected environments, a
> hash-pinned InvarLock installation and deployment scripts owned by your team.

## Files and responsibilities

This directory has the layout of a separate repository that consumes an
InvarLock evidence pack. It keeps verifier-owned review material separate from
submitted evidence and uses two protected GitHub environments:

- `release-review` verifies the signed evidence against independent anchors and
  issues a verifier-signed receipt;
- `production` reauthenticates that receipt against production-owned inputs and
  deploys only the exact subject identity named by the approval record.

The repository owner supplies these files before enabling the workflow:

- `incoming/evidence/`, delivered by the evaluation operator;
- `review/requirements/invarlock-verifier.txt`, a hash-pinned lock for the
  selected InvarLock release and its complete verifier dependency closure;
- `deployment/fetch-approved-candidate.sh`, which resolves a candidate and
  proves its InvarLock artifact identity equals the approved subject digest;
- `deployment/deploy-candidate.sh`, which accepts that exact artifact and the
  authenticated approval record.

The checked-in policy and approval-input JSON are public demonstration inputs
for the retained Inspect AI transaction. A real consumer replaces them with
recipient-controlled material and stores the complete approval-input JSON in
the protected `INVARLOCK_DEPLOYMENT_APPROVAL_INPUTS_JSON` secret. The verifier
key, policy digest, artifact identities, schedule, runtime identities, evidence
signer, and verifier identity are not derived from submitted evidence.

[`deployment-approval.yml`](.github/workflows/deployment-approval.yml) pins the
InvarLock composite action and every third-party action to immutable commits.
Copy this directory into its own repository, add the four consumer-owned
surfaces above, and replace the demonstration anchors before using it for a
deployment.

## Configure and run the workflow

1. In `release-review`, configure the `INVARLOCK_VERIFIER_KEY_PEM` secret and
   the expected artifact, schedule, runtime, evidence-signer and verifier-identity
   variables named in the workflow. Obtain these expectations independently of
   `incoming/evidence/`.
2. In `production`, set `INVARLOCK_DEPLOYMENT_APPROVAL_INPUTS_JSON` to the complete
   independently approved recipient inputs. Configure environment protection
   rules appropriate to the deployment.
3. Deliver the evidence to `incoming/evidence/` through your controlled artifact
   acquisition step, then manually dispatch the workflow.

The first job uploads the evidence and signed verification receipt. The second
downloads them, checks the receipt again and writes
`review-artifact/deployment-approval.json`. Only after that check does it resolve
the approved candidate and invoke your deployment adapter. Job success or an
artifact download alone is not deployment authorization.

Before connecting a deployment, try the
[local receipt example](../README.md#run-the-example). Its command checks the
committed Inspect fixture without GitHub credentials, inference or deployment.

## What the tests establish

The repository tests execute the composite action's command steps from an
isolated copy of this consumer layout and cover a valid signed pack and a
tampered pack. Distribution smoke tests separately run the copied receipt
consumer after installing only the built core wheel. Verification against the
hosted package is repeated only after that version has been published.
