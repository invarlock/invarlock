# CI deployment approval

This example turns a successful InvarLock evidence verification into a
separately authenticated deployment decision. It deliberately uses two trust
domains:

- `release-review` owns evidence-verification anchors and the verifier signing
  key;
- `production` owns deployment approval inputs and independently checks the
  signed receipt before running the deployment command.

[`github-actions/deployment-approval.yml`](github-actions/deployment-approval.yml)
is a consumer workflow template. Replace the example policy, requirements,
candidate, and deployment-script paths with repository-owned equivalents. Keep
the action and third-party actions pinned to immutable commits.

`verify_deployment_receipt.py` accepts an
`invarlock/deployment-approval-inputs-v1` object containing exactly:

- baseline and subject artifact digests;
- baseline and subject runtime digests;
- schedule and policy digests;
- the authorized evidence-signer fingerprint; and
- the authorized verifier identity and fingerprint.

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

Run the behavioral tests with:

```bash
python -m pytest -q tests/examples/test_ci_deployment_approval_example.py
```

The test uses the committed acceptance-handoff evidence and signed receipt. It
also changes verifier and policy anchors to prove that artifact transfer or job
success alone cannot authorize deployment.
