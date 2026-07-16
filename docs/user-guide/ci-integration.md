# CI integration

The InvarLock evidence-gate action verifies one immutable evidence pack against
verifier-controlled anchors, writes a separately signed receipt, renders the
authenticated report, and uploads the review material as one workflow artifact.

!!! tip "User guide"

    **Outcome:** Add a fail-closed GitHub Actions gate for an existing
    `evidence-pack-v1` directory.

    **Audience:** Release engineers and verifier operators automating an
    independent acceptance decision.

    **Prerequisites:** An evidence pack, an independently managed policy and
    trust anchors, an installed matching InvarLock version, and a verifier key
    available to the job without being stored in the evidence artifact.

The action runs the same public transactions documented by the
[CLI reference](../reference/cli.md):

1. `invarlock verify EVIDENCE` authenticates and replays the bundle using the
   supplied artifact identities, schedule, policy, runtime digests, and
   evidence-signer fingerprint, then writes a verifier-signed receipt.
2. `invarlock report EVIDENCE --html PATH --explain` authenticates the bundle
   and writes a human view without changing the evidence.
3. The action uploads the evidence directory, verification JSON, signed
   receipt, and HTML report. It never uploads the verifier private key.

## Inputs

Every trust input must come from verifier-controlled configuration rather than
from the submitted evidence directory.

| Input | Required | Purpose |
| --- | --- | --- |
| `evidence` | Yes | Immutable `evidence-pack-v1` directory to review |
| `policy` | Yes | Independently approved acceptance-policy JSON |
| `expected-baseline-artifact` | Yes | Approved baseline artifact-identity digest |
| `expected-subject-artifact` | Yes | Approved subject artifact-identity digest |
| `expected-schedule` | Yes | Approved canonical schedule digest |
| `expected-baseline-runtime` | Yes | Approved baseline outer-image digest |
| `expected-subject-runtime` | Yes | Approved subject outer-image digest |
| `expected-signer` | Yes | Authorized Ed25519 evidence-signer fingerprint |
| `verifier-signing-key` | Yes | Path to the verifier-owned private-key file |
| `verifier-identity` | Yes | Stable identity recorded in the signed receipt |
| `python` | No | Python executable; defaults to `python` |
| `receipt-output` | No | Signed receipt path outside evidence |
| `verify-output` | No | Machine-readable verification-result path |
| `html-output` | No | Self-contained HTML report path outside evidence |
| `artifact-name` | No | Name of the uploaded workflow artifact |
| `fail-on-verify` | No | Whether a nonzero verifier result fails the job; defaults to `true` |

Output paths are no-clobber. Use new names for every review attempt and keep
all three outside the signed evidence directory. The policy and verifier key
must also resolve outside evidence; the action refuses an unsafe layout before
it renders or uploads any review artifact. Path inputs must be plain paths:
control characters, leading or trailing whitespace, and upload-pattern
characters (`*`, `?`, `[`, `]`, `{`, `}`, `!`, `~`, `\`, and `#`) are
rejected so the artifact uploader cannot expand one approved path into
additional files.
The submitted evidence must also be a real directory with no symbolic links;
this is checked before any failed-review material becomes uploadable.

## Complete workflow

This example assumes an earlier evaluation job or a controlled checkout placed
the submitted pack at `incoming/evidence`. Store the verifier private key as a
GitHub Actions secret. Store the expected artifact, schedule, runtime, and
evidence-signer anchors as protected environment variables or retrieve them
from an independently managed approval system.

Pin the action to a pinned InvarLock commit rather than a mutable branch.
Replace `PINNED_INVARLOCK_COMMIT` with its full commit SHA.

```yaml
name: Verify InvarLock evidence

on:
  workflow_dispatch:

permissions:
  contents: read

jobs:
  verify-evidence:
    runs-on: ubuntu-latest
    environment: release-review
    steps:
      - name: Check out review inputs
        uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0

      - name: Set up Python
        uses: actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1
        with:
          python-version: "3.13"

      - name: Install the pinned verifier environment
        run: |
          python -m pip install --require-hashes \
            -r review/requirements/invarlock-verifier.txt

      - name: Materialize the verifier key
        env:
          VERIFIER_KEY_PEM: ${{ secrets.INVARLOCK_VERIFIER_KEY_PEM }}
        run: |
          umask 077
          printf '%s\n' "$VERIFIER_KEY_PEM" > "$RUNNER_TEMP/invarlock-verifier.pem"

      - name: Authenticate the acceptance policy
        env:
          EXPECTED_POLICY_SHA256: ${{ vars.INVARLOCK_POLICY_SHA256 }}
        run: |
          python - <<'PY'
          import hashlib
          import os
          from pathlib import Path

          policy = Path("policy/acceptance.json").read_bytes()
          observed = "sha256:" + hashlib.sha256(policy).hexdigest()
          if observed != os.environ["EXPECTED_POLICY_SHA256"]:
              raise SystemExit("acceptance policy digest mismatch")
          PY

      - name: Verify and render evidence
        uses: invarlock/invarlock/.github/actions/invarlock-report-gate@PINNED_INVARLOCK_COMMIT
        with:
          evidence: incoming/evidence
          policy: policy/acceptance.json
          expected-baseline-artifact: ${{ vars.INVARLOCK_BASELINE_ARTIFACT_DIGEST }}
          expected-subject-artifact: ${{ vars.INVARLOCK_SUBJECT_ARTIFACT_DIGEST }}
          expected-schedule: ${{ vars.INVARLOCK_SCHEDULE_DIGEST }}
          expected-baseline-runtime: ${{ vars.INVARLOCK_BASELINE_RUNTIME_DIGEST }}
          expected-subject-runtime: ${{ vars.INVARLOCK_SUBJECT_RUNTIME_DIGEST }}
          expected-signer: ${{ vars.INVARLOCK_EVIDENCE_SIGNER_FINGERPRINT }}
          verifier-signing-key: ${{ runner.temp }}/invarlock-verifier.pem
          verifier-identity: release-verifier
          receipt-output: reports/invarlock/verification.receipt.json
          verify-output: reports/invarlock/verification.result.json
          html-output: reports/invarlock/evidence.html
          artifact-name: candidate-invarlock-evidence
```

The verifier-owned requirements file must pin the complete verifier dependency
closure and carry hashes verified against the intended release artifacts; a
version constraint alone is not a supply-chain identity. For a released action,
use that release's pinned immutable commit SHA. A tag is easier to read but a
full commit pin makes action-source substitution visible in review.
`INVARLOCK_POLICY_SHA256` must likewise be a protected verifier-owned digest,
not a value read from the submitted checkout.

## Outputs and status

The default action outputs are:

```text
reports/invarlock/
├── evidence.html
├── verification.receipt.json
└── verification.result.json
```

The uploaded artifact contains those files plus the submitted evidence
directory. Their roles remain distinct:

| Output | Authority |
| --- | --- |
| Evidence directory | Signed canonical evidence and comparison report |
| `verification.result.json` | Machine-readable result for this verifier run |
| `verification.receipt.json` | Verifier-signed decision and independent anchors |
| `evidence.html` | Reproducible human rendering; not an acceptance record |

Verification JSON is written even when the verifier returns nonzero. The action
then attempts the authenticated renderer and uploads available review material.
With the default `fail-on-verify: true`, any nonzero verification result fails
the job after artifact collection. Automation must require job success and
validate the signed receipt; it must not infer acceptance from HTML or artifact
upload success.

The verification action writes the signed receipt and selected review artifacts. It does not create MLflow tags, model-card fragments, or
release-review Markdown. Downstream systems should consume the verified JSON
and validate the signed receipt through the
[Python API](../reference/api-guide.md#verify-a-signed-receipt).

## Trust and secret handling

- Never derive `expected-signer` from `manifest.signature.json` in the pack.
- Never copy either expected runtime digest from the submitted runtime
  manifests.
- Do not upload or cache the verifier private key.
- Protect the policy, anchor variables, environment approvals, and workflow
  revision from the evidence-signing role.
- Retain failed attempts according to the review policy; use fresh receipt and
  report paths for a retry.
- Validate a received receipt against the expected verifier identity and
  fingerprint before another system relies on it.

Continue with [Evidence and verification](evidence-and-verification.md) for the
handoff and receipt verifier rules and [Key management](key-management.md) for
rotation and incident response.
