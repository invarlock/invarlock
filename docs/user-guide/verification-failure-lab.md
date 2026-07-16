# Verification failure lab

This lab turns policy and trust failures into observable command behavior. It
uses the checked-in offline example, so it exercises real signing, bundle
integrity, policy replay, receipt signing, and report authentication without
downloading a model.

!!! tip "User guide"

    **Outcome:** Reproduce an authentic report-local policy rejection plus wrong-anchor, tamper, extra-file, and receipt-authorization failures.
    **Audience:** Verifier operators, CI authors, and maintainers validating fail-closed behavior.
    **Prerequisites:** A source checkout, the core package installed, a shell with `mktemp`, and permission to create a disposable working directory.

## Create a disposable valid transaction

```bash
WORK_DIR="$(mktemp -d)"
cp -R examples "$WORK_DIR/examples"
cd "$WORK_DIR/examples"

python generate_keys.py --output-dir .keys
invarlock evaluate request.yaml --signing-key .keys/evidence-signer.pem

EVIDENCE_SIGNER_FINGERPRINT="$(tr -d '\n' < .keys/evidence-signer.fingerprint)"
BASELINE_ARTIFACT="$(python -c 'import json; print(json.load(open("trusted-inputs/input-digests.json"))["baseline_artifact"])')"
SUBJECT_ARTIFACT="$(python -c 'import json; print(json.load(open("trusted-inputs/input-digests.json"))["subject_artifact"])')"
SCHEDULE="$(python -c 'import json; print(json.load(open("trusted-inputs/input-digests.json"))["canonical_schedule"])')"
BASELINE_RUNTIME='sha256:1111111111111111111111111111111111111111111111111111111111111111'
SUBJECT_RUNTIME='sha256:2222222222222222222222222222222222222222222222222222222222222222'

invarlock verify artifacts/evidence \
  --policy policy/acceptance.json \
  --expected-baseline-artifact "$BASELINE_ARTIFACT" \
  --expected-subject-artifact "$SUBJECT_ARTIFACT" \
  --expected-schedule "$SCHEDULE" \
  --expected-baseline-runtime "$BASELINE_RUNTIME" \
  --expected-subject-runtime "$SUBJECT_RUNTIME" \
  --expected-signer "$EVIDENCE_SIGNER_FINGERPRINT" \
  --receipt accepted.receipt.json \
  --verifier-signing-key .keys/verifier.pem \
  --verifier-identity failure-lab-verifier \
  --json
```

The control verification exits `0`, reports `ok: true`, and writes a signed
acceptance receipt outside `artifacts/evidence`. Keep this original directory
unchanged; each failure starts from a separate copy.

## Authentic policy rejection

The second checked-in request uses the same schedule, policy, identities, and
runtimes. Its subject answers one of two records incorrectly, so the independently
replayed exact-match delta is `-50` percentage points against a required minimum
of `0`. Produce its immutable evidence, then verify it under the same anchors:

```bash
invarlock evaluate rejected-request.yaml --signing-key .keys/evidence-signer.pem

if invarlock verify artifacts/rejected-evidence \
  --policy policy/acceptance.json \
  --expected-baseline-artifact "$BASELINE_ARTIFACT" \
  --expected-subject-artifact "$SUBJECT_ARTIFACT" \
  --expected-schedule "$SCHEDULE" \
  --expected-baseline-runtime "$BASELINE_RUNTIME" \
  --expected-subject-runtime "$SUBJECT_RUNTIME" \
  --expected-signer "$EVIDENCE_SIGNER_FINGERPRINT" \
  --receipt policy-rejected.receipt.json \
  --verifier-signing-key .keys/verifier.pem \
  --verifier-identity failure-lab-verifier \
  --json; then
  echo 'unexpected policy acceptance' >&2
  exit 1
fi
```

Expected result: the evidence remains authentic and integrity-valid, the
replayed policy verdict is `fail`, the command exits nonzero, and
`policy-rejected.receipt.json` is a verifier-signed rejection. This is a valid
measurement outcome rather than an infrastructure or cryptographic failure.

## Wrong evidence signer anchor

```bash
WRONG_SIGNER='sha256:0000000000000000000000000000000000000000000000000000000000000000'

if invarlock verify artifacts/evidence \
  --policy policy/acceptance.json \
  --expected-baseline-artifact "$BASELINE_ARTIFACT" \
  --expected-subject-artifact "$SUBJECT_ARTIFACT" \
  --expected-schedule "$SCHEDULE" \
  --expected-baseline-runtime "$BASELINE_RUNTIME" \
  --expected-subject-runtime "$SUBJECT_RUNTIME" \
  --expected-signer "$WRONG_SIGNER" \
  --receipt wrong-signer.receipt.json \
  --verifier-signing-key .keys/verifier.pem \
  --verifier-identity failure-lab-verifier \
  --json; then
  echo 'unexpected acceptance' >&2
  exit 1
fi
```

Expected result: nonzero verification, an explicit signer mismatch, and a
signed rejection receipt when the transaction reached receipt issuance. The
receipt's existence is not acceptance; its signed verdict is rejection.

## Changed authenticated bytes

```bash
cp -R artifacts/evidence tampered-evidence
chmod u+w tampered-evidence/reports/evaluation.report.json
printf '\n' >> tampered-evidence/reports/evaluation.report.json

if invarlock report tampered-evidence; then
  echo 'unexpected report success' >&2
  exit 1
fi
```

Expected result: reporting rejects before rendering because the report digest
no longer matches the closed signed inventory. Do not repair the manifest or
checksums to follow the changed report; that would create a different unsigned
artifact.

## Extra file in the closed bundle

```bash
cp -R artifacts/evidence extra-file-evidence
chmod u+w extra-file-evidence
touch extra-file-evidence/unexpected.txt

if invarlock report extra-file-evidence; then
  echo 'unexpected report success' >&2
  exit 1
fi
```

Expected result: closed-inventory rejection. Receipts and HTML belong beside,
not inside, the evidence directory.

## Wrong runtime anchor

```bash
WRONG_RUNTIME='sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'

if invarlock verify artifacts/evidence \
  --policy policy/acceptance.json \
  --expected-baseline-artifact "$BASELINE_ARTIFACT" \
  --expected-subject-artifact "$SUBJECT_ARTIFACT" \
  --expected-schedule "$SCHEDULE" \
  --expected-baseline-runtime "$WRONG_RUNTIME" \
  --expected-subject-runtime "$SUBJECT_RUNTIME" \
  --expected-signer "$EVIDENCE_SIGNER_FINGERPRINT" \
  --receipt wrong-runtime.receipt.json \
  --verifier-signing-key .keys/verifier.pem \
  --verifier-identity failure-lab-verifier \
  --json; then
  echo 'unexpected acceptance' >&2
  exit 1
fi
```

Expected result: integrity can remain true while external runtime acceptance
fails. This distinction is why the verifier receives runtime anchors instead
of copying them from the pack.

## Unauthorized receipt signer

The accepted receipt embeds a verifier public key, but that key does not
self-authorize. Verify the receipt through the Python facade with a deliberately
wrong expected verifier fingerprint:

```python
from pathlib import Path

from invarlock.engine import verify_signed_verification_receipt

result = verify_signed_verification_receipt(
    Path("accepted.receipt.json"),
    Path("artifacts/evidence"),
    policy_path=Path("policy/acceptance.json"),
    expected_runtime_digests={
        "baseline": "sha256:" + "1" * 64,
        "subject": "sha256:" + "2" * 64,
    },
    expected_pack_signer_fingerprint=Path(
        ".keys/evidence-signer.fingerprint"
    ).read_text(encoding="utf-8").strip(),
    expected_verifier_identity="failure-lab-verifier",
    expected_verifier_fingerprint="sha256:" + "0" * 64,
)
assert not result.ok
assert result.errors
```

Expected result: the receipt signature may be cryptographically valid while
authorization fails. Obtain the real expected verifier fingerprint through a
separate trust registry before relying on a receipt.

## Interpret the outcomes

| Failure | Evidence signature may still be valid? | Integrity may still be valid? | Acceptance |
| --- | --- | --- | --- |
| Exact-match regression | Yes | Yes | Reject under the independently replayed policy |
| Wrong evidence signer anchor | Yes, for the embedded key | No under the verifier's required signer anchor | Reject |
| Changed report bytes | No for the changed inventory | No | Reject |
| Extra file | Original signed files may be unchanged | No closed inventory | Reject |
| Wrong runtime anchor | Yes | Yes | Reject |
| Wrong verifier authorization | Receipt signature may be valid | Pack may be valid | Reject |

Delete the disposable `WORK_DIR` after review. Never run tamper exercises on a
published pack or overwrite a signed receipt that has been distributed.

Continue with [Evidence and verification](evidence-and-verification.md) for the
full trust model and [Troubleshooting](troubleshooting.md) for production
recovery.
