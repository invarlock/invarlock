# Acceptance attestations

An acceptance attestation transports one InvarLock technical decision about
one exact derived artifact. It is an in-toto Statement v1 inside a DSSE
envelope. It does not represent a recipient's final approval, exception,
quorum, governance action, or deployment decision.

!!! info "Reference"

    - **Surface:** `https://invarlock.dev/attestations/acceptance/v1`
      in-toto/DSSE transport
    - **Stability:** Closed v1 predicate and recipient-policy contracts
    - **Use this page when:** Producing, transporting, or independently
      evaluating a portable acceptance attestation

The predicate format is `invarlock/acceptance-predicate-v1`; the envelope
payload type is `application/vnd.in-toto+json`.

## Handoff model

The primary workflow is:

> producer evaluation → portable signed evidence → independent recipient
> verification → recipient-controlled acceptance

The producer evaluates a baseline and subject on the same authenticated
schedule, exports the detailed evidence and signed receipt, then wraps that
receipt for standard attestation transport. The recipient authenticates the
DSSE signer from its own trust registry, binds the in-toto subject to the exact
artifact bytes or an independently obtained digest, authenticates the embedded
receipt, and applies its current policy.

Existing attestation policy engines can authenticate and policy-evaluate an
InvarLock acceptance attestation without a custom InvarLock service or plugin;
full semantic replay uses the InvarLock verifier.

Generic attestation engines do not reproduce InvarLock's paired metric,
statistics, policy arithmetic, or evidence-pack replay.

## Statement and predicate

The in-toto `subject` contains exactly the derived artifact name and SHA-256
content digest. The predicate repeats that exact binding and carries:

- exact subject and baseline artifact identities, identity digests, content
  digests, and digest kinds;
- the InvarLock release, evidence-pack, comparison-report, and receipt
  versions;
- the schedule format, digest, and evaluation-source identity;
- the built-in metric identity or complete scorer-extension identity;
- the evaluated policy identity and digest;
- the technical verdict copied from the signed receipt;
- evaluation, receipt, and attestation timestamps;
- receipt and envelope signer identities, key fingerprints, and their
  relationship; and
- the complete signed InvarLock receipt.

The closed source schema is
`contracts/acceptance_predicate.schema.json`; the byte-identical packaged copy
ships with the core wheel.

## Authoritative receipt and v0.13 wrapping

Version 1 uses `receipt.representation: embedded`. The embedded signed
InvarLock receipt is the authoritative replayable technical result. Its digest
is SHA-256 over its canonical JSON bytes. The surrounding predicate is a
portable, policy-friendly projection whose duplicated fields must agree with
the receipt. A digest reference is not an alternative v1 representation.

Existing v0.13 receipt formats are wrapped without modification or relabeling.
Their original format remains in `contracts.receipt` and in the embedded
receipt. Because v0.13 receipts did not record an issuance time,
`timestamps.receipt_issued_at` remains `null`; the wrapper does not manufacture
historical metadata. The attestation issuance time is new transport metadata.

The verifier authenticates the inner receipt independently and checks its
technical verdict, artifact anchors, schedule digest, policy digest, contract
version, and signer against the outer predicate. Any contradiction rejects the
attestation, even if the modified envelope has a valid outer signature.

## Signer relationship

The receipt signer is the technical verifier. The envelope signer is the
producer or other party transporting that verified result.

- `same_signer` means both the identity and Ed25519 public-key fingerprint are
  identical.
- `countersigned` means either differs.

The relationship is descriptive and checked against the two signer objects.
Recipient policy independently decides whether countersigned receipts are
allowed. Trust in the DSSE signer does not replace authentication of the
embedded receipt signer.

## Canonical bytes and signatures

All signed JSON uses UTF-8, lexicographically sorted object keys, no
insignificant whitespace, no non-finite numbers, unescaped Unicode, and one
trailing line feed. SHA-256 digests are lowercase and use the `sha256:` prefix.

The DSSE payload is the canonical in-toto Statement bytes. Ed25519 signs the
DSSE pre-authentication encoding:

```text
DSSEv1 <type-byte-length> <payloadType> <payload-byte-length> <payload>
```

Lengths count bytes, not characters. The envelope contains one signature whose
`keyid` is the SHA-256 fingerprint of the raw Ed25519 public key. Verification
rejects a non-canonical payload even when it decodes to the same JSON value.

## Recipient policy and exact subject binding

The closed recipient-policy schema is
`contracts/recipient_acceptance_policy.schema.json`. It controls the expected
predicate TypeURI, trusted envelope signer identity and fingerprint, signer
status, freshness and clock skew, allowed InvarLock contract versions,
required technical verdict, and whether countersigning is allowed.

Exact subject binding is a separate, mandatory recipient input: provide either
the expected `sha256:` digest or the artifact path to hash using the predicate's
declared artifact digest kind. Providing neither or both fails closed.

```python
from datetime import UTC, datetime
from pathlib import Path

from invarlock.engine import verify_acceptance_attestation

decision = verify_acceptance_attestation(
    Path("acceptance.dsse.json"),
    trusted_public_keys={
        "sha256:<producer-key-fingerprint>": Path("producer.public.pem")
    },
    recipient_policy=Path("recipient-policy.json"),
    subject_artifact_path=Path("artifact"),
    now=datetime.now(tz=UTC),
)
if not decision.accepted:
    raise SystemExit("; ".join(decision.errors))
```

The reference policy is
`examples/acceptance-handoff/recipient-policy.example.json`. Replace its
placeholder fingerprint and policy values with recipient-controlled inputs.
The complete offline journey and committed package are documented in
`examples/acceptance-handoff/README.md`.

`verify_acceptance_attestation` authenticates and policy-evaluates the portable
envelope. The authoritative full replay entry point remains
`invarlock verify EVIDENCE --trust-profile PROFILE`; the equivalent Python
surface is `invarlock.engine.verify_evidence`.
