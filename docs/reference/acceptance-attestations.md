# Acceptance attestations

An acceptance attestation transports one InvarLock technical decision about
one exact derived artifact. It is an in-toto Statement v1 inside a DSSE
envelope. It does not represent a recipient's final approval, exception,
quorum, governance action, or deployment decision.

!!! info "Reference"

    - **Surface:** `https://invarlock.dev/attestations/acceptance/v2`
      standards-shaped in-toto/DSSE transport
    - **Stability:** Closed v2 predicate and recipient-policy contracts
    - **Use this page when:** Producing, transporting, or independently
      evaluating a portable acceptance attestation

The predicate format is `invarlock/acceptance-predicate-v2`; the envelope
payload type is `application/vnd.in-toto+json`.

## Handoff model

The primary workflow is:

> authenticated evaluation → portable signed evidence → independent technical
> verification → recipient-controlled acceptance

The evaluation operator evaluates a baseline and subject on the same
authenticated schedule and exports the detailed signed evidence. An independent
technical verifier checks that evidence and signs the receipt. The envelope
signer wraps the receipt for standard attestation transport. The recipient
authenticates the DSSE signer from its own trust registry, binds the in-toto
subject to the exact artifact bytes or an independently obtained digest,
authenticates the embedded receipt, and applies its current policy.

The envelope is standards-shaped in-toto/DSSE transport. The repository
maintains a standalone conformance example that authenticates the envelope and
embedded receipt, then applies recipient policy with OPA/Rego and CUE without an
InvarLock service or policy-engine plugin. The policy engines consume an
authenticated projection; they do not perform raw signature verification or
full evidence replay themselves. See
[Policy-engine interoperability](policy-engine-interop.md). Full semantic replay
uses the InvarLock verifier.

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
- the complete parsed signed InvarLock receipt plus the exact supplied receipt
  bytes as authenticated base64.

The closed source schema is
`contracts/acceptance_predicate.schema.json`; the byte-identical packaged copy
ships with the core wheel.

## Authoritative receipt and v0.13 wrapping

Version 2 uses `receipt.representation: embedded`. The embedded signed
InvarLock receipt is the authoritative replayable technical result.
`receipt.raw_base64` preserves the exact supplied receipt-file bytes, and
`receipt.digest` is SHA-256 over those decoded bytes. `receipt.content` is the
parsed policy projection and must decode to the same JSON value. The
surrounding predicate duplicates selected fields, all of which must agree with
the authenticated receipt.

Existing v0.13 receipt formats are wrapped without modification or relabeling.
Their original bytes are recoverable byte-for-byte from `receipt.raw_base64`,
and their original format remains in `contracts.receipt` and the parsed
receipt. Because v0.13 receipts did not authenticate an issuance time,
`timestamps.receipt_issued_at` remains `null`; the wrapper rejects any attempt
to manufacture that historical metadata. `evaluation_completed_at` is
wrapper-supplied context, not an authoritative freshness input.
`attestation_issued_at` is new envelope-transport metadata.

The verifier authenticates the inner receipt independently and checks its
technical verdict, artifact anchors, schedule digest, policy digest, contract
version, and signer against the outer predicate. Any contradiction rejects the
attestation, even if the modified envelope has a valid outer signature.

## Signer relationship

The receipt signer is the technical verifier. The envelope signer is the party
transporting that verified result.

- `same_signer` means both the identity and Ed25519 public-key fingerprint are
  identical.
- `countersigned` means either differs.

The relationship is descriptive and checked against the two signer objects.
Recipient policy independently decides whether countersigned receipts are
allowed. Trust in the DSSE signer does not replace authentication of the
embedded receipt signer. Recipient policy therefore has a separate
`trusted_receipt_verifiers` registry. An optional
`expected_receipt_trust_profile_digest` pins the verifier-owned trust profile
recorded by the receipt. Unknown or revoked receipt verifiers fail closed even
when the outer envelope signer is trusted.

## Canonical bytes and signatures

The DSSE payload and each receipt statement's signed bytes use UTF-8,
lexicographically sorted object keys, no insignificant whitespace, no
non-finite numbers, unescaped Unicode, and one trailing line feed. The supplied
receipt file itself may use different valid JSON formatting; its exact bytes
are retained separately. SHA-256 digests are lowercase and use the `sha256:`
prefix.

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
`contracts/recipient_acceptance_policy.schema.json`. It independently controls
trusted envelope signers, trusted receipt verifiers, optional receipt
trust-profile pinning, signer status, allowed InvarLock contract versions,
required technical verdict, and whether countersigning is allowed.
Each trust registry requires unique identity/fingerprint pairs. Duplicate pairs
invalidate the policy regardless of record order or status, and verification
requires exactly one matching trust record for each authenticated signer.

Freshness has two independent limits. `max_envelope_age_seconds` applies to
`attestation_issued_at`. `max_evidence_age_seconds`, when non-null, applies
only to the receipt-authenticated `receipt_issued_at`. A missing authoritative
evidence timestamp rejects under an evidence-age constraint. An envelope signer
cannot make old or undated evidence fresh merely by creating a new envelope.

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
        "sha256:<envelope-signer-fingerprint>": Path("envelope-signer.public.pem")
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
