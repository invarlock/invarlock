# Policy-engine interoperability

The maintained policy-engine example demonstrates signature verification and a
bounded set of recipient-policy checks without an InvarLock service or
policy-engine plugin. It is not the full recipient-acceptance verifier.

> **Reference**
>
> **Surface:** Standalone DSSE verification followed by OPA/Rego or CUE
> evaluation of the example's recipient-policy subset
>
> **Stability:** Maintained example over the acceptance v2 predicate and
> `invarlock/acceptance-policy-input-v1`
>
> **Use this page when:** Studying standalone acceptance-envelope authentication
> and external policy-engine integration

## Data flow

```text
acceptance.dsse.json + envelope-signer public key
recipient policy + expected subject name/digest + evaluation time
                  |
                  v
standalone DSSE and receipt verification
                  |
                  v
invarlock/acceptance-policy-input-v1
                  |
                  +----> OPA/Rego decision
                  |
                  +----> CUE validation
```

The standalone verifier checks:

- strict envelope structure and canonical Statement and embedded receipt
  representations;
- the DSSE payload type, key ID, Ed25519 signature, and public-key
  fingerprint;
- the embedded receipt digest, public key, Ed25519 signature, verifier
  identity, and fingerprint; and
- agreement among the signed receipt verdict, predicate verdict, and signed
  signer projections.

OPA and CUE then enforce the expected predicate type and subject, allowed
InvarLock release contract, an active matching envelope signer and receipt
verifier, envelope freshness, and required technical verdict. Only native
receipt v1/v2 formats are supported; captured and judge receipts are not.

## Convert and evaluate an envelope

Run from the repository root with Python's `cryptography` package and the pinned
OPA/CUE executables installed. Obtain recipient policy, expected subject identity
and evaluation time independently of the submitted envelope. The subject digest
below is lowercase SHA-256 hexadecimal without the `sha256:` prefix.

```bash
set -e
python examples/policy-engine-interop/verify_envelope.py \
  --envelope acceptance.dsse.json \
  --envelope-key envelope-signer.public.pem \
  --recipient-policy recipient-policy.json \
  --expected-subject-name "${EXPECTED_SUBJECT_NAME:?Set the approved subject name}" \
  --expected-subject-sha256 "${EXPECTED_SUBJECT_SHA256:?Set the approved subject digest}" \
  --now "${EVALUATION_TIME:?Set the recipient evaluation time with a timezone}" \
  > policy-input.json

opa eval --format raw \
  --data examples/policy-engine-interop/policy/acceptance.rego \
  --input policy-input.json data.invarlock.acceptance.decision

cue vet examples/policy-engine-interop/policy/acceptance.cue policy-input.json
```

Continue to policy evaluation only if conversion succeeds, and protect the
generated input from modification before consumption. OPA returns an `allow`
boolean and reason codes; a successful `opa eval` exit alone does not mean
`allow: true`. CUE exits successfully when the input satisfies the example's
constraints. Neither result establishes checks outside this example's scope.

## Conformance fixtures

The committed corpus covers positive, policy-rejected, tampered-subject,
untrusted-signer, stale-envelope, and unsupported-contract inputs. The fixture
named `stale-evidence` changes envelope age, not receipt-authenticated evidence
age. OPA returns an explicit allow/deny decision with reason codes. CUE treats
the same positive input as valid and rejects all five negative inputs.

```bash
make acceptance-policy-interop
```

The exact tested tool versions live in
`examples/policy-engine-interop/tool-versions.json`. See the [example
README](https://github.com/invarlock/invarlock/tree/main/examples/policy-engine-interop)
for pinned installation commands.

## Assurance boundary

Rego and CUE evaluate an authenticated projection; they do not themselves
perform raw Ed25519 verification. This division is explicit in the example and
its tests. The verifier is a local reference executable, not an InvarLock
service or hidden plugin.

The example does not enforce every field of the full recipient policy. It omits
receipt-authenticated evidence age, clock-skew allowance, the receipt trust-profile
pin, countersigning restrictions, and duplicate-free, exactly-one signer lookup.
Its receipt-to-predicate checks cover verdict and signer projections, not all
artifact, schedule, policy and contract bindings checked by the full verifier.
Passing the example therefore does not establish full recipient acceptance.

Use the [acceptance-attestation verifier](acceptance-attestations.md) for the
complete portable recipient contract. That verifier is itself distinct from
replaying the complete evidence pack with `invarlock verify`.
