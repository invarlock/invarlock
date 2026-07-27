# Policy-engine interoperability

The maintained policy-engine example demonstrates that a recipient can
authenticate an InvarLock acceptance envelope and apply current policy without
an InvarLock service or policy-engine plugin.

!!! info "Reference"

    - **Envelope:** DSSE with an in-toto Statement v1 payload
    - **Predicate:** `https://invarlock.dev/attestations/acceptance/v2`
    - **Engines:** Open Policy Agent v1.17.0 and CUE v0.16.1
    - **Boundary:** Standalone Ed25519 verifier to authenticated JSON policy
      input; no `invarlock` import or network call

## Data flow

```text
acceptance.dsse.json + producer public key
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

- canonical envelope, Statement, and embedded receipt representations;
- the DSSE payload type, key ID, Ed25519 signature, and public-key
  fingerprint;
- the embedded receipt digest, public key, Ed25519 signature, verifier
  identity, and fingerprint; and
- agreement among the signed receipt verdict, predicate verdict, and signed
  signer projections.

OPA and CUE then enforce the expected predicate type and subject, allowed
InvarLock release contract, active envelope signer and receipt verifier,
attestation freshness, and required technical verdict.

## Conformance fixtures

The committed corpus covers positive, policy-rejected, tampered-subject,
untrusted-signer, stale-evidence, and unsupported-contract inputs. OPA returns
an explicit allow/deny decision with reason codes. CUE treats the same positive
input as valid and rejects all five negative inputs.

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

The result answers whether a portable historical technical decision is
acceptable under current recipient policy. It does not replay the complete
evidence pack. Use `invarlock verify` for full technical replay.
