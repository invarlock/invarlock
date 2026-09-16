# Standalone policy-engine interoperability

This example feeds authenticated acceptance facts from the committed in-toto
Statement and DSSE envelope into two external policy engines:

- Open Policy Agent v1.17.0 with Rego; and
- CUE v0.16.1.

OPA and CUE do not provide a raw Ed25519 DSSE-verification primitive. The local
`verify_envelope.py` boundary therefore authenticates the DSSE envelope and
embedded signed verification receipt, checks envelope structure, canonical
Statement and receipt representations, and verdict/signer projections, then
emits one JSON input. It imports no InvarLock module,
starts no service, and performs no network request. The policy engines then
apply the example's recipient-controlled signer, subject, envelope-freshness,
contract-version, and technical-verdict rules. This is a bounded integration
example, not the full recipient-acceptance verifier. It supports native receipt
v1/v2 formats, not captured or judge receipts.

Run the pinned conformance matrix:

```bash
gopath="$(go env GOPATH)"
go install github.com/open-policy-agent/opa@v1.17.0
go install cuelang.org/go/cmd/cue@v0.16.1
make acceptance-policy-interop \
  OPA="${gopath}/bin/opa" \
  CUE="${gopath}/bin/cue"
```

The six fixtures are:

| Fixture | OPA | CUE |
| --- | --- | --- |
| Positive authenticated delivery | Allow | Valid |
| Recipient policy rejection | Deny | Invalid |
| Tampered subject | Deny | Invalid |
| Untrusted envelope signer | Deny | Invalid |
| Stale envelope (`stale-evidence` fixture) | Deny | Invalid |
| Unsupported InvarLock contract | Deny | Invalid |

Regenerate fixtures only with `python
examples/policy-engine-interop/build_fixtures.py`. The maintained target first
uses `--check` to prove the committed inputs still derive from the signed
golden envelope.

The example omits receipt-authenticated evidence age, clock-skew allowance,
receipt trust-profile pins, countersigning restrictions, and duplicate-free,
exactly-one signer lookup. Its receipt-to-predicate checks cover verdict and
signer projections, not all artifact, schedule, policy and contract bindings.
Passing these fixtures does not establish the full portable acceptance contract.

See the [interoperability reference](../../docs/reference/policy-engine-interop.md)
for conversion and OPA/CUE commands with independent policy, subject and time
inputs. Use the [acceptance-attestation verifier](../../docs/reference/acceptance-attestations.md)
for full recipient acceptance, or `invarlock verify` to replay the complete
evidence pack.
