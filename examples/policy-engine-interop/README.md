# Standalone policy-engine interoperability

This example feeds authenticated acceptance facts from the committed in-toto
Statement and DSSE envelope into two external policy engines:

- Open Policy Agent v1.17.0 with Rego; and
- CUE v0.16.1.

OPA and CUE do not provide a raw Ed25519 DSSE-verification primitive. The local
`verify_envelope.py` boundary therefore authenticates the DSSE envelope and
embedded signed verification receipt, checks their canonical representations
and cross-projections, and emits one JSON input. It imports no InvarLock module,
starts no service, and performs no network request. The policy engines then
apply recipient-controlled trust, subject, freshness, contract-version, and
technical-verdict rules.

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
| Stale evidence | Deny | Invalid |
| Unsupported InvarLock contract | Deny | Invalid |

Regenerate fixtures only with `python
examples/policy-engine-interop/build_fixtures.py`. The maintained target first
uses `--check` to prove the committed inputs still derive from the signed
golden envelope.

This is acceptance interoperability, not complete evidence replay. The
standalone verifier establishes envelope and receipt authenticity; OPA or CUE
applies current recipient policy to that authenticated projection. An
InvarLock verifier remains necessary when a recipient wants to replay every
evidence-pack invariant rather than consume the portable acceptance result.
