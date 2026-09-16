# Standalone policy-engine interoperability

Use this example when you want an existing policy engine to consume a signed
model-evaluation handoff. A small Python program checks the signatures and
produces JSON; OPA or CUE then decides whether that JSON satisfies the example's
policy rules. No model execution or InvarLock service is needed.

> **Outcome:** Run six fixtures through OPA and CUE and see one allowed input
> alongside five rejected inputs under the example policy.
>
> **Audience:** Developers integrating evaluation evidence into policy tooling.
>
> **Prerequisites:** The matching checkout, Python with `cryptography` installed,
> Make, and the pinned OPA/CUE executables below. Building those executables also
> requires a compatible Go installation and network access.

The example feeds authenticated facts from the committed in-toto Statement and
DSSE envelope into two external policy engines:

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

## Run the fixtures

From the repository root, install the pinned tools if they are not already
available, then run the fixture checks:

```bash
gopath="$(go env GOPATH)"
go install github.com/open-policy-agent/opa@v1.17.0
go install cuelang.org/go/cmd/cue@v0.16.1
make acceptance-policy-interop \
  OPA="${gopath}/bin/opa" \
  CUE="${gopath}/bin/cue"
```

The Make target first checks that the JSON fixtures match their signed source,
then runs both engines. Expect one line per fixture, including
`positive: opa=true cue=true`. All five negative fixtures print `false` for
both engines; that is the expected successful result. The fixture run is
offline once its tools and Python dependency are installed.

The six fixtures are:

| Fixture | OPA | CUE |
| --- | --- | --- |
| Positive authenticated delivery | Allow | Valid |
| Recipient policy rejection | Deny | Invalid |
| Tampered subject | Deny | Invalid |
| Untrusted envelope signer | Deny | Invalid |
| Stale envelope (`stale-evidence` fixture) | Deny | Invalid |
| Unsupported InvarLock contract | Deny | Invalid |

## Understand the files and scope

`verify_envelope.py` produces the authenticated JSON input. The files under
`policy/` implement the OPA and CUE rules, `fixtures/` contains their inputs,
and `run.py` compares both engines' results with `fixtures/expectations.json`.
To check your own envelope, follow the conversion commands in the
[interoperability reference](../../docs/reference/policy-engine-interop.md#convert-and-evaluate-an-envelope)
and supply your own independently approved policy, subject identity and time.

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
