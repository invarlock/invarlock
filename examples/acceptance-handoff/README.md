# Offline acceptance handoff

This example executes one complete, service-free handoff:

1. an evaluation operator identifies the exact baseline and subject artifacts;
2. the evaluation operator imports authenticated per-record results, recomputes the
   paired metric, and signs the evidence pack;
3. an independent technical verifier checks the pack and signs a verification
   receipt;
4. an envelope signer wraps that receipt in an in-toto Statement and DSSE
   envelope;
5. a recipient authenticates the envelope, binds it to the artifact bytes,
   and applies its current acceptance policy.

Run it with:

```bash
make example-acceptance-handoff
```

The command uses a new temporary workspace and needs no network or running
InvarLock service. It demonstrates acceptance plus fail-closed behavior for a
stricter current policy, a wrong artifact, tampered evidence, a tampered
envelope, an unknown envelope signer, a revoked envelope signer, an unknown
receipt verifier, a stale envelope, a missing authoritative evidence timestamp,
and a contradiction between the receipt and envelope.
On success it prints the fixture decision, the rejected-scenario count, and
the exact paths to the signed evidence, verifier receipt, acceptance envelope,
scenario results, and retained workspace.

## Committed package

[`golden/`](golden/) is the compact generated package used by the
release-blocking compatibility and acceptance tests. It contains:

- the exact subject artifact;
- the current signed evidence pack and verification receipt;
- the standards-shaped in-toto/DSSE acceptance envelope, including the exact
  supplied receipt bytes;
- the envelope-signer and verifier public keys;
- the evaluated release policy, current recipient policy, and independent
  technical anchors; and
- the expected scenario results.

Regenerate it only through:

```bash
PYTHONPATH=src:. python examples/run_acceptance_handoff.py --write-golden
```

Generation refuses to overwrite an existing package. Delete or move the old
directory only when intentionally refreshing the corpus, then review the
byte-for-byte test diff.

The deterministic private keys used while generating this public fixture are
test material. They are never copied into the package and must not be used for
real artifacts. A production recipient supplies separate envelope-signer and
receipt-verifier trust registries, freshness rules, supported contract
versions, and exact artifact bytes or digest. Within each registry, every
identity/fingerprint pair must be unique: duplicate pairs invalidate the policy
regardless of order or status, and each authenticated signer must match exactly
one trust record. Because the retained receipt format has no authenticated
issuance timestamp, a recipient that requires an evidence-age limit rejects
them; a new envelope cannot renew that missing history.

Historical technical verification answers whether signed evidence still
satisfies its recorded contract. Present-day acceptability is a
separate decision made under the recipient's current policy. A stricter
recipient can therefore reject an authentic historical pass.

The envelope provides standards-shaped in-toto/DSSE transport. The sibling
[`policy-engine-interop`](../policy-engine-interop/) example authenticates this
exact envelope with a standalone verifier, then demonstrates current recipient
policy in Open Policy Agent and CUE without an InvarLock service or import.
Full evidence-pack semantic replay still uses the InvarLock verifier.
