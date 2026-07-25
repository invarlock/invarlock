# Offline acceptance handoff

This example executes one complete, service-free handoff:

1. a producer creates exact baseline and subject artifacts;
2. the producer imports authenticated per-record results, recomputes the
   paired metric, and signs the evidence pack;
3. an independent technical verifier checks the pack and signs a v0.13
   receipt;
4. the producer wraps that receipt in an in-toto Statement and DSSE envelope;
5. a recipient authenticates the envelope, binds it to the artifact bytes,
   and applies its current acceptance policy.

Run it with:

```bash
make example-acceptance-handoff
```

The command uses a new temporary workspace and needs no network or running
InvarLock service. It demonstrates acceptance plus fail-closed behavior for a
stricter current policy, a wrong artifact, tampered evidence, a tampered
envelope, an unknown signer, a revoked signer, stale evidence, and a
contradiction between the receipt and envelope.

## Committed package

[`golden/`](golden/) is the compact producer-generated package used by the
release-blocking compatibility and acceptance tests. It contains:

- the exact subject artifact;
- the signed evidence pack and v0.13 verification receipt;
- the in-toto/DSSE acceptance envelope;
- the producer and verifier public keys;
- the evaluated producer policy, current recipient policy, and independent
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
real artifacts. A production recipient supplies its own trust registry,
freshness rule, supported contract versions, and exact artifact bytes or
digest.

Historical technical verification answers whether the signed v0.13 evidence
still satisfies its recorded contract. Present-day acceptability is a
separate decision made under the recipient's current policy. A stricter
recipient can therefore reject an authentic historical pass.

Existing attestation policy engines can authenticate and policy-evaluate an
InvarLock acceptance attestation without a custom InvarLock service or plugin;
full semantic replay uses the InvarLock verifier.
