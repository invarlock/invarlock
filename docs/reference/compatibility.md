# Compatibility covenant

InvarLock treats the v0.13 technical decision contract as a permanent
verification boundary:

- **Verification support: permanent.**
- **Dossier ingestion support: permanent.**
- **Acceptance outcome: current recipient policy.**

!!! info "Reference"

    - **Surface:** v0.13 evidence packs and signed verification receipts
    - **Stability:** Original parsing, binding, replay, and decision semantics
    - **Use this page when:** Retaining historical receipts or deciding whether
      an older result is acceptable now

## What remains stable

A v0.13 receipt remains parseable and independently verifiable under its
original semantics. Verification continues to bind the exact baseline and
subject artifact identities, canonical paired schedule, per-record evidence,
metric or authorized scorer, evaluated policy, runtime identities, evidence
signer, and verifier signer. The verifier recomputes the recorded comparison
and rejects tampering or semantic contradictions.

The historical result recorded by those authenticated bytes is immutable.
Later software does not reinterpret, upgrade, or silently relabel a v0.13
evidence pack, report, or receipt as a newer contract. A new representation may
embed or reference the historical object, but it must retain the original
format identifier and bytes. Acceptance predicate v2 embeds the exact supplied
receipt-file bytes in `receipt.raw_base64`; `receipt.digest` authenticates
those decoded bytes rather than a new serialization.

Future dossier implementations may ingest v0.13 receipts as explicitly
versioned, first-class inputs. This covenant makes them **ingestible**; it does
not require a current recipient to accept them.

## What remains a current decision

A recipient applies its current policy after authenticating the historical
technical result. That decision may depend on:

- independently configured envelope-signer and receipt-verifier trust anchors;
- active or revoked signer status;
- separate envelope-age and authoritative evidence-age limits, plus clock skew;
- supported InvarLock contract versions and signature or digest algorithms;
- exact binding to the artifact supplied by the recipient; and
- the technical verdict required by the recipient.

An authentic historical pass can therefore be rejected today. This does not
change the recorded result; it distinguishes historical technical verification
from present-day acceptability.

## Release-blocking evidence

The committed v0.13 corpus lives at
`tests/fixtures/compatibility/v0.13.0/corpus.json` and points only to tracked
files under `examples/acceptance-handoff/golden/`. `make compatibility-test`
checks immutable fixture digests, original-semantic replay, signed-receipt
verification, tamper rejection, semantic inconsistency rejection, and
protection against contract relabeling.

This covenant is deliberately limited to the tested v0.13 atomic
baseline-versus-subject contract. It does not promise that every historical
algorithm or artifact will satisfy every future recipient policy.
