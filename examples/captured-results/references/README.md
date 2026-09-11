# Captured comparison references

These references retain signed `invarlock/evidence-pack-v2` captured comparisons,
independent trust inputs and signed verification receipts. They support offline
recipient replay with the installed CLI. Their recorded-score assurance is
separate from the native runtime transactions in the [public index](../../../public_evidence/README.md).

| Reference | Paired records | Recorded decision | Scoring assurance |
| --- | ---: | --- | --- |
| [K2 Horizon 32B routing prompts](k2-32b-routing/README.md) | 4,000 | Regression | Recorded external scores |

The full finite schedule is retained, including adverse results. A successfully
replayed rejection means the evidence is intact and the declared policy rejects
the subject; it does not mean acceptance or runtime qualification.
