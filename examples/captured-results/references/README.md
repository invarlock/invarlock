# Captured comparison references

These references retain signed `invarlock/evidence-pack-v2` captured comparisons,
independent trust inputs and signed verification receipts. They support offline
recipient replay with the installed CLI. Their captured provenance is separate
from the native runtime transactions in the [public index](../../../public_evidence/README.md).

| Reference | Paired records | Recorded decision | Scoring assurance |
| --- | ---: | --- | --- |
| [K2 Horizon 32B routing prompts](k2-32b-routing/README.md) | 4,000 | Regression | Recorded external scores |
| [Harness likelihood](harness-likelihood/README.md) | 6 | Pass (same-model conformance) | Real reference-continuation likelihoods; captured provenance |
| [Mistral 7B base to instruction-tuned](mistral-7b-likelihood/README.md) | 400 | Regression | Real reference-continuation likelihoods; captured provenance |
| [Mistral 7B local HTTP comparison](../../hosted-service/references/mistral-7b-http/README.md) | 400 | Pass (no absolute quality floor) | Real HTTP responses with literal exact match; captured service provenance |

Each reference retains its complete fixed schedule and declared scope. The
routing schedule includes adverse results. A successfully
replayed rejection means the evidence is intact and the declared policy rejects
the subject; it does not mean acceptance or runtime qualification.
