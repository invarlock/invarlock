# Captured comparison references

Use these examples to inspect actual model comparisons without running the
models again. Each linked page explains the task, the baseline and subject, the
recorded decision and how to check the result with an installed InvarLock package.
Offline replay needs no GPU or provider account.

A **pair** is a baseline and subject measurement for the same task. A
**regression** result means the subject did not meet that comparison's declared
policy; it is not a broken example.

These references retain signed `invarlock/evidence-pack-v2` captured comparisons,
independent trust inputs and signed verification receipts. They support offline
recipient replay with the installed CLI. Their captured provenance is separate
from the native runtime transactions in the [public index](../../../public_evidence/README.md).

| Reference | Paired records | Recorded decision | What was measured |
| --- | ---: | --- | --- |
| [K2 Horizon 32B routing prompts](k2-32b-routing/README.md) | 4,000 | Regression | Task scores supplied by the original routing evaluator |
| [Harness likelihood](harness-likelihood/README.md) | 6 | Pass (same-model conformance) | Likelihood of the specified reference text |
| [Mistral 7B base to instruction-tuned](mistral-7b-likelihood/README.md) | 400 | Regression | Likelihood of the specified reference text |
| [Mistral 7B local HTTP comparison](../../hosted-service/references/mistral-7b-http/README.md) | 400 | Pass (no absolute quality floor) | Literal exact match of answers returned over HTTP |

Each reference retains its complete fixed schedule and declared scope. The
routing schedule includes adverse results. A successfully
replayed rejection means the evidence is intact and the declared policy rejects
the subject; it does not mean acceptance or runtime qualification.
