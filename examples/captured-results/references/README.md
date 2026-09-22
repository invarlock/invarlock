# Captured comparison references

Use these examples to inspect actual model comparisons without running the
models again. Each linked page explains the task, the baseline and subject, the
recorded decision and how to check the result with an installed InvarLock package.
Offline replay needs no GPU or provider account.

For a model-change walkthrough, start with the
[Mistral likelihood rejection](mistral-7b-likelihood/README.md): it compares a base
checkpoint with its instruction-tuned counterpart on the same prose continuations.
For an existing experiment pipeline, the [Langfuse replay](langfuse/README.md)
shows how complete records carry that result into InvarLock. For a prompt change
rated against a task rubric, use the separate
[held-out extraction comparison](../../judge-measurements/references/k2-32b-luna-xhigh-heldout/README.md#task-and-proposed-change).

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
| [Harness likelihood](harness-likelihood/README.md) | 6 | Pass (same-model conformance) | Integration check using authored cases and the same tiny checkpoint on both sides |
| [Mistral 7B base to instruction-tuned](mistral-7b-likelihood/README.md) | 400 | Regression | Likelihood of the specified reference text |
| [Mistral 7B local HTTP comparison](../../hosted-service/references/mistral-7b-http/README.md) | 400 | Pass (no absolute quality floor) | HTTP integration with observed accuracy of 5% and 6% |
| [Langfuse experiment replay](langfuse/README.md) | 400 per comparison | Preserves the original HTTP pass and NLL regression | SDK handoff of the same retained facts, without new model inference |

Each reference retains its complete fixed schedule and declared scope. The
routing schedule includes adverse results. A successfully
replayed rejection means the evidence is intact and the declared policy rejects
the subject; it does not mean acceptance or runtime qualification.
