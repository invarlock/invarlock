# Local HTTP comparison reference

This reference retains a real HTTP capture and offline verification journey for
two distinct full 7B checkpoints. It demonstrates the captured-service workflow,
not qualification of an external provider or a production workload.

Use it when connecting an HTTP model endpoint to evaluation and recipient
verification. The actual change is from the base Mistral checkpoint to its
instruction-tuned counterpart behind the same requested service alias. The
result supports checking that the declared capture, pairing and verification
workflow works; the low task scores do not justify deploying either service.
Start with the retained [report](current/report.md), then follow
[offline reproduction](#offline-reproduction-and-limits). For a substantive
likelihood regression on the same checkpoint change, use the separate
[Mistral NLL reference](../../../captured-results/references/mistral-7b-likelihood/README.md).

## Result and interpretation

The corrected campaign made 400 requests to each deployment. All 800 requests
returned HTTP 200, with no transport errors or missing measurements. A separate
installed core environment reconstructed the captured runs, verified the signed
evidence against independently prepared inputs, and issued a signed receipt.

| Measurement | Baseline | Candidate |
| --- | --- | --- |
| Checkpoint | Mistral-7B-v0.1 | Mistral-7B-Instruct-v0.1 |
| Literal first-whitespace-field exact matches | 20 of 400 | 24 of 400 |
| Observed score | 5% | 6% |

The observed paired change is +1 percentage point. Its paired 95% confidence
interval is approximately [-1.444, 3.526] percentage points. The declared policy
permits a regression of up to 2 percentage points, requires 400 pairs, and limits
interval width to 10 percentage points. Those comparative requirements pass.
There is no absolute quality floor in this demonstration policy. Its 2-point
tolerance and 10-point precision limit illustrate comparative checks, not a
service-specific error budget. Low absolute scores remain visible and do not
establish adequate task quality.

The service generates eight greedy tokens and returns the first whitespace-delimited
field of its decoded continuation. Attached punctuation is preserved: `insurance`
and `insurance?"` are different outputs. The scorer compares that literal string
with the declared reference. **These scores are not standard LAMBADA benchmark
accuracy.** No projection, threshold or case selection was changed after observing
the corrected campaign results.

## Data and execution

The campaign reuses the existing 400 public LAMBADA passages and case IDs from
`examples/integrations/evaluator_transaction/lambada_qwen35_deployment_400.jsonl`.
Its source bytes have SHA-256
`e4a0e431b8b64130cbbf6e8fb3ed7b5769744d18ca6499d2088f2e1b3fb36dda`.
Inputs and order are unchanged; reference strings use the declared whitespace-strip
projection. The earlier subset's eligibility and length selection remain part of
its limitations. It is not a representative production sample or a general model
ranking dataset. Preserve the upstream dataset provenance and terms when sharing.

Both checkpoints are full float16 local models, served sequentially through a
loopback HTTP endpoint with an unchanged requested alias and decoding configuration.
The response reports the observed model ID but exposes no revision. The complete
checkpoint inventory and loading protocol are retained separately. The captured
identity uses `artifact_digest: null`: its service descriptor identifies the
declared observation, not hidden model weights. This locally controlled model
inventory does not establish external-provider identity assurance.

The checkpoint-loading protocol comes from the separate likelihood study. Only
its model inventory and loader are reused here; the service source explicitly
overrides the MPS allocation fraction to 0.60. Its likelihood cases and policy
are not this HTTP campaign's cases or policy. `capture/protocol.json` defines
the HTTP campaign, and `capture/runtime-environment.json` records a later
inventory of the unchanged model environment for reproduction.

The baseline window precedes the candidate window. Each request has its literal
request, raw response, extracted value, token use and timing retained. The policy
and collector sources were pinned before the corrected calls. The protocol sets
per-deployment bounds of 400 calls, 3,200 output tokens, 30 seconds per request,
16 KiB per response and two hours per observation window.

## Retained files

- `capture/` contains the corrected protocol, both complete captures, and exact
  collector, journey, service and checkpoint-loading source material.
- `evidence/` is the original signed pack. Its bytes and original decision are
  preserved.
- `verification.receipt.json` and the public keys retain the original separate
  recipient verification result. No private signing keys are included.
- `report.html`, `report.md` and `report.xml` retain the original rendering.
- `current/` contains a later offline verification and presentation-only rendering
  from the source identified in `reference.json`, using the same immutable pack
  and no new model calls.
- `reference.json` inventories exact bytes, source/wheel identity, independent
  run/request expectations and the scope of each validation.

The wheel metadata remains `0.15.0` in this development tree. The recorded source
commit and wheel digest identify the implementation used here; these capabilities
must not be attributed to the published v0.15.0 release.

## Earlier attempts

Three startup attempts admitted no model requests: one lacked MPS access, and two
exceeded the initial allocation cap during model loading. Their original logs are
retained in the local campaign workspace and identified by hashes in the inventory.

An earlier 400-call baseline-only capture is retained in `earlier-attempt/` with
its original protocol and executed helper sources. A collector audit found admission,
credential-handling, timing and revision-observation issues before candidate capture.
The corrected helper and protocol were frozen, and both sides were collected under
that corrected campaign. The old attempt has no candidate or comparison result and
was not selected or discarded based on a comparison outcome. Its baseline generated
token sequences match the corrected baseline. The separate likelihood reference's
adverse result remains unchanged.

## Offline reproduction and limits

Build the recorded source into a wheel and install that exact wheel into two fresh
environments. Follow `examples/hosted-service/README.md` to run the retained journey
with `capture/protocol.json`, `capture/baseline.json` and `capture/subject.json`, using
their physical SHA-256 pins from an independently approved copy of `reference.json`.
Use a new output directory. Export, evaluation, verification and reporting require
neither the model server nor provider credentials. Do not execute the retained
service source merely to inspect or verify this evidence.

A repeated journey creates new example signer keys and therefore a new signed pack;
its normalized inputs and analysis should match the retained result. To verify the
original pack, use its separately approved signer fingerprint, baseline/subject run
digests, request digest and original policy. The documentation's local example
key exchange does not establish organizational independence or operational signer
enrollment.

Verification checks these retained observations. It does not recheck the running
service, independently attest execution or authorize deployment. This reference
does not qualify NLL over this HTTP endpoint, judge quality, agent trajectories,
continuous alerting or any external hosted provider.
