# Retained evaluator transactions

This directory retains compact, independently replayable evidence for LM
Evaluation Harness and Inspect AI. Each transaction package contains:

- a complete signed evidence pack with 400 paired records and native evaluator
  provenance;
- an independently signed verifier receipt and the evaluated policy;
- a builder-signed OCI image attestation and builder public key; and
- `transaction.json`, which records external verification anchors, the expected
  policy outcome, evaluator identity, and exact build identities.

Model snapshots, runtime images, private keys, temporary paths, and generated
HTML are excluded. Signed evidence binds the model and runtime identities, and
HTML can be regenerated offline. Fixture signing keys demonstrate the format;
production acceptance requires recipient-owned keys, policies, and anchors.

## Retained set

| Package | Evaluator | Model comparison | Purpose | Signed outcome |
| --- | --- | --- | --- | --- |
| [`qwen35-lm-evaluation-harness`](qwen35-lm-evaluation-harness/) | LM Evaluation Harness | Qwen3.5 9B Base → post-trained | Current-model flagship | Integrity valid; policy rejected |
| [`qwen35-inspect-ai`](qwen35-inspect-ai/) | Inspect AI | Qwen3.5 9B Base → post-trained | Independent flagship evaluator | Integrity valid; policy rejected |
| [`gemma4-lm-evaluation-harness`](gemma4-lm-evaluation-harness/) | LM Evaluation Harness | Gemma 4 12B IT → official unquantized QAT-Q4 | Cross-family portability | Integrity valid; policy rejected |
| [`deployment-approval-inspect-ai`](deployment-approval-inspect-ai/) | Inspect AI | Qwen3 0.6B Base → post-trained | Passing CI approval example | Policy passed |

A rejected policy outcome is not an integrity failure. The retained receipt
authenticates `integrity_ok: true`, the exact `policy_verdict`, and verification
status. Offline verification requires each package to reproduce the declared
outcome; it rejects outcome drift, bad signatures, changed records, or changed
trust anchors.

## Proof map

The generic qualification evidence remains separate from model-running signed
transactions. For each evaluator, it binds the maintained adapter and upstream
entry point, then demonstrates deterministic replay over 102 shared model
outputs:

### Inspect AI

1. [Qualification profile](../artifacts/inspect-ai/profile.json),
   [retained upstream output](../artifacts/inspect-ai/upstream-output.json),
   [qualification export](../artifacts/inspect-ai/export.json), and
   [qualification result](../artifacts/inspect-ai/qualification-result.json).
2. [Shared-output import replay](../authoritative/artifacts/inspect-ai/import-replay.json).
3. Qwen3.5 [evidence manifest](qwen35-inspect-ai/evidence/manifest.json),
   [verification receipt](qwen35-inspect-ai/verification.receipt.json),
   [builder attestation](qwen35-inspect-ai/build-attestation.json), and
   [transaction anchors](qwen35-inspect-ai/transaction.json).
4. Deployment-approval [evidence manifest](deployment-approval-inspect-ai/evidence/manifest.json),
   [verification receipt](deployment-approval-inspect-ai/verification.receipt.json),
   [builder attestation](deployment-approval-inspect-ai/build-attestation.json),
   and [transaction anchors](deployment-approval-inspect-ai/transaction.json).
5. The [runnable integration](../../integrations/inspect-ai/README.md) produces
   new transactions, and the [CI example](../../ci/README.md) consumes the
   passing receipt under separate recipient-controlled anchors.

### LM Evaluation Harness

1. [Qualification profile](../artifacts/lm-evaluation-harness/profile.json),
   [retained upstream output](../artifacts/lm-evaluation-harness/upstream-output.json),
   [qualification export](../artifacts/lm-evaluation-harness/export.json), and
   [qualification result](../artifacts/lm-evaluation-harness/qualification-result.json).
2. [Shared-output import replay](../authoritative/artifacts/lm-evaluation-harness/import-replay.json).
3. Qwen3.5 [evidence manifest](qwen35-lm-evaluation-harness/evidence/manifest.json),
   [verification receipt](qwen35-lm-evaluation-harness/verification.receipt.json),
   [builder attestation](qwen35-lm-evaluation-harness/build-attestation.json),
   and [transaction anchors](qwen35-lm-evaluation-harness/transaction.json).
4. Gemma 4 [evidence manifest](gemma4-lm-evaluation-harness/evidence/manifest.json),
   [verification receipt](gemma4-lm-evaluation-harness/verification.receipt.json),
   [builder attestation](gemma4-lm-evaluation-harness/build-attestation.json),
   and [transaction anchors](gemma4-lm-evaluation-harness/transaction.json).
5. The [runnable integration](../../integrations/lm-evaluation-harness/README.md)
   produces both current-model profiles.

The [qualification matrix](../../../docs/reference/evaluator-qualification.md#qualification-matrix)
reports adapter support, replay authority, and retained transactions as
independent properties. These demonstrations are scoped to the pinned
exact-match workflows.

## Current-model results

The Qwen3.5 transactions used the same ordered 400-record schedule. LM
Evaluation Harness and Inspect AI produced identical normalized record digests
and scores for all 400 baseline and all 400 subject records. The
[flagship comparison](flagship-comparison.json) records that agreement without
assigning an additional acceptance decision.

| Transaction | Baseline | Subject | Point change | Paired 95% interval | Width | Policy |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Qwen3.5 / LM Evaluation Harness | 55.5% | 53.0% | −2.5 pp | [−6.42, 1.43] pp | 7.85 pp | Reject |
| Qwen3.5 / Inspect AI | 55.5% | 53.0% | −2.5 pp | [−6.42, 1.43] pp | 7.85 pp | Reject |
| Gemma 4 / LM Evaluation Harness | 44.0% | 42.5% | −1.5 pp | [−3.84, 0.84] pp | 4.68 pp | Reject |

All three current-model packs passed integrity, record-count, interval-width,
and side-accuracy checks. Their conservative regression rule requires the
confidence lower bound to be at least −2 percentage points, so each authentic
transaction was correctly rejected.

Run complete offline verification with:

```bash
make evaluator-qualification
```

## Operational footprint

Independent CPU verification does not execute an evaluator or model. Reference
measurements use seven measured runs after one warmup and report exact package
sizes for the committed artifacts:

<!-- signed-transaction-costs:start -->
Measured on arm64 macOS with Python 3.12.13:

| Transaction | Evidence bytes | Package bytes | Verify and issue receipt | Render HTML |
| --- | ---: | ---: | ---: | ---: |
| Deployment approval / Inspect AI | 1,320,166 | 1,326,335 | 467.699 ms | 125.297 ms |
| Gemma 4 / LM Evaluation Harness | 1,194,453 | 1,200,972 | 478.270 ms | 68.112 ms |
| Qwen3.5 / Inspect AI | 1,163,736 | 1,169,978 | 438.673 ms | 71.554 ms |
| Qwen3.5 / LM Evaluation Harness | 1,199,597 | 1,206,116 | 256.628 ms | 47.250 ms |
<!-- signed-transaction-costs:end -->

`Verify and issue receipt` performs complete semantic evidence replay against
independent anchors and writes a fresh Ed25519-signed receipt. `Render HTML`
authenticates the evidence and writes a self-contained report. Model execution,
evaluator execution, container construction, package installation, and network
access are excluded.

Reproduce the measurement from a source checkout with:

```bash
PYTHONPATH=src python \
  examples/evaluator-qualification/measure_signed_transactions.py --runs 7
```

Timings describe the recorded environment and are not a performance guarantee;
compare them only when the machine, Python version, package version, and run
count are also recorded.
