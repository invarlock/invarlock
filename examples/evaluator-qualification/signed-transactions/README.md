# Retained flagship evaluator transactions

This directory retains the compact release evidence for LM Evaluation Harness
and Inspect AI. Each package contains:

- the complete signed evidence pack with 102 paired records and native
  evaluator provenance;
- the independently signed verifier receipt;
- the independent evaluated policy;
- the builder-signed OCI image attestation and builder public key; and
- `transaction.json`, which records the independently supplied verification
  anchors and exact build identities.

Model snapshots, runtime images, private keys, temporary paths, and generated
HTML are deliberately excluded. The model and image identities remain bound by
the signed evidence and build attestation. HTML can be regenerated from the
evidence pack.

The signing keys used for these public fixtures are demonstration material,
not production trust roots. The packages prove the retained transaction and
support offline replay; production acceptance requires recipient-owned keys,
policies, and anchors.

The build attestations remain in the exact signed format emitted by their
recorded OCI runs. Offline verification accepts that retained format without
rewriting or re-signing it; newly executed transactions emit only the current
evaluator-named format.

## Flagship proof map

For each flagship, the map connects three complementary demonstrations: the
small upstream conformance qualification, the 102-record model-output import
replay, and the separately executed signed OCI journey. They share the reviewed
profile and evaluator-neutral contracts; the retained conformance output is not
claimed as the source of the signed evidence pack.

### Inspect AI

1. [Qualification profile](../artifacts/inspect-ai/profile.json) binds the
   maintained adapter, upstream package, runner, and dependency declaration.
2. [Retained upstream output](../artifacts/inspect-ai/upstream-output.json) is
   normalized into the [qualification export](../artifacts/inspect-ai/export.json)
   and independently checked in the
   [qualification result](../artifacts/inspect-ai/qualification-result.json).
3. The [102-record import replay](../authoritative/artifacts/inspect-ai/import-replay.json)
   demonstrates verdict-authoritative replay over retained model outputs.
4. The signed journey retains the [evidence manifest](inspect-ai/evidence/manifest.json),
   [verification receipt](inspect-ai/verification.receipt.json),
   [builder attestation](inspect-ai/build-attestation.json), and
   [transaction anchors](inspect-ai/transaction.json).
5. The [runnable integration](../../integrations/inspect-ai/README.md) produces
   the transaction, and the [deployment approval example](../../ci/README.md)
   consumes its signed receipt under separate recipient-controlled anchors.

### LM Evaluation Harness

1. [Qualification profile](../artifacts/lm-evaluation-harness/profile.json)
   binds the maintained adapter, upstream package, runner, and dependency
   declaration.
2. [Retained upstream output](../artifacts/lm-evaluation-harness/upstream-output.json)
   is normalized into the
   [qualification export](../artifacts/lm-evaluation-harness/export.json) and
   independently checked in the
   [qualification result](../artifacts/lm-evaluation-harness/qualification-result.json).
3. The [102-record import replay](../authoritative/artifacts/lm-evaluation-harness/import-replay.json)
   demonstrates verdict-authoritative replay over retained model outputs.
4. The signed journey retains the
   [evidence manifest](lm-evaluation-harness/evidence/manifest.json),
   [verification receipt](lm-evaluation-harness/verification.receipt.json),
   [builder attestation](lm-evaluation-harness/build-attestation.json), and
   [transaction anchors](lm-evaluation-harness/transaction.json).
5. The [runnable integration](../../integrations/lm-evaluation-harness/README.md)
   produces the transaction.

The [qualification matrix](../../../docs/reference/evaluator-qualification.md#qualification-matrix)
records adapter support, replay authority, and retained signed-journey maturity
as independent properties. None of these demonstrations generalizes beyond the
pinned exact-match workflow.

Run the offline verification through:

```bash
make evaluator-qualification
```

## Operational footprint

The retained transactions are compact and their independent CPU verification
does not execute either evaluator or model. This reference measurement used
seven measured runs after one warmup on arm64 macOS with Python 3.12.12:

| Transaction | Records | Evidence bytes | Complete package bytes | Verify and issue receipt, median | Render HTML, median |
| --- | ---: | ---: | ---: | ---: | ---: |
| LM Evaluation Harness | 102 | 277,079 | 283,562 | 94.0 ms | 38.1 ms |
| Inspect AI | 102 | 250,958 | 257,185 | 92.0 ms | 31.6 ms |

`Verify and issue receipt` performs complete semantic evidence replay against
independent anchors and writes a fresh Ed25519-signed receipt. `Render HTML`
authenticates the evidence and writes the self-contained report. Model
execution, evaluator execution, container construction, package installation,
and network access are not included.

Reproduce the measurement from a source checkout with:

```bash
PYTHONPATH=src python \
  examples/evaluator-qualification/measure_signed_transactions.py --runs 7
```

The byte counts are exact for the committed packages. Timings describe this
reference environment and are not a performance guarantee; compare results
only when the machine, Python version, package version, and run count are also
recorded.
