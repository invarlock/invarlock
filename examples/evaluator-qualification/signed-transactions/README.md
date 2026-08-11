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

Run the offline verification through:

```bash
make evaluator-qualification
```
