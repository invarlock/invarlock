# Dense magnitude-pruned BYOE subject

This directory is a small public BYOE fixture for InvarLock's release-gate claim.
It records a baseline-vs-subject report where the subject is an external checkpoint
reference, not a model produced by a built-in InvarLock edit plugin.

- External edit type: `magnitude_prune`
- Artifact class: `validation_subject_checkpoint`
- Model weights vendored: `false`
- Deployable optimized backend claim: `false`

Verify it with:

```bash
invarlock verify --profile release --assurance strict \
  public_evidence/byoe_examples/magnitude_prune_byoe/evaluation.report.json
```

The fixture proves report/verifier wiring for an external BYOE subject. It does
not claim runtime compression, packed storage, or a production edit backend.
