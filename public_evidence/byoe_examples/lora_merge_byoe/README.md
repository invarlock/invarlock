# LoRA-merged BYOE subject

This directory is a small public BYOE fixture for InvarLock's strict-evidence
workflow.
It records a baseline-vs-subject report where the subject is an external
checkpoint reference; built-in InvarLock edit plugins are outside this fixture's
scope.

- External edit type: `lora_merge`
- Artifact class: `validation_subject_checkpoint`
- Model weights vendored: `false`
- Deployable optimized backend claim: `false`

Verify it with:

```bash
invarlock verify --profile release --assurance strict \
  public_evidence/byoe_examples/lora_merge_byoe/evaluation.report.json
```

The fixture proves report/verifier wiring for an external BYOE subject. Its
scope is artifact verification for the edited checkpoint reference; runtime
compression, packed storage, and production edit-backend behavior are outside
this fixture.
