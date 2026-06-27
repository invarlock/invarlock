# Fine-tuned BYOE subject

This directory is a small public BYOE fixture for InvarLock's strict-evidence
workflow. It records a baseline-vs-subject report where the subject is an
external fine-tuned checkpoint reference; the training recipe and weights remain
outside the fixture.

- External edit type: `fine_tune`
- Artifact class: `validation_subject_checkpoint`
- Model weights vendored: `false`
- Deployable optimized backend claim: `false`
- Edit provenance: descriptive BYOE metadata
- Evaluation realism: teacher-forced log-prob proxy, not live generation
- Delta availability: `hash_only`

Verify it with:

```bash
invarlock verify --profile release --assurance strict \
  public_evidence/byoe_examples/fine_tune_byoe/evaluation.report.json
```

The fixture validates report/verifier wiring for an external BYOE subject. Edit
success, fine-tuning quality, model safety, locality, and robustness require
separate benchmark evidence.
