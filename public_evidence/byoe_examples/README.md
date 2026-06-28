# BYOE Public Evidence Examples

These fixtures demonstrate that InvarLock validates baseline-vs-subject
checkpoint comparisons regardless of which external edit workflow produced the
subject. They are intentionally BYOE examples, not new built-in edit plugins.

| Example | External edit type | Purpose |
| --- | --- | --- |
| `magnitude_prune_byoe/` | Dense magnitude pruning | Pruning evidence point for BYOE strict-evidence wiring. |
| `lora_merge_byoe/` | LoRA-merged checkpoint | Adapter-merge evidence point with optional edit provenance and edit-impact scenario labels. |
| `fine_tune_byoe/` | Fine-tuned checkpoint | Fine-tune evidence point with optional realism, topology, and delta/privacy metadata. |

Each example includes `evaluation.report.json`, `runtime.manifest.json`, and
`checkpoint_refs.json`. The reports verify under the strict release profile, and
the checkpoint references make clear that model weights are external references.

Their scope is verification wiring and artifact structure; deployable
compression, sparse runtime acceleration, and packed quantized storage are
outside these fixture examples.

The LoRA merge and fine-tune fixtures demonstrate optional descriptive metadata
for BYOE subjects. Those fields help reviewers identify the upstream edit
family, scenario labels, evaluation realism, topology, and delta/privacy
disclosure. Verifier verdicts remain the strict release-profile checks over each
report.
