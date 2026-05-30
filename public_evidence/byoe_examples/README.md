# BYOE Public Evidence Examples

These fixtures demonstrate that InvarLock validates baseline-vs-subject
checkpoint comparisons regardless of which external edit workflow produced the
subject. They are intentionally BYOE examples, not new built-in edit plugins.

| Example | External edit type | Purpose |
| --- | --- | --- |
| `magnitude_prune_byoe/` | Dense magnitude pruning | Pruning proof point for BYOE release-gate wording. |
| `lora_merge_byoe/` | LoRA merge / fine-tune-derived checkpoint | Adapter-merge proof point for fine-tuned/BYOE wording. |

Each example includes `evaluation.report.json`, `runtime.manifest.json`, and
`checkpoint_refs.json`. The reports verify under the strict release profile, and
the checkpoint references make clear that model weights are external references.

These fixtures do not claim deployable compression, sparse runtime acceleration,
or packed quantized storage.
