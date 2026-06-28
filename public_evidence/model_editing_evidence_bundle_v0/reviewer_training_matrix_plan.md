# Reviewer Training Matrix Plan

The v0 public bundle gives reviewers a compact release-evidence view across
quantization, pruning, LoRA/adapter-merge, and fine-tuned subjects. The current
public lanes mix real tiny-model runs and BYOE subject fixtures so the bundle
stays small, reproducible, and free of vendored model weights.

## Current Coverage

| Edit family | Public lane mode | Reviewer use |
| --- | --- | --- |
| Quantization | Real tiny-model run | Check release/strict report verification for a quantized subject. |
| Magnitude prune | Real tiny-model external edit run | Check release/strict report verification and signed evidence-pack packaging for an externally produced subject. |
| LoRA merge | Public BYOE subject fixture | Check optional edit provenance and edit-impact metadata for an adapter-merge subject reference. |
| Fine-tune | Public BYOE subject fixture | Check optional evaluation-realism, topology, and delta/privacy metadata for a fine-tuned subject reference. |

The evidence-pack harness also exercises generated LoRA and fine-tune
validation-subject lanes. Those lanes are useful for deterministic parity and
regression coverage. A real training matrix answers a different reviewer
question: whether the same reporting and verification flow remains clean for
subjects produced by common training/adaptation pipelines.

## Recommended Optional Matrix

Start with publishable tiny-model runs, then add larger CUDA-capable validation
runs when reviewer needs justify the extra runtime:

- PEFT LoRA train-and-merge subject, evaluated as a saved subject checkpoint.
- Full fine-tune subject, evaluated as a saved subject checkpoint.
- At least one control lane where the baseline and subject are equivalent.
- Optional larger-family lanes for reviewer-facing breadth when runtime budget
  permits.

Each lane should publish or retain:

- `evaluation.report.json` verified with the release profile and strict
  assurance.
- `runtime.manifest.json` and `checkpoint_refs.json`.
- `evidence.meta.json` when the lane is committed under `public_evidence/`.
- Hash inventory for reviewer-facing artifacts.
- Clear statement of whether weights are vendored; default to no vendored
  weights.

Training quality, locality, robustness, and safety results should be reported
only when the lane includes benchmark evidence for those questions.
