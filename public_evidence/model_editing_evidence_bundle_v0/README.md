# Model-Editing Evidence Bundle v0

This publishable evidence bundle groups small InvarLock evidence lanes for common
post-edit subject types. It is a curated index over existing public fixtures and
real tiny-model runs; it does not vendor model weights.

The bundle demonstrates release-evidence wiring for baseline-vs-subject
comparisons across quantization, pruning, LoRA/adapter-merge, and fine-tuned
subjects. The lane notes identify the artifact mode used by each example and
the benchmark evidence that can be paired with the verified reports when a
review needs edit-quality, locality, robustness, or safety results.

Use `manifest.json` to locate each lane's evaluation report, runtime manifest,
checkpoint references, and evidence note. Use `verification_summary.json` for
the deterministic hash inventory and release/strict verification status for the
bundle lanes.
