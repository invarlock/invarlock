# Integration Examples

This directory contains first-party, optional examples for attaching InvarLock
regression evidence to model-edit workflows owned by external tools.

Each target example should stay small enough to review in a pull request and
should point generated outputs to local ignored directories. Model weights,
large reports, and downloaded datasets belong outside the repository.

## Current Status

The shared scaffold is the only active surface in this first pass. Target
directories should land one at a time after their runnable path or compatibility
blocker is understood.

| Target | Initial status | First useful artifact |
| --- | --- | --- |
| GPTQModel | Backend validation pending | Post-GPTQ baseline-vs-subject comparison |
| torchao | Save/load boundary validation pending | Post-quantization HF-loadable subject comparison |
| LLM Compressor / vLLM | Compatibility validation pending | Compression sidecar evidence before deployment |
| PEFT | Merged-subject path has public BYOE precedent | Base-vs-merged-adapter comparison |
| Hugging Face Optimum | Target path selection pending | Model-card evidence section for a quantized subject |
| LM Evaluation Harness | Sidecar positioning pending | Task-eval artifacts beside InvarLock regression evidence |
| bitsandbytes | Optional backend/platform validation pending | BNB-loaded subject provenance and report comparison |

## Shared Assets

- `_shared/evidence-scope.md` defines the claim boundary for integration
  examples.
- `_shared/expected-artifacts.md` lists the review artifacts each runnable
  example should produce.
- `_shared/run_invarlock_compare.sh` is a reusable baseline-vs-subject wrapper
  for HF-loadable paths.

## Example Lifecycle

1. Confirm the optional backend and adapter status with `invarlock doctor` and
   `invarlock advanced plugins list --json`.
2. Create or reference a subject checkpoint from the external tool.
3. Run `invarlock evaluate` against the baseline and subject.
4. Run `invarlock verify --json` and render `evaluation.html`.
5. Record the output paths and any backend limitations in the target README.

Use target examples as public review aids before opening external issues or
docs pull requests. The target README should make the runnable status explicit:
`runnable`, `exploratory-host`, or `compatibility-investigation`.
