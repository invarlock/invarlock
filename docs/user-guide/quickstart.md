# InvarLock Quickstart Guide

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Complete the core evaluation workflow in a few commands. |
| **Audience** | New users running their first evaluation. |
| **Requires** | `pip install invarlock` for verify/report/evidence-pack flows; add `invarlock[hf]` only for Hugging Face-backed `evaluate`. |
| **Network** | Use `--allow-network` on `evaluate` when a run needs model or dataset downloads. |
| **Next step** | [Compare & evaluate](compare-and-evaluate.md) for production use. |

This guide keeps the public front door first: `evaluate`, `verify`, and
`report html`. The default path produces a machine-readable evaluation report.
The minimal install is enough for verification, report rendering, and
evidence-pack inspection. Add `invarlock[hf]` only when you want the evaluate path
to load Hugging Face models. Reach for `report generate` and `report explain`
after the core path is already green.

If any terms are unfamiliar, see the [Glossary](../assurance/glossary.md).

## Quick Start

### 1. Prepare the environment

```bash
pip install invarlock

# Optional: only for evaluate with Hugging Face-backed models
pip install "invarlock[hf]"

invarlock doctor
```

The default strict evaluation path also needs a running Docker or Podman engine
that the current account can invoke. Check it with `docker info` or `podman
info`; `invarlock doctor` covers Python, dependency, and accelerator
diagnostics rather than replacing the engine check.

Wheel-only verification path:
`invarlock verify --profile release --assurance strict --baseline /path/to/baseline/report.json --policy-pack /path/to/acceptance/policy-pack.json --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST" /path/to/evaluation.report.json`,
`invarlock report html -i /path/to/evaluation.report.json -o /path/to/evaluation.html`,
and `invarlock report explain --evaluation-report /path/to/evaluation.report.json`.

### 2. Evaluate a baseline against a subject

Set the subject to a checkpoint actually produced by your external edit
pipeline. This production template intentionally does not label two unrelated
pretrained checkpoints as a baseline/edit pair.

```bash
BASELINE_CHECKPOINT=/path/to/original-checkpoint
EDITED_SUBJECT_CHECKPOINT=/path/to/checkpoint-produced-by-your-edit-pipeline

INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline "$BASELINE_CHECKPOINT" \
  --subject "$EDITED_SUBJECT_CHECKPOINT" \
  --baseline-adapter auto --subject-adapter auto \
  --profile ci \
  --assurance strict \
  --verbose \
  --report-out reports/eval
```

`evaluate` uses the runtime container by default unless you explicitly pass
`--execution-mode host` for a host-side workflow. Container-backed runs emit
`reports/eval/runtime.manifest.json` next to `evaluation.report.json`. For a
host-side bypass, verify the resulting report with
`invarlock verify --runtime-provenance host --assurance off ...`.
With `--verbose`, evaluation prints `Baseline report: ...`; the same path is
recorded at `provenance.baseline.report_path`. Its report-local `Status: PASS`
remains provisional when `assurance.verdict=pending_verifier`.

For a remote model ID, strict evaluation requires the corresponding
`--baseline-revision` or `--subject-revision` as an immutable 40–64 character
lowercase hexadecimal commit. Local checkpoint directories are bound by an
automatic content-tree digest.

Evidence-pack verification works from an installed wheel and does not require a
repo checkout:

```bash
TRUSTED_RUNTIME_IMAGE_DIGEST='sha256:REPLACE_WITH_REVIEWED_64_HEX_DIGEST'
ACCEPTANCE_POLICY_PACK='/path/to/acceptance/policy-pack.json'
invarlock advanced evidence-pack verify <pack> \
  --strict --report-assurance strict \
  --policy-pack "$ACCEPTANCE_POLICY_PACK" \
  --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST"
```

### 3. Verify the evaluation report

```bash
# Container/default evaluate output
BASELINE_RUN_REPORT=/path/to/baseline/run/report.json
invarlock verify \
  --profile ci \
  --assurance strict \
  --baseline "$BASELINE_RUN_REPORT" \
  --policy-pack "$ACCEPTANCE_POLICY_PACK" \
  --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST" \
  reports/eval/evaluation.report.json

# Host evaluate output
invarlock verify --runtime-provenance host --assurance off reports/eval/evaluation.report.json
```

The verifier re-checks schema, paired math, gate results, and the adjacent
runtime manifest before you promote results. Use the host form only
when the evaluation itself ran with `--execution-mode host`.
Obtain `ACCEPTANCE_POLICY_PACK` and `TRUSTED_RUNTIME_IMAGE_DIGEST` from reviewed
policy channels, not from the report bundle being verified. The submitted bundle cannot
make those independent trust anchors by building the pack itself or copying the
manifest digest; policy tooling serializes an authorized decision but does not
grant authorization. See [Policy packs](../reference/contracts.md#policy-packs)
and the [Runtime Provenance Guide](../security/runtime-provenance-guide.md).
Use the complete baseline run `report.json` emitted by `evaluate`; strict mode
requires it for PPL and accuracy and rejects hand-written metric fragments.

Only a strict verifier exit `0` is strict acceptance. It is distinct from the
report-local policy `PASS` emitted during evaluation.

Artifact model:

| Artifact | Produced by | Primary consumers |
| --- | --- | --- |
| `evaluation.report.json` | `invarlock evaluate`, `invarlock report generate --format report` | `invarlock verify`, `invarlock report html`, `invarlock report validate`, `invarlock report explain --evaluation-report`, `invarlock advanced runtime-verify` |
| `report.json` | Baseline/subject run directories under `runs/...` | `invarlock report generate`, `invarlock report explain --subject-report ... --baseline-report ...` |

### 4. Render shareable HTML

```bash
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
```

Optional: explain gate decisions directly from the evaluation bundle with
`invarlock report explain --evaluation-report reports/eval/evaluation.report.json`.

If you only have the run reports, the lower-level form remains:
`invarlock report explain --subject-report runs/subject/report.json --baseline-report runs/source/report.json`.

## Execution Notes

- Enable downloads per command with `--allow-network`.
- For offline reads after warming caches, use `HF_DATASETS_OFFLINE=1`.
- `--execution-mode host` is the explicit host bypass for `evaluate`.
- `verify` expects `runtime.manifest.json` for container-backed evaluation outputs.
- `--profile ci` currently expands causal-LM windows to `240/240`; `release`
  expands them to `400/400`.

## Advanced And Demo Flows

The built-in `quant_rtn` edit is a dequantized-weight synthetic edit fixture for
demos and smoke tests. It is not evidence that an external quantizer produced a
deployable quantized checkpoint; the production onboarding path is BYOE as
shown above.

```bash
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline gpt2 \
  --subject gpt2 \
  --baseline-adapter auto --subject-adapter auto \
  --profile ci \
  --execution-mode host \
  --assurance off \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --edit-config configs/overlays/edits/quant_rtn/8bit_attn.yaml \
  --report-out reports/demo
```

Advanced commands live under `invarlock advanced`:

```bash
invarlock advanced plugins list
invarlock advanced evidence-pack verify <pack> \
  --strict --report-assurance strict \
  --policy-pack "$ACCEPTANCE_POLICY_PACK" \
  --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST"
invarlock advanced policy --help
invarlock advanced calibrate --help
```

Use Python extras such as `pip install "invarlock[awq,gptq]"` when you need
optional backends. The `awq` and `gptq` extras use GPTQModel-backed subject
loading.

## Repo Maintainer Path

If you are working from a repository checkout and want the local image-backed
smoke flows, build the runtime image after the basic front door works:

`make runtime-image`, `make container-default-smoke`, and
`make container-front-door-smoke`.

Podman users can prepare the same image explicitly with:
`make runtime-image-podman` and `make runtime-smoke-podman`.

## Core Concepts

### Workflow

- **Evaluate**: compare baseline and subject with deterministic pairing
- **Verify**: fail closed on malformed or missing-provenance evaluation outputs
- **Report**: render HTML or explain gate decisions from existing artifacts

### Guards

- **Invariants**: verify structural properties are preserved
- **Spectral**: check spectral norm bounds for stability
- **Variance**: monitor activation variance shifts
- **RMT**: apply random-matrix-theory-based validation

### Devices

`--device auto` probes CUDA, then MPS, then CPU. Override it explicitly when
validating portability or troubleshooting accelerator issues.

## Next Steps

- See [CLI Reference](../reference/cli.md) for command details
- Read [Compare & evaluate](compare-and-evaluate.md) for the primary production workflow
- Review [reports](../reference/reports.md) for schema and validation details
- See [Reading a report](reading-report.md) for interpretation guidance
- Read the [Device Support note](getting-started.md#device-support) for CPU and Apple Silicon guidance

> Note: presets under `configs/` are repo-first assets. When using a wheel-only
> install, prefer direct flags instead of preset paths unless you also cloned
> the repository.
