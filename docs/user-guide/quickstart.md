# InvarLock Quickstart Guide

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Complete the core evaluation workflow in a few commands. |
| **Audience** | New users running their first evaluation. |
| **Requires** | `invarlock[hf]` for Hugging Face-backed evaluation. |
| **Network** | `INVARLOCK_ALLOW_NETWORK=1` for model and dataset downloads. |
| **Next step** | [Compare & evaluate](compare-and-evaluate.md) for production use. |

This guide focuses on the public core CLI: `evaluate`, `verify`, and
`report html`. The default path is baseline versus subject evaluation with a
machine-readable evaluation report, deterministic pairing, and an attested
runtime manifest.

If any terms are unfamiliar, see the [Glossary](../assurance/glossary.md).

## Quick Start

### 1. Prepare the environment

```bash
pip install "invarlock[hf]"

# Repo checkout only: build the local runtime image once for attested runs
make runtime-image

invarlock doctor
```

### 2. Evaluate a baseline against a subject

```bash
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --baseline gpt2 \
  --subject /path/to/edited \
  --adapter auto \
  --profile ci \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --report-out reports/eval
```

`evaluate` uses the secure-default runtime container unless you explicitly pass
`--mode local` for a trusted host-side workflow. Attested runs emit
`reports/eval/runtime.manifest.json` next to `evaluation.report.json`.

### 3. Verify the evaluation report

```bash
invarlock verify reports/eval/evaluation.report.json
```

The verifier re-checks schema, paired math, gate results, and the adjacent
runtime manifest before you promote results.

### 4. Render shareable HTML

```bash
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
```

Directory inputs to `invarlock report` are only accepted when they contain
canonical `report.json` or `evaluation.report.json`; otherwise pass the exact
file path.

Optional: explain gate decisions directly from the evaluation artifacts.

```bash
invarlock report explain --report runs/subject/report.json --baseline runs/source/report.json
```

## Execution Notes

- Enable downloads per command with `INVARLOCK_ALLOW_NETWORK=1`.
- For offline reads after warming caches, use `HF_DATASETS_OFFLINE=1`.
- `--mode local` is the explicit trusted-host bypass for `evaluate`.
- `verify` expects `runtime.manifest.json` for attested evaluation outputs.

## Advanced And Demo Flows

The built-in `quant_rtn` edit still ships for demos and smoke tests, but it is
no longer the primary onboarding path.

```bash
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --baseline gpt2 \
  --subject gpt2 \
  --adapter auto \
  --profile ci \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --edit-config configs/overlays/edits/quant_rtn/8bit_attn.yaml \
  --report-out reports/demo
```

Advanced commands live under `invarlock advanced`:

```bash
invarlock advanced plugins list
invarlock advanced proof-pack verify <pack> --strict
invarlock advanced policy --help
invarlock advanced calibrate --help
```

Use Python extras such as `pip install "invarlock[awq,gptq]"` when you need
optional backends.

## Core Concepts

### Workflow

- **Evaluate**: compare baseline and subject with deterministic pairing
- **Verify**: fail closed on malformed or unattested evaluation outputs
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
