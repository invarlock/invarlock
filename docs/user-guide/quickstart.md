# InvarLock Quickstart Guide

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Complete the core evaluation workflow in a few commands. |
| **Audience** | New users running their first evaluation. |
| **Requires** | `pip install invarlock` for verify/report/proof-pack flows; add `invarlock[hf]` only for Hugging Face-backed `evaluate`. |
| **Network** | Use `--allow-network` on `evaluate` when a run needs model or dataset downloads. |
| **Next step** | [Compare & evaluate](compare-and-evaluate.md) for production use. |

This guide keeps the public front door first: `evaluate`, `verify`, and
`report html`. The default path produces a machine-readable evaluation report.
The minimal install is enough for verification, report rendering, and
proof-pack inspection. Add `invarlock[hf]` only when you want the evaluate path
to load Hugging Face models. Reach for `report generate` and `report explain`
after the core path is already green.

If any terms are unfamiliar, see the [Glossary](../assurance/glossary.md).

## Quick Start

### 1. Prepare the environment

```bash
pip install invarlock

# Optional: only for evaluate with Hugging Face-backed models
pip install "invarlock[hf]"

# Repo checkout only: build the local runtime image once for container-backed runs
make runtime-image

# Podman users can prepare the same image explicitly with Podman
make runtime-image-podman
make runtime-smoke-podman

invarlock doctor
```

### 2. Evaluate a baseline against a subject

```bash
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline gpt2 \
  --subject /path/to/edited \
  --adapter auto \
  --profile ci \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --report-out reports/eval
```

Repo presets usually ship with small YAML window counts so the same files stay
usable for local smokes. Keep using those presets, but pair them with
`--profile ci` or `--profile release` when you need balanced-tier evaluations
to meet the normal token-floor gates.

`evaluate` uses the secure-default runtime container unless you explicitly pass
`--execution-mode trusted-local` for a trusted host-side workflow. Container-backed runs emit
`reports/eval/runtime.manifest.json` next to `evaluation.report.json`. For a
trusted host-side bypass, verify the resulting report with
`invarlock verify --runtime-provenance trusted-local ...`.

Proof-pack verification works from an installed wheel and does not require a
repo checkout:

```bash
invarlock advanced proof-pack verify <pack> --strict
```

### 3. Verify the evaluation report

```bash
# Container/default evaluate output
invarlock verify reports/eval/evaluation.report.json

# Trusted-local evaluate output
invarlock verify --runtime-provenance trusted-local reports/eval/evaluation.report.json
```

The verifier re-checks schema, paired math, gate results, and the adjacent
runtime manifest before you promote results. Use the trusted-local form only
when the evaluation itself ran with `--execution-mode trusted-local`.

`invarlock report generate` and `invarlock report explain` expect canonical
`report.json` inputs. `invarlock report html` expects canonical
`evaluation.report.json`. Directory inputs are command-specific and ambiguous
directories are rejected.

### 4. Render shareable HTML

```bash
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
```

Directory inputs are command-specific: `invarlock report explain` expects a
directory containing canonical `report.json`, while `invarlock report html`
expects a directory containing canonical `evaluation.report.json`.

Optional: explain gate decisions directly from the run reports.

```bash
invarlock report explain \
  --subject-report runs/subject/report.json \
  --baseline-report runs/source/report.json
```

`invarlock report explain` expects run reports (`report.json`), not the
generated `evaluation.report.json` bundle. Use `invarlock verify` for the
paired evaluation report.

## Execution Notes

- Enable downloads per command with `--allow-network`.
- For offline reads after warming caches, use `HF_DATASETS_OFFLINE=1`.
- `--execution-mode trusted-local` is the explicit trusted-local bypass for `evaluate`.
- `verify` expects `runtime.manifest.json` for container-backed evaluation outputs.
- `--profile ci` currently expands causal-LM windows to `240/240`; `release`
  expands them to `400/400`.

## Advanced And Demo Flows

The built-in `quant_rtn` edit ships for demos and smoke tests, but the primary
onboarding path is the secure-default evaluate flow shown above.

```bash
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
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
optional backends. On Python 3.13+ stacks, `gptq` may still require a vendor
wheel or a supported older interpreter because upstream `auto-gptq` packaging
remains narrower than the core InvarLock support matrix.

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
