# Getting Started

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Install InvarLock and complete the core evaluate → verify → report flow. |
| **Audience** | New users setting up their first local or CI evaluation. |
| **Python** | 3.12+ recommended (CI uses 3.13). |
| **Install** | `pip install "invarlock[hf]"` for Hugging Face-backed evaluation. |
| **Next step** | [Quickstart](quickstart.md) for copy-paste commands. |

This guide covers installation, environment setup, and the smallest useful
InvarLock workflow: compare a baseline against a subject, verify the attested
report, and render HTML for review. The same top-level loop also underpins the
included image-text path when you use the explicit multimodal preset and
provider configuration.

## Install InvarLock

```bash
# Minimal core (no torch; CLI + schema/verification tools)
pip install invarlock

# Recommended for model-loading and evaluation workflows
pip install "invarlock[hf]"

# Full extras bundle
pip install "invarlock[all]"
```

### Install via pipx

```bash
pipx install --python python3.12 "invarlock[hf]"
```

## Initialize Environment

```bash
conda create -n invarlock python=3.12 -y
conda activate invarlock
pip install "invarlock[hf]"
```

## Verify Installation

```bash
invarlock doctor
```

## Network Access

InvarLock blocks outbound network by default. When you need to download models
or datasets, opt in per command with `INVARLOCK_ALLOW_NETWORK=1`:

```bash
INVARLOCK_ALLOW_NETWORK=1 invarlock evaluate \
  --baseline gpt2 \
  --subject distilgpt2 \
  --adapter auto \
  --profile ci
```

For offline use, pre-download assets and enforce offline reads with
`HF_DATASETS_OFFLINE=1`. You can also relocate your Hugging Face cache via
`HF_HOME` and `HF_DATASETS_CACHE`.

## First Evaluation

The default `evaluate` path is attested: model-loading steps run inside the
runtime container and emit `runtime.manifest.json` beside the evaluation
report.

```bash
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --baseline gpt2 \
  --subject /path/to/edited \
  --adapter auto \
  --profile ci \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --report-out reports/eval
```

## Verify And Render

```bash
invarlock verify reports/eval/evaluation.report.json
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
```

These commands validate the paired math, schema, and runtime attestation, then
render a shareable HTML artifact from the same report.

## Execution Modes

- `evaluate` defaults to the runtime container (`--assurance attested`).
- Use `--assurance trusted-local` only for trusted host-side workflows that intentionally
  bypass container execution.
- `verify` expects `runtime.manifest.json` next to attested evaluation reports.

## Learning Paths

| Persona | Path |
| --- | --- |
| **First-time user** | Getting Started → [Quickstart](quickstart.md) → [Compare & evaluate](compare-and-evaluate.md) |
| **Python developer** | Getting Started → [Primary Metric Smoke](primary-metric-smoke.md) → [API Guide](../reference/api-guide.md) |
| **Custom data user** | Getting Started → [Bring Your Own Data](bring-your-own-data.md) → [Config Gallery](config-gallery.md) |
| **Validation engineer** | Getting Started → [Proof Packs](proof-packs.md) → [Proof Packs Internals](proof-packs-internals.md) |
| **Security auditor** | Getting Started → [Threat Model](../security/threat-model.md) → [Best Practices](../security/best-practices.md) |

## Advanced Workflows

The simplified public CLI keeps the core path at the top level. Non-core
surfaces live under `invarlock advanced`:

- `invarlock advanced proof-pack ...`
- `invarlock advanced policy ...`
- `invarlock advanced plugins ...`
- `invarlock advanced calibrate ...`

Optional adapter and backend installs use Python extras such as
`pip install "invarlock[awq,gptq]"`; they are not managed through CLI
install or uninstall commands.

## Device Support

InvarLock defaults to `--device auto`, probing **CUDA → MPS → CPU** in that
order. All guard calculations and reports are device-agnostic; CUDA is
recommended for larger release-tier workloads, while CPU and MPS remain useful
for local smoke and portability runs.

- `invarlock doctor` reports detected accelerators.
- Use `--device cpu` to force portability runs.
- Use `--profile ci_cpu` for a reduced-window CPU preset when you need a fast
  validation lane.

## Next Steps

| I want to... | Start here |
| --- | --- |
| evaluate my own edited checkpoint workflow | [Compare & evaluate (BYOE)](compare-and-evaluate.md) |
| understand the CLI commands | [Quickstart](quickstart.md) |
| bring my own evaluation dataset | [Bring Your Own Data](bring-your-own-data.md) |
| see example outputs | [Example Reports](example-reports.md) |
| understand what's in a report | [Reading a report](reading-report.md) |
| use InvarLock programmatically | [API Guide](../reference/api-guide.md) |
| understand the assurance scope | [Assurance Case](../assurance/00-assurance-case.md) |
| set up secure production deployment | [Security Best Practices](../security/best-practices.md) |
