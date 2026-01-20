# InvarLock Quickstart Guide

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Get started with InvarLock certification in minutes. |
| **Audience** | New users running their first certification. |
| **Requires** | `invarlock[hf]` for HF adapter workflows. |
| **Network** | `INVARLOCK_ALLOW_NETWORK=1` for model/dataset downloads. |
| **Next step** | [Compare & Certify](compare-and-certify.md) for production workflows. |

This guide helps you get started with InvarLock (Edit‑agnostic robustness certificates for weight edits)
quickly. Every run flows through the **GuardChain**
(invariants → spectral → RMT → variance) and produces a machine-readable safety
certificate with drift, guard-overhead, and policy digests.
If any terms are unfamiliar, see the [Glossary](../assurance/glossary.md).

Note: For installation and environment setup, see Getting Started. This page focuses on core commands and workflow.

Tip: Enable downloads per command when fetching models/datasets:
`INVARLOCK_ALLOW_NETWORK=1 invarlock certify ...`
For offline reads after warming caches: `HF_DATASETS_OFFLINE=1`.

Adapter‑based commands shown below (for example, `invarlock run` on HF
checkpoints or `invarlock certify` with `--adapter auto`) assume you have
installed an appropriate extra such as `invarlock[hf]` or `invarlock[adapters]`.

## Quick Start

### 1. List Available Plugins

```bash
# List all plugins
invarlock plugins

# List specific categories
invarlock plugins edits
invarlock plugins guards
invarlock plugins adapters
```

See [Plugin Workflow](plugins.md) for extending adapters and guards, or use Compare & Certify (BYOE) when you
already have two checkpoints.

**Safety tip:** After any run that produces a certificate, execute
`invarlock verify reports/cert/evaluation.cert.json`. The verifier re-checks paired
log‑space math, guard‑overhead (<= 1%), drift gates, and schema compliance before
you promote results.

### 2. Run a Simple Edit or Compare & Certify

Use the built‑in RTN quantization preset (demo), or prefer Compare & Certify (BYOE):

```bash
# RTN quantization (smoke, demo edit overlay)
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock certify \
  --baseline gpt2 \
  --subject gpt2 \
  --adapter auto \
  --profile ci \
  --tier balanced \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --edit-config configs/overlays/edits/quant_rtn/8bit_attn.yaml

# Compare & Certify (recommended)
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock certify \
  --baseline gpt2 \
  --subject /path/to/edited \
  --adapter auto \
  --profile ci \
  --preset configs/presets/causal_lm/wikitext2_512.yaml

# Explain decisions and render HTML (includes Primary Metric Tail gate details)
invarlock report explain --report runs/edited/report.json --baseline runs/source/report.json
invarlock report html -i reports/cert/evaluation.cert.json -o reports/cert/evaluation.html
```

### 3. Generate Reports

```bash
# Generate JSON report
invarlock report --run runs/20240118_143022 --format json

# Generate all formats
invarlock report --run runs/20240118_143022 --format all

# Generate safety certificate (requires baseline)
invarlock report --run runs/20240118_143022 --format cert --baseline runs/baseline
```

## Core Concepts

### Edits

- **RTN Quantization** (built‑in, demo): Reduce precision using
  Round‑To‑Nearest quantization
- **Compare & Certify (BYOE)** (recommended): Provide baseline + subject checkpoints and certify

### Guards

- **Invariants**: Verify structural properties are preserved
- **Spectral**: Check spectral norm bounds for stability
- **Variance**: Monitor activation variance changes
- **RMT**: Random Matrix Theory-based validation
- **Guard Overhead**: Comparison against the bare baseline to ensure the
  GuardChain adds <= 1% perplexity overhead (captured under
  `validation.guard_overhead_*` in certificates)

### Adapters

- **HF GPT-2**: HuggingFace GPT-2 model support
- Extensible to other architectures via plugin system

## Configuration (quant_rtn example)

Create a YAML configuration file:

```yaml
model:
  id: "gpt2"
  adapter: "hf_causal"
  device: "auto"  # mirrors the CLI default (--device auto)

dataset:
  provider: "wikitext2"
  seq_len: 128

edit:
  name: "quant_rtn"
  plan:
    bitwidth: 8
    per_channel: true
    group_size: 128
    clamp_ratio: 0.005

guards:
  order: ["invariants", "spectral"]
```

By default `invarlock run` uses `--device auto`, which selects CUDA, then Apple
Silicon (MPS), then CPU. Override it explicitly (`--device cpu`, `--device mps`,
etc.) when validating portability or troubleshooting driver issues.

## Next Steps

- See [CLI Reference](../reference/cli.md) for detailed command options
- Check [Configuration Schema](../reference/config-schema.md) for all config options
- Review [Certificates](../reference/certificates.md) for schema and validation details
- See [Reading a Certificate](reading-certificate.md) for guidance
- Read the [Device Support note](getting-started.md#device-support) if you plan to run on CPU or Apple Silicon
- Learn about [Guard Contracts](../assurance/04-guard-contracts.md) for guard behavior details

> Note: presets and the tiny-matrix script are repo-first assets (not shipped in wheels)
> Clone the repository if you want to reference presets under `configs/` or use the matrix script
> Otherwise, pass flags directly (no preset) for CLI-only flows
