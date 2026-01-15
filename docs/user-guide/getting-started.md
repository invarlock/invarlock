# Getting Started

This quick guide walks through installation, environment setup, and the first automated certification loop.

## Install InvarLock

```bash
# Minimal core (no torch; CLI + config/schema tools)
pip install invarlock

# Recommended (HF adapter + evaluation stack for certify/run)
pip install "invarlock[hf]"

# Full (all extras)
pip install "invarlock[all]"
```

### Install via pipx (recommended isolation)

```bash
pipx install --python python3.12 "invarlock[hf]"
```

## Initialize Environment

```bash
conda create -n invarlock python=3.12 -y
conda activate invarlock
# Core + HF stack in this env
pip install "invarlock[hf]"
```

## Verify Installation

```bash
invarlock doctor
```

## Network Access

InvarLock blocks outbound network by default. When you need to download models or
datasets, opt in per run with `INVARLOCK_ALLOW_NETWORK=1`:

```bash
INVARLOCK_ALLOW_NETWORK=1 invarlock run -c configs/presets/causal_lm/wikitext2_512.yaml --profile ci
```

For offline use, pre‑download assets and enforce offline reads with
`HF_DATASETS_OFFLINE=1`. You can also relocate your HF cache via
`HF_HOME`/`HF_DATASETS_CACHE`.

## Run The Automation Loop

Use the prebuilt workflow to capture a baseline and execute the edit stack:

```bash
make cert-loop
```

For more hands-on examples, see the [Example Reports](example-reports.md).

See also: [Compare & Certify (BYOE)](compare-and-certify.md) for a universal
baseline→subject→certificate workflow when you already have two checkpoints.

## Fast Smoke Runs

For quick local/CI checks, enable an approximate capacity pass to shorten
dataset prep:

```bash
INVARLOCK_CAPACITY_FAST=1 invarlock run -c configs/presets/causal_lm/wikitext2_512.yaml --profile ci
```

Note: this skips full capacity/dedupe work; don’t use for release evidence.

## Compare & Certify First

```bash
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock certify \
  --source gpt2 \
  --edited /path/to/edited \
  --adapter auto \
  --profile ci \
  --preset configs/presets/causal_lm/wikitext2_512.yaml
```

Notes

- Prefer Compare & Certify (BYOE) for production. Use `--edit-config` overlays for quick smokes.

## Device Support

InvarLock defaults to `--device auto`, probing **CUDA → MPS → CPU** in that order.
All guard calculations and certificates are device-agnostic; we continuously
exercise CPU paths on Linux and macOS runners, document MPS fallbacks for
Apple Silicon, and treat CUDA as optional-but-recommended for release-tier
baselines. Native Windows is not supported; use WSL2 or a Linux container
if you need to run InvarLock from a Windows host. When in doubt:

- `invarlock doctor` reports the detected accelerators.
- Use `--device cpu` to force portability runs, or `--profile ci_cpu` to exercise the reduced-window telemetry preset.
- Keep `INVARLOCK_OMP_THREADS` >= 4 for long CPU jobs to avoid multi-hour baselines.

## Next Steps

Choose your path based on your workflow:

| I want to... | Start here |
|--------------|------------|
| Certify my own edited model (BYOE) | [Compare & Certify (BYOE)](compare-and-certify.md) |
| Understand the CLI commands | [Quickstart](quickstart.md) |
| Bring my own evaluation dataset | [Bring Your Own Data](bring-your-own-data.md) |
| See example outputs | [Example Reports](example-reports.md) |
| Understand what's in a certificate | [Reading a Certificate](reading-certificate.md) |
| Use InvarLock programmatically | [API Guide](../reference/api-guide.md) |
| Understand the safety guarantees | [Safety Case](../assurance/00-safety-case.md) |
| Set up secure production deployment | [Security Best Practices](../security/best-practices.md) |
