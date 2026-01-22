# Proof Packs

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Hardware-agnostic validation runs that bundle certificates into portable artifacts. |
| **Audience** | CI operators producing validation evidence across GPU topologies. |
| **Requires** | GPU capable of fitting selected models; HF cache or network for model download. |
| **Outputs** | Proof pack directory with certificates, reports, checksums, and optional GPG signature. |
| **Source of truth** | `scripts/proof_packs/run_suite.sh`, `scripts/proof_packs/run_pack.sh`. |

Proof packs are hardware-agnostic validation runs that bundle InvarLock certificates,
summary reports, and verification metadata into a portable artifact. They replace the
B200-specific validation harness with a suite that can run on any NVIDIA GPU topology
that can fit the selected models.

## Entrypoint Guide

| Script | Purpose | Output | Use When |
| --- | --- | --- | --- |
| `run_pack.sh` | Full proof pack: runs suite + packages artifacts | Proof pack directory with manifest + checksums | Default: distributable validation evidence |
| `run_suite.sh` | Suite execution only | Reports + certs under the run directory | Development/debugging, iterative runs |
| `verify_pack.sh` | Validate an existing proof pack | Verification status | Validating received proof packs |

## Quick Start

```bash
# RECOMMENDED: Full proof pack with verification artifacts
PACK_TUNED_EDIT_PARAMS_FILE=./scripts/proof_packs/tuned_edit_params.json \
  ./scripts/proof_packs/run_pack.sh --suite subset --net 1

# Development/debugging only (runs the suite, but does not build a proof pack)
./scripts/proof_packs/run_suite.sh --suite subset --resume

# Verify an existing proof pack
./scripts/proof_packs/verify_pack.sh --pack ./proof_pack_runs/subset_20250101_000000/proof_pack
```

Note: clean edits require tuned preset parameters. Either set
`PACK_TUNED_EDIT_PARAMS_FILE` or place the file at
`scripts/proof_packs/tuned_edit_params.json`.

## How It Works

This page focuses on running proof packs. For the internal task graph,
scheduler flow, and artifacts, see [Proof Pack Internals](proof-packs-internals.md).

## Suites

Model suites live in `scripts/proof_packs/suites.sh`. You can also override individual
models via `MODEL_1`–`MODEL_8`.

| Suite | Models | Notes |
| --- | --- | --- |
| `subset` | `mistralai/Mistral-7B-v0.1` | Single-GPU friendly |
| `full` | 7B–72B ungated models | Multi-GPU recommended |

## Network & Model Revisions

Proof packs require pinned model revisions for reproducibility:

- Use `--net 1` on the first run to preflight and pin revisions in
  `OUTPUT_DIR/state/model_revisions.json`.
- Offline runs use `--net 0` (default) and error if the cache is missing.
- The `PACK_NET` environment variable is exported as `1` or `0` to gate `HF_*_OFFLINE` settings.

## Output Layout

A suite run writes artifacts under `OUTPUT_DIR` (default: `./proof_pack_runs/<suite>_<timestamp>`):

- `reports/final_verdict.txt` + `reports/final_verdict.json`
- `analysis/determinism_repeats.json` (when `--repeats` is used)
- `*/certificates/**/evaluation.cert.json`

`run_pack.sh` copies curated artifacts into a pack directory (default
`OUTPUT_DIR/proof_pack`) and organizes them as:

- `results/final_verdict.txt` + `results/final_verdict.json`
- `results/**/determinism_repeats.json` (if present)
- `certs/<model>/<edit>/<run>/evaluation.cert.json`
- `certs/**/evaluation.html` + `certs/**/verify.json`
- `README.md`, `manifest.json`, `checksums.sha256`
- `manifest.json.asc` if GPG signing is available

## Edit Provenance Labels

Certificates record the edit algorithm used:

| Label | When to Use |
| --- | --- |
| `noop` | Baseline model with no edit applied |
| `quant_rtn`, `magnitude_prune`, etc. | Using InvarLock's built-in edit functions |
| `custom` | BYOE (Bring-Your-Own-Edit) pre-edited models |

For BYOE workflows, use `--edit-label custom` or let InvarLock infer from the model path.

## Determinism

Use `--determinism strict` to disable TF32 and cuDNN benchmarks and align with
strict InvarLock presets. `--repeats N` reruns a single edit N times and records
a drift summary in `results/determinism_repeats.json`.

## Signing & Verification

`run_pack.sh` signs `manifest.json` when `gpg` is available. To skip signing,
set `PACK_GPG_SIGN=0`. Use `verify_pack.sh` to validate checksums, signatures,
and certificate integrity.
