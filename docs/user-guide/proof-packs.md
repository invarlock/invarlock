# Proof Packs

Proof packs are hardware-agnostic validation runs that bundle InvarLock certificates,
summary reports, and verification metadata into a portable artifact. They replace the
B200-specific validation harness with a suite that can run on any NVIDIA GPU topology
that can fit the selected models.

## Quick Start

```bash
# Run the subset suite (offline by default)
./scripts/proof_packs/run_suite.sh --suite subset

# Run the full suite and build a proof pack
./scripts/proof_packs/run_pack.sh --suite full --net 1

# Verify an existing proof pack
./scripts/proof_packs/verify_pack.sh --pack ./proof_pack_runs/subset_20250101_000000/proof_pack
```

## How It Works

This page focuses on running proof packs. For the internal task graph,
scheduler flow, and artifacts, see [Proof Pack Internals](proof-packs-internals.md).

## Suites

Model suites live in `scripts/proof_packs/suites.sh`. You can also override individual
models via `MODEL_1`–`MODEL_8`.

| Suite | Models | Notes |
| --- | --- | --- |
| `subset` | `mistralai/Mistral-7B-v0.1`, `Qwen/Qwen2.5-14B` | Single-GPU friendly |
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
- `analysis/eval_results.csv` + `analysis/guard_sensitivity_matrix.csv`
- `*/certificates/**/evaluation.cert.json`

`run_pack.sh` copies curated artifacts into a pack directory (default
`OUTPUT_DIR/proof_pack`) and organizes them as:

- `results/final_verdict.txt` + `results/final_verdict.json`
- `results/eval_results.csv` + `results/guard_sensitivity_matrix.csv` (if present)
- `certs/<model>/<edit>/<run>/evaluation.cert.json`
- `certs/**/evaluation.html` + `certs/**/verify.json`
- `README.md`, `manifest.json`, `checksums.sha256`
- `manifest.json.asc` if GPG signing is available

## Determinism

Use `--determinism strict` to disable TF32 and cuDNN benchmarks and align with
strict InvarLock presets. `--repeats N` reruns a single edit N times and records
a drift summary in `results/determinism_repeats.json`.

## Signing & Verification

`run_pack.sh` signs `manifest.json` when `gpg` is available. To skip signing,
set `PACK_GPG_SIGN=0`. Use `verify_pack.sh` to validate checksums, signatures,
and certificate integrity.
