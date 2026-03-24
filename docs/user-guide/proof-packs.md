# Proof Packs

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Hardware-agnostic validation runs that bundle reports into portable evidence artifacts. |
| **Audience** | CI operators producing validation evidence across GPU topologies. |
| **Requires** | Active repo environment, GPU capable of fitting selected models, and HF cache or network for model download. Secure-default runs also require an OCI container engine. |
| **Outputs** | Proof pack directory with reports, checksums, and optional GPG signature. |
| **Source of truth** | `scripts/proof_packs/run_suite.sh`, `scripts/proof_packs/run_pack.sh`. |

Proof packs are hardware-agnostic validation runs that bundle InvarLock reports,
summary reports, and verification metadata into a portable evidence artifact. They replace the
B200-specific validation harness with a suite that can run on any NVIDIA GPU topology
that can fit the selected models.

By default, a proof pack is integrity-checked and report-verified. Treat it as
proof-grade only when the manifest is signed, the pack is verified in strict
verification mode, the bundled clean reports retain their `runtime.manifest.json`
attestation, and the final verdict is PASS.

Operationally, proof packs are a maintainer smoke test that also emits reusable
evidence data. The same run should let maintainers catch regressions, let third parties
verify reported outcomes, and provide structured outputs for downstream analysis.

> Terminology: the proof-pack suite includes a run-scoped **Preset Derivation**
> phase (`CALIBRATION_RUN -> GENERATE_PRESET`) that writes
> `calibrated_preset_<model>.yaml/json` for that suite run. It does not directly
> modify global `runtime/tiers.yaml`. For global tier policy tuning, use
> `invarlock calibrate ...` (see [Tier Policy Tuning CLI](../reference/calibration.md)).
> Calibration entrypoints still use the secure-default runtime container unless
> a trusted local workflow opts into `--allow-host-execution`.

## Entrypoint Guide

| Script | Purpose | Output | Use When |
| --- | --- | --- | --- |
| `run_pack.sh` | Full proof pack: runs suite + packages artifacts | Proof pack directory with manifest + checksums | Default: distributable validation evidence |
| `run_suite.sh` | Suite execution only | Reports under the run directory | Development/debugging, iterative runs |
| `verify_pack.sh` | Validate an existing proof pack | Verification status | Validating received proof packs |
| `invarlock proof-pack inspect` | Read-only proof-pack summary | Manifest/integrity/report inventory summary | Auditing a received pack without nested report verification |
| `invarlock proof-pack build` | Assemble a proof pack from existing artifacts | Proof pack directory with manifest + checksums | Packaging already-produced verdicts, metadata, and reports |
| `invarlock proof-pack verify` | Package-native proof-pack verification | Verification status + optional JSON | Validating received proof packs from a wheel install |

## Quick Start

```bash
# In a repo checkout, install the CLI into the active environment once.
make dev-install

# Secure-default proof-pack runs call `invarlock run` / `evaluate` under the
# runtime container. Build it once per checkout.
make runtime-image

# RECOMMENDED: Full proof pack with verification artifacts
INVARLOCK_ALLOW_REMOTE_CODE=1 \
PACK_TUNED_EDIT_PARAMS_FILE=./scripts/proof_packs/tuned_edit_params.json \
  ./scripts/proof_packs/run_pack.sh --suite subset --net 1

# Trusted local host workflow (skips the attested container path)
INVARLOCK_ALLOW_REMOTE_CODE=1 \
INVARLOCK_ALLOW_HOST_EXECUTION=1 \
PACK_TUNED_EDIT_PARAMS_FILE=./scripts/proof_packs/tuned_edit_params.json \
  ./scripts/proof_packs/run_pack.sh --suite subset --net 1

# Development/debugging only (runs the suite, but does not build a proof pack)
INVARLOCK_ALLOW_REMOTE_CODE=1 \
  ./scripts/proof_packs/run_suite.sh --suite subset --resume

# Inspect a received proof pack without nested report verification
invarlock proof-pack inspect ./proof_pack_runs/subset_20250101_000000/proof_pack --json

# Build a proof pack from existing artifacts
invarlock proof-pack build ./tmp/proof_pack \
  --final-verdict ./reports/final_verdict.json \
  --source-repo ./metadata/source_repo.json \
  --environment ./metadata/environment.json \
  --material model_revisions=./metadata/model_revisions.json \
  --report ./runs/model/evaluation.report.json

# Verify an existing proof pack
invarlock proof-pack verify ./proof_pack_runs/subset_20250101_000000/proof_pack --strict
```

Note: clean edits require tuned preset parameters. Either set
`PACK_TUNED_EDIT_PARAMS_FILE` or place the file at
`scripts/proof_packs/tuned_edit_params.json`.

The proof-pack shell wrappers do not expose a top-level
`--allow-host-execution` flag. For trusted local host execution, set
`INVARLOCK_ALLOW_HOST_EXECUTION=1` in the environment before calling
`run_pack.sh` or `run_suite.sh`. Otherwise, the underlying model-loading
commands use the secure-default runtime container path and expect `docker` or
`podman`, plus a locally built `invarlock-runtime:local` image from
`make runtime-image`.

Validated secure-default parity contract for proof-pack wrappers:

- Wrapper-provided `INVARLOCK_CONFIG_ROOT` and `INVARLOCK_STORE_EVAL_WINDOWS`
  now survive delegated `invarlock run` / `evaluate` calls.
- External cache and temp overrides such as `HF_HOME`, `HF_HUB_CACHE`,
  `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, `TMPDIR`, and `TMP` remain visible
  inside the runtime container.
- Staged or external `--preset`, `--baseline-report`, and `--edit-config`
  inputs are mounted automatically before delegation.
- Configs that rely on `!include` outside the config directory must set
  `INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE=1`; otherwise the proof-pack wrapper
  now fails before launch instead of starting an unusable container job.

Bulk proof-pack entrypoints now default to `SKIP_FLASH_ATTN=true` and
`PACK_BASELINE_STORAGE_MODE=snapshot_copy`. That is the safe default for remote
secure-default runs. Only opt back into flash-attn builds or
`snapshot_symlink` baselines when you intentionally want the extra complexity.

## How It Works

This page focuses on running proof packs. For the internal task graph,
scheduler flow, and artifacts, see [Proof Pack Internals](proof-packs-internals.md).

## Suites

Model suites live in `scripts/proof_packs/suites.sh`. You can also override individual
models via `MODEL_1`–`MODEL_8`.

| Suite | Models | Notes |
| --- | --- | --- |
| `subset` | `mistralai/Mistral-7B-v0.1` | Single-GPU friendly |
| `showcase` | 7B–14B ungated models | Multi-GPU recommended; adds guard-focused scenarios |
| `workshop3` | 7B–32B ungated models | Workshop-friendly 3-model suite (architecture diversity) |
| `full` | 7B–72B ungated models | Multi-GPU recommended |

Storage note: a default `subset` run on Mistral-7B typically needs about 42 GB
of model-weight space on the output filesystem with the default
`PACK_BASELINE_STORAGE_MODE=snapshot_symlink` when the Hugging Face cache lives
on the same filesystem as `OUTPUT_DIR`, or about 28 GB if the cache is on a
separate volume. `snapshot_copy` is heavier at about 56 GB. The suite's disk
preflight also enforces `MIN_FREE_DISK_GB` headroom (200 GB by default).

Scenario selection is driven by `scripts/proof_packs/scenarios.json`. Scenarios can
optionally declare `suites: ["subset", "showcase", "full", ...]`; during execution the
suite writes the effective (filtered) manifest to `OUTPUT_DIR/state/scenarios.json`,
and both task generation and final verdict compilation use that state manifest.
`--scenario-ids` filters that manifest before queue generation, and the runtime now
honors one-sided selections exactly: clean-only, stress-only, or single-scenario
smokes no longer expand back to the default 8 edit scenarios. Disk estimation uses
the same filtered state manifest, so storage preflight reflects the selected
scenario set rather than the suite defaults.

## Network & Model Revisions

Proof packs require pinned model revisions for reproducibility:

- Use `--net 1` on the first run to preflight and pin revisions in
  `OUTPUT_DIR/state/model_revisions.json`.
- Offline runs use `--net 0` (default) and error if the cache is missing.
- The `PACK_NET` environment variable is exported as `1` or `0` to gate `HF_*_OFFLINE` settings.
- Bulk proof-pack runs also require `INVARLOCK_ALLOW_REMOTE_CODE=1`; the
  entrypoint now fails fast before queue creation when that opt-in is missing.

## Promotion Sentinels

For Qwen2.5-14B promotion work, use the maintained sentinel helper from a fresh
repo work tree:

```bash
INVARLOCK_ALLOW_REMOTE_CODE=1 \
INVARLOCK_ALLOW_NETWORK=1 \
  ./scripts/proof_packs/run_qwen14_sentinels.sh \
    --run-dir /path/to/proof_pack_run \
    --model-name qwen__qwen2.5-14b
```

What it checks:

- saved-model direct evaluate for `quant_4bit_clean`
- saved-model direct evaluate for `prune_12pct_clean`
- the promotion-grade public quant smoke (`quant_4bit_clean` + `invarlock verify`)

Acceptance for these sentinels is load-path completion, not scientific PASS:

- `evaluation.report.json` must be emitted for each sentinel
- the public quant smoke must also produce `verify.json`
- a primary-metric `FAIL` is acceptable for this infrastructure/load-path gate

Use a fresh work tree on remote hosts. If you intentionally run from a checkout
that is not the editable install used by `.venv`, either reinstall the checkout
or run with `PYTHONPATH=src` so `invarlock` uses the intended source tree.

## Output Layout

A suite run writes artifacts under `OUTPUT_DIR` (default: `./proof_pack_runs/<suite>_<timestamp>`):

- `reports/final_verdict.txt` + `reports/final_verdict.json`
- `reports/category_summary.json`
- `reports/guard_signal_summary.json`
- `reports/guard_intervention_summary.json` (non-failing remediation signals, e.g. spectral caps + VE probe)
- `reports/scenario_signal_summary.json`
- `analysis/determinism_repeats.json` (when `--repeats` is used)
- `*/reports/**/evaluation.report.json`

`run_pack.sh` copies curated artifacts into a pack directory (default
`OUTPUT_DIR/proof_pack`) and organizes them as:

- `results/final_verdict.txt` + `results/final_verdict.json`
- `results/**/category_summary.json`, `results/**/guard_signal_summary.json`, `results/**/guard_intervention_summary.json`, `results/**/scenario_signal_summary.json`
- `results/**/determinism_repeats.json` (if present)
- `reports/<model>/<edit>/<run>/evaluation.report.json`
- `reports/**/rmt_probe.json` (optional sidecar; emitted by some scenarios, e.g. `rmt_norm_noise`)
- `reports/**/ve_probe.json` (optional sidecar; emitted by VE demo scenarios, e.g. `ve_mlp_scale_skew`)
- `reports/**/evaluation.html` + `reports/**/verify.json`
- `README.md`, `manifest.json`, `checksums.sha256`
- `manifest.json.asc` if GPG signing is available
- `metadata/source_repo.json`, `metadata/environment.json`, and other input metadata sidecars when present

Pack assembly is atomic at the directory level. `run_pack.sh` stages the pack in
a hidden sibling temporary directory and only renames it into the final
`proof_pack/` path after manifest generation, checksum sealing, optional HTML
export, and optional signing succeed. Failed pack builds do not leave a partial
pack behind at the final destination.

## Edit Provenance Labels

reports record the edit algorithm used:

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

## Signing & Verification (Evidence vs Proof-Grade)

`manifest.json` includes `checksums_sha256_digest` (sha256 of `checksums.sha256`) so a
signed manifest cryptographically binds the checksums file (and thus all hashed artifacts).
Newer packs also carry a repo-native attestation block in the same signed manifest:
`builder`, `subject`, `invocation`, `environment`, and digest-backed `materials`.
Signed packs also record `signing_key_fingerprint` for audit trails.

The manifest contract is published at `contracts/proof_pack_manifest.schema.json`.
`invarlock proof-pack verify` validates this schema before checksum and signature verification so
malformed proof packs fail deterministically.

Installed wheels now ship the public contracts and support package-native
inspection, assembly, and verification via `invarlock proof-pack inspect`,
`invarlock proof-pack build`, and `invarlock proof-pack verify`. The repo shell verifier remains
available for maintainers using the proof-pack harness directly.

Use the package-native subcommands:

- `invarlock proof-pack inspect <dir>`
  - Summarizes manifest validity, checksum coverage, attestation references, report inventory, and strict-readiness.
  - Does not run nested `invarlock verify`; use this for quick received-artifact triage.
- `invarlock proof-pack build <out> --final-verdict <json> --report <report> [...more --report]`
  - Packages existing JSON artifacts into a proof pack and pre-verifies the supplied clean reports with `invarlock verify`.
  - Intended for wheel users packaging already-produced evidence, not for running the full suite.
- `invarlock proof-pack verify <dir>`

- Default: `invarlock proof-pack verify <dir>`
  - Verifies `checksums_sha256_digest`, validates digest-backed manifest references, validates `checksums.sha256`, and runs `invarlock verify`.
  - Warns (but does not fail) if the pack is unsigned; this is evidence-grade verification.
- Strict (recommended for distributable evidence): `invarlock proof-pack verify <dir> --strict`
  - Fails if `manifest.json.asc` is missing, `gpg` verification fails, or extra files exist outside `checksums.sha256`.
  - Repo-harness alternative: `PACK_STRICT_MODE=1 scripts/proof_packs/verify_pack.sh --pack <dir>`.

`invarlock proof-pack verify` returns structured exit codes:

- `0`: verified successfully
- `2`: invalid usage or unsupported flag combination
- `3`: missing pack directory or required files
- `4`: manifest format or schema validation failure
- `5`: signature verification failure
- `6`: integrity failure (`checksums_sha256_digest`, `checksums.sha256`,
  digest-backed manifest references, or strict extra-file checks)
- `7`: report verification failure (`invarlock verify`)

Reviewer checklist:

- `invarlock proof-pack verify <dir> --strict` returns `0`
- `jq -e . <dir>/manifest.json` succeeds
- `sha256sum -c <dir>/checksums.sha256` succeeds
- `gpg --verify <dir>/manifest.json.asc <dir>/manifest.json` succeeds when the
  pack is published as signed evidence
- `manifest.json` includes builder, subject, invocation, environment, and
  material digests for the distributed pack

For proof-grade attestation, require all three: signed manifest, strict verification, and PASS final verdict.

To skip signing during pack creation, set `PACK_GPG_SIGN=0`. To require signing, set `PACK_STRICT_MODE=1`.
