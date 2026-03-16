# Proof Packs

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Hardware-agnostic validation runs that bundle reports into portable evidence artifacts. |
| **Audience** | CI operators producing validation evidence across GPU topologies. |
| **Requires** | GPU capable of fitting selected models; HF cache or network for model download. |
| **Outputs** | Proof pack directory with reports, reports, checksums, and optional GPG signature. |
| **Source of truth** | `scripts/proof_packs/run_suite.sh`, `scripts/proof_packs/run_pack.sh`. |

Proof packs are hardware-agnostic validation runs that bundle InvarLock reports,
summary reports, and verification metadata into a portable evidence artifact. They replace the
B200-specific validation harness with a suite that can run on any NVIDIA GPU topology
that can fit the selected models.

By default, a proof pack is evidence-grade (integrity + report verification). Treat it
as proof-grade only when the manifest is signed, the pack is verified in strict verification mode, and the final verdict is PASS.

Operationally, proof packs are a maintainer smoke test that also emits reusable
evidence data. The same run should let maintainers catch regressions, let third parties
verify reported outcomes, and provide structured outputs for downstream analysis.

> Terminology: the proof-pack suite includes a run-scoped **Preset Derivation**
> phase (`CALIBRATION_RUN -> GENERATE_PRESET`) that writes
> `calibrated_preset_<model>.yaml/json` for that suite run. It does not directly
> modify global `runtime/tiers.yaml`. For global tier policy tuning, use
> `invarlock calibrate ...` (see [Tier Policy Tuning CLI](../reference/calibration.md)).

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
- `certs/<model>/<edit>/<run>/evaluation.report.json`
- `certs/**/rmt_probe.json` (optional sidecar; emitted by some scenarios, e.g. `rmt_norm_noise`)
- `certs/**/ve_probe.json` (optional sidecar; emitted by VE demo scenarios, e.g. `ve_mlp_scale_skew`)
- `certs/**/evaluation.html` + `certs/**/verify.json`
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
`verify_pack.sh` validates this schema before checksum and signature verification so
malformed proof packs fail deterministically.

This verifier path is repo-first today: wheel installs do not include
`contracts/proof_pack_manifest.schema.json` or `scripts/proof_packs/verify_pack.sh`.
For proof-pack validation, clone the repository and run the verifier from the
checked-out tree.

Use `verify_pack.sh`:

- Default: `scripts/proof_packs/verify_pack.sh --pack <dir>`
  - Verifies `checksums_sha256_digest`, validates digest-backed manifest references, validates `checksums.sha256`, and runs `invarlock verify`.
  - Warns (but does not fail) if the pack is unsigned; this is evidence-grade verification.
- Strict (recommended for distributable evidence): `scripts/proof_packs/verify_pack.sh --pack <dir> --strict`
  - Fails if `manifest.json.asc` is missing, `gpg` verification fails, or extra files exist outside `checksums.sha256`.
  - Alternative: set `PACK_STRICT_MODE=1` (e.g., `PACK_STRICT_MODE=1 scripts/proof_packs/verify_pack.sh --pack <dir>`).

`verify_pack.sh` returns structured exit codes:

- `0`: verified successfully
- `2`: invalid usage or unsupported flag combination
- `3`: missing pack directory or required files
- `4`: manifest format or schema validation failure
- `5`: signature verification failure
- `6`: integrity failure (`checksums_sha256_digest`, `checksums.sha256`,
  digest-backed manifest references, or strict extra-file checks)
- `7`: cert verification failure (`invarlock verify`)

Reviewer checklist:

- `scripts/proof_packs/verify_pack.sh --pack <dir> --strict` returns `0`
- `jq -e . <dir>/manifest.json` succeeds
- `sha256sum -c <dir>/checksums.sha256` succeeds
- `gpg --verify <dir>/manifest.json.asc <dir>/manifest.json` succeeds when the
  pack is published as signed evidence
- `manifest.json` includes builder, subject, invocation, environment, and
  material digests for the distributed pack

For proof-grade attestation, require all three: signed manifest, strict verification, and PASS final verdict.

To skip signing during pack creation, set `PACK_GPG_SIGN=0`. To require signing, set `PACK_STRICT_MODE=1`.
