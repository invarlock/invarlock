# Evidence Packs

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Hardware-agnostic validation runs that bundle reports into portable evidence artifacts. |
| **Audience** | CI operators producing validation evidence across GPU topologies. |
| **Requires** | Active repo environment, GPU capable of fitting selected models, and HF cache or network for model download. Default runtime-container runs also require an OCI container engine. |
| **Outputs** | Evidence pack directory with reports, checksums, and an optional package-native signature bundle. |
| **Source of truth** | `scripts/evidence_packs/run_suite.sh`, `scripts/evidence_packs/run_pack.sh`, `src/invarlock/evidence_pack.py`, `src/invarlock/cli/commands/evidence_pack.py`, `src/invarlock/reporting/verify_contract.py`. |

Evidence packs are hardware-agnostic validation runs that bundle InvarLock reports,
summary reports, and verification metadata into a portable evidence artifact. They replace the
B200-specific validation harness with a suite that can run on any NVIDIA GPU topology
that can fit the selected models.

By default, an evidence pack is integrity-checked and report-verified. Treat it as
strong distributable evidence only when the manifest is signed, the pack is
verified in strict verification mode, the bundled clean reports retain their
`runtime.manifest.json` provenance sidecar, and the final verdict is PASS.

Operationally, evidence packs are a maintainer smoke test that also emits reusable
evidence data. The same run should let maintainers catch regressions, let
received packs be re-verified, and provide structured outputs for later
analysis.

> Terminology: the evidence-pack suite includes a run-scoped **Preset Derivation**
> phase (`CALIBRATION_RUN -> GENERATE_PRESET`) that writes
> `calibrated_preset_<model>.yaml/json` for that suite run. It does not directly
> modify global `runtime/tiers.yaml`. For global tier policy tuning, use
> `invarlock advanced calibrate ...` (see [Tier Policy Tuning CLI](../reference/calibration.md)).
> Calibration entrypoints use the runtime container by default unless
> a repo-only workflow opts into local host execution.

## Entrypoint Guide

| Script | Purpose | Output | Use When |
| --- | --- | --- | --- |
| `run_pack.sh` | Repo-only full evidence-pack harness: runs suite + packages artifacts | Evidence pack directory with manifest + checksums | Maintainer/distributor workflow from a repo checkout |
| `run_suite.sh` | Repo-only suite harness | Reports under the run directory | Development/debugging, iterative runs |
| `verify_pack.sh` | Repo-only shell verifier | Verification status | Validating received evidence packs from a repo checkout |
| `invarlock advanced evidence-pack inspect` | Read-only evidence-pack summary | Manifest/integrity/report inventory summary | Auditing a received pack without nested report verification |
| `invarlock advanced evidence-pack build` | Assemble an evidence pack from existing artifacts | Evidence pack directory with manifest + checksums | Packaging already-produced verdicts, metadata, and reports |
| `invarlock advanced evidence-pack verify` | Package-native evidence-pack verification | Verification status + optional JSON | Validating received evidence packs from a wheel install |

## Quick Start

```bash
# In a repo checkout, install the CLI into the active environment once.
make dev-install

# Evidence-pack shell wrappers are advanced repo workflows. They call a repo-only
# Python config runner plus `invarlock evaluate` under the default
# runtime container. Build it once per checkout.
make runtime-image

# RECOMMENDED: Full evidence pack with verification artifacts
INVARLOCK_ALLOW_REMOTE_CODE=1 \
PACK_TUNED_EDIT_PARAMS_FILE=./scripts/evidence_packs/tuned_edit_params.json \
  ./scripts/evidence_packs/run_pack.sh --suite subset --net 1

# Host-side workflow for these repo-only wrappers (skips the default
# container-backed path)
INVARLOCK_ALLOW_REMOTE_CODE=1 \
INVARLOCK_ALLOW_HOST_EXECUTION=1 \
PACK_TUNED_EDIT_PARAMS_FILE=./scripts/evidence_packs/tuned_edit_params.json \
  ./scripts/evidence_packs/run_pack.sh --suite subset --net 1

# Development/debugging only (runs the suite, but does not build an evidence pack)
INVARLOCK_ALLOW_REMOTE_CODE=1 \
  ./scripts/evidence_packs/run_suite.sh --suite subset --resume

# Inspect a received evidence pack without nested report verification
invarlock advanced evidence-pack inspect ./evidence_pack_runs/subset_20250101_000000/evidence_pack --json

# Build an evidence pack from existing artifacts
invarlock advanced evidence-pack keygen ./tmp/evidence_pack_signing_key.pem

invarlock advanced evidence-pack build ./tmp/evidence_pack \
  --final-verdict ./reports/final_verdict.json \
  --source-repo ./metadata/source_repo.json \
  --environment ./metadata/environment.json \
  --material model_revisions=./metadata/model_revisions.json \
  --report ./runs/model/evaluation.report.json \
  --signing-key ./tmp/evidence_pack_signing_key.pem

# Verify an existing evidence pack
invarlock advanced evidence-pack verify ./evidence_pack_runs/subset_20250101_000000/evidence_pack --strict --report-assurance strict
```

Each `--report` must be an explicit `evaluation.report.json` file path. The
builder also requires `runtime.manifest.json` next to each supplied report so
packaged evidence preserves runtime provenance.

Note: clean edits require tuned preset parameters. Either set
`PACK_TUNED_EDIT_PARAMS_FILE` or place the file at
`scripts/evidence_packs/tuned_edit_params.json`.

The evidence-pack shell wrappers do not expose the public core
`--execution-mode` / `--runtime-provenance` flags directly. For host-side
execution in these repo-only wrappers, set `INVARLOCK_ALLOW_HOST_EXECUTION=1`
in the environment before calling `run_pack.sh` or `run_suite.sh`.
Installed-wheel/public workflows should use
`invarlock evaluate --execution-mode host` instead. Otherwise, the
underlying model-loading commands follow the default runtime-container path
and expect an OCI container engine such as `podman` or `docker`, plus a locally
built `invarlock-runtime:local` image from `make runtime-image`. If both engines
are installed, set
`INVARLOCK_CONTAINER_ENGINE=podman` to force Podman.

Validated container-path parity contract for evidence-pack wrappers:

- Wrapper-provided `INVARLOCK_CONFIG_ROOT` and `INVARLOCK_STORE_EVAL_WINDOWS`
  survive delegated repo-only config-runner / `evaluate` calls.
- External cache and temp overrides such as `HF_HOME`, `HF_HUB_CACHE`,
  `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, `TMPDIR`, and `TMP` remain visible
  inside the runtime container.
- Staged or external `--preset`, `--baseline-report`, and `--edit-config`
  inputs are mounted automatically before delegation.
- Configs that rely on `!include` outside the config directory must set
  `INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE=1`; otherwise the evidence-pack wrapper
  fails before launch instead of starting an unusable container job.

Bulk evidence-pack entrypoints default to `SKIP_FLASH_ATTN=true` and
`PACK_BASELINE_STORAGE_MODE=snapshot_copy`. That is the safe default for remote
default runtime-container runs. The pinned flash-attn installer is not shipped
in the current dependency surface because the audited package has an unpatched deserialization
advisory; keep eager attention unless your environment owner separately accepts
and pins a remediated build. Only opt back into `snapshot_symlink` baselines
when you intentionally want the extra complexity.

## How It Works

This page focuses on running evidence packs. For the internal task graph,
scheduler flow, and artifacts, see [Evidence Pack Internals](evidence-packs-internals.md).

## Suites

Model suites live in `scripts/evidence_packs/run_suite.sh`. You can also override individual
models via `MODEL_1`–`MODEL_8`.

| Suite | Models | Notes |
| --- | --- | --- |
| `subset` | `mistralai/Mistral-7B-v0.1` | Single-GPU friendly |
| `showcase` | `mistralai/Mistral-7B-v0.1`, `Qwen/Qwen2.5-14B`, `Qwen/Qwen2.5-32B` | Multi-GPU recommended; adds guard-focused scenarios |
| `workshop3` | `mistralai/Mistral-7B-v0.1`, `mistralai/Mixtral-8x7B-v0.1`, `01-ai/Yi-34B` | Workshop-friendly 3-model suite (architecture diversity) |
| `full` | `mistralai/Mistral-7B-v0.1`, `Qwen/Qwen2.5-14B`, `Qwen/Qwen2.5-32B`, `01-ai/Yi-34B`, `mistralai/Mixtral-8x7B-v0.1`, `Qwen/Qwen1.5-72B` | Multi-GPU recommended |

Storage note: a default `subset` run on Mistral-7B typically needs about 56 GB
of model-weight space on the output filesystem with the wrapper default
`PACK_BASELINE_STORAGE_MODE=snapshot_copy`. If you explicitly opt into
`snapshot_symlink`, the same run typically needs about 42 GB when the Hugging
Face cache lives on the same filesystem as `OUTPUT_DIR`, or about 28 GB if the
cache is on a separate volume. The suite's disk preflight also enforces
`MIN_FREE_DISK_GB` headroom (200 GB by default).

Scenario selection is driven by `scripts/evidence_packs/scenarios.json`. Scenarios can
optionally declare `suites: ["subset", "showcase", "full", ...]`; during execution the
suite writes the effective (filtered) manifest to `OUTPUT_DIR/state/scenarios.json`,
and both task generation and final verdict compilation use that state manifest.
`--scenario-ids` filters that manifest before queue generation, and the runtime
honors one-sided selections exactly: clean-only, stress-only, or single-scenario
smokes do not expand back to the default 8 edit scenarios. Disk estimation uses
the same filtered state manifest, so storage preflight reflects the selected
scenario set rather than the suite defaults.

## Evidence Pack Artifact Taxonomy

Evidence packs do not package model weights by default. The run directory may
contain edited subject checkpoints under each model's `models/` directory, while
the evidence pack contains reports and digest-backed sidecars about those
subjects. Scenarios declare one artifact class:

| Artifact class | Meaning |
| --- | --- |
| `validation_subject_checkpoint` | A Hugging Face checkpoint-shaped validation subject, not an optimized runtime backend |
| `deployable_optimized_subject` | A backend-specific deployable subject with packed storage and smoke evidence |
| `fault_injection_fixture` | An intentionally invalid or degraded fixture used to test detection |
| `evidence_only_pack` | Evidence without a corresponding edited subject artifact |

Current RTN, FP8, pruning, and low-rank evidence-pack edits are validation
subjects. They validate InvarLock behavior on external subject checkpoints; they
do not claim runtime memory reduction or packed deployment storage.

## Network & Model Revisions

Evidence packs require pinned model revisions for reproducibility:

- Use `--net 1` on the first run to preflight and pin revisions in
  `OUTPUT_DIR/state/model_revisions.json`.
- Offline runs use `--net 0` (default) and error if the cache is missing.
- The `PACK_NET` environment variable is exported as `1` or `0` to gate `HF_*_OFFLINE` settings.
- Bulk evidence-pack runs also require `INVARLOCK_ALLOW_REMOTE_CODE=1`; the
  entrypoint fails fast before queue creation when that opt-in is missing.

## Promotion Sentinels

For Qwen2.5-14B promotion work, run the evidence-pack campaign with
`PACK_CLEANUP_MODELS=0`, then use the maintained sentinel helper from a fresh
repo work tree. The sentinel helper reloads the saved validation subjects, so
the default cleanup mode removes the directories it needs.

```bash
PACK_CLEANUP_MODELS=0 \
INVARLOCK_ALLOW_REMOTE_CODE=1 \
INVARLOCK_ALLOW_NETWORK=1 \
  ./scripts/evidence_packs/run_qwen14_sentinels.sh \
    --run-dir /path/to/evidence_pack_run \
    --model-name qwen__qwen2.5-14b
```

What it checks:

- saved-model direct evaluate for `quant_4bit_clean`
- saved-model direct evaluate for `prune_clean`
- the promotion-grade public quant smoke (`quant_4bit_clean` + `invarlock verify`)

Acceptance for these sentinels is load-path completion, not scientific PASS:

- `evaluation.report.json` must be emitted for each sentinel
- the public quant smoke must also produce `verify.json`
- evaluate and verify commands must exit zero
- the helper defaults to `--profile dev --assurance off`; a primary-metric `FAIL`
  inside the emitted report is acceptable for this infrastructure/load-path gate

Use a fresh work tree on remote hosts. If you intentionally run from a checkout
that is not the editable install used by `.venv`, either reinstall the checkout
or run with `PYTHONPATH=src` so `invarlock` uses the intended source tree.

## Output Layout

A suite run writes artifacts under `OUTPUT_DIR` (default: `./evidence_pack_runs/<suite>_<timestamp>`):

- `reports/final_verdict.txt` + `reports/final_verdict.json`
- `reports/category_summary.json`
- `reports/guard_signal_summary.json`
- `reports/guard_intervention_summary.json` (non-failing remediation signals, e.g. spectral caps + VE probe)
- `reports/scenario_signal_summary.json`
- `analysis/determinism_repeats.json` (when `--repeats` is used)
- `analysis/evaluation_optimization_summary.json`
- `*/reports/**/evaluation.report.json`

`run_pack.sh` copies curated artifacts into a pack directory (default
`OUTPUT_DIR/evidence_pack`) and organizes them as:

- `results/final_verdict.txt` + `results/final_verdict.json`
- `results/**/category_summary.json`, `results/**/guard_signal_summary.json`, `results/**/guard_intervention_summary.json`, `results/**/scenario_signal_summary.json`
- `results/**/determinism_repeats.json` (if present)
- `results/**/edit_artifact_summary.json`
- `results/analysis/evaluation_optimization_summary.json`
- `reports/<model>/<edit>/<run>/evaluation.report.json`
- `reports/<model>/<edit>/<run>/edit_metadata.json`
- `reports/**/rmt_probe.json` (optional sidecar; emitted by some scenarios, e.g. `rmt_norm_noise`)
- `reports/**/ve_probe.json` (optional sidecar; emitted by VE demo scenarios, e.g. `ve_mlp_scale_skew`)
- `reports/**/deployable_artifact_validation.json`, `backend_inventory.json`, `memory_report.json`, `load_smoke.json`, and `inference_smoke.json` for deployable scenarios
- `reports/**/evaluation.html` + `reports/**/verify.json`
- `README.md` (reviewer summary), `manifest.json`, `checksums.sha256`
- `manifest.signature.json` when the pack is signed
- `metadata/source_repo.json`, `metadata/environment.json`, and other input metadata sidecars when present

Pack assembly is atomic at the directory level. `run_pack.sh` stages the pack in
a hidden sibling temporary directory and only renames it into the final
`evidence_pack/` path after manifest generation, checksum sealing, optional HTML
export, and optional signing succeed. Failed pack builds do not leave a partial
pack behind at the final destination.

## Evaluation Loop Controls

The default suite keeps one evaluation task per scenario so scheduler behavior
and GPU placement remain easy to inspect. Batch edit creation still reduces edit
generation overhead by loading the baseline once for multiple validation edits,
but each evaluated subject checkpoint is loaded by its own `evaluate_EDIT` task.

`PACK_DEFER_REPORT_RENDERING=1` keeps `evaluation.report.json` and required
sidecars, but skips optional markdown/reviewer rendering in the hot path.
`run_pack.sh --release-review` enables this by default; pack-level HTML export
and verification still run unless explicitly disabled.

Every run writes `results/analysis/evaluation_optimization_summary.json`. It
records timing files discovered under the run directory, deferred-render counts,
baseline-report reuse counts, and nested run timing totals from
`evaluate_timing.json`. Use it as scheduling and regression telemetry, not as a
standalone throughput claim.

## Edit Provenance Labels

reports record the edit algorithm used:

| Label | When to Use |
| --- | --- |
| `noop` | Baseline model with no edit applied |
| `quant_rtn`, `magnitude_prune`, etc. | Using InvarLock's built-in edit functions |
| `custom` | BYOE (Bring-Your-Own-Edit) pre-edited models |

For BYOE workflows, use `--edit-label custom` or let InvarLock infer from the model path.

Evidence-pack edit artifacts also write `edit_metadata.json`. The metadata uses
`schema: invarlock/evidence-pack-edit-metadata-v1` and records the scenario
artifact class, edit type, storage format, deployment flags, parameters, and
coverage. Evidence-pack task execution validates this metadata before evaluating
an edited subject, and pack verification checks that metadata agrees with
`metadata/scenarios.json`.

| Edit Type | Artifact class | Deployable optimization? |
| --- | --- | --- |
| RTN dequantized external-subject simulation | validation subject checkpoint | No |
| FP8 dequantized external-subject simulation | validation subject checkpoint | No |
| Dense magnitude-pruned checkpoint | validation subject checkpoint | No sparse runtime |
| Dense low-rank-SVD approximated checkpoint | validation subject checkpoint | No factorized runtime |

## Deployable Edit Lane

Deployable edit scenarios are separate from the default validation suite. They
are opt-in because backends such as bitsandbytes, GPTQ, and AWQ depend
on specific PyTorch, CUDA, kernel, architecture, and package versions.

There is no default deployable scenario in the OSS evidence-pack suite. A
deployable lane should be added only after its backend has a generator or
BYOE-loading path that passes reload, inference, inventory, and memory/storage
checks on a supported GPU stack.

A deployable scenario must produce backend metadata, backend inventory,
reload-smoke evidence, inference-smoke evidence, storage or memory evidence, an
InvarLock evaluation report, and verification output. The evidence pack still
does not include model weights unless explicitly configured; it includes
digest-backed evidence about the deployable artifact that was validated.

## Determinism

Use `--determinism strict` to disable TF32 and cuDNN benchmarks and align with
strict InvarLock presets. `--repeats N` reruns a single edit N times and records
a drift summary in `analysis/determinism_repeats.json` in the run output; packed
bundles copy it to `results/analysis/determinism_repeats.json`.

## Signing & Verification (Evidence vs Strict Signed Verification)

`manifest.json` includes `checksums_sha256_digest` (sha256 of `checksums.sha256`) so a
signed manifest cryptographically binds the checksums file (and thus all hashed artifacts).
Newer packs also carry a signed provenance block in the same manifest:
`builder`, `subject`, `invocation`, `environment`, and digest-backed `materials`.
The manifest also records a derived `evidence_level` (`low`/`medium`/`high`) so
reviewers can triage bundles quickly without replacing the underlying strict signed checks.
Package-native signed packs store the detached Ed25519 signature bundle in
`manifest.signature.json` and record `signing_key_fingerprint` in the manifest
for audit trails.

Signature verification confirms that `manifest.json` has not changed since the
holder of the matching private key signed it. To also prove signer authenticity,
pin the expected signer fingerprint or use a local trust store. Without pinning,
the verifier reports the signer fingerprint for review, but a different key can
sign a different pack.

```bash
invarlock advanced evidence-pack verify <dir> \
  --strict \
  --expected-fingerprint sha256:<64-hex-chars> \
  --report-assurance strict
```

The package-native verifier also accepts `--trust-store <json>`. If the flag is
omitted and `~/.config/invarlock/trusted-signers.json` exists, that file is used.
The trust store may be either a JSON list of fingerprints or an object with a
`trusted_signers` or `fingerprints` list; list entries may be strings or objects
with a `fingerprint` field.

The manifest contract is published at `contracts/evidence_pack_manifest.schema.json`.
`invarlock advanced evidence-pack verify` validates this schema before checksum and signature verification so
malformed evidence packs fail deterministically.

The current manifest format is `evidence-pack-v1`. The schema-required core is
`format`, `checksums_sha256`, and `checksums_sha256_digest`; builder, subject,
invocation, environment, material, signing, and nested report-verification
fields are additive provenance fields. Strong distributable evidence should
include those provenance fields even when a minimal schema-valid pack omits
them.

Installed wheels ship the public contracts and support package-native
inspection, key generation, assembly, and verification via `invarlock advanced evidence-pack inspect`,
`invarlock advanced evidence-pack keygen`, `invarlock advanced evidence-pack build`,
and `invarlock advanced evidence-pack verify`. The package-native CLI does not
depend on external signature binaries for evidence-pack verification.

Use the package-native subcommands:

- `invarlock advanced evidence-pack inspect <dir>`
  - Summarizes manifest validity, checksum coverage, signed provenance references, report inventory, and strict-readiness.
  - Does not run nested `invarlock verify`; use this for quick received-artifact triage.
- `invarlock advanced evidence-pack keygen <private-key.pem>`
  - Generates an Ed25519 signing key pair for package-native evidence-pack signatures.
- `invarlock advanced evidence-pack build <out> --final-verdict <json> --report <report> [...more --report]`
  - Packages existing JSON artifacts into an evidence pack and pre-verifies the supplied clean reports with `invarlock verify`.
  - Add `--signing-key <private-key.pem>` to produce `manifest.signature.json`.
  - Intended for installed-package packaging of already-produced evidence, not for running the full suite.
  - The repo maintainer harness signs by default as well; set `PACK_SIGN_MANIFEST=0` only when you intentionally need an unsigned pack.
- `invarlock advanced evidence-pack verify <dir>`

- Default: `invarlock advanced evidence-pack verify <dir>`
  - Verifies `checksums_sha256_digest`, validates digest-backed manifest references, validates `checksums.sha256`, requires a signed `manifest.signature.json`, and runs `invarlock verify`.
  - Fails closed if the pack is unsigned or if signature verification cannot run.
- Strict (recommended for distributable evidence): `invarlock advanced evidence-pack verify <dir> --strict --report-assurance strict`
  - Adds fail-closed checks for extra files outside `checksums.sha256` on top of the default signed-manifest requirement.
  - `--strict` is pack-integrity strictness; `--report-assurance strict` requires every bundled clean report to satisfy strict report assurance.
  - Add `--expected-fingerprint sha256:<64-hex-chars>` or `--trust-store <json>` when accepting evidence from a specific signer.
  - Repo-harness alternative: `PACK_STRICT_MODE=1 scripts/evidence_packs/verify_pack.sh --pack <dir> --report-assurance strict --expected-fingerprint sha256:<64-hex-chars>`.

`invarlock advanced evidence-pack verify` returns structured exit codes:

- `0`: verified successfully
- `2`: invalid usage or unsupported flag combination
- `3`: missing pack directory or required files
- `4`: manifest format or schema validation failure
- `5`: signature verification failure
- `6`: integrity failure (`checksums_sha256_digest`, `checksums.sha256`,
  digest-backed manifest references, or strict extra-file checks)
- `7`: report verification failure (`invarlock verify`)

Reviewer checklist:

- [ ] `invarlock advanced evidence-pack verify <dir> --strict --report-assurance strict` returns `0`
- [ ] Verification is signer-pinned with `--expected-fingerprint` or a trust
  store when authenticity matters outside the producing workspace
- [ ] `jq -e . <dir>/manifest.json` succeeds
- [ ] `sha256sum -c <dir>/checksums.sha256` succeeds
- [ ] `jq -e . <dir>/manifest.signature.json` succeeds when the pack is
  published as signed evidence
- [ ] `manifest.json` includes builder, subject, invocation, environment, and
  material digests for the distributed pack

For strong distributable evidence, require all three: signed manifest, strict
verification, and PASS final verdict.

## Troubleshooting

- **Missing tuned-edit parameters**: clean edits require tuned preset
  parameters. Either set `PACK_TUNED_EDIT_PARAMS_FILE` or place the file at
  `scripts/evidence_packs/tuned_edit_params.json`.
- **Container engine not found**: install Podman or Docker and a locally built
  `invarlock-runtime:local` image (`make runtime-image`); pick one explicitly
  with `INVARLOCK_CONTAINER_ENGINE=podman` or `INVARLOCK_CONTAINER_ENGINE=docker`.
- **Disk preflight fails**: increase free space to at least `MIN_FREE_DISK_GB`
  (200 GB by default) or relocate `OUTPUT_DIR` / the HF cache to a separate
  volume.
- **Missing remote-code opt-in**: bulk evidence-pack runs require
  `INVARLOCK_ALLOW_REMOTE_CODE=1`; the entrypoint fails fast before queue
  creation when that opt-in is missing.
- **Unsigned pack rejected**: `verify` fails closed without
  `manifest.signature.json`. Either re-sign with
  `invarlock advanced evidence-pack keygen` + `--signing-key`, or accept that
  the bundle is not assurance-grade.
- **YAML `!include` outside config dir**: set
  `INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE=1` or move the included file under
  the config directory.

## Observability

- `manifest.json` records `checksums_sha256_digest`, builder, subject,
  invocation, environment, material digests, and a derived `evidence_level`.
- `manifest.signature.json` records the detached Ed25519 signature when the
  pack is signed.
- `results/final_verdict.json` and per-model summary JSON files surface the
  scenario verdicts that drove the pack-level pass/fail.
- `reports/**/runtime.manifest.json` sidecars preserve runtime provenance for
  every clean report in the pack.

## Related Documentation

- [Evidence Pack Internals](evidence-packs-internals.md) — task graph, scheduler flow, and artifacts
- [CLI Reference](../reference/cli.md) — `invarlock advanced evidence-pack` subcommands
- [Public Contracts](../reference/contracts.md) — manifest schema and JSON output envelopes
- [Runtime Provenance Guide](../security/runtime-provenance-guide.md) — manifest requirements for strict bundles
- [Trust Model](../assurance/14-trust-model.md) — strict assurance scope
- [Tier Policy Tuning CLI](../reference/calibration.md) — global tier policy calibration
- [Threat Model](../security/threat-model.md) — security assumptions for distributable evidence
