# CLI Reference

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Command-line interface for evaluation, verification, reporting, and advanced maintenance flows. |
| **Audience** | Operators running InvarLock from a terminal or CI. |
| **Primary commands** | `evaluate`, `verify`, `report`, `doctor`, `advanced`, `version`. |
| **Runtime verifier** | `invarlock advanced runtime-verify` for direct runtime manifest checks. |
| **Requires** | `invarlock[hf]` for model-loading workflows; extra backends are installed via Python extras. |
| **Network** | Offline by default; use `evaluate --allow-network` when a run needs model or dataset downloads. |
| **Source of truth** | `src/invarlock/cli/app.py`, `src/invarlock/cli/commands/*.py`. |

Most users only need a narrow top-level surface:

1. `invarlock evaluate`
2. `invarlock verify`
3. `invarlock report html`

Everything else is either diagnostics (`doctor`) or explicitly advanced
(`invarlock advanced ...`).

## First-Touch Surfaces

These entrypoints are the ones users hit first when orienting themselves in a
fresh install or wheel-only environment:

| Surface | Why it matters |
| --- | --- |
| `invarlock --help` | Top-level discovery of the supported public command set |
| `invarlock --version` | Confirms the installed package and schema pairing |
| `invarlock report --help` | Shows the report subcommands without requiring run artifacts |
| `invarlock advanced --help` | Lists the advanced maintenance namespace before drilling into subcommands |
| `invarlock advanced calibrate --help` | Establishes that calibration lives under `advanced` rather than the core loop |
| `invarlock advanced runtime-verify --help` | Wheel-native runtime-manifest verification for existing report bundles |

## Quick Start

```bash
# Install the Hugging Face-backed evaluation stack
pip install "invarlock[hf]"

# Compare a local baseline against its externally edited subject
BASELINE_CHECKPOINT=/path/to/original-checkpoint
EDITED_SUBJECT_CHECKPOINT=/path/to/edited-checkpoint
invarlock evaluate --allow-network \
  --baseline "$BASELINE_CHECKPOINT" \
  --subject "$EDITED_SUBJECT_CHECKPOINT" \
  --baseline-adapter auto --subject-adapter auto \
  --profile ci \
  --assurance strict

# Validate the bundle using a digest pinned in an independent policy channel
BASELINE_RUN_REPORT=/path/to/baseline/run/report.json
ACCEPTANCE_POLICY_PACK=/path/to/acceptance/policy-pack.json
invarlock verify --profile ci --assurance strict \
  --baseline "$BASELINE_RUN_REPORT" \
  --policy-pack "$ACCEPTANCE_POLICY_PACK" \
  --expected-runtime-image-digest "$EXPECTED_RUNTIME_IMAGE_DIGEST" \
  reports/eval/evaluation.report.json

# Render shareable HTML
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
invarlock report explain --evaluation-report reports/eval/evaluation.report.json
invarlock report export -i reports/eval/evaluation.report.json --format mlflow-tags
```

## Security Defaults

- `evaluate` defaults to `--execution-mode container`, which delegates model-loading work
  into the runtime container.
- `evaluate` defaults to `--assurance strict`, which requires CI/release profile,
  balanced/conservative tier, canonical guard order, and complete run evidence.
  The emitted strict report remains `pending_verifier` until a later `verify`
  call supplies the retained baseline report, acceptance policy pack, and
  independent runtime-image pin.
- Use `--execution-mode host` only for host-side workflows that intentionally
  bypass the container boundary. Host mode is non-assurance unless
  `--assurance off` is explicit.
- `verify` expects `runtime.manifest.json` beside container-backed evaluation
  outputs. Strict mode also requires an independently supplied `--policy-pack` and
  `--expected-runtime-image-digest` from independent policy channels, and fails
  closed when a required input is missing.
- `verify --assurance report` is the default: strict is enforced when the report
  claims strict. Use `verify --assurance strict` to require strict on every
  report input.
- Network access remains opt-in through `evaluate --allow-network`.

## Task To Command Map

| Task | Command | Output |
| --- | --- | --- |
| Compare baseline vs subject | `invarlock evaluate` | `reports/eval/evaluation.report.json` plus `runtime.manifest.json` for container-backed runs |
| Validate an evaluation report | `invarlock verify` | Exit code plus human or JSON verification output |
| Render HTML from an evaluation report | `invarlock report html` | HTML file |
| Explain gate decisions from an evaluation bundle or explicit run reports | `invarlock report explain` | Human-readable explanation |
| Export evidence to CI and registry handoff formats | `invarlock report export` | MLflow tag JSON, model-card Markdown, or release-review Markdown |
| Inspect environment health | `invarlock doctor` | Human or JSON diagnostics |
| Evidence-pack, policy, plugin, or calibration workflows | `invarlock advanced ...` | Advanced artifacts and diagnostics |

## Artifact Outputs Matrix

| Command | Writes `runs/` | Writes `reports/` | Notes |
| --- | --- | --- | --- |
| `invarlock evaluate` | Yes (`--out`, default `runs/`) | Yes (`--report-out`, default `reports/eval`) | Produces the paired evaluation report bundle |
| `invarlock verify` | No | No | Reads existing evaluation report JSON |
| `invarlock report html` | No | Yes (`--output`) | Renders HTML from an existing report |
| `invarlock report explain` | No | No | Explains `evaluation.report.json` directly; also accepts explicit `--subject-report` and `--baseline-report` when you need to rebuild from raw run artifacts |
| `invarlock report export` | No | Optional (`--output`) | Exports MLflow tags, model-card Markdown, or release-review Markdown from an existing evaluation report |
| `invarlock doctor` | No | No | Diagnostics only |
| `invarlock advanced evidence-pack` | Depends on subcommand | Depends on subcommand | Advanced evidence packaging |
| `invarlock advanced policy` | Depends on subcommand | No | Advanced policy-pack tooling |
| `invarlock advanced plugins` | No | No | Read-only plugin discovery and explanation |
| `invarlock advanced calibrate` | Yes | Yes | Advanced tier-policy calibration workflows |

## Top-Level Command Index

| Command | Purpose |
| --- | --- |
| `invarlock evaluate` | Compare baseline and subject checkpoints with deterministic pairing |
| `invarlock verify` | Verify evaluation reports against schema, pairing, gates, report/manifest binding, and runtime-image claim rules |
| `invarlock report` | Explain, render, and validate existing report artifacts |
| `invarlock doctor` | Diagnose environment and configuration issues |
| `invarlock advanced` | Advanced evidence-pack, policy, plugin, and calibration workflows |
| `invarlock version` | Show the installed version |
| `invarlock advanced runtime-verify` | Verify an evaluation report against its sibling `runtime.manifest.json` |

Exit codes: `0=success`, `1=generic failure`, `2=usage/schema/config failure`,
`3=hard abort` for profile-aware fail-closed paths. Advanced evidence-pack
verification also uses `4-7` for package failure classes and `8` for a
non-assurance integrity-only diagnostic; `8` is never report-verification
success.

## Stable vs Experimental Commands

| Stability class | Commands | Contract |
| --- | --- | --- |
| Stable core workflow | `invarlock evaluate`, `invarlock verify`, `invarlock report html`, `invarlock report explain`, `invarlock report export`, `invarlock report validate`, `invarlock doctor`, `invarlock version` | Documented command names, documented options, exit-code meaning, and artifact paths are stable within the current CLI policy. |
| Stable JSON automation | `invarlock doctor --json`, `invarlock verify --json`, `invarlock advanced runtime-verify --json`, `invarlock advanced plugins list --json`, `invarlock advanced plugins adapters --json`, `invarlock advanced evidence-pack verify --json`, `invarlock advanced evidence-catalog validate --json`, `invarlock advanced policy verify --json` | Required envelope fields and `format_version` values are stable; optional fields are additive. |
| Stable advanced verifiers | `invarlock advanced runtime-verify`, `invarlock advanced evidence-pack inspect`, `invarlock advanced evidence-pack verify`, `invarlock advanced evidence-catalog validate`, `invarlock advanced policy build`, `invarlock advanced policy verify`, `invarlock advanced plugins list`, `invarlock advanced plugins adapters` | Public operational commands outside the core user loop. Their documented behavior is maintained, while additional subcommands may evolve faster. |
| Experimental or maintainer-only | `invarlock advanced calibrate`, repo scripts under `scripts/`, package-internal config runners, undocumented flags, and local harness entrypoints | Useful for development, calibration, and release work; not covered by the public CLI stability contract until documented as stable here. |

## `invarlock evaluate`

Purpose: compare a baseline against a subject and emit an evaluation report.

Common options:

- `--baseline`: baseline checkpoint path or model ID
- `--subject`: subject checkpoint path or model ID
- `--baseline-revision`: immutable 40–64 character lowercase hexadecimal
  commit for a remote baseline. Strict evaluation requires this when
  `--baseline` is not a local checkpoint directory.
- `--subject-revision`: immutable 40–64 character lowercase hexadecimal commit
  for a remote subject. Strict evaluation requires this when `--subject` is not
  a local checkpoint directory. Local checkpoint directories are bound by a
  deterministic content-tree SHA-256 identity instead.

Strict local identity intentionally performs three full checkpoint-tree reads:
one during command planning to bind the requested input, then one immediately
before and one immediately after model loading to detect substitution. This can
add material storage I/O for large checkpoints; immutable remote revisions avoid
local content hashing. Throughput depends on the checkpoint layout, filesystem,
cache state, and storage device; benchmark the target environment rather than
relying on a published host-specific estimate:

```bash
python scripts/checks/benchmark_checkpoint_identity.py \
  /path/to/local-checkpoint --repeat 3 --json
```

- `--baseline-report`: reuse a stored baseline report by passing the explicit
  `report.json` file path that captured the baseline windows. Reused reports
  must match the requested baseline model, profile, tier, adapter family,
  assurance mode, and dataset/window-plan fields.
- `--baseline-adapter`: baseline-side adapter name or `auto`
- `--subject-adapter`: subject-side adapter name or `auto`
- `--profile`: `ci`, `release`, or another included profile
- `--tier`: tier label for policy context
- `--preset`: optional repo preset path
- `--out`: run-artifact directory
- `--report-out`: evaluation report directory
- `--execution-mode container|host`: execution policy for `evaluate`.
  `container` keeps model loading inside the runtime container; `host`
  allows host-side execution and produces host artifacts that should
  be verified with `verify --runtime-provenance host --assurance off`.
- `--assurance strict|off`: strict is the default assurance contract;
  off is for exploratory/dev reports outside the assurance-evidence surface.
- `--edit-config`: optional demo/smoke edit overlay such as `quant_rtn`

Example:

```bash
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline gpt2 \
  --subject distilgpt2 \
  --baseline-adapter auto --subject-adapter auto \
  --profile ci \
  --execution-mode host \
  --assurance off \
  --report-out reports/eval
```

## `invarlock verify`

Purpose: verify existing evaluation report JSON files against the implemented
artifact and policy contracts. This command
checks report schema, primary-metric recomputation, paired-window consistency,
policy gates, strict-assurance claims when requested, and report/manifest
binding through the sibling `runtime.manifest.json`.

Arguments:

- `REPORTS...`: one or more evaluation report JSON paths or directories containing
  canonical `evaluation.report.json`

Common options:

- `--baseline`: required whenever strict assurance is enforced (explicitly or
  because the input report claims strict). Supply the complete baseline
  `report.json` emitted by the noop baseline run; metric-only and evaluation
  report fragments are rejected. It is optional only when strict assurance is
  not being enforced.
- `--policy-pack`: required whenever strict assurance is enforced. Supply the
  independently maintained `policy-pack-v1` JSON/YAML artifact; strict verification
  rejects thresholds authorized only by the submitted report.
- `--tolerance`: finite recompute tolerance in `[0, 1e-9]`; larger, negative,
  NaN, and infinite values are rejected so callers cannot disable recomputation
- `--profile`: profile-aware validation mode
- `--assurance report|strict|off`: `report` enforces strict only for reports
  claiming strict; `strict` requires every input to claim and pass strict;
  `off` skips strict assurance policy checks.
- `--warning-policy pass|fail`: keep guard warnings advisory (`pass`, default)
  or fail verification when baseline-relative guard warnings are present (`fail`).
- `--runtime-provenance container|host`: runtime provenance policy for
  the supplied report artifacts
- `--expected-runtime-image-digest sha256:...`: compare the manifest's claimed
  image digest to a value obtained independently. Required by strict assurance.
  This is image-identity claim matching, not proof that the image ran.
- `--json`: emit a single JSON envelope

Each result's `verification.receipt` identifies the exact subject and baseline
bytes by SHA-256, their provider digests, the verifier version, normalized
profile/assurance inputs, and the expected runtime-image digest. The receipt
sets `signed=false`; authenticate or sign the containing evidence pack when an
authenticated receipt is required.

Example:

```bash
invarlock verify --json \
  --expected-runtime-image-digest "$EXPECTED_RUNTIME_IMAGE_DIGEST" \
  reports/eval/evaluation.report.json
```

Use strict warning mode when you want to fail an otherwise policy-passing edit
because a guard signal changed relative to the baseline:

```bash
invarlock verify --warning-policy fail reports/eval/evaluation.report.json
```

## `invarlock report`

Purpose: operate on existing report artifacts through explicit subcommands.

Core subcommands:

- `invarlock report generate`
  - Generate human-readable report output from existing run reports
  - Options: `--run`, `--compare-run-report`, `--baseline-run-report`,
    `--format`, `--output`
- `invarlock report html`
  - Render an evaluation report to HTML
  - Options: `-i/--input`, `-o/--output`, `--embed-css`, `--force`
- `invarlock report explain`
  - Explain gates and primary-metric behavior from the preferred evaluation
    bundle input, or from explicit subject/baseline run reports when needed
  - Options: `--evaluation-report`, `--subject-report`, `--baseline-report`
- `invarlock report export`
  - Export an existing evaluation report for CI and registry handoff surfaces
  - Formats: `mlflow-tags`, `model-card-md`, `release-review-md`
  - Options: `-i/--evaluation-report`, `--format`, `-o/--output`,
    `--policy-profile`, `--report-url`, `--evidence-url`, `--verify-result`,
    `--force`
  - `--verify-result` uses only the verifier result item whose `id` matches
    the resolved evaluation report path, requires a strict `verify-v1` receipt
    bound to the exact exported report bytes, and rejects stale or malformed
    verifier JSON. Current unsigned receipts are exported only as
    `receipt_bound_untrusted` metadata, never as a verified pass.
- `invarlock report validate`
  - Validate a report JSON against the v1 schema
- Directory inputs are command-specific:
  - `report generate` and `report explain` accept directories containing
    canonical `report.json`
  - `report html` and `report validate` accept directories containing
    canonical `evaluation.report.json`
  - `report explain --evaluation-report` accepts directories containing
    canonical `evaluation.report.json`
  - `verify` accepts directories containing canonical `evaluation.report.json`
    and optional baselines containing canonical `report.json` or
    `evaluation.report.json`
    - A directory containing only `report.json` is a raw run directory, not a
      verifier bundle. Generate `evaluation.report.json` first with
      `invarlock report generate --run <subject report.json>
      --baseline-run-report <baseline report.json> --format report -o <output-dir>`.
    - If a directory contains both canonical filenames, it is ambiguous and
      rejected; pass the exact file path instead.

Example:

```bash
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
invarlock report explain --evaluation-report reports/eval/evaluation.report.json
invarlock report explain \
  --subject-report runs/subject/report.json \
  --baseline-report runs/baseline/report.json
```

## `invarlock doctor`

Purpose: environment diagnostics that remain light-import safe.

Common options:

- `--json`
- `--profile`
- `--tier`
- `--baseline-report`
- `--subject-report`
- `--strict`
- Report inputs accept an explicit JSON file path or a directory containing
  canonical `report.json` or `evaluation.report.json`; ambiguous directories
  with both canonical files are rejected and require an explicit file path.

Example:

```bash
invarlock doctor --json
```

## `invarlock advanced`

Purpose: advanced and maintenance-oriented workflows that are intentionally
outside the core user loop, except for the explicitly versioned JSON contracts
listed below.

Subcommands:

- `invarlock advanced evidence-pack`
  - Inspect and verify evidence packs, including exact catalog-bound sets
- `invarlock advanced evidence-catalog`
  - Validate the checked public evidence catalog
- `invarlock advanced policy`
  - Build and verify policy-pack artifacts
- `invarlock advanced plugins`
  - Read-only plugin discovery and explanation
- `invarlock advanced calibrate`
  - Tier-policy calibration and sweep tooling
- `invarlock advanced runtime-verify`
  - Low-level runtime-manifest verification for an existing report

Examples:

```bash
invarlock advanced evidence-pack verify <pack> \
  --strict \
  --report-assurance strict \
  --policy-pack acceptance-policy-pack.json \
  --expected-runtime-image-digest "$EXPECTED_RUNTIME_IMAGE_DIGEST"
invarlock advanced evidence-pack inspect <pack> --json
invarlock advanced evidence-pack verify-set --help
invarlock advanced evidence-catalog validate \
  contracts/evidence_catalog_v1.json \
  --json
invarlock advanced policy verify policy-pack.json --json
invarlock advanced plugins list --json
invarlock advanced calibrate --help
invarlock advanced runtime-verify --report reports/eval/evaluation.report.json --manifest reports/eval/runtime.manifest.json
```

`advanced evidence-pack verify --skip-verify` is a diagnostic-only integrity
inspection. It returns the distinct nonzero status `8`, with
`integrity_ok=true`, `reports_verified=false`, and `ok=false` when integrity
checks complete. It is rejected with `--strict`, strict report assurance, or
CI/release profiles; use normal report verification for assurance.

## Plugins & Entry Points

`invarlock advanced plugins` lists built-in and optional adapters, guards,
edits, datasets, and related entry points without mutating the active Python
environment.

Available read-only flows include:

- `invarlock advanced plugins list`
- `invarlock advanced plugins adapters`
- `invarlock advanced plugins guards`
- `invarlock advanced plugins edits`

Optional backends are installed through normal Python packaging, for example:

```bash
pip install "invarlock[hf]"
pip install "invarlock[awq,gptq]"
```

Plugin install and uninstall commands are not part of the CLI surface.

## `invarlock advanced runtime-verify`

Purpose: low-level runtime evidence verification for an existing evaluation
report and runtime manifest. This command validates the manifest contract,
declared container fields, image-digest presence, and report SHA-256 binding.
With `--expected-runtime-image-digest`, it also compares the declared image
identity with a caller-supplied pin. Without that pin, a successful result is
`manifest_bound`, not independently anchored. Neither mode attests actual
container execution; `invarlock verify` separately owns primary-metric gates,
paired-window math, and strict-assurance report policy.

Common options:

- `--report`: path to `evaluation.report.json`
- `--manifest`: path to `runtime.manifest.json`
- `--expected-runtime-image-digest`: optional independent `sha256:...` image pin
- `--json`: emit a machine-readable `runtime-verify-v1` envelope

Example:

```bash
invarlock advanced runtime-verify \
  --report reports/eval/evaluation.report.json \
  --manifest reports/eval/runtime.manifest.json \
  --expected-runtime-image-digest "$EXPECTED_RUNTIME_IMAGE_DIGEST"
```

The JSON envelope distinguishes `binding_verified`,
`expected_digest_matched`, and `trust_status`. The latter two describe a
caller-supplied digest comparison only; they are not execution attestation.

## JSON Output

Stable machine-readable output is available on these surfaces:

| Command | Format version | Stability |
| --- | --- | --- |
| `invarlock doctor --json` | `doctor-v1` | Required envelope fields are stable. |
| `invarlock verify --json` | `verify-v1` | Required envelope fields and exit-code meaning are stable. |
| `invarlock advanced runtime-verify --json` | `runtime-verify-v1` | Runtime-manifest verification envelope is stable. |
| `invarlock advanced plugins list --json` | `plugins-v1` | Plugin catalog envelope and contract catalog keys are stable. |
| `invarlock advanced plugins adapters --json` | `plugins-v1` | Adapter rows and contract catalog keys are stable. |
| `invarlock advanced evidence-pack verify --json` | `evidence-pack-verify-v1` | Evidence-pack verification envelope is stable. |
| `invarlock advanced evidence-catalog validate --json` | `evidence-catalog-validate-v1` | Catalog validation envelope is stable. |
| `invarlock advanced policy verify --json` | `policy-pack-verify-v1` | Policy-pack verification envelope is stable. |

These commands emit a single JSON object suitable for CI parsing. Within a
format version, new optional fields may be added and consumers should ignore
unknown fields. Removing a required field, renaming a required field, changing a
field type, or changing pass/fail exit-code meaning requires a new format
version.

## Command Layout

- The public top level is `evaluate`, `verify`, `report`, `doctor`,
  `advanced`, and `version`.
- Evidence-pack, policy, plugin, and calibration workflows live under
  `invarlock advanced ...`.
- Host execution for the core evaluation path is expressed as
  `--execution-mode host`.
- Internal delegated config execution uses a package-internal config-runner
  module, not a public CLI command.
- Optional runtime backends are installed with Python extras instead of CLI
  install and uninstall commands.

## Related Documentation

- [Getting Started](../user-guide/getting-started.md)
- [Quickstart](../user-guide/quickstart.md)
- [Compare & evaluate (BYOE)](../user-guide/compare-and-evaluate.md)
- [Reports Reference](reports.md) — Schema, telemetry, and HTML export
- [Configuration Schema](config-schema.md)
- [Environment Variables](env-vars.md)
- [Public Contracts](contracts.md)
- [Troubleshooting](../user-guide/troubleshooting.md) — Error codes and recovery
- [Runtime Provenance Guide](../security/runtime-provenance-guide.md)
- [One Run Lifecycle](one-run-lifecycle.md) — Stage map for a single run
