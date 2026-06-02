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

# Compare a baseline against a subject
invarlock evaluate --allow-network \
  --baseline gpt2 \
  --subject distilgpt2 \
  --baseline-adapter auto --subject-adapter auto \
  --profile ci \
  --assurance strict

# Validate the container-backed evaluation bundle
invarlock verify --assurance strict reports/eval/evaluation.report.json

# Render shareable HTML
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
invarlock report explain --evaluation-report reports/eval/evaluation.report.json
```

## Security Defaults

- `evaluate` defaults to `--execution-mode container`, which delegates model-loading work
  into the runtime container.
- `evaluate` defaults to `--assurance strict`, which requires CI/release profile,
  balanced/conservative tier, canonical guard order, complete evidence, and
  verified runtime provenance.
- Use `--execution-mode host` only for host-side workflows that intentionally
  bypass the container boundary. Host mode is non-assurance unless
  `--assurance off` is explicit.
- `verify` expects `runtime.manifest.json` beside container-backed evaluation outputs
  and fails closed when required runtime provenance is missing.
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
| Inspect environment health | `invarlock doctor` | Human or JSON diagnostics |
| Evidence-pack, policy, plugin, or calibration workflows | `invarlock advanced ...` | Advanced artifacts and diagnostics |

## Artifact Outputs Matrix

| Command | Writes `runs/` | Writes `reports/` | Notes |
| --- | --- | --- | --- |
| `invarlock evaluate` | Yes (`--out`, default `runs/`) | Yes (`--report-out`, default `reports/eval`) | Produces the paired evaluation report bundle |
| `invarlock verify` | No | No | Reads existing evaluation report JSON |
| `invarlock report html` | No | Yes (`--output`) | Renders HTML from an existing report |
| `invarlock report explain` | No | No | Prefers `evaluation.report.json`, then auto-resolves linked run reports; also accepts explicit `--subject-report` and `--baseline-report` |
| `invarlock doctor` | No | No | Diagnostics only |
| `invarlock advanced evidence-pack` | Depends on subcommand | Depends on subcommand | Advanced evidence packaging |
| `invarlock advanced policy` | Depends on subcommand | No | Advanced policy-pack tooling |
| `invarlock advanced plugins` | No | No | Read-only plugin discovery and explanation |
| `invarlock advanced calibrate` | Yes | Yes | Advanced tier-policy calibration workflows |

## Top-Level Command Index

| Command | Purpose |
| --- | --- |
| `invarlock evaluate` | Compare baseline and subject checkpoints with deterministic pairing |
| `invarlock verify` | Verify evaluation reports against schema, pairing, and runtime provenance rules |
| `invarlock report` | Explain, render, and validate existing report artifacts |
| `invarlock doctor` | Diagnose environment and configuration issues |
| `invarlock advanced` | Advanced evidence-pack, policy, plugin, and calibration workflows |
| `invarlock version` | Show the installed version |
| `invarlock advanced runtime-verify` | Verify an evaluation report against its sibling `runtime.manifest.json` |

Exit codes: `0=success`, `1=generic failure`, `2=usage/schema/config failure`,
`3=hard abort` for profile-aware fail-closed paths.

## Stable vs Experimental Commands

| Stability class | Commands | Contract |
| --- | --- | --- |
| Stable core workflow | `invarlock evaluate`, `invarlock verify`, `invarlock report html`, `invarlock report explain`, `invarlock report validate`, `invarlock doctor`, `invarlock version` | Documented command names, documented options, exit-code meaning, and artifact paths are stable within the current CLI policy. |
| Stable JSON automation | `invarlock doctor --json`, `invarlock verify --json`, `invarlock advanced runtime-verify --json`, `invarlock advanced plugins list --json`, `invarlock advanced plugins adapters --json`, `invarlock advanced evidence-pack verify --json`, `invarlock advanced policy verify --json` | Required envelope fields and `format_version` values are stable; optional fields are additive. |
| Stable advanced verifiers | `invarlock advanced runtime-verify`, `invarlock advanced evidence-pack inspect`, `invarlock advanced evidence-pack verify`, `invarlock advanced policy build`, `invarlock advanced policy verify`, `invarlock advanced plugins list`, `invarlock advanced plugins adapters` | Public operational commands outside the core user loop. Their documented behavior is maintained, while additional subcommands may evolve faster. |
| Experimental or maintainer-only | `invarlock advanced calibrate`, repo scripts under `scripts/`, package-internal config runners, undocumented flags, and local harness entrypoints | Useful for development, calibration, and release work; not covered by the public CLI stability contract until promoted here. |

## `invarlock evaluate`

Purpose: compare a baseline against a subject and emit an evaluation report.

Common options:

- `--baseline`: baseline checkpoint path or model ID
- `--subject`: subject checkpoint path or model ID
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
  --report-out reports/eval
```

## `invarlock verify`

Purpose: verify existing evaluation report JSON files end to end. This command
checks report schema, primary-metric recomputation, paired-window consistency,
policy gates, strict-assurance claims when requested, and runtime provenance
through the report's sibling `runtime.manifest.json`.

Arguments:

- `REPORTS...`: one or more evaluation report JSON paths or directories containing
  canonical `evaluation.report.json`

Common options:

- `--baseline`: optional baseline report for comparison flows
- `--tolerance`: float tolerance for recompute checks
- `--profile`: profile-aware validation mode
- `--assurance report|strict|off`: `report` enforces strict only for reports
  claiming strict; `strict` requires every input to claim and pass strict;
  `off` skips strict assurance policy checks.
- `--runtime-provenance container|host`: runtime provenance policy for
  the supplied report artifacts
- `--json`: emit a single JSON envelope

Example:

```bash
invarlock verify --json reports/eval/evaluation.report.json
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
  - Inspect, build, sign/keygen, and verify evidence packs
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
invarlock advanced evidence-pack verify <pack> --strict --report-assurance strict
invarlock advanced policy verify policy-pack.json --json
invarlock advanced plugins list --json
invarlock advanced calibrate --help
invarlock advanced runtime-verify --report reports/eval/evaluation.report.json --manifest reports/eval/runtime.manifest.json
```

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

Purpose: low-level runtime provenance verification for an existing evaluation
report and runtime manifest. This command validates the manifest contract,
container execution fields, image digest presence, and the report SHA-256
binding. It is scoped to runtime provenance; `invarlock verify` owns
primary-metric gates, paired-window math, and strict-assurance report policy.

Common options:

- `--report`: path to `evaluation.report.json`
- `--manifest`: path to `runtime.manifest.json`
- `--json`: emit a machine-readable `runtime-verify-v1` envelope

Example:

```bash
invarlock advanced runtime-verify \
  --report reports/eval/evaluation.report.json \
  --manifest reports/eval/runtime.manifest.json
```

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
