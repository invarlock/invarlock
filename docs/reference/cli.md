# CLI Reference

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Command-line interface for evaluation, verification, reporting, and advanced maintenance flows. |
| **Audience** | Operators running InvarLock from a terminal or CI. |
| **Primary commands** | `evaluate`, `verify`, `report`, `doctor`, `advanced`, `version`. |
| **Companion entrypoint** | `invarlock-runtime-verify` for direct runtime manifest checks. |
| **Requires** | `invarlock[hf]` for model-loading workflows; extra backends are installed via Python extras. |
| **Network** | Offline by default; use `evaluate --allow-network` when a run needs model or dataset downloads. |
| **Source of truth** | `src/invarlock/cli/app.py`, `src/invarlock/cli/commands/*.py`, `src/invarlock/cli/runtime_verify.py`. |

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
| `invarlock-runtime-verify --help` | Wheel-native runtime-manifest verification companion for existing report bundles |

## Quick Start

```bash
# Install the Hugging Face-backed evaluation stack
pip install "invarlock[hf]"

# Compare a baseline against a subject
invarlock evaluate --allow-network \
  --baseline gpt2 \
  --subject distilgpt2 \
  --adapter auto \
  --profile ci

# Validate the container-backed evaluation bundle
invarlock verify reports/eval/evaluation.report.json

# Render shareable HTML
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
invarlock report explain \
  --subject-report runs/subject/report.json \
  --baseline-report runs/source/report.json
```

## Security Defaults

- `evaluate` defaults to `--execution-mode container`, which delegates model-loading work
  into the runtime container.
- Use `--execution-mode host` only for host-side workflows that intentionally
  bypass the container boundary.
- `verify` expects `runtime.manifest.json` beside container-backed evaluation outputs
  and fails closed when required runtime provenance is missing.
- Network access remains opt-in through `evaluate --allow-network`.

## Task To Command Map

| Task | Command | Output |
| --- | --- | --- |
| Compare baseline vs subject | `invarlock evaluate` | `reports/eval/evaluation.report.json` plus `runtime.manifest.json` for container-backed runs |
| Validate an evaluation report | `invarlock verify` | Exit code plus human or JSON verification output |
| Render HTML from an evaluation report | `invarlock report html` | HTML file |
| Explain gate decisions from run reports | `invarlock report explain` | Human-readable explanation |
| Inspect environment health | `invarlock doctor` | Human or JSON diagnostics |
| Evidence-pack, policy, plugin, or calibration workflows | `invarlock advanced ...` | Advanced artifacts and diagnostics |

## Artifact Outputs Matrix

| Command | Writes `runs/` | Writes `reports/` | Notes |
| --- | --- | --- | --- |
| `invarlock evaluate` | Yes (`--out`, default `runs/`) | Yes (`--report-out`, default `reports/eval`) | Produces the paired evaluation report bundle |
| `invarlock verify` | No | No | Reads existing evaluation report JSON |
| `invarlock report html` | No | Yes (`--output`) | Renders HTML from an existing report |
| `invarlock report explain` | No | No | Reads existing baseline and subject run report JSON files (not evaluation.report.json) |
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
| `invarlock-runtime-verify` | Verify an evaluation report against its sibling `runtime.manifest.json` |

Exit codes: `0=success`, `1=generic failure`, `2=usage/schema/config failure`,
`3=hard abort` for profile-aware fail-closed paths.

## `invarlock evaluate`

Purpose: compare a baseline against a subject and emit an evaluation report.

Common options:

- `--baseline`: baseline checkpoint path or model ID
- `--subject`: subject checkpoint path or model ID
- `--baseline-report`: reuse a stored baseline report by passing the explicit
  `report.json` file path that captured the baseline windows
- `--adapter`: adapter name or `auto`
- `--profile`: `ci`, `release`, or another included profile
- `--tier`: tier label for policy context
- `--preset`: optional repo preset path
- `--out`: run-artifact directory
- `--report-out`: evaluation report directory
- `--execution-mode container|host`: execution policy for `evaluate`.
  `container` keeps model loading inside the runtime container; `host`
  allows host-side execution and produces host artifacts that should
  be verified with `verify --runtime-provenance host`.
- `--edit-config`: optional demo/smoke edit overlay such as `quant_rtn`

Example:

```bash
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline gpt2 \
  --subject /path/to/edited \
  --adapter auto \
  --profile ci \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --report-out reports/eval
```

## `invarlock verify`

Purpose: verify existing evaluation report JSON files.

Arguments:

- `REPORTS...`: one or more evaluation report JSON paths or directories containing
  canonical `evaluation.report.json`

Common options:

- `--baseline`: optional baseline report for comparison flows
- `--tolerance`: float tolerance for recompute checks
- `--profile`: profile-aware validation mode
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
  - Explain gate decisions from run reports, not the generated evaluation bundle
  - Explain gates and primary-metric behavior for a subject report versus a
    baseline report
  - Options: `--subject-report`, `--baseline-report`
- `invarlock report validate`
  - Validate a report JSON against the v1 schema
- Directory inputs are command-specific:
  - `report generate` and `report explain` accept directories containing
    canonical `report.json`
  - `report html` and `report validate` accept directories containing
    canonical `evaluation.report.json`
  - `verify` accepts directories containing canonical `evaluation.report.json`
    and optional baselines containing canonical `report.json` or
    `evaluation.report.json`
    - If a directory contains both canonical filenames, it is ambiguous and
      rejected; pass the exact file path instead.

Example:

```bash
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
invarlock report explain \
  --subject-report runs/subject/report.json \
  --baseline-report runs/source/report.json
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
outside the core product contract.

Subcommands:

- `invarlock advanced evidence-pack`
  - Inspect, build, and verify evidence packs
- `invarlock advanced policy`
  - Build and verify policy-pack artifacts
- `invarlock advanced plugins`
  - Read-only plugin discovery and explanation
- `invarlock advanced calibrate`
  - Tier-policy calibration and sweep tooling

Examples:

```bash
invarlock advanced evidence-pack verify <pack> --strict
invarlock advanced policy verify policy-pack.json --json
invarlock advanced plugins list --json
invarlock advanced calibrate --help
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

## `invarlock-runtime-verify`

Purpose: package-native runtime provenance verification for an existing
evaluation report and its sibling runtime manifest.

Arguments:

- `--report`: path to `evaluation.report.json`
- `--manifest`: path to `runtime.manifest.json`
- `--json`: emit a machine-readable `runtime-verify-v1` envelope

Example:

```bash
invarlock-runtime-verify \
  --report reports/eval/evaluation.report.json \
  --manifest reports/eval/runtime.manifest.json
```

## JSON Output

Stable machine-readable output is available on the verification and advanced
plugin surfaces.

- `invarlock verify --json`
- `invarlock advanced plugins list --json`
- `invarlock advanced evidence-pack verify --json`
- `invarlock advanced policy verify --json`

These commands emit a single JSON object suitable for CI parsing.

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
- [reports](reports.md)
- [Public Contracts](contracts.md)
