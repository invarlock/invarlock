# CLI Reference

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Command-line interface for evaluation, verification, reporting, and advanced maintenance flows. |
| **Audience** | Operators running InvarLock from a terminal or CI. |
| **Primary commands** | `evaluate`, `verify`, `report`, `doctor`, `advanced`, `version`. |
| **Requires** | `invarlock[hf]` for model-loading workflows; extra backends are installed via Python extras. |
| **Network** | Offline by default; enable downloads per command with `INVARLOCK_ALLOW_NETWORK=1`. |
| **Source of truth** | `src/invarlock/cli/app.py`, `src/invarlock/cli/commands/*.py`. |

The public product surface is intentionally narrow:

1. `invarlock evaluate`
2. `invarlock verify`
3. `invarlock report html`

Everything else is either diagnostics (`doctor`) or explicitly advanced
(`invarlock advanced ...`).

## Quick Start

```bash
# Install the Hugging Face-backed evaluation stack
pip install "invarlock[hf]"

# Compare a baseline against a subject
INVARLOCK_ALLOW_NETWORK=1 invarlock evaluate \
  --baseline gpt2 \
  --subject distilgpt2 \
  --adapter auto \
  --profile ci

# Validate the attested evaluation bundle
invarlock verify reports/eval/evaluation.report.json

# Render shareable HTML
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
invarlock report explain --report runs/subject/report.json --baseline runs/source/report.json
```

## Security Defaults

- `evaluate` defaults to `--mode attested`, which delegates model-loading work
  into the runtime container.
- Use `--mode local` only for trusted host-side workflows that intentionally
  bypass the container boundary.
- `verify` expects `runtime.manifest.json` beside attested evaluation outputs
  and fails closed when required attestation is missing.
- Network access remains opt-in through `INVARLOCK_ALLOW_NETWORK=1`.

## Task To Command Map

| Task | Command | Output |
| --- | --- | --- |
| Compare baseline vs subject | `invarlock evaluate` | `reports/eval/evaluation.report.json` plus `runtime.manifest.json` for attested runs |
| Validate an evaluation report | `invarlock verify` | Exit code plus human or JSON verification output |
| Render HTML from an evaluation report | `invarlock report html` | HTML file |
| Explain gate decisions from run reports | `invarlock report explain` | Human-readable explanation |
| Inspect environment health | `invarlock doctor` | Human or JSON diagnostics |
| Proof-pack, policy, plugin, or calibration workflows | `invarlock advanced ...` | Advanced artifacts and diagnostics |

## Artifact Outputs Matrix

| Command | Writes `runs/` | Writes `reports/` | Notes |
| --- | --- | --- | --- |
| `invarlock evaluate` | Yes (`--out`, default `runs/`) | Yes (`--report-out`, default `reports/eval`) | Produces the paired evaluation report bundle |
| `invarlock verify` | No | No | Reads existing evaluation report JSON |
| `invarlock report html` | No | Yes (`--output`) | Renders HTML from an existing report |
| `invarlock report explain` | No | No | Reads existing baseline and subject run report JSON files (not evaluation.report.json) |
| `invarlock doctor` | No | No | Diagnostics only |
| `invarlock advanced proof-pack` | Depends on subcommand | Depends on subcommand | Advanced evidence packaging |
| `invarlock advanced policy` | Depends on subcommand | No | Advanced policy-pack tooling |
| `invarlock advanced plugins` | No | No | Read-only plugin discovery and explanation |
| `invarlock advanced calibrate` | Yes | Yes | Advanced tier-policy calibration workflows |

## Top-Level Command Index

| Command | Purpose |
| --- | --- |
| `invarlock evaluate` | Compare baseline and subject checkpoints with deterministic pairing |
| `invarlock verify` | Verify evaluation reports against schema, pairing, and attestation rules |
| `invarlock report` | Explain, render, and validate existing report artifacts |
| `invarlock doctor` | Diagnose environment and configuration issues |
| `invarlock advanced` | Advanced proof-pack, policy, plugin, and calibration workflows |
| `invarlock version` | Show the installed version |

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
- `--profile`: `ci`, `release`, or another shipped profile
- `--tier`: tier label for policy context
- `--preset`: optional repo preset path
- `--out`: run-artifact directory
- `--report-out`: evaluation report directory
- `--mode attested|local`: execution mode for model-loading steps
- `--edit-config`: optional demo/smoke edit overlay such as `quant_rtn`

Example:

```bash
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
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

- `REPORTS...`: one or more report JSON paths

Common options:

- `--baseline`: optional baseline report for comparison flows
- `--tolerance`: float tolerance for recompute checks
- `--profile`: profile-aware validation mode
- `--json`: emit a single JSON envelope

Example:

```bash
invarlock verify --json reports/eval/evaluation.report.json
```

## `invarlock report`

Purpose: operate on existing report artifacts.

Core subcommands:

- `invarlock report html`
  - Render an evaluation report to HTML
  - Options: `-i/--input`, `-o/--output`, `--embed-css`, `--force`
- `invarlock report explain`
  - Explain gate decisions from run reports, not the generated evaluation bundle
  - Explain gates and primary-metric behavior for a subject report versus a
    baseline report
- `invarlock report validate`
  - Validate a report JSON against the current schema
- `invarlock report verify`
  - Re-run verification through the report namespace when needed
- Directory inputs to `report` commands are only accepted when they contain a
  canonical `report.json` or `evaluation.report.json`; otherwise pass an
  explicit file path.

Example:

```bash
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
invarlock report explain --report runs/subject/report.json --baseline runs/source/report.json
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
  canonical `report.json` or `evaluation.report.json`.

Example:

```bash
invarlock doctor --json
```

## `invarlock advanced`

Purpose: advanced and maintenance-oriented workflows that are intentionally
outside the core product contract.

Subcommands:

- `invarlock advanced proof-pack`
  - Inspect, build, and verify proof-pack evidence bundles
- `invarlock advanced policy`
  - Build and verify policy-pack artifacts
- `invarlock advanced plugins`
  - Read-only plugin discovery and explanation
- `invarlock advanced calibrate`
  - Tier-policy calibration and sweep tooling

Examples:

```bash
invarlock advanced proof-pack verify <pack> --strict
invarlock advanced policy verify policy-pack.json --json
invarlock advanced plugins list --json
invarlock advanced calibrate --help
```

## Plugins & Entry Points

`invarlock advanced plugins` lists built-in and optional adapters, guards,
edits, datasets, and related entry points without mutating the current Python
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

## JSON Output

Stable machine-readable output is available on the verification and advanced
plugin surfaces.

- `invarlock verify --json`
- `invarlock advanced plugins list --json`
- `invarlock advanced proof-pack verify --json`
- `invarlock advanced policy verify --json`

These commands emit a single JSON object suitable for CI parsing.

## Command Layout

- The public top level is `evaluate`, `verify`, `report`, `doctor`,
  `advanced`, and `version`.
- Proof-pack, policy, plugin, and calibration workflows live under
  `invarlock advanced ...`.
- Trusted host execution for the core evaluation path is expressed as
  `--mode local`.
- Optional runtime backends are installed with Python extras instead of CLI
  install and uninstall commands.

## Related Documentation

- [Getting Started](../user-guide/getting-started.md)
- [Quickstart](../user-guide/quickstart.md)
- [Compare & evaluate (BYOE)](../user-guide/compare-and-evaluate.md)
- [reports](reports.md)
- [Public Contracts](contracts.md)
