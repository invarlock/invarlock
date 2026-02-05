# CLI Reference

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Command-line interface for certification, verification, and reporting. |
| **Audience** | Operators running InvarLock from terminal/CI. |
| **Primary commands** | `evaluate`, `verify`, `report`, `run`, `plugins`, `doctor`. |
| **Requires** | `invarlock[hf]` for HF workflows; optional extras for quantized adapters. |
| **Network** | Offline by default; enable per command with `INVARLOCK_ALLOW_NETWORK=1`. |
| **Source of truth** | `src/invarlock/cli/app.py`, `src/invarlock/cli/commands/*.py`. |

## Contents

1. [Quick Start](#quick-start)
2. [Concepts](#concepts)
3. [Reference](#reference)
   - [Artifact outputs matrix](#artifact-outputs-matrix)
   - [Early Stops (CI/Release)](#early-stops-cirelease)
   - [Measurement Contracts](#measurement-contracts-gpumps-first)
   - [Command Index](#command-index)
4. [Quickstart Commands](#quickstart-commands)
5. [JSON Output](#json-output-verify-and-plugins)
6. [Compare & evaluate](#compare-evaluate)
7. [Profile Reference](#profile-reference-ci-vs-release)
8. [Security Defaults](#security-defaults)
9. [Troubleshooting](#troubleshooting)
10. [Related Documentation](#related-documentation)

## Quick Start

```bash
# Install core HF stack
pip install "invarlock[hf]"

# Compare & evaluate two checkpoints
INVARLOCK_ALLOW_NETWORK=1 invarlock evaluate --baseline gpt2 --subject gpt2

# Validate a report
invarlock verify reports/eval/evaluation.report.json
```

## Concepts

- **Pairing**: `evaluate` records baseline windows and enforces pairing in CI/Release.
- **Profiles**: `--profile ci|release|ci_cpu` controls window counts and determinism.
- **Tiers**: `--tier balanced|conservative` selects guard thresholds from `tiers.yaml`.
- **Offline-first**: downloads are opt-in; local paths work without network.
For definitions of common terms (pairing, tier policy, primary metric), see the
[Glossary](../assurance/glossary.md).

### Task → Command map

| Task | Command | Output |
| --- | --- | --- |
| Compare baseline vs subject | `invarlock evaluate` | `runs/` reports + `reports/eval` report. |
| Single-model run report | `invarlock run` | `report.json` + `events.jsonl`. |
| Validate report | `invarlock verify` | Exit code + validation messages. |
| Explain / HTML / compare | `invarlock report` | Rendered reports/evals. |
| Inspect environment | `invarlock plugins` / `invarlock doctor` | Plugin diagnostics. |

## Reference

InvarLock groups commands by task. The recommended path is Compare & evaluate (baseline ↔ subject):

```bash
invarlock evaluate --baseline <BASELINE_MODEL> --subject <SUBJECT_MODEL>
```

### Artifact outputs matrix

| Command | Writes `runs/` | Writes `reports/` | Emits report | Notes |
| --- | --- | --- | --- | --- |
| `invarlock evaluate` | Yes (`--out`, default `runs/`) | Yes (`--report-out`, default `reports/eval`) | Yes | Emits cert even on degraded PM (`E111`). |
| `invarlock run` | Yes (`--out`) | No | No | Produces `report.json` + `events.jsonl`. |
| `invarlock report` | No | Yes (`--output`) | Optional (`--format report/html`) | Renders from existing reports. |
| `invarlock verify` | No | No | No | Reads report JSON(s). |
| `invarlock plugins` / `doctor` | No | No | No | Diagnostics only. |

### CLI → Report → report → Verify

| Command | Report output | report output | Verify behavior |
| --- | --- | --- | --- |
| `invarlock run` | `report.json`, `events.jsonl` | None | Use `invarlock report` or `verify` later. |
| `invarlock evaluate` | `report.json` (baseline + subject) | `evaluation.report.json` | Exit `3` in CI/Release on pairing/gate failures. |
| `invarlock report --format report` | None (reads reports) | `evaluation.report.json` | Same verify rules as `evaluate`. |
| `invarlock verify` | None | None | Schema + pairing + profile gates. |

Note on presets and scripts

- Presets and scripts in this repository (`configs/`, `scripts/`) are not
  shipped in wheels.
- When installing from PyPI, prefer flag‑only `invarlock evaluate` (no preset
  paths), or clone this repo to use presets and matrix scripts.

Top‑level commands:

| Command             | Description                                                               |
| ------------------- | ------------------------------------------------------------------------- |
| `invarlock evaluate` | evaluate two checkpoints (baseline vs subject) with pinned windows         |
| `invarlock verify`  | Verify report JSONs against schema and pairing math                  |
| `invarlock report`  | Operations on reports and reports (explain, html, validate, compare) |
| `invarlock run`     | Advanced: single‑model evaluation to produce a report                     |
| `invarlock plugins` | Manage optional backends; list available guards/edits/adapters            |
| `invarlock doctor`  | Perform environment diagnostics                                           |

Exit codes: `0=success · 1=generic failure · 2=schema invalid · 3=hard abort`
([INVARLOCK:EXXX]) in ci/release.

### Early Stops (CI/Release)

InvarLock stops early in CI/Release profiles when evidence would be invalid,
failing fast with a profile‑aware exit code (`3`). Dev runs still emit
artifacts and exit with `1` to aid debugging.

- Primary metric degraded or non‑finite (evaluate only)
  - Where: after the edited run in `invarlock evaluate`.
  - Error: `[INVARLOCK:E111] Primary metric degraded or non‑finite (...)`.
  - Behavior: emits the report, then exits with a profile‑aware code.
  - Action: try an accelerator (mps/cuda), force float32, reduce
    `plan.max_modules`, lower the evaluation batch size.

- Pairing schedule mismatch (`E001`) when window matching fails
  (`window_match_fraction != 1.0`, `window_overlap_fraction > 0`), window
  counts diverge after stratification, the run is unpaired while a baseline
  is provided, or paired windows collapse (`paired_windows <= 0`).

Notes

- `invarlock run` in CI/Release logs a warning if the bare primary metric is
  non‑finite and continues to produce a report; it does not raise `E111`.
- `invarlock evaluate` always emits a report before exiting on `E111`.

For details on windowing, pairing, and tier minima, see
`docs/assurance/02-coverage-and-pairing.md` and
`docs/assurance/09-tier-v1-calibration.md`.

### Measurement Contracts (GPU/MPS-first)

InvarLock’s guards are approximation-only and accelerator-first (CUDA/MPS).
Each report records the measurement contract (estimator + sampling policy)
used to produce guard statistics.

- Recorded under:
  - `resolved_policy.spectral.measurement_contract` / `resolved_policy.rmt.measurement_contract`
  - `spectral.measurement_contract_hash` / `rmt.measurement_contract_hash`
- In CI/Release, `invarlock verify --profile ci|release` enforces:
  - measurement contract present, and
  - baseline/subject pairing (`*_measurement_contract_match = true`).

`assurance.mode` and per-guard `guards.{spectral,rmt}.mode` are not supported;
configs containing them are rejected.

### Quickstart Commands

```bash
# Core HF adapter + evaluation stack
pip install "invarlock[hf]"

# Optional GPU kernels / optimised kernels
pip install "invarlock[gpu]"

# Optional PTQ backends (install together with hf/gpu extras)
pip install "invarlock[awq,gptq]"

# Compare & evaluate two checkpoints (hero path)
invarlock evaluate --baseline gpt2 --subject gpt2-quant

# Force CPU execution when no accelerator is available (baseline smoke)
invarlock run -c configs/presets/causal_lm/wikitext2_512.yaml \
  --profile release --tier balanced --device cpu --out runs/baseline_cpu

# Explain decisions, compare, and render HTML
invarlock report explain --report runs/subject/report.json --baseline runs/baseline/report.json
invarlock report --run runs/subject/report.json --compare runs/baseline/report.json -o reports/compare
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html

# Validate a report
invarlock verify reports/eval/evaluation.report.json
```

Use `invarlock plugins` to review available adapters, edits, and guards.

Core installs (`pip install invarlock`) keep the CLI entry points
(`invarlock --help`, `invarlock version`) torch‑free; adapter‑based flows
(`invarlock evaluate`, `invarlock run` with HF adapters) require extras such as
`"invarlock[hf]"` or `"invarlock[adapters]"`.

### Command Index

Exhaustive command map with brief descriptions and notable options.

#### Top-level

- `invarlock` (global)
  - Options: `--install-completion`, `--show-completion`, `--version/-V`, `--help`
  - Summary: evaluate model changes with deterministic pairing and safety gates.
  - Quick path: `invarlock evaluate --baseline <MODEL> --subject <MODEL>`.
  - Tip: enable downloads per command with `INVARLOCK_ALLOW_NETWORK=1`.
  - Version: `invarlock --version` prints the CLI version (and report schema when available) and exits.

- `invarlock evaluate`
  - Purpose: Compare & evaluate (BYOE). Emits an evaluation report.
  - Options: `--baseline/--source`, `--subject/--edited`, `--adapter`,
    `--profile`, `--tier`, `--preset`, `--out`, `--report-out`, `--edit-config`.

- `invarlock verify`
  - Purpose: Verify report JSON(s) against schema, pairing math, and gates.
  - Args: `reportS...`
  - Options: `--baseline`, `--tolerance`, `--profile`, `--json`.

- `invarlock run`
  - Purpose: Execute pipeline from a YAML config (edit + guards + reports).
  - Options: `--config/-c`, `--device`, `--profile`, `--out`, `--edit`, `--tier`,
    `--metric-kind`, `--probes`, `--until-pass`, `--max-attempts`, `--timeout`,
    `--baseline`, `--no-cleanup`, `--timing`, `--telemetry`.

- `invarlock report` (group)
  - Purpose: Operations on reports/evalificates (verify, explain, html, validate).
  - Default (no subcommand): generate report(s) from a run.
  - Options (default callback): `--run`, `--format (json|md|html|cert|all)`,
    `--compare`, `--baseline`, `--output/-o`.
  - Subcommands:
    - `invarlock report verify` — recompute/verify metrics for report/cert.
      - Args: `reportS...`
      - Options: `--baseline`, `--tolerance`, `--profile`, `--json`.
    - `invarlock report explain` — explain gates for report vs baseline (primary metric ratio,
      Primary Metric Tail (ΔlogNLL), drift, and guard overhead when available).
    - `invarlock report html` — render report JSON to HTML.
      - Options: `-i/--input`, `-o/--output`, `--embed-css/--no-embed-css`, `--force`.
    - `invarlock report validate` — validate report JSON against current schema (v1).
      - Args: `report` (path to report JSON).

- `invarlock plugins` (group)
  - Purpose: Manage optional backends; list adapters/guards/edits.
  - Subcommands:
    - `invarlock plugins list [CATEGORY]` — show plugins for a category or all.
      - CATEGORY: `adapters|guards|edits|datasets|plugins|all` (default all).
      - Options: `--json`, `--verbose`, `--explain <name>`, adapters-only
        `--hide-unsupported/--show-unsupported`.
    - `invarlock plugins adapters` — list adapter plugins.
      - Options: `--only`, `--verbose`, `--json`, `--explain`,
        `--hide-unsupported/--show-unsupported`.
    - `invarlock plugins guards` — list guard plugins.
      - Options: `--only`, `--verbose`, `--json`.
    - `invarlock plugins edits` — list edit plugins.
      - Options: `--only`, `--verbose`, `--json`.
    - `invarlock plugins install NAMES...` — install extras/backends.
      - Options: `--upgrade/-U`, `--dry-run` (default), `--apply`.
    - `invarlock plugins uninstall NAMES...` — uninstall extras/backends.
      - Options: `--yes/-y`, `--dry-run` (default), `--apply`.

- `invarlock doctor`
  - Purpose: Health checks for environment and configuration.
  - Options: `--config/-c`, `--profile`, `--baseline`, `--json`, `--tier`,
    `--baseline-report`, `--subject-report`, `--strict`.

- `invarlock version`
  - Purpose: Show version (and schema when available).
  - Alias: `invarlock --version` / `-V`.

Evidence debug

- Set `INVARLOCK_EVIDENCE_DEBUG=1` to write a tiny guards_evidence.json next to the
  generated report and include a pointer in `manifest.json`. This contains
  only small policy knobs (no large arrays) and is safe to enable locally.

#### Plugins & Entry Points

`invarlock plugins` lists plugins without importing them and includes:


- Name and version (when known)
- Module path
- Entry point group/name (e.g., `invarlock.adapters:hf_causal`)
- Status and any extras hints (e.g., `invarlock[adapters]`)

Built-in entry points include:

- Adapters: `hf_causal`, `hf_mlm`, `hf_causal`
  - Convenience: `adapter: auto` resolves to a concrete adapter (`hf_causal`/`hf_causal`/`hf_mlm`) from the model's `config.json`.
- Edits: `quant_rtn`
  - Guards: `invariants`, `spectral`, `rmt`, `variance`

If you see an extras hint like `invarlock[adapters]`, install the extra to enable
richer functionality:

```bash
pip install "invarlock[adapters]"
```

Adapter listing defaults:

- `invarlock plugins adapters` hides platform‑unsupported adapters by default (clean
  view on macOS/CPU). Add `--show-unsupported` to include them.
- Filters and views:
  - `--only {ready,missing,core,optional}`
  - `--verbose` (adds module + entry point columns)
  - `--json` (machine‑readable)
  - `--explain <name>` (details for one adapter)

Extras helpers:

- Install: `invarlock plugins install <gptq|awq|gpu|adapters>` (adds the right extras)
- Uninstall: `invarlock plugins uninstall <gptq|awq|gpu>` (removes backend packages)

#### JSON Output (verify and plugins)

The CLI provides stable, single-line JSON envelopes for scripting and CI.

##### verify --json (format: verify-v1)

Envelope example:

```json
{
  "format_version": "verify-v1",
  "summary": { "ok": true, "reason": "ok" },
  "report": { "count": 1 },
  "results": [
    {
      "id": "reports/eval/evaluation.report.json",
      "schema_version": "v1",
      "kind": "ppl_causal",
      "ok": true,
      "reason": "ok",
      "ratio_vs_baseline": 1.002,
      "ci": [0.995, 1.010]
    }
  ],
  "resolution": { "exit_code": 0 },
  "component": "cli",
  "ts": "2025-01-01T00:00:00Z"
}
```

Notes:

- Exactly one JSON object is printed when `--json` is used.
- Exit codes: `0=pass`, `1=policy_fail`, `2=malformed`.
- `results[]` contains one element per input report; fields remain present
  with `null` when unknown.

Recompute details

The verifier includes a best‑effort recompute summary to help debug the primary metric:

- `recompute.family` — which family was checked: `accuracy` or `ppl` (or `other` if not applicable)
- `recompute.ok` — `true` when the recomputed value matches `primary_metric.final` within tolerance
- `recompute.reason` — `"mismatch"` when values differ, `"skipped"` when the
  report lacks the inputs (e.g., no counts or windows)

Example (accuracy):

```json
{
  "results": [
    {
      "kind": "accuracy",
      "ok": true,
      "recompute": { "family": "accuracy", "ok": true, "reason": null }
    }
  ]
}
```

Example (ppl):

```json
{
  "results": [
    {
      "kind": "ppl_causal",
      "ok": false,
      "recompute": { "family": "ppl", "ok": false, "reason": "mismatch" }
    }
  ]
}
```

Troubleshooting recompute mismatches

When `recompute.ok` is false (reason `"mismatch"`), the verifier found a
disagreement between the report’s recorded primary metric and what can be
derived from the embedded inputs. Common causes and quick fixes:

- Accuracy mismatches:
  - Cause: `metrics.classification.{n_correct,n_total}` don’t match `primary_metric.final`.
  - Fix: ensure counts reflect the same evaluation slice as the PM (preview/final),
    and that the PM kind is `accuracy` (or `vqa_accuracy`). If you changed counts,
    regenerate the report.
- PPL mismatches:
  - Cause: `evaluation_windows.final.{logloss,token_counts}` don’t correspond to the
    displayed `primary_metric.final`.
  - Fix: verify the windows used for the PM match those stored in the cert (same
    window IDs and counts). Regenerate the cert if windows changed.
- Baseline reference drift:
  - Cause: report’s `baseline_ref.primary_metric.final` doesn’t reflect the baseline
    actually used when computing the ratio.
  - Fix: keep the baseline report next to the cert or regenerate the cert with the
    intended baseline.
- Tolerance/precision:
  - Cause: Very small floating‑point differences.
  - Fix: pass a slightly larger `--tolerance`; the verifier uses it when comparing
    recomputed vs displayed values.

If recompute is `"skipped"`, the report doesn’t include the inputs needed for
this quick check. The verifier still checks schema and pairing math.

##### plugins list --json (format: plugins-v1)

Adapters example:

```json
{
  "format_version": "plugins-v1",
  "category": "adapters",
  "items": [
    {
      "name": "hf_causal",
      "kind": "adapter",
      "module": "invarlock.adapters.hf_causal",
      "entry_point": "invarlock.adapters.hf_causal:Adapter",
      "origin": "builtin",
      "backend": { "name": "transformers", "version": "4.43.0" }
    }
  ]
}
```

Guards/Edits example (no `backend` key):

```json
{
  "format_version": "plugins-v1",
  "category": "guards",
  "items": [
    {
      "name": "variance",
      "kind": "guard",
      "module": "invarlock.guards.variance",
      "entry_point": "invarlock.guards.variance:Guard",
      "origin": "builtin"
    }
  ]
}
```

All plugins (adapters + guards + edits):

```bash
invarlock plugins list plugins --json
```

Deterministic sort: `name, kind, module, entry_point`. Unknown categories exit
with code `2`.

##### plugins list (tables)

Default invocation shows all categories in rich tables:

```text
$ invarlock plugins list
             Guard Plugins — ready: 5 · missing-extras: 0
┏━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Name        ┃ Origin ┃ Mode  ┃ Backend ┃ Version ┃ Status / Action ┃
┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ invariants  │ Core   │ Guard │ —       │ —       │ ✅ Ready        │
│ rmt         │ Core   │ Guard │ —       │ —       │ ✅ Ready        │
│ spectral    │ Core   │ Guard │ —       │ —       │ ✅ Ready        │
│ variance    │ Core   │ Guard │ —       │ —       │ ✅ Ready        │
├─────────────┼────────┼───────┼─────────┼─────────┼─────────────────┤
│ hello_guard │ Plugin │ Guard │ —       │ —       │ ✅ Ready        │
└─────────────┴────────┴───────┴─────────┴─────────┴─────────────────┘
            Edit Plugins — ready: 2 · missing-extras: 0
┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Name      ┃ Origin ┃ Mode ┃ Backend ┃ Version ┃ Status / Action ┃
┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ noop      │ Core   │ Edit │ —       │ —       │ ✅ Ready        │
│ quant_rtn │ Core   │ Edit │ —       │ —       │ ✅ Ready        │
└───────────┴────────┴──────┴─────────┴─────────┴─────────────────┘
       Adapters — ready: 7 · auto: 2 · missing-extras: 0 · unsupported: 0
┏━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Adapter        ┃ Origin ┃ Mode         ┃ Backend      ┃ Version  ┃ Status / Action                     ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ hf_mlm        │ Core   │ Adapter      │ transformers │ ==<ver>  │ ✅ Ready                            │
│ hf_causal        │ Core   │ Adapter      │ transformers │ ==<ver>  │ ✅ Ready                            │
│ hf_causal       │ Core   │ Adapter      │ transformers │ ==<ver>  │ ✅ Ready                            │
│ hf_causal_onnx        │ Core   │ Adapter      │ transformers │ ==<ver>  │ ✅ Ready                            │
│ hf_seq2seq          │ Core   │ Adapter      │ transformers │ ==<ver>  │ ✅ Ready                            │
├────────────────┼────────┼──────────────┼──────────────┼──────────┼─────────────────────────────────────┤
│ hf_auto │ Core   │ Auto‑matcher │ transformers │ ==<ver>  │ 🧩 Auto (selects best hf_* adapter) │
│ hf_auto    │ Core   │ Auto‑matcher │ transformers │ ==<ver>  │ 🧩 Auto (selects best hf_* adapter) │
└────────────────┴────────┴──────────────┴──────────────┴──────────┴─────────────────────────────────────┘
Hints: add --only ready|core|optional|auto|unsupported · use --json for scripting · use adapters (plural)
                          Dataset Providers
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Provider          ┃ Network   ┃ Kind    ┃ Params               ┃ Status / Action ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ local_jsonl       │ No        │ text    │ path[, text_field]   │ ✓ Available     │
│ local_jsonl_pairs │ No        │ pairs   │ path[, input_field,  │ ✓ Available     │
│                   │           │         │ target_field]        │                 │
│ seq2seq           │ No        │ seq2seq │ -                    │ ✓ Available     │
│ synthetic         │ No        │ text    │ -                    │ ✓ Available     │
│ wikitext2         │ Cache/Net │ text    │ -                    │ ✓ Available     │
│ hf_seq2seq        │ Yes       │ seq2seq │ dataset_name[,       │ ✓ Available     │
│                   │           │         │ split, input_field,  │                 │
│                   │           │         │ target_field]        │                 │
│ hf_text           │ Yes       │ text    │ dataset_name[,       │ ✓ Available     │
│                   │           │         │ split, text_field]   │                 │
└───────────────────┴───────────┴─────────┴──────────────────────┴─────────────────┘
```

Notes:

- Counts and versions vary by environment (installed extras, OS).
- Use filters for stable views, for example:
  - `invarlock plugins adapters --only core`
  - `invarlock plugins adapters --only auto`
- Use `--hide-unsupported/--show-unsupported` to toggle platform‑gated adapters.

#### Quant (RTN) or Compare & evaluate examples

```bash
# Baseline (CI, GPT-2 small)
invarlock run -c configs/presets/causal_lm/wikitext2_512.yaml \
  --profile ci --tier balanced

# Compare & evaluate (recommended)
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --baseline gpt2 \
  --subject /path/to/edited \
  --adapter auto \
  --profile ci \
  --preset configs/presets/causal_lm/wikitext2_512.yaml

# Demo edit overlay (quant_rtn)
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --baseline gpt2 \
  --subject gpt2 \
  --adapter auto \
  --profile ci \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --edit-config configs/overlays/edits/quant_rtn/8bit_attn.yaml
```

### Minimal Configuration (quant_rtn)

```yaml
model:
  id: gpt2
  adapter: hf_causal
dataset:
  provider: wikitext2
  seq_len: 768
  stride: 768
  preview_n: 200
  final_n: 200
edit:
  name: quant_rtn
guards:
  spectral:
    enabled: true
  variance:
    tier: balanced
auto:
  tier: balanced
  probes: 0
```

### Compare & evaluate

Compare a subject against a baseline with pinned windows. This is the single
recommended workflow. Optionally, you can run the in‑repo demo edit
(`quant_rtn`) via `--edit-config` to produce a subject for smoke/demos.

```bash
# Compare & evaluate (BYOE checkpoints)
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --source <hf_dir_or_id> \
  --edited <hf_dir_or_id> \
  --adapter auto \
  --profile ci \
  --out runs \
  --report-out reports/eval

# Optional (demo): run the in‑repo quant_rtn edit to produce a subject
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --source <hf_dir_or_id> \
  --edited <hf_dir_or_id> \
  --adapter auto \
  --profile ci \
  --edit-config configs/overlays/edits/quant_rtn/8bit_attn.yaml
```

Behavior:

- Runs a baseline on `--source` and records windows.
- Runs the subject model with windows pinned via `--baseline` pairing.
- Emits a report JSON under `--report-out`.

Baseline reuse (skip Phase 1/3):

- Provide `--baseline-report <path>` to reuse a previously generated baseline `report.json` and skip the baseline evaluation phase.
- The baseline report must be from a no-op run (`edit.name == "noop"`) and must include stored evaluation windows (set `INVARLOCK_STORE_EVAL_WINDOWS=1` when producing it).

```bash
# 1) Produce a reusable baseline report once
INVARLOCK_STORE_EVAL_WINDOWS=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --source <hf_dir_or_id> \
  --edited <hf_dir_or_id> \
  --adapter auto \
  --profile ci \
  --tier balanced \
  --out runs/baseline_once \
  --report-out reports/eval_baseline_once

# 2) Reuse it for many subjects (skips baseline evaluation)
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --baseline-report runs/baseline_once/source \
  --source <hf_dir_or_id> \
  --edited <hf_dir_or_id> \
  --adapter auto \
  --profile ci \
  --tier balanced
```

See also: User Guide → Scripts & Utilities for preparing checkpoints
(state_dict → HF, GPTQ/AWQ export).

#### Expected Outcomes

- Quant RTN edits aim for ≤ 1.10× perplexity drift under the balanced CI profile.
- Guard verdicts surface in `report.json` and the report bundle; run
  `invarlock verify` for a one-shot policy check that enforces the schema, ratio
  math, and paired-window guarantees.
- Typical GPT‑2 small runs complete within ~5 minutes on a modern GPU or Apple
  Silicon. CPU runs are slower but supported via `--device cpu`.

### Helpful Options

| Flag                                             | Description                                                       |
| ------------------------------------------------ | ----------------------------------------------------------------- |
| --tier {balanced,conservative,aggressive,none}   | Applies tier-specific guard thresholds.                           |
| --profile {ci,release,ci_cpu}                    | Selects evaluation window counts and bootstrap depth.             |
| --probes N                                       | Enables micro-probes for exploratory analysis (default 0 for CI). |
| --out PATH                                       | Overrides the run output directory.                               |
| --baseline-report PATH                            | Reuse baseline `report.json` and skip baseline evaluation (pinned windows required). |
| --device {cpu,cuda,mps,auto}                     | Overrides device selection.                                       |

`--device auto` mirrors the default CLI behavior and attempts CUDA, then MPS
(Apple Silicon), then CPU. The resolved device is echoed in the run banner
(e.g., `Device resolved: auto → mps`) and recorded under `meta.device` in the
resulting report/report for audit trails.

### Profile Reference (CI vs Release)

| Profile                     | Preview Windows (`dataset.preview_n`) | Final Windows (`dataset.final_n`) | Bootstrap Replicates (`eval.bootstrap.replicates`) | Notes                                                                                                                                                                                                         |
| --------------------------- | ------------------------------------- | --------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CI (balanced defaults)      | 200                                   | 200                               | 1200                                               | Provided by the built-in CI profile defaults (or `INVARLOCK_CI_PREVIEW/FINAL` env overrides) when `--profile ci` is used.                                                                                     |
| Release                     | 400                                   | 400                               | 3200                                               | Set by the packaged release profile (`invarlock._data.runtime/profiles/release.yaml`); also raises the VE calibration cap to 320 windows. Override via `INVARLOCK_CONFIG_ROOT/runtime/profiles/release.yaml`. |
| CI CPU telemetry (optional) | 120                                   | 120                               | 1200 (inherits)                                    | Packaged `ci_cpu.yaml` (`invarlock._data.runtime/profiles/ci_cpu.yaml`) trims window counts and forces `model.device=cpu`. Override via `INVARLOCK_CONFIG_ROOT/runtime/profiles/ci_cpu.yaml`.                 |

When a profile is supplied, the values above override the dataset/eval blocks
in your base config before the run starts. Keep the profile metadata
(`/context.policy_snapshot`) with the report when you archive release
evidence.

For automation loops see the
[Getting Started guide](../user-guide/getting-started.md), the
[Example Reports](../user-guide/example-reports.md), and the
[Artifact Layout](artifacts.md) reference for retention guidelines.

### Security Defaults

- Outbound network access is disabled by default. Set `INVARLOCK_ALLOW_NETWORK=1`
  when a run needs to download models or datasets.
- YAML `!include` is restricted to files under the config directory by default.
  Set `INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE=1` to permit out-of-tree includes.
- Use `invarlock.security.secure_tempdir()` for scratch space with 0o700 permissions and automatic cleanup.
- JSONL event logs redact sensitive keys (tokens, secrets, passwords) and attach the run ID for auditability.
- Memory/perf levers:
  - `INVARLOCK_SNAPSHOT_MODE={auto|bytes|chunked}` controls how the model snapshot
    is taken for retries. In `auto` (default), InvarLock estimates snapshot size and
    chooses bytes or chunked based on available RAM and disk. `bytes` keeps the
    snapshot in memory; `chunked` writes per-parameter files to disk to minimize
    peak RAM. If `bytes` snapshotting fails (e.g., due to memory pressure), the
    CLI will attempt `chunked` snapshotting when the adapter supports it; otherwise
    it falls back to reload-per-attempt.
  - `INVARLOCK_SNAPSHOT_AUTO_RAM_FRACTION` tunes the auto mode (default `0.4` →
    choose chunked when snapshot size ≥ 40% of available RAM).
  - `INVARLOCK_STORE_EVAL_WINDOWS=0` disables token/attention caching during eval,
    and `INVARLOCK_EVAL_DEVICE=cpu` forces evaluation to run on CPU if needed.
  - Window difficulty stratification uses a byte‑level n‑gram scorer by default
    and runs fully offline.

#### Snapshot Mode Controls (Config)

Retries reuse a single loaded model and reset its state via snapshot/restore
between attempts. You can control snapshot strategy in your run config (takes
precedence over env):

```yaml
context:
  snapshot:
    mode: auto                # auto | bytes | chunked
    ram_fraction: 0.4         # choose chunked when snapshot ≥ fraction × available RAM
    threshold_mb: 768         # fallback when RAM not detectable
    disk_free_margin_ratio: 1.2  # require 20% headroom for chunked on disk
    temp_dir: /tmp            # where to place chunked snapshots
```

Notes:

- `mode` decides bytes vs chunked vs auto selection.
- In auto mode, InvarLock estimates snapshot bytes from tensor sizes and compares to
  available RAM. If large and disk has room, chunked is used; otherwise bytes.
- The retry loop (including the guard-overhead “bare” run) restores from the
  same snapshot for reproducible comparisons without reloading the model.

## Troubleshooting

- **`DEPENDENCY-MISSING` errors**: install the required extras (see Quick Start).
- **Pairing failures (`E001`)**: ensure baseline `report.json` preserves
  `evaluation_windows` and uses the same dataset settings.
- **Non-finite metrics**: lower batch size or force `dtype=float32`.

## Observability

- Reports land under `runs/<name>/<timestamp>/report.json`.
- reports are emitted under `reports/` via `invarlock report --format report`.
- JSON output modes (`--json`) provide stable machine-readable envelopes.

## Related Documentation

- [Configuration Schema](config-schema.md)
- [Dataset Providers](datasets.md)
- [Environment Variables](env-vars.md)
- [reports](reports.md) — Schema, telemetry, and HTML export
