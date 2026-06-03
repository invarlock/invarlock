# Example Reports

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Show how to generate and interpret InvarLock reports. |
| **Audience** | Users learning the evaluation workflow. |
| **Outputs** | `evaluation.report.json`, `evaluation_report.md`, `report.json`, and `runtime.manifest.json` for container-backed outputs. |
| **Requires** | `invarlock[hf]` for HF adapter workflows. |

InvarLock emits both machine-readable reports and human-friendly summaries.
Use the steps below to reproduce representative artifacts from this repository version.

## Read The Bundle First

For most reviewers, the primary artifact is `evaluation.report.json`, not the
lower-level run reports. Use it as the front door:

```bash
invarlock verify reports/quant8_demo/evaluation.report.json
invarlock report html -i reports/quant8_demo/evaluation.report.json -o reports/quant8_demo/evaluation.html
invarlock report explain --evaluation-report reports/quant8_demo/evaluation.report.json
```

Artifact model:

| Artifact | What it contains | Typical next step |
| --- | --- | --- |
| `evaluation.report.json` | Paired evaluation outcome, validation block, policy/provenance summary | `verify`, `report html`, `report explain --evaluation-report` |
| `report.json` | One run's raw metrics, guard telemetry, and execution artifacts | `report generate`, explicit `report explain --subject-report ... --baseline-report ...` |

## 1. Generate a report Bundle

The command below shows the default runtime-container path. It writes a
container-backed `runtime.manifest.json` next to `evaluation.report.json`.
Public host-side workflows use `--execution-mode host` and should verify the
resulting report with
`invarlock verify --runtime-provenance host --assurance off ...`.
This reproduction uses repo-owned preset and overlay files so it matches the
example artifacts checked into this repository version; wheel-only installs
should start with [Getting Started](getting-started.md) for the first evaluation
run, then come back here once they already have an evaluation bundle.

```bash
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline sshleifer/tiny-gpt2 \
  --subject  sshleifer/tiny-gpt2 \
  --baseline-adapter auto --subject-adapter auto \
  --profile release \
  --tier balanced \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --edit-config configs/overlays/edits/quant_rtn/8bit_full.yaml \
  --out runs/quant8_demo \
  --report-out reports/quant8_demo
```

The command writes `evaluation.report.json`, `evaluation_report.md`, and
`runtime.manifest.json` under `reports/quant8_demo/`.
Each report contains:

- Model and edit metadata (model id, adapter, commit hash, edit plan)
- Drift / perplexity / RMT verdicts with paired bootstrap confidence intervals
- Guard diagnostics (spectral, variance, invariants) including predictive-gate notes
- Policy digest capturing tier thresholds and calibration choices

## 2. Create a Narrative Summary

```bash
# The report already includes a markdown summary:
cat reports/quant8_demo/evaluation_report.md

# To regenerate markdown from run reports, pass edited + baseline:
invarlock report generate \
  --run <edited_report.json> \
  --baseline-run-report <baseline_report.json> \
  --format markdown
```

The markdown report mirrors the report content but highlights:

- Baseline vs edited perplexity series
- Guard outcomes with links to supporting metrics
- Checklist of gates (PASS/FAIL) suitable for change-control review

## 3. Shareable Attachments

HTML report chrome:

```text
Header -> Summary chips -> Quick links rail -> Canonical report body
```

That layout is intentional: reviewers should be able to confirm overall status,
jump directly to the gate or provenance section they care about, and still read
the unchanged canonical report content underneath.

For audits, collect the following files:

| File | Purpose |
|------|---------|
| `runs/<name>/**/report.json` | Execution log, metrics, and guard telemetry |
| `reports/<name>/evaluation.report.json` | Machine-readable evaluation report |
| `reports/<name>/runtime.manifest.json` | Runtime provenance for container-backed outputs |
| `reports/<name>/evaluation_report.md` | Human-friendly summary for reviewers |

Reports remain valid only for the same baseline reference, pairing assumptions,
dataset/tokenizer context, and scoped claim surface, and only while
`invarlock verify --json reports/<name>/evaluation.report.json` continues to pass
against the adjacent `runtime.manifest.json`.
