# Example Reports

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Show how to generate and interpret InvarLock reports. |
| **Audience** | Users learning the evaluation workflow. |
| **Outputs** | `evaluation.report.json`, `evaluation_report.md`, `report.json`, and `runtime.manifest.json` for attested outputs. |
| **Requires** | `invarlock[hf]` for HF adapter workflows. |

InvarLock emits both machine-readable reports and human-friendly summaries.
Use the steps below to reproduce representative artifacts from this repository version.

## 1. Generate a report Bundle

The command below shows the secure-default runtime-container path. It writes an
attested `runtime.manifest.json` next to `evaluation.report.json`. Trusted
public host-side workflows use `--assurance trusted-local` and should verify the
resulting report with `invarlock verify --assurance trusted-local ...`.

```bash
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --baseline sshleifer/tiny-gpt2 \
  --subject  sshleifer/tiny-gpt2 \
  --adapter auto \
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
invarlock report --run <edited_report.json> --baseline <baseline_report.json> --format markdown
```

The markdown report mirrors the report content but highlights:

- Baseline vs edited perplexity series
- Guard outcomes with links to supporting metrics
- Checklist of gates (PASS/FAIL) suitable for change-control review

## 3. Shareable Attachments

For audits, collect the following files:

| File | Purpose |
|------|---------|
| `runs/<name>/**/report.json` | Execution log, metrics, and guard telemetry |
| `reports/<name>/evaluation.report.json` | Machine-readable evaluation report |
| `reports/<name>/runtime.manifest.json` | Runtime attestation for secure-default outputs |
| `reports/<name>/evaluation_report.md` | Human-friendly summary for reviewers |

Reports remain valid only for the same baseline reference, pairing assumptions,
dataset/tokenizer context, and scoped claim surface, and only while
`invarlock verify --json reports/<name>/evaluation.report.json` continues to pass
against the adjacent `runtime.manifest.json`.
