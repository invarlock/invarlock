# Example Reports

InvarLock emits both machine-readable certificates and human-friendly summaries.
Use the steps below to reproduce representative artifacts from the current release.

## 1. Generate a Certificate Bundle

```bash
invarlock run -c configs/tasks/causal_lm/release_cpu.yaml \
  --profile release --tier balanced --out runs/quant8_demo_baseline

invarlock run -c configs/edits/quant_rtn/8bit_full.yaml \
  --profile release --tier balanced \
  --baseline runs/quant8_demo_baseline/report.json \
  --out runs/quant8_demo

invarlock report --run runs/quant8_demo \
  --baseline runs/quant8_demo_baseline/report.json \
  --format cert \
  --output reports/quant8_demo
```

The last command writes `evaluation.cert.json` and `evaluation_certificate.md` under `reports/quant8_demo/`.
Each certificate contains:

- Model and edit metadata (model id, adapter, commit hash, edit plan)
- Drift / perplexity / RMT verdicts with paired bootstrap confidence intervals
- Guard diagnostics (spectral, variance, invariants) including predictive-gate notes
- Policy digest capturing tier thresholds and calibration choices

## 2. Create a Narrative Summary

```bash
invarlock report --run runs/quant8_demo --format markdown
```

The markdown report mirrors the certificate content but highlights:

- Baseline vs edited perplexity series
- Guard outcomes with links to supporting metrics
- Checklist of gates (PASS/FAIL) suitable for change-control review

## 3. Shareable Attachments

For audits, collect the following files:

| File | Purpose |
|------|---------|
| `runs/<name>/report.json` | Execution log, metrics, and guard telemetry |
| `reports/<name>/evaluation.cert.json` | Signed compliance payload |
| `reports/<name>/evaluation_certificate.md` | Human-friendly summary for reviewers |

Certificates remain valid as long as `invarlock verify reports/<name>/evaluation.cert.json` passes with the original baseline reference.
