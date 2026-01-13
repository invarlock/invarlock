# Example Reports

InvarLock emits both machine-readable certificates and human-friendly summaries.
Use the steps below to reproduce representative artifacts from the current release.

## 1. Generate a Certificate Bundle

```bash
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock certify \
  --baseline sshleifer/tiny-gpt2 \
  --subject  sshleifer/tiny-gpt2 \
  --adapter auto \
  --profile release \
  --tier balanced \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --edit-config configs/overlays/edits/quant_rtn/8bit_full.yaml \
  --out runs/quant8_demo \
  --cert-out reports/quant8_demo
```

The command writes `evaluation.cert.json` and `evaluation_certificate.md` under `reports/quant8_demo/`.
Each certificate contains:

- Model and edit metadata (model id, adapter, commit hash, edit plan)
- Drift / perplexity / RMT verdicts with paired bootstrap confidence intervals
- Guard diagnostics (spectral, variance, invariants) including predictive-gate notes
- Policy digest capturing tier thresholds and calibration choices

## 2. Create a Narrative Summary

```bash
# The certificate already includes a markdown summary:
cat reports/quant8_demo/evaluation_certificate.md

# To regenerate markdown from run reports, pass edited + baseline:
invarlock report --run <edited_report.json> --baseline <baseline_report.json> --format markdown
```

The markdown report mirrors the certificate content but highlights:

- Baseline vs edited perplexity series
- Guard outcomes with links to supporting metrics
- Checklist of gates (PASS/FAIL) suitable for change-control review

## 3. Shareable Attachments

For audits, collect the following files:

| File | Purpose |
|------|---------|
| `runs/<name>/**/report.json` | Execution log, metrics, and guard telemetry |
| `reports/<name>/evaluation.cert.json` | Signed compliance payload |
| `reports/<name>/evaluation_certificate.md` | Human-friendly summary for reviewers |

Certificates remain valid as long as `invarlock verify reports/<name>/evaluation.cert.json` passes with the original baseline reference.
