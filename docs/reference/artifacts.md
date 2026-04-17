# Artifact Layout

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Explain where evaluation outputs and reports live. |
| **Audience** | Operators archiving evidence and CI outputs. |
| **Scope** | `runs/` scratch outputs and `reports/` long-lived evidence. |
| **Source of truth** | `src/invarlock/core/evaluate_contract.py`, `src/invarlock/core/evaluate_plan.py`, `src/invarlock/reporting/report_make.py`, `src/invarlock/reporting/report_bundle.py`, `src/invarlock/reporting/report_console.py`, `src/invarlock/reporting/report_files.py`, `src/invarlock/cli/commands/evaluate.py`. |

## Quick Start

```bash
# Compare baseline and subject on the secure-default runtime path
invarlock evaluate --allow-network \
  --baseline gpt2 \
  --subject gpt2 \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --report-out reports/eval

# Render HTML from the emitted evaluation bundle
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
```

Model-loading commands use the secure-default runtime container unless a trusted
`invarlock evaluate --execution-mode trusted-local` workflow explicitly bypasses it.

## Concepts

- `runs/` is scratch space: evaluate emits baseline/subject working artifacts there.
- `reports/` is evidence: archive `evaluation.report.json` and `runtime.manifest.json`
  for audit, plus any HTML or proof-pack outputs you distribute.
- evaluation bundles reference baseline/subject report artifacts; keep them
  together to preserve pairing and make later review easier.

### Command outputs

| Command | Writes | What to archive |
| --- | --- | --- |
| `invarlock evaluate` | `runs/`, `reports/<name>/evaluation.report.json`, `runtime.manifest.json` | Evaluation report bundle plus runtime provenance for container-backed runs. |
| `invarlock report html` | `reports/<name>/evaluation.html` | Optional (can be rebuilt). |

## Reference

### Evaluate scratch outputs (`runs/`)

```text
runs/
  baseline/
    ...
  subject/
    ...
```

### Evaluation reports (`reports/`)

```text
reports/
  eval/
    evaluation.report.json
    runtime.manifest.json
    evaluation.html
```

### Archive checklist

- Keep `evaluation.report.json` with `runtime.manifest.json`.
- Retain HTML exports only when you need reviewer-friendly artifacts.
- Retain scratch `runs/` only if debugging or rebuilding derived artifacts.
- Prune timestamped `runs/` once evidence is archived.

| Artifact | Why archive | Required for verify |
| --- | --- | --- |
| `evaluation.report.json` | Evaluation report snapshot | Yes |
| `runtime.manifest.json` | Runtime provenance for secure-default outputs | Yes |
| `events.jsonl` | Debugging timeline | No |
| `evaluation.html` | Human review | No |

### Seeds, hashes, and policy digests

- `report.meta.seeds` includes Python/NumPy/Torch seeds.
- `report.meta.tokenizer_hash` and dataset digests support pairing verification.
- reports record `policy_digest` and resolved tier policy snapshots.

### Cleanup checklist

1. Copy `evaluation.report.json` and `runtime.manifest.json` into `reports/`
   for retention.
2. Keep any referenced baseline/subject artifacts alongside derived reports for
   pairing checks and `report explain`.
3. Remove stale timestamped runs once evidence is archived.

## Troubleshooting

- **Missing pairing artifacts**: `report explain` and some advanced workflows
  need the baseline/subject artifacts referenced by the evaluation bundle.
- **Large run dirs**: prune old timestamped runs after archiving reports.

## Observability

- `evaluation.report.json` is the canonical distribution artifact.
- scratch run artifacts provide per-phase logs for debugging when needed.

## Related Documentation

- [reports](reports.md) — Schema, telemetry, and HTML export
- [CLI Reference](cli.md)
