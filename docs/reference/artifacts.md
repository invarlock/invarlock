# Artifact Layout

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Explain where evaluation outputs and reports live. |
| **Audience** | Operators archiving evidence and CI outputs. |
| **Scope** | `runs/` scratch outputs and `reports/` long-lived evidence. |
| **Source of truth** | `src/invarlock/core/evaluate_contract.py`, `src/invarlock/core/evaluate_plan.py`, `src/invarlock/reporting/report_make.py`, `src/invarlock/reporting/report_bundle.py`, `src/invarlock/reporting/report_summary.py`, `src/invarlock/cli/commands/evaluate.py`. |

## Quick Start

```bash
# Compare baseline and subject on the default runtime-container path
invarlock evaluate --allow-network \
  --baseline gpt2 \
  --subject gpt2 \
  --report-out reports/eval

# Render HTML from the emitted evaluation bundle
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
invarlock report explain --evaluation-report reports/eval/evaluation.report.json
invarlock report export -i reports/eval/evaluation.report.json --format release-review-md
```

Model-loading commands use the runtime container by default unless a
host-side `invarlock evaluate --execution-mode host` workflow explicitly
bypasses it.

Repo-owned presets under `configs/` remain available for maintainers, but the
quick-start path above stays wheel-compatible by using direct flags only.

## Concepts

- `runs/` is scratch space: evaluate emits baseline/subject working artifacts there.
- `reports/` is evidence: archive `evaluation.report.json` and `runtime.manifest.json`
  for audit, plus any HTML or evidence-pack outputs you distribute.
- evaluation bundles may reference baseline/subject report artifacts; keep them
  together when you want regeneration, deeper provenance review, or low-level
  run telemetry, but `evaluation.report.json` is the canonical portable artifact
  for verification, rendering, validation, and explanation.

### Command outputs

| Command | Writes | What to archive |
| --- | --- | --- |
| `invarlock evaluate` | `runs/`, `reports/<name>/evaluation.report.json`, `runtime.manifest.json` | Evaluation report bundle plus runtime provenance for container-backed runs. |
| `invarlock report html` | `reports/<name>/evaluation.html` | Optional (can be rebuilt). |
| `invarlock report export` | Optional output path for `mlflow-tags.json`, `model-card-invarlock.md`, or `release-review.md` | Optional release/registry convenience output (can be rebuilt). |

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
    mlflow-tags.json
    model-card-invarlock.md
    release-review.md
    invarlock-verify.json
```

### Archive checklist

- Keep `evaluation.report.json` with `runtime.manifest.json`.
- Retain HTML exports only when you need human-readable artifacts.
- Retain scratch `runs/` only if debugging or rebuilding derived artifacts.
- Prune timestamped `runs/` once evidence is archived.

| Artifact | Why archive | Required for verify |
| --- | --- | --- |
| `evaluation.report.json` | Evaluation report snapshot | Yes |
| `runtime.manifest.json` | Runtime provenance for container-backed outputs | Yes |
| `events.jsonl` | Debugging timeline | No |
| `evaluation.html` | Human review | No |
| `mlflow-tags.json` | Registry tag handoff | No |
| `model-card-invarlock.md` | Model-card evidence block | No |
| `release-review.md` | Release packet | No |
| `invarlock-verify.json` | Stored CI verify output | No |

### Seeds, hashes, and policy digests

- `report.meta.seeds` includes Python/NumPy/Torch seeds.
- `report.meta.tokenizer_hash` and dataset digests support pairing verification.
- reports record `policy_digest` and resolved tier policy snapshots.

### Cleanup checklist

1. Copy `evaluation.report.json` and `runtime.manifest.json` into `reports/`
   for retention.
2. Keep any referenced baseline/subject artifacts alongside derived reports when
   you need regeneration or low-level run telemetry.
3. Remove stale timestamped runs once evidence is archived.

## Troubleshooting

- **Missing pairing artifacts**: `report explain --evaluation-report` works from
  `evaluation.report.json`; use explicit `--subject-report`/`--baseline-report`
  only when you need to rebuild the explanation from raw run artifacts.
- **Large run dirs**: prune old timestamped runs after archiving reports.

## Observability

- `evaluation.report.json` is the canonical distribution artifact.
- scratch run artifacts provide per-phase logs for debugging when needed.

## Related Documentation

- [reports](reports.md) — Schema, telemetry, and HTML export
- [CLI Reference](cli.md)
