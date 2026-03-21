# Artifact Layout

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Explain where run outputs and reports live. |
| **Audience** | Operators archiving evidence and CI outputs. |
| **Scope** | `runs/` scratch outputs and `reports/` long-lived evidence. |
| **Source of truth** | CLI run/report commands (`src/invarlock/cli/commands/run.py`). |

## Quick Start

```bash
# Baseline run on the secure-default runtime path
invarlock run -c configs/presets/causal_lm/wikitext2_512.yaml --out runs/baseline

# Generate report
invarlock report --run runs/baseline/report.json --format report --output reports/baseline
```

Model-loading commands use the secure-default runtime container unless a trusted
workflow explicitly opts into `--allow-host-execution`.

## Concepts

- `runs/` is scratch space: timestamped run directories with `report.json` + `events.jsonl`.
- `reports/` is evidence: copy `report.json`, `evaluation.report.json`, and
  `runtime.manifest.json` for audit when they are emitted.
- reports reference baseline reports; keep them together to preserve pairing.

### Command outputs

| Command | Writes | What to archive |
| --- | --- | --- |
| `invarlock run` | `runs/<name>/<timestamp>/report.json`, `events.jsonl` | Baseline + subject `report.json`. |
| `invarlock report --format report` | `reports/<name>/evaluation.report.json`, `runtime.manifest.json` | report + baseline report + runtime manifest. |
| `invarlock report html` | `reports/<name>/evaluation.html` | Optional (can be rebuilt). |

## Reference

### Run outputs (`runs/`)

```text
runs/
  baseline/
    20251010_182515/
      report.json
      events.jsonl
  quant8/
    20251010_151826/
      report.json
      events.jsonl
```

### Reports and reports (`reports/`)

```text
reports/
  baseline/
    report.json
  quant8_balanced/
    evaluation.report.json
    runtime.manifest.json
    report.json
```

### Archive checklist

- Move baseline + subject `report.json` into `reports/`.
- Keep `evaluation.report.json` with the baseline report and
  `runtime.manifest.json`.
- Retain `events.jsonl` only if debugging; HTML exports are optional.
- Prune timestamped `runs/` once evidence is archived.

| Artifact | Why archive | Required for verify |
| --- | --- | --- |
| `report.json` (baseline + subject) | Metrics, windows, provenance | Yes |
| `evaluation.report.json` | Evaluation report snapshot | Yes |
| `runtime.manifest.json` | Runtime attestation for secure-default outputs | Yes |
| `events.jsonl` | Debugging timeline | No |
| `evaluation.html` | Human review | No |

### Seeds, hashes, and policy digests

- `report.meta.seeds` includes Python/NumPy/Torch seeds.
- `report.meta.tokenizer_hash` and dataset digests support pairing verification.
- reports record `policy_digest` and resolved tier policy snapshots.

### Cleanup checklist

1. Copy `report.json`, `evaluation.report.json`, and `runtime.manifest.json`
   into `reports/` for retention.
2. Keep baseline reports alongside derived reports for pairing checks.
3. Remove stale timestamped runs once evidence is archived.

## Troubleshooting

- **Missing baseline report**: reports cannot be validated without the
  baseline `report.json`; keep it alongside the report.
- **Large run dirs**: prune old timestamped runs after archiving reports.

## Observability

- `report.json` is the canonical source for metrics/guards.
- `events.jsonl` provides per-phase logs for debugging.

## Related Documentation

- [reports](reports.md) — Schema, telemetry, and HTML export
- [CLI Reference](cli.md)
