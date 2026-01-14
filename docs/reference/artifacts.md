# Artifact Layout

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Explain where run outputs and certificates live. |
| **Audience** | Operators archiving evidence and CI outputs. |
| **Scope** | `runs/` scratch outputs and `reports/` long-lived evidence. |
| **Source of truth** | CLI run/report commands (`src/invarlock/cli/commands/run.py`). |

## Quick Start

```bash
# Run baseline
invarlock run -c configs/presets/causal_lm/wikitext2_512.yaml --out runs/baseline

# Generate certificate
invarlock report --run runs/baseline/report.json --format cert --output reports/baseline
```

## Concepts

- `runs/` is scratch space: timestamped run directories with `report.json` + `events.jsonl`.
- `reports/` is evidence: copy `report.json` and certificates for audit.
- Certificates reference baseline reports; keep them together to preserve pairing.

### Command outputs

| Command | Writes | What to archive |
| --- | --- | --- |
| `invarlock run` | `runs/<name>/<timestamp>/report.json`, `events.jsonl` | Baseline + subject `report.json`. |
| `invarlock report --format cert` | `reports/<name>/evaluation.cert.json` | Certificate + baseline report. |
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

### Reports and certificates (`reports/`)

```text
reports/
  baseline/
    report.json
  quant8_balanced/
    evaluation.cert.json
    report.json
```

### Archive checklist

- Move baseline + subject `report.json` into `reports/`.
- Keep `evaluation.cert.json` with the baseline report.
- Retain `events.jsonl` only if debugging; HTML exports are optional.
- Prune timestamped `runs/` once evidence is archived.

| Artifact | Why archive | Required for verify |
| --- | --- | --- |
| `report.json` (baseline + subject) | Metrics, windows, provenance | Yes |
| `evaluation.cert.json` | Safety certificate snapshot | Yes |
| `events.jsonl` | Debugging timeline | No |
| `evaluation.html` | Human review | No |

### Seeds, hashes, and policy digests

- `report.meta.seeds` includes Python/NumPy/Torch seeds.
- `report.meta.tokenizer_hash` and dataset digests support pairing verification.
- Certificates record `policy_digest` and resolved tier policy snapshots.

### Cleanup checklist

1. Copy `report.json` and `evaluation.cert.json` into `reports/` for retention.
2. Keep baseline reports alongside derived certificates for pairing checks.
3. Remove stale timestamped runs once evidence is archived.

## Troubleshooting

- **Missing baseline report**: certificates cannot be validated without the
  baseline `report.json`; keep it alongside the certificate.
- **Large run dirs**: prune old timestamped runs after archiving certificates.

## Observability

- `report.json` is the canonical source for metrics/guards.
- `events.jsonl` provides per-phase logs for debugging.

## Related Documentation

- [Certificate Schema (v1)](certificate-schema.md)
- [CLI Reference](cli.md)
- [Exporting Certificates (HTML)](exporting-certificates-html.md)
