# Certificate Telemetry Fields

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Document telemetry fields embedded in certificates. |
| **Audience** | Operators validating latency/memory characteristics. |
| **Source of truth** | `report.json` telemetry values copied into certificates. |

## Quick Start

```bash
invarlock verify reports/telemetry/evaluation.cert.json
jq '.telemetry' reports/telemetry/evaluation.cert.json
```

## Concepts

- Telemetry values are copied from `report.json` and always include the device.
- CPU telemetry sweeps are collected via `scripts/run_cpu_telemetry.sh`.

## Reference

| JSON Pointer | Meaning | Notes |
| --- | --- | --- |
| `/telemetry/device` | Execution device (`cpu`, `mps`, `cuda`). | Mirrors `meta.device`. |
| `/telemetry/latency_ms_per_tok` | Mean latency per token. | ms/token. |
| `/telemetry/memory_mb_peak` | Peak resident memory. | MiB. |
| `/telemetry/preview_total_tokens` | Tokens processed in preview. | Derived from windows. |
| `/telemetry/final_total_tokens` | Tokens processed in final. | Derived from windows. |
| `/telemetry/throughput_tok_per_s` | Average throughput. | Present when available. |

## Troubleshooting

- **Telemetry missing**: ensure the run completed successfully and check
  `report.metrics` for latency/memory values.

## Observability

- `report.json` contains `metrics.latency_ms_per_tok` and `metrics.memory_mb_peak`.
- `telemetry.summary_line` is emitted when `INVARLOCK_TELEMETRY=1`.

## Related Documentation

- [Artifact Layout](artifacts.md)
- [Certificate Schema (v1)](certificate-schema.md)
- [CLI Reference](cli.md)
