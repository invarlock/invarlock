# Programmatic Quickstart

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Minimal Python example for running InvarLock without the CLI. |
| **Audience** | Developers running small scripted experiments. |
| **Supported surface** | `CoreRunner.execute` and adapters/guards/edits from core packages. |
| **Requires** | `invarlock[adapters]` for HF adapters; `invarlock[edits]` for built-in edits; `invarlock[guards]` for guard math. |
| **Network** | Offline by default; CLI runs use `evaluate --allow-network`, while programmatic callers set `INVARLOCK_ALLOW_NETWORK=1` for downloads. |

## Quick Start

```python
from invarlock.adapters.auto import HF_Auto_Adapter
from invarlock.core.api import RunConfig
from invarlock.core.runner import CoreRunner
from invarlock.edits import RTNQuantEdit
from invarlock.guards.invariants import InvariantsGuard

adapter = HF_Auto_Adapter()
model = adapter.load_model("gpt2", device="auto")

edit = RTNQuantEdit(bitwidth=8, per_channel=True, group_size=128, clamp_ratio=0.005)
guards = [InvariantsGuard()]

config = RunConfig(device="auto")
report = CoreRunner().execute(model, adapter, edit, guards, config)

print("status:", report.status)
```

## Concepts

- Prefer the CLI for full workflows (pairing, reports, reproducibility).
- Programmatic runs follow the same pipeline phases and produce a
  `RunReport` object.
- Pass `calibration_data` to `CoreRunner.execute` for real primary-metric values.
- CLI workflows use `evaluate --allow-network`; programmatic runs set
  `INVARLOCK_ALLOW_NETWORK=1` when using remote model IDs.

## Reference

- `CoreRunner.execute` signature: see [API Guide](api-guide.md).
- Built-in adapters: see [Model Adapters](model-adapters.md).
- Guard policies and tiers: see [Guards](guards.md).

## Troubleshooting

- **Dependency missing**: install `invarlock[adapters]` or `invarlock[guards]`.
- **Downloads blocked**: for the CLI, rerun with `evaluate --allow-network`;
  for programmatic code, set `INVARLOCK_ALLOW_NETWORK=1`.

## Observability

- Inspect `report.meta`, `report.guards`, and `report.metrics`.
- For report generation, use `invarlock.reporting.report_make.make_report`, then
  persist the evaluation bundle with `invarlock.reporting.report_bundle.save_evaluation_bundle`.

## Related Documentation

- [API Guide](api-guide.md)
- [CLI Reference](cli.md)
- [Compare & evaluate (BYOE)](../user-guide/compare-and-evaluate.md)
- [Primary Metric Smoke](../user-guide/primary-metric-smoke.md)
