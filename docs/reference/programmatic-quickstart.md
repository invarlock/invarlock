# Programmatic Quickstart

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Minimal Python example for running InvarLock without the CLI. |
| **Audience** | Developers running small scripted experiments. |
| **Supported surface** | `CoreRunner.execute` and adapters/guards/edits from core packages. |
| **Requires** | `invarlock[adapters]` for HF adapters; `invarlock[edits]` for built-in edits; `invarlock[guards]` for guard math. |
| **Network** | Offline by default; set `INVARLOCK_ALLOW_NETWORK=1` for downloads. |

## Quick Start

```python
from invarlock.adapters import HF_Causal_Auto_Adapter
from invarlock.core.api import RunConfig
from invarlock.core.runner import CoreRunner
from invarlock.edits import RTNQuantEdit
from invarlock.guards import InvariantsGuard

adapter = HF_Causal_Auto_Adapter()
model = adapter.load_model("gpt2", device="auto")

edit = RTNQuantEdit(bitwidth=8, per_channel=True, group_size=128, clamp_ratio=0.005)
guards = [InvariantsGuard()]

config = RunConfig(device="auto")
report = CoreRunner().execute(model, adapter, edit, guards, config)

print("status:", report.status)
```

## Concepts

- Prefer the CLI for full workflows (pairing, certificates, reproducibility).
- Programmatic runs still follow the same pipeline phases and produce a
  `RunReport` object.
- Pass `calibration_data` to `CoreRunner.execute` for real primary-metric values.
- Enable downloads per run with `INVARLOCK_ALLOW_NETWORK=1` when using remote
  model IDs.

## Reference

- `CoreRunner.execute` signature: see [API Guide](api-guide.md).
- Built-in adapters: see [Model Adapters](model-adapters.md).
- Guard policies and tiers: see [Guards](guards.md).

## Troubleshooting

- **Dependency missing**: install `invarlock[adapters]` or `invarlock[guards]`.
- **Downloads blocked**: use `INVARLOCK_ALLOW_NETWORK=1` for HF downloads.

## Observability

- Inspect `report.meta`, `report.guards`, and `report.metrics`.
- For certificate generation, use `invarlock.assurance.make_certificate`.

## Related Documentation

- [API Guide](api-guide.md)
- [CLI Reference](cli.md)
- [Compare & Certify (BYOE)](../user-guide/compare-and-certify.md)
- [Primary Metric Smoke](../user-guide/primary-metric-smoke.md)
