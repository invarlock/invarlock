# Programmatic Quickstart

Minimal, self-contained example for running InvarLock from Python. Prefer the CLI
for end-to-end workflows; use the programmatic surface for small, scripted
experiments.

## Minimal run (demo edit)

```python
from invarlock.core.runner import CoreRunner
from invarlock.core.api import RunConfig
from invarlock.adapters import HF_Causal_Auto_Adapter
from invarlock.edits import RTNQuantEdit
from invarlock.guards import InvariantsGuard

# 1) Resolve adapter and load a small model (auto selects hf_gpt2/llama/bert)
adapter = HF_Causal_Auto_Adapter()
model = adapter.load_model("gpt2", device="auto")

# 2) Use the built-in demo edit (RTN INT8) and a minimal guard set
edit = RTNQuantEdit(bitwidth=8, per_channel=True, group_size=128, clamp_ratio=0.005)
guards = [InvariantsGuard()]

# 3) Configure runtime (device, context). For full control, see API Guide.
cfg = RunConfig(device="auto")

# 4) Execute the pipeline and inspect the report
runner = CoreRunner()
report = runner.execute(model, adapter, edit, guards, cfg)

print("device:", report.meta.get("device"))
print("seed bundle:", report.meta.get("seeds"))
print("edit name:", report.edit.get("name"))
```

Notes

- Network is disabled by default. Enable downloads per run with:
  `INVARLOCK_ALLOW_NETWORK=1 python your_script.py`.
- The runner reuses a single loaded model and snapshots state for retries.
  See config docs for `context.snapshot.*` controls.

## Next steps

- [Compare & Certify (BYOE)](../user-guide/compare-and-certify.md)
- [Primary Metric Smoke](../user-guide/primary-metric-smoke.md)
- [API Guide](./api-guide.md)
- [CLI Reference](./cli.md)
