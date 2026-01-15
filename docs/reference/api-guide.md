# API Guide

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Programmatic interface for running the InvarLock pipeline and generating certificates. |
| **Audience** | Python callers building scripted workflows or integrations. |
| **Supported surface** | `CoreRunner.execute`, `RunConfig`, `ModelAdapter`, `ModelEdit`, `Guard`, `invarlock.assurance` helpers. |
| **Requires** | `invarlock[adapters]` for HF adapters, `invarlock[edits]` for built-in edits, `invarlock[guards]` for guard math, `invarlock[eval]` for dataset providers. |
| **Network** | Offline by default; set `INVARLOCK_ALLOW_NETWORK=1` to download models or datasets. |
| **Inputs** | Model instance, adapter, edit, guard list, `RunConfig`, optional calibration data. |
| **Outputs / Artifacts** | `RunReport` object; optional event logs/checkpoints; certificates via `make_certificate`. |
| **Source of truth** | `src/invarlock/core/runner.py`, `src/invarlock/core/api.py`, `src/invarlock/assurance/__init__.py`. |

## Quick Start

```python
from invarlock.adapters import HF_Causal_Auto_Adapter
from invarlock.core.api import RunConfig
from invarlock.core.runner import CoreRunner
from invarlock.edits import RTNQuantEdit
from invarlock.guards import InvariantsGuard, SpectralGuard

adapter = HF_Causal_Auto_Adapter()
model = adapter.load_model("gpt2", device="auto")

edit = RTNQuantEdit(bitwidth=8, per_channel=True, group_size=128, clamp_ratio=0.005)
guards = [InvariantsGuard(), SpectralGuard(sigma_quantile=0.95, deadband=0.10)]

config = RunConfig(device="auto")
report = CoreRunner().execute(model, adapter, edit, guards, config)

print("status:", report.status)
print("primary metric:", report.metrics.get("primary_metric"))
```

> For real primary-metric values, pass `calibration_data` (see Concepts). Without it,
> the runner falls back to lightweight mock metrics so the pipeline can still finish.

## Concepts

- **Pipeline phases**: prepare → guard prepare → edit → guard validate → eval → finalize/rollback.
- **Calibration data**: indexable batches (list/sequence) with `input_ids`, optional
  `attention_mask`, and optional `labels`. Preview/final windows are sliced from this sequence.
- **Auto configuration**: `auto_config` controls tier/policy resolution and is recorded
  under `report.meta["auto"]` for certificate generation.
- **Snapshots**: retries use snapshot/restore; configure via
  `context.snapshot.*` when using YAML configs.
- **Certificates**: generated from `RunReport` + baseline report via
  `invarlock.assurance.make_certificate`.

### Responsibility lanes

| Lane | Responsibility |
| --- | --- |
| User code | Build `RunConfig`, call `execute`, consume `RunReport`. |
| CoreRunner | Orchestrate phases, apply edit, assemble status + metrics. |
| Adapter | Load/describe model, snapshot/restore. |
| Guards | `prepare`/`validate`, return action (warn/rollback/abort). |
| Eval | Build windows, compute primary metric + tail metrics. |
| Certificate | `make_certificate(report, baseline)` for verification. |

Note: CoreRunner coordinates each lane.

## Reference

### CoreRunner.execute

`CoreRunner.execute` is the only supported entry point for programmatic runs.

```python
report = CoreRunner().execute(
    model,
    adapter,
    edit,
    guards,
    config,
    calibration_data=calibration_data,
    auto_config=auto_config,
    edit_config=edit_config,
    preview_n=preview_n,
    final_n=final_n,
)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `model` | `Any` | Loaded model instance. |
| `adapter` | `ModelAdapter` | Adapter that can describe/snapshot/restore the model. |
| `edit` | `ModelEdit` or `EditLike` | Edit operation to apply. |
| `guards` | `list[Guard]` | Guard instances to validate after edit. |
| `config` | `RunConfig` | Runtime settings (device, thresholds, event logs). |
| `calibration_data` | `Any` | Optional calibration batches for evaluation. |
| `auto_config` | `dict[str, Any]` | Optional tier/policy hints (recorded into report meta). |
| `edit_config` | `dict[str, Any]` | Overrides passed to `edit.apply(...)`. |
| `preview_n` / `final_n` | `int \| None` | Override preview/final counts; defaults to slicing calibration data. |

### RunConfig

`RunConfig` controls runtime behavior in the core runner.

| Field | Default | Notes |
| --- | --- | --- |
| `device` | `"auto"` | Resolves to CUDA → MPS → CPU. |
| `max_pm_ratio` | `1.5` | Max acceptable primary-metric ratio before rollback. |
| `spike_threshold` | `2.0` | Catastrophic spike ratio for immediate rollback. |
| `event_path` | `None` | Path to JSONL event log (optional). |
| `checkpoint_interval` | `0` | 0 disables checkpoints. |
| `dry_run` | `False` | Skip mutations, still produce report. |
| `verbose` | `False` | Enables extra logging. |
| `context` | `{}` | Free-form context passed to guards/eval. |

### Auto config hints

`auto_config` is recorded in `report.meta["auto"]` and used for tier resolution.

| Key | Meaning |
| --- | --- |
| `enabled` | Whether auto mode is enabled. |
| `tier` | Tier label (`balanced`, `conservative`, `aggressive`). |
| `probes` | Micro-probe count (0–10). |
| `target_pm_ratio` | Target ratio for auto tuning (CLI default: 2.0). |

### RunReport fields

| Field | Description |
| --- | --- |
| `meta` | Execution metadata (device, seeds, config snapshot). |
| `edit` | Edit metadata and deltas. |
| `guards` | Guard results keyed by guard name. |
| `metrics` | Primary metric + telemetry values. |
| `evaluation_windows` | Captured preview/final windows (if enabled). |
| `status` | `pending`, `running`, `success`, `failed`, or `rollback`. |
| `error` | Error string when `status=failed`. |
| `context` | Run context propagated to guards/eval. |

### Failure outcomes

| Outcome | Trigger | RunReport evidence |
| --- | --- | --- |
| Warn | Guard returns `action: warn`. | `report.guards[].warnings`; `report.status = success`. |
| Rollback | Guard failures or primary-metric gates fail. | `report.status = rollback`; `report.meta.rollback_reason`. |
| Abort | Exceptions or `action: abort`. | `report.status = failed`; `report.error`. |

### Interfaces

`ModelAdapter`, `ModelEdit`, and `Guard` are defined in `invarlock.core.api`.

```python
from invarlock.core.api import Guard, ModelAdapter, ModelEdit

class CustomGuard(Guard):
    name = "custom_guard"

    def prepare(self, model, adapter, calib, policy):
        return {"ready": True}

    def validate(self, model, adapter, context):
        return {"passed": True, "action": "warn", "metrics": {"ok": 1}}
```

Notes:

- The runner calls `prepare(...)` when the guard implements it (`GuardWithPrepare`).
- `validate(...)` is always called during the guard phase.
- Optional lifecycle helpers (`before_edit`, `after_edit`, `finalize`) are only
  invoked when you manage guards manually (for example via `GuardChain`).

### GuardChain helper

`GuardChain` provides lifecycle helpers for manually coordinating guard calls:

```python
from invarlock.core.api import GuardChain

chain = GuardChain([guard])
chain.prepare_all(model, adapter, calib, policy_config)
chain.before_edit_all(model)
chain.after_edit_all(model)
chain.finalize_all(model)
```

### Calibration data format

Calibration batches should be indexable and yield dict-like objects:

```python
batch = {
    "input_ids": [[101, 102, 103]],
    "attention_mask": [[1, 1, 1]],
    # optional
    "labels": [[101, 102, 103]],
}
```

If your calibration data is an iterator without `__len__`, set
`INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE=1` to allow the runner to materialize it.

### Evaluation window helpers

You can build calibration batches from dataset providers:

```python
from invarlock.eval.data import get_provider

provider = get_provider("wikitext2")
preview, final = provider.windows(
    tokenizer,
    preview_n=64,
    final_n=64,
    seq_len=512,
    stride=512,
)

calibration = [
    {"input_ids": ids, "attention_mask": mask}
    for ids, mask in zip(
        preview.input_ids + final.input_ids,
        preview.attention_masks + final.attention_masks,
        strict=False,
    )
]
```

### Certificates (assurance helpers)

```python
from invarlock.assurance import make_certificate, render_certificate_markdown, validate_certificate

certificate = make_certificate(report, baseline_report)
validate_certificate(certificate)
print(render_certificate_markdown(certificate))
```

### Exceptions

Core exceptions live in `invarlock.core.exceptions`:

- `ModelLoadError`, `AdapterError`, `EditError`, `GuardError`, `ConfigError`
- `InvarlockError` (base class)

## Troubleshooting

- **`DEPENDENCY-MISSING` during adapter load**: install the matching extra (e.g.,
  `pip install "invarlock[adapters]"`) and retry.
- **`No calibration data provided` warnings**: pass `calibration_data` to
  `CoreRunner.execute` (or use the CLI, which handles datasets automatically).
- **Calibration data not indexable**: pass a list/sequence or set
  `INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE=1` to allow materialization.
- **Guard prepare failures in CI/Release**: adjust guard policies or set
  `INVARLOCK_GUARD_PREPARE_STRICT=0` for local debugging only.

## Observability

- `RunReport.meta`, `RunReport.guards`, `RunReport.metrics`, and
  `RunReport.evaluation_windows` are the canonical inspection points (windows can
  be omitted when `INVARLOCK_STORE_EVAL_WINDOWS=0`).
- If `RunConfig.event_path` is set, an event log is written as JSONL.
- Certificates from `make_certificate` can be validated with
  `invarlock.assurance.validate_certificate` or the CLI `invarlock verify`.

## Related Documentation

- [Programmatic Quickstart](programmatic-quickstart.md)
- [CLI Reference](cli.md)
- [Configuration Schema](config-schema.md)
- [Dataset Providers](datasets.md)
- [Guards](guards.md)
- [Certificates](certificates.md) — Schema, telemetry, and HTML export
- [Determinism Contracts](../assurance/08-determinism-contracts.md) — Reproducibility guarantees
- [Observability](observability.md) — Monitoring and telemetry
