# API Guide

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Programmatic interface for running the InvarLock pipeline and generating reports. |
| **Audience** | Python callers building scripted workflows or integrations. |
| **Supported surface** | Stable contract surfaces remain CLI/report/contract-read paths; `CoreRunner.execute`, `RunConfig`, `ModelAdapter`, `ModelEdit`, `Guard`, and direct reporting helpers are advanced/non-stable. |
| **Requires** | `invarlock[adapters]` for HF adapters, `invarlock[edits]` for built-in edits, `invarlock[guards]` for guard math, `invarlock[eval]` for dataset providers. |
| **Network** | Offline by default; CLI runs use `evaluate --allow-network`, while Python callers set `INVARLOCK_ALLOW_NETWORK=1` to download models or datasets. |
| **Inputs** | Model instance, adapter, edit, guard list, `RunConfig`, optional calibration data. |
| **Outputs / Artifacts** | `RunReport` object; optional event logs/checkpoints; evaluation bundles via `invarlock.reporting.make_report(...)` and `report_bundle.save_evaluation_bundle(...)`. |
| **Source of truth** | `src/invarlock/core/runner.py`, `src/invarlock/core/api.py`, `src/invarlock/cli/config_execution.py`, `src/invarlock/reporting/report_make.py`, `src/invarlock/reporting/report_make_assembly.py`, `src/invarlock/reporting/report_bundle.py`, `src/invarlock/reporting/report_summary.py`, `src/invarlock/reporting/report_schema.py`. |

## Quick Start

```python
from invarlock.adapters.auto import HF_Auto_Adapter
from invarlock.core.api import RunConfig
from invarlock.core.runner import CoreRunner
from invarlock.edits import RTNQuantEdit
from invarlock.guards.invariants import InvariantsGuard
from invarlock.guards.spectral import SpectralGuard

adapter = HF_Auto_Adapter()
model = adapter.load_model("gpt2", device="auto")

edit = RTNQuantEdit(bitwidth=8, per_channel=True, clamp_ratio=0.005)
guards = [InvariantsGuard(), SpectralGuard(sigma_quantile=0.95, deadband=0.10)]

config = RunConfig(device="auto")
report = CoreRunner().execute(model, adapter, edit, guards, config)

print("status:", report.status)
print("primary metric:", report.metrics.get("primary_metric"))
```

> For real primary-metric values, pass `calibration_data` (see Concepts). Without it,
> the runner uses lightweight mock metrics so the pipeline can finish.

## Concepts

- **Pipeline phases**: prepare → guard prepare → edit → guard validate → eval → finalize/rollback.
- **Calibration data**: indexable batches (list/sequence) with `input_ids`, optional
  `attention_mask`, and optional `labels`. Preview/final windows are sliced from this sequence.
- **Auto configuration**: `auto_config` controls tier/policy resolution and is recorded
  under `report.meta["auto"]` for report generation.
- **Snapshots**: retries use snapshot/restore; configure via
  `context.snapshot.*` when using YAML configs.
- **reports**: generated from `RunReport` + baseline report via
  `invarlock.reporting.make_report`, then persisted as an
  evaluation bundle with `invarlock.reporting.report_bundle.save_evaluation_bundle`.
- **Verification**: CLI-side `invarlock verify` enforces
  `runtime.manifest.json` runtime provenance for container-backed outputs in
  addition to schema and pairing checks.

### Responsibility lanes

| Lane | Responsibility |
| --- | --- |
| User code | Build `RunConfig`, call `execute`, consume `RunReport`. |
| CoreRunner | Orchestrate phases, apply edit, assemble status + metrics. |
| Adapter | Load/describe model, snapshot/restore. |
| Guards | `prepare`/`validate`, return typed decisions (`allow`/`monitor`/`rollback`/`block`). |
| Eval | Build windows, compute primary metric + tail metrics. |
| report | `make_report(report, baseline)` + `save_evaluation_bundle(...)` for evaluation-bundle generation. |

Note: CoreRunner coordinates each lane.

## Reference

### CoreRunner.execute

`CoreRunner.execute` is the primary entry point for advanced/non-stable
programmatic runs.

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
| `dry_run` | `False` | Skip mutations and produce a report. |
| `verbose` | `False` | Enables extra logging. |
| `context` | `{}` | Free-form context passed to guards/eval. |

### Auto config hints

`auto_config` is recorded in `report.meta["auto"]` and used for tier resolution.

| Key | Meaning |
| --- | --- |
| `enabled` | Whether auto mode is enabled. |
| `tier` | Tier label (`balanced`, `conservative`, `aggressive`). |
| `probes` | Micro-probe count (0–10). |
| `target_pm_ratio` | Target ratio for auto tuning (default: 1.0 when unset). |

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
| Monitor | Guard returns `decision: monitor`. | `report.guards[].decision = monitor`; `report.status = success`. |
| Rollback | Guard returns `decision: rollback`, or guard/primary-metric gates fail. | `report.status = rollback`; `report.meta.rollback_reason`. |
| Failed | Unrecoverable runner exception. | `report.status = failed`; `report.error`. |

### Interfaces

`ModelAdapter`, `ModelEdit`, and `Guard` are defined in `invarlock.core.api`.

```python
from invarlock.core.api import Guard, ModelAdapter, ModelEdit

class CustomGuard(Guard):
    name = "custom_guard"

    def prepare(self, model, adapter, calib, policy):
        return {"ready": True}

    def validate(self, model, adapter, context):
        return {"passed": True, "decision": "monitor", "metrics": {"ok": 1}}
```

Notes:

- The runner calls `prepare(...)` when the guard implements it (`GuardWithPrepare`).
- `validate(...)` is always called during the guard phase.
- `validate(...)` should emit the typed decision vocabulary:
  `allow`, `monitor`, `rollback`, or `block`.
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

### reports (canonical helpers)

```python
from invarlock.reporting.render import render_report_markdown
from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_schema import validate_report

report = make_report(report, baseline_report)
validate_report(report)
print(render_report_markdown(report))
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
  `context.run.strict_guard_prepare: false` for local debugging only.

## Observability

- `RunReport.meta`, `RunReport.guards`, `RunReport.metrics`, and
  `RunReport.evaluation_windows` are the canonical inspection points (windows can
  be omitted when `INVARLOCK_STORE_EVAL_WINDOWS=0`).
- If `RunConfig.event_path` is set, an event log is written as JSONL.
- reports from `make_report` can be validated with
  `invarlock.reporting.report_schema.validate_report` or the CLI `invarlock verify`.

## Related Documentation

- [Programmatic Quickstart](programmatic-quickstart.md)
- [CLI Reference](cli.md)
- [Configuration Schema](config-schema.md)
- [Dataset Providers](datasets.md)
- [Guards](guards.md)
- [reports](reports.md) — Schema, telemetry, and HTML export
- [Determinism Contracts](../assurance/08-determinism-contracts.md) — Reproducibility guarantees
- [Observability](observability.md) — Monitoring and telemetry
