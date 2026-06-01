# Observability Tests

Tests for InvarLock's telemetry, logging, metrics, health, and export surfaces.

## Module Map

| Module | Primary tests | Notes |
| --- | --- | --- |
| `core.py` | `test_observability_core.py` | Event emission and run context behavior. |
| `metrics.py` | `test_observability_metrics.py`, `test_observability_coverage_health_and_metrics.py` | Counters, gauges, histograms, and branch edges. |
| `alerting.py` | `test_observability_alerting.py`, `test_observability_alerting_import_safety.py`, `test_observability_coverage_alerting_and_exporters.py` | Alert dispatch, formatting, and import safety. |
| `exporters.py` | `test_observability_export_manager.py`, `test_observability_exported_metrics.py`, `test_observability_coverage_alerting_and_exporters.py` | JSONL/export manager behavior and optional exporter paths. |
| `health.py` | `test_observability_health.py`, `test_observability_coverage_health_and_metrics.py` | Dependency checks and system health summaries. |
| `utils.py` | `test_observability_utils.py`, `test_observability_coverage_utils.py` | Shared utility behavior and edge paths. |
| exceptions | `test_exceptions.py` | Exception formatting and handling. |

## Running Tests

```bash
PYTHONPATH=src pytest -q tests/observability
PYTHONPATH=src pytest -q tests/observability/test_observability_core.py
PYTHONPATH=src pytest -q tests/observability/test_observability_coverage_alerting_and_exporters.py
```

## Coverage

Observability modules are part of the core coverage surface enforced by
`make coverage-enforce`. Keep targeted branch tests in the
`test_observability_coverage_*.py` files when adding edge behavior, and avoid
reintroducing broad status tables that drift from the actual test tree.
