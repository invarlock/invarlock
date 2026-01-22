# Observability

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Runtime monitoring, health checking, and telemetry collection for InvarLock operations. |
| **Audience** | Operators running production workloads and developers debugging performance issues. |
| **Supported surface** | `MonitoringManager`, `HealthChecker`, `TelemetryCollector`, `MetricsRegistry`. |
| **Requires** | `psutil` (included in base install); `torch` for GPU monitoring. |
| **Network** | Fully offline; no external telemetry is sent. |
| **Source of truth** | `src/invarlock/observability/*.py`. |

## Quick Start

```python
from invarlock.observability import MonitoringManager, MonitoringConfig

# Start monitoring with default config
config = MonitoringConfig(
    metrics_interval=10.0,
    health_check_interval=30.0,
)
monitor = MonitoringManager(config)
monitor.start()

# Record an operation
monitor.record_operation("certify", duration=45.2, model_id="gpt2")

# Get current status
status = monitor.get_status()
print(status["health_status"])

# Stop when done
monitor.stop()
```

## Concepts

- **Metrics**: counters, gauges, and histograms tracked during runs.
- **Health checks**: periodic checks for system resources (CPU, memory, disk, GPU).
- **Telemetry**: operation tracking with start/end times and metadata.
- **Alerting**: configurable thresholds for warnings (not external notifications).

### Component Architecture

| Component | Responsibility |
| --- | --- |
| `MonitoringManager` | Central coordinator; starts/stops monitoring threads. |
| `MetricsRegistry` | Stores counters, gauges, and histograms by name. |
| `HealthChecker` | Runs periodic health checks on system components. |
| `TelemetryCollector` | Tracks operation lifecycles (start → end). |
| `ResourceMonitor` | Collects CPU, memory, disk, and GPU usage. |
| `PerformanceMonitor` | Tracks operation durations and percentiles. |
| `AlertManager` | Evaluates threshold rules and records violations. |

## Reference

### MonitoringConfig

```python
from invarlock.observability import MonitoringConfig

config = MonitoringConfig(
    # Collection intervals (seconds)
    metrics_interval=10.0,
    health_check_interval=30.0,
    resource_check_interval=5.0,
    
    # Data retention
    metrics_retention_hours=24,
    max_events=10000,
    
    # Alerting
    enable_alerting=True,
    alert_channels=[],
    
    # Export settings
    prometheus_enabled=False,
    prometheus_port=9090,
    json_export_enabled=True,
    json_export_path="./monitoring",
    
    # Resource thresholds (percent)
    cpu_threshold=80.0,
    memory_threshold=85.0,
    gpu_memory_threshold=90.0,
    
    # Performance monitoring
    latency_percentiles=[50, 90, 95, 99],
    slow_request_threshold=30.0,
)
```

### Default Metrics

InvarLock registers these metrics automatically:

| Metric | Type | Description |
| --- | --- | --- |
| `'invarlock.operations.total'` | Counter | Total operations by type and status. |
| `'invarlock.errors.total'` | Counter | Total errors by type. |
| `'invarlock.edits.applied'` | Counter | Total edits applied. |
| `'invarlock.guards.triggered'` | Counter | Guard triggers. |
| `'invarlock.operation.duration'` | Histogram | Operation duration distribution. |
| `'invarlock.edit.duration'` | Histogram | Edit operation duration. |
| `'invarlock.guard.duration'` | Histogram | Guard execution duration. |
| `'invarlock.memory.usage'` | Gauge | Current memory usage. |
| `'invarlock.gpu.memory.usage'` | Gauge | Current GPU memory usage. |
| `'invarlock.cpu.usage'` | Gauge | Current CPU usage. |
| `'invarlock.model.parameters'` | Gauge | Model parameter count. |
| `'invarlock.model.size_mb'` | Gauge | Model size in MB. |
| `'invarlock.model.loads'` | Counter | Model loads. |

### Health Checks

The `HealthChecker` runs these checks by default:

| Check | Status Thresholds | Details |
| --- | --- | --- |
| `memory` | WARNING > 80%, CRITICAL > 90% | System RAM usage. |
| `cpu` | WARNING > 85%, CRITICAL > 95% | CPU utilization. |
| `disk` | WARNING > 85%, CRITICAL > 95% | Disk space on `/`. |
| `gpu` | WARNING > 85%, CRITICAL > 95% | GPU memory (if CUDA available). |
| `pytorch` | CRITICAL on failure | PyTorch functionality test. |

InvarLock-specific checks (via `InvarLockHealthChecker`):

| Check | Description |
| --- | --- |
| `adapters` | Verifies adapter classes can be instantiated. |
| `guards` | Verifies guard classes can be instantiated. |
| `dependencies` | Checks for torch, transformers, numpy, psutil. |

### Health Status API

```python
from invarlock.observability import InvarLockHealthChecker

checker = InvarLockHealthChecker()

# Check all components
results = checker.check_all()
for name, health in results.items():
    print(f"{name}: {health.status.value} - {health.message}")

# Get overall status
overall = checker.get_overall_status()
print(f"Overall: {overall.value}")

# Get summary
summary = checker.get_summary()
print(summary["status_counts"])
```

### TelemetryCollector

Track operation lifecycles:

```python
from invarlock.observability import MonitoringManager, TelemetryCollector

monitor = MonitoringManager()
telemetry = TelemetryCollector(monitor)

# Start tracking
op_id = telemetry.start_operation(
    "op-123",
    "certify",
    model_id="gpt2",
    profile="ci",
)

# ... perform operation ...

# End tracking
telemetry.end_operation(
    op_id,
    status="success",
    ratio_vs_baseline=1.02,
)

# Get stats
stats = telemetry.get_operation_stats()
print(f"Success rate: {stats['success_rate']:.2%}")
```

### ResourceMonitor

```python
from invarlock.observability.core import ResourceMonitor, MonitoringConfig
from invarlock.observability import MetricsRegistry

config = MonitoringConfig()
metrics = MetricsRegistry()
resource_monitor = ResourceMonitor(metrics, config)

# Get current usage
usage = resource_monitor.get_current_usage()
print(f"CPU: {usage['cpu_percent']:.1f}%")
print(f"Memory: {usage['memory_percent']:.1f}%")
print(f"Disk free: {usage['disk_free_gb']:.1f} GB")

# Check thresholds
warnings = resource_monitor.check_thresholds()
for warning in warnings:
    print(f"WARNING: {warning}")
```

### JSON Export

When `json_export_enabled=True`, metrics are periodically exported to
`json_export_path`:

```bash
ls ./monitoring/
# metrics_20260115_120000.json
# metrics_20260115_121000.json
```

### Health HTTP Endpoint

For containerized deployments:

```python
from invarlock.observability.health import create_health_endpoint

HTTPServer, HealthHandler = create_health_endpoint()
if HTTPServer:
    server = HTTPServer(("0.0.0.0", 8080), HealthHandler)
    server.serve_forever()
```

Access at `http://localhost:8080/health`:

```json
{
  "overall_status": "healthy",
  "total_components": 8,
  "status_counts": {
    "healthy": 8,
    "warning": 0,
    "critical": 0,
    "unknown": 0
  },
  "components": {
    "memory": { "status": "healthy", "message": "Memory usage normal: 45.2%" },
    "cpu": { "status": "healthy", "message": "CPU usage normal: 12.3%" }
  }
}
```

## Troubleshooting

- **High memory warnings**: reduce batch size or use `--device cpu` for smaller
  footprint.
- **GPU memory critical**: clear CUDA cache between runs or use chunked snapshots.
- **Health check failures**: run `invarlock doctor` for detailed diagnostics.
- **Missing metrics**: ensure monitoring is started before operations.

## Observability in CLI

The CLI doesn't start full monitoring by default, but you can enable telemetry:

```bash
# Enable single-line telemetry summary
INVARLOCK_TELEMETRY=1 invarlock certify --baseline gpt2 --subject gpt2
```

Reports include telemetry under `report.metrics`:

- `latency_ms_per_tok` — mean latency per token
- `memory_mb_peak` — peak memory during run
- `throughput_tok_per_s` — average throughput

Certificates copy these to the `telemetry` block.

## Related Documentation

- [CLI Reference](cli.md)
- [Certificates](certificates.md) — Schema, telemetry, and HTML export
- [Environment Variables](env-vars.md)
