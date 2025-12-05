# Observability Tests

Tests for InvarLock's telemetry, logging, and monitoring capabilities.

## Module Coverage

| Module | Test File | Status | Description |
|--------|-----------|--------|-------------|
| `core.py` | `test_observability_core.py` | 🔴 Needs expansion | Event logging, run context |
| `metrics.py` | `test_observability_metrics.py` | 🔴 Needs expansion | Counters, gauges, histograms |
| `alerting.py` | `test_observability_alerting.py` | 🔴 Needs expansion | Alert dispatch and formatting |
| `exporters.py` | `test_observability_exporters.py` | 🔴 Needs expansion | JSONL, OpenTelemetry export |
| `health.py` | `test_observability_health.py` | 🔴 Needs expansion | System health checks |
| `utils.py` | `test_observability_utils.py` | 🔴 Needs expansion | Utility functions |
| exceptions | `test_exceptions.py` | ✅ Exists | Exception handling |

## Key Observability Features

### Event Logging (`core.py`)
- Structured event emission with timestamps
- Run context management (run_id, session_id)
- Log level filtering

### Metrics (`metrics.py`)
- Counter tracking (increments)
- Gauge tracking (arbitrary values)
- Histogram tracking (distributions)
- Thread-safe metric collection

### Alerting (`alerting.py`)
- Alert severity levels (INFO, WARN, ERROR, CRITICAL)
- Alert dispatch to configured handlers
- Alert formatting for different outputs

### Exporters (`exporters.py`)
- JSONL file export for events
- Console pretty-printing
- OpenTelemetry integration (optional)

### Health Checks (`health.py`)
- System health status
- Dependency availability checks
- Resource utilization monitoring

## Running Tests

```bash
# All observability tests
PYTHONPATH=src pytest tests/observability/ -v

# Specific module
PYTHONPATH=src pytest tests/observability/test_observability_core.py -v
```

## Coverage Targets

- **Target: ≥85% branch coverage** for `core.py`, `metrics.py`, `exporters.py`
- **Target: ≥70% coverage** for `alerting.py`, `health.py`, `utils.py`

## Test Patterns

### Event Testing
```python
def test_event_emission():
    """Test that events are properly structured and logged."""
    obs = create_observer()
    obs.log_event("test_event", level="INFO", data={"key": "value"})
    events = obs.get_events()
    assert len(events) == 1
    assert events[0]["type"] == "test_event"
```

### Metric Testing
```python
def test_counter_increment():
    """Test counter incrementing."""
    metrics = MetricsCollector()
    metrics.increment("operation_count", 1)
    assert metrics.get("operation_count") == 1
```

### Exporter Testing
```python
def test_jsonl_export(tmp_path):
    """Test JSONL file export."""
    exporter = JSONLExporter(tmp_path / "events.jsonl")
    exporter.export({"type": "test", "ts": "..."})
    content = (tmp_path / "events.jsonl").read_text()
    assert "test" in content