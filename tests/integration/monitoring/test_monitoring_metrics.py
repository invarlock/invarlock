"""Unit tests for invarlock.observability.metrics."""

from __future__ import annotations

import itertools

import pytest

from invarlock.observability import metrics


def _set_fake_time(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    """Patch time.time() in metrics module with deterministic sequence."""
    iterator = itertools.chain(values, itertools.repeat(values[-1]))
    monkeypatch.setattr(metrics.time, "time", lambda: next(iterator))


def test_metric_value_sets_timestamp_when_missing(monkeypatch: pytest.MonkeyPatch):
    _set_fake_time(monkeypatch, [123.456])
    metric_value = metrics.MetricValue(value=5, labels={}, timestamp=None)
    assert metric_value.timestamp == pytest.approx(123.456)


def test_counter_increment_get_and_reset(monkeypatch: pytest.MonkeyPatch):
    _set_fake_time(monkeypatch, [1.0, 2.0, 3.0])
    counter = metrics.Counter("requests_total")

    # Default value is zero for unseen labels
    assert counter.get() == 0.0

    # Increment with unordered labels to exercise sorting logic
    counter.inc(amount=1.5, labels={"b": "2", "a": "1"})
    assert counter.get(labels={"a": "1", "b": "2"}) == pytest.approx(1.5)

    # Reset only affects matching label set
    counter.reset(labels={"a": "1", "b": "2"})
    assert counter.get(labels={"a": "1", "b": "2"}) == 0.0

    # Timestamps are captured via MetricValue when exporting
    counter.inc()
    values = counter.get_all()
    assert len(values) == 2  # default label key plus specific key
    labels_sets = {frozenset(v.labels.items()) for v in values}
    assert labels_sets == {frozenset(), frozenset({("a", "1"), ("b", "2")})}
    for mv in values:
        assert mv.timestamp >= 1.0


def test_counter_labels_roundtrip():
    key = metrics.Counter._labels_to_key({"b": "2", "a": "1"})
    assert key == "a=1|b=2"

    labels = metrics.Counter._key_to_labels(key)
    assert labels == {"a": "1", "b": "2"}

    # Uneven pair is ignored and empty key returns empty dict
    assert metrics.Counter._key_to_labels("foo|bar=baz") == {"bar": "baz"}
    assert metrics.Counter._key_to_labels("") == {}


def test_gauge_operations(monkeypatch: pytest.MonkeyPatch):
    _set_fake_time(monkeypatch, [10.0, 11.0])
    gauge = metrics.Gauge("temperature_celsius")
    gauge.set(25.0, labels={"city": "paris"})
    gauge.inc(3.0, labels={"city": "paris"})
    gauge.dec(5.0, labels={"city": "paris"})

    assert gauge.get(labels={"city": "paris"}) == pytest.approx(23.0)
    all_values = gauge.get_all()
    assert len(all_values) == 1
    assert all_values[0].labels == {"city": "paris"}
    assert all_values[0].value == pytest.approx(23.0)


def test_histogram_statistics_and_percentiles(monkeypatch: pytest.MonkeyPatch):
    """
    Lightweight, deterministic check for histogram stats and percentiles.
    Uses a tiny, fixed sample set to avoid long runtimes and flakiness.
    """
    histogram = metrics.Histogram("request_latency", buckets=[0.1, 1.0, 10.0])
    for v in (0.05, 0.5, 2.0):
        histogram.observe(v)

    # Percentiles should be stable and within [min, max]
    p50 = histogram.get_percentile(50)
    assert p50 == pytest.approx(0.5, rel=1e-6)

    stats = histogram.get_stats()
    assert stats["count"] == 3
    assert stats["sum"] == pytest.approx(2.55)
    assert stats["mean"] == pytest.approx(2.55 / 3)
    assert stats["min"] == pytest.approx(0.05)
    assert stats["max"] == pytest.approx(2.0)

    # Reported percentiles are monotonic and bounded
    assert stats["p50"] == pytest.approx(p50)
    assert (
        stats["min"]
        <= stats["p50"]
        <= stats["p90"]
        <= stats["p95"]
        <= stats["p99"]
        <= stats["max"]
    )

    # Bucket counts reflect cumulative behavior
    buckets = histogram.get_buckets()
    assert buckets[0.1] == 1  # only first observation
    assert buckets[1.0] == 2  # first two observations
    assert buckets[10.0] == 3  # all observations

    # Empty histogram returns defaults quickly
    empty_hist = metrics.Histogram("empty")
    assert empty_hist.get_percentile(90) == 0.0
    assert empty_hist.get_stats() == {}


def test_histogram_retains_recent_samples_only():
    histogram = metrics.Histogram("rolling")
    for value in range(10_200):
        histogram.observe(float(value))

    observations = histogram._observations[""]
    assert len(observations) == 10_000
    assert observations[0] == pytest.approx(200.0)
    assert observations[-1] == pytest.approx(10_199.0)


def test_timer_context_records_duration(monkeypatch: pytest.MonkeyPatch):
    _set_fake_time(monkeypatch, [1.0, 1.5, 2.0])
    timer = metrics.Timer("handler_duration")

    with timer.time(labels={"handler": "ping"}):
        pass

    # Manual record for second branch
    timer.record(0.75, labels={"handler": "ping"})

    stats = timer.get_stats(labels={"handler": "ping"})
    assert stats["count"] == 2
    assert stats["min"] <= stats["max"]
    assert stats["sum"] == pytest.approx(stats["mean"] * stats["count"])

    # No recording occurs if context was never started
    context = metrics.TimerContext(timer)
    context.__exit__(None, None, None)
    post_stats = timer.get_stats(labels={"handler": "ping"})
    assert post_stats["count"] == 2


def test_metrics_registry_registration_and_retrieval(monkeypatch: pytest.MonkeyPatch):
    _set_fake_time(monkeypatch, [42.0, 43.0, 44.0, 45.0, 46.0])
    registry = metrics.MetricsRegistry()

    counter = registry.register_counter("requests")
    counter.inc()
    assert registry.register_counter("requests") is counter

    gauge = registry.register_gauge("temperature")
    gauge.set(30.0)
    with pytest.raises(ValueError):
        registry.register_counter("temperature")

    histogram = registry.register_histogram("latency")
    histogram.observe(0.2)

    timer = registry.register_timer("duration")
    timer.record(0.5)

    # get_* create metrics lazily
    assert isinstance(registry.get_counter("new_counter"), metrics.Counter)
    with pytest.raises(ValueError):
        registry.get_counter("temperature")

    assert isinstance(registry.get_gauge("new_gauge"), metrics.Gauge)
    with pytest.raises(ValueError):
        registry.get_gauge("requests")

    assert isinstance(registry.get_histogram("new_hist"), metrics.Histogram)
    with pytest.raises(ValueError):
        registry.get_histogram("duration")

    assert isinstance(registry.get_timer("new_timer"), metrics.Timer)
    with pytest.raises(ValueError):
        registry.get_timer("latency")

    all_metrics = registry.get_all_metrics()
    assert all_metrics["requests"]["type"] == "counter"
    assert all_metrics["requests"]["values"][0]["value"] == pytest.approx(1.0)
    assert all_metrics["temperature"]["values"][0]["value"] == pytest.approx(30.0)
    assert all_metrics["latency"]["stats"]["count"] == 1
    assert all_metrics["duration"]["stats"]["count"] >= 1

    assert sorted(registry.list_metrics())
    registry.remove_metric("new_counter")
    assert "new_counter" not in registry.list_metrics()

    registry.clear_all()
    assert registry.list_metrics() == []


def test_operation_and_resource_factory_helpers():
    registry = metrics.MetricsRegistry()

    ops_metrics = metrics.create_operation_metrics(registry, "sync")
    assert isinstance(ops_metrics["counter"], metrics.Counter)
    assert isinstance(ops_metrics["timer"], metrics.Timer)
    assert isinstance(ops_metrics["errors"], metrics.Counter)
    assert isinstance(ops_metrics["success_rate"], metrics.Gauge)

    resource_metrics = metrics.create_resource_metrics(registry)
    assert all(
        isinstance(metric, metrics.Gauge) for metric in resource_metrics.values()
    )


pytestmark = pytest.mark.integration
