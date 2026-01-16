"""
Tests for observability metrics collection and registry.

This module tests:
- MetricType enum
- MetricValue dataclass
- Counter metric operations
- Gauge metric operations
- Histogram metric operations
- Timer metric and context manager
- MetricsRegistry registration and retrieval
- Utility functions for metric creation
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

# =============================================================================
# MetricType Tests
# =============================================================================


@pytest.mark.unit
class TestMetricType:
    """Tests for MetricType enum."""

    def test_metric_types_exist(self):
        """Test all expected metric types exist."""
        from invarlock.observability.metrics import MetricType

        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.TIMER.value == "timer"


# =============================================================================
# MetricValue Tests
# =============================================================================


@pytest.mark.unit
class TestMetricValue:
    """Tests for MetricValue dataclass."""

    def test_creation(self):
        """Test creating a metric value."""
        from invarlock.observability.metrics import MetricValue

        value = MetricValue(value=42.0, labels={"env": "test"}, timestamp=1000.0)

        assert value.value == 42.0
        assert value.labels == {"env": "test"}
        assert value.timestamp == 1000.0

    def test_auto_timestamp(self):
        """Test timestamp is auto-set if not provided."""
        from invarlock.observability.metrics import MetricValue

        before = time.time()
        value = MetricValue(value=1.0, labels={}, timestamp=None)
        after = time.time()

        # The __post_init__ should set timestamp
        assert value.timestamp >= before
        assert value.timestamp <= after


# =============================================================================
# Counter Tests
# =============================================================================


@pytest.mark.unit
class TestCounter:
    """Tests for Counter metric."""

    def test_increment_default(self):
        """Test incrementing by default amount (1)."""
        from invarlock.observability.metrics import Counter

        counter = Counter("test_counter", "A test counter")

        counter.inc()
        assert counter.get() == 1.0

        counter.inc()
        assert counter.get() == 2.0

    def test_increment_custom_amount(self):
        """Test incrementing by custom amount."""
        from invarlock.observability.metrics import Counter

        counter = Counter("test_counter")

        counter.inc(5.0)
        assert counter.get() == 5.0

        counter.inc(2.5)
        assert counter.get() == 7.5

    def test_increment_with_labels(self):
        """Test incrementing with labels."""
        from invarlock.observability.metrics import Counter

        counter = Counter("test_counter")

        counter.inc(labels={"env": "prod"})
        counter.inc(labels={"env": "dev"})
        counter.inc(labels={"env": "prod"})

        assert counter.get(labels={"env": "prod"}) == 2.0
        assert counter.get(labels={"env": "dev"}) == 1.0
        assert counter.get(labels={}) == 0.0  # No labels = separate series

    def test_get_all(self):
        """Test getting all counter values."""
        from invarlock.observability.metrics import Counter

        counter = Counter("test_counter")

        counter.inc(labels={"env": "prod"})
        counter.inc(2.0, labels={"env": "dev"})

        all_values = counter.get_all()

        assert len(all_values) == 2
        values_dict = {str(v.labels): v.value for v in all_values}
        assert "{'env': 'prod'}" in values_dict or "env=prod" in str(all_values)

    def test_reset(self):
        """Test resetting counter."""
        from invarlock.observability.metrics import Counter

        counter = Counter("test_counter")

        counter.inc(10.0, labels={"env": "prod"})
        counter.reset(labels={"env": "prod"})

        assert counter.get(labels={"env": "prod"}) == 0.0

    def test_thread_safety(self):
        """Test counter is thread-safe."""
        from invarlock.observability.metrics import Counter

        counter = Counter("test_counter")
        num_threads = 10
        increments_per_thread = 100

        def increment():
            for _ in range(increments_per_thread):
                counter.inc()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(increment) for _ in range(num_threads)]
            for f in futures:
                f.result()

        assert counter.get() == num_threads * increments_per_thread

    def test_labels_to_key_conversion(self):
        """Test label key serialization is consistent."""
        from invarlock.observability.metrics import Counter

        labels = {"env": "prod", "region": "us-east"}
        key = Counter._labels_to_key(labels)

        # Should be sorted and consistent
        assert "env=prod" in key
        assert "region=us-east" in key

    def test_key_to_labels_conversion(self):
        """Test key to labels deserialization."""
        from invarlock.observability.metrics import Counter

        key = "env=prod|region=us-east"
        labels = Counter._key_to_labels(key)

        assert labels == {"env": "prod", "region": "us-east"}

    def test_empty_key_to_labels(self):
        """Test empty key returns empty labels."""
        from invarlock.observability.metrics import Counter

        labels = Counter._key_to_labels("")
        assert labels == {}


# =============================================================================
# Gauge Tests
# =============================================================================


@pytest.mark.unit
class TestGauge:
    """Tests for Gauge metric."""

    def test_set_value(self):
        """Test setting gauge value."""
        from invarlock.observability.metrics import Gauge

        gauge = Gauge("test_gauge", "A test gauge")

        gauge.set(42.0)
        assert gauge.get() == 42.0

        gauge.set(10.0)
        assert gauge.get() == 10.0

    def test_increment(self):
        """Test incrementing gauge."""
        from invarlock.observability.metrics import Gauge

        gauge = Gauge("test_gauge")

        gauge.inc()
        assert gauge.get() == 1.0

        gauge.inc(5.0)
        assert gauge.get() == 6.0

    def test_decrement(self):
        """Test decrementing gauge."""
        from invarlock.observability.metrics import Gauge

        gauge = Gauge("test_gauge")
        gauge.set(10.0)

        gauge.dec()
        assert gauge.get() == 9.0

        gauge.dec(3.0)
        assert gauge.get() == 6.0

    def test_with_labels(self):
        """Test gauge with labels."""
        from invarlock.observability.metrics import Gauge

        gauge = Gauge("test_gauge")

        gauge.set(100.0, labels={"host": "server1"})
        gauge.set(200.0, labels={"host": "server2"})

        assert gauge.get(labels={"host": "server1"}) == 100.0
        assert gauge.get(labels={"host": "server2"}) == 200.0

    def test_get_all(self):
        """Test getting all gauge values."""
        from invarlock.observability.metrics import Gauge

        gauge = Gauge("test_gauge")

        gauge.set(1.0, labels={"env": "prod"})
        gauge.set(2.0, labels={"env": "dev"})

        all_values = gauge.get_all()
        assert len(all_values) == 2

    def test_thread_safety(self):
        """Test gauge is thread-safe."""
        from invarlock.observability.metrics import Gauge

        gauge = Gauge("test_gauge")
        gauge.set(0.0)

        def increment_decrement():
            for _ in range(100):
                gauge.inc()
                gauge.dec()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(increment_decrement) for _ in range(10)]
            for f in futures:
                f.result()

        # After equal increments and decrements, should be 0
        assert gauge.get() == 0.0


# =============================================================================
# Histogram Tests
# =============================================================================


@pytest.mark.unit
class TestHistogram:
    """Tests for Histogram metric."""

    def test_observe(self):
        """Test observing values."""
        from invarlock.observability.metrics import Histogram

        histogram = Histogram("test_histogram", "A test histogram")

        histogram.observe(0.1)
        histogram.observe(0.5)
        histogram.observe(1.0)

        stats = histogram.get_stats()
        assert stats["count"] == 3
        assert stats["sum"] == 1.6

    def test_default_buckets(self):
        """Test default bucket configuration."""
        from invarlock.observability.metrics import Histogram

        histogram = Histogram("test_histogram")

        expected_buckets = [
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
        ]
        assert histogram.buckets == expected_buckets

    def test_custom_buckets(self):
        """Test custom bucket configuration."""
        from invarlock.observability.metrics import Histogram

        custom_buckets = [1.0, 5.0, 10.0, 50.0, 100.0]
        histogram = Histogram("test_histogram", buckets=custom_buckets)

        assert histogram.buckets == custom_buckets

    def test_bucket_counts(self):
        """Test bucket counting."""
        from invarlock.observability.metrics import Histogram

        histogram = Histogram("test_histogram", buckets=[1.0, 5.0, 10.0])

        histogram.observe(0.5)  # Goes in 1.0, 5.0, 10.0
        histogram.observe(3.0)  # Goes in 5.0, 10.0
        histogram.observe(7.0)  # Goes in 10.0
        histogram.observe(15.0)  # Goes in none

        buckets = histogram.get_buckets()
        assert buckets[1.0] == 1
        assert buckets[5.0] == 2
        assert buckets[10.0] == 3

    def test_percentiles(self):
        """Test percentile calculation."""
        from invarlock.observability.metrics import Histogram

        histogram = Histogram("test_histogram")

        # Add values 1-100
        for i in range(1, 101):
            histogram.observe(float(i))

        # Percentile calculation returns value at percentile index
        # For 100 values, p50 index is around 50, so value should be near 50
        p50 = histogram.get_percentile(50)
        p90 = histogram.get_percentile(90)
        p99 = histogram.get_percentile(99)

        assert 49 <= p50 <= 52  # Allow some tolerance
        assert 89 <= p90 <= 92
        assert 98 <= p99 <= 100

    def test_percentile_empty(self):
        """Test percentile on empty histogram."""
        from invarlock.observability.metrics import Histogram

        histogram = Histogram("test_histogram")

        assert histogram.get_percentile(50) == 0.0

    def test_get_stats(self):
        """Test comprehensive statistics."""
        from invarlock.observability.metrics import Histogram

        histogram = Histogram("test_histogram")

        for i in range(1, 11):
            histogram.observe(float(i))

        stats = histogram.get_stats()

        assert stats["count"] == 10
        assert stats["sum"] == 55.0
        assert stats["mean"] == 5.5
        assert stats["min"] == 1.0
        assert stats["max"] == 10.0
        assert "p50" in stats
        assert "p90" in stats
        assert "p95" in stats
        assert "p99" in stats

    def test_get_stats_empty(self):
        """Test stats on empty histogram."""
        from invarlock.observability.metrics import Histogram

        histogram = Histogram("test_histogram")

        stats = histogram.get_stats()
        assert stats == {}

    def test_observations_limited(self):
        """Test observations are limited to 10000."""
        from invarlock.observability.metrics import Histogram

        histogram = Histogram("test_histogram")

        for i in range(11000):
            histogram.observe(float(i))

        # Should keep last 10000
        stats = histogram.get_stats()
        assert stats["min"] == 1000.0

    def test_with_labels(self):
        """Test histogram with labels."""
        from invarlock.observability.metrics import Histogram

        histogram = Histogram("test_histogram")

        histogram.observe(1.0, labels={"endpoint": "/api"})
        histogram.observe(2.0, labels={"endpoint": "/api"})
        histogram.observe(5.0, labels={"endpoint": "/health"})

        api_stats = histogram.get_stats(labels={"endpoint": "/api"})
        health_stats = histogram.get_stats(labels={"endpoint": "/health"})

        assert api_stats["count"] == 2
        assert health_stats["count"] == 1


# =============================================================================
# Timer Tests
# =============================================================================


@pytest.mark.unit
class TestTimer:
    """Tests for Timer metric."""

    def test_record(self):
        """Test recording duration directly."""
        from invarlock.observability.metrics import Timer

        timer = Timer("test_timer", "A test timer")

        timer.record(1.5)
        timer.record(2.5)

        stats = timer.get_stats()
        assert stats["count"] == 2
        assert stats["sum"] == 4.0

    def test_context_manager(self):
        """Test using timer as context manager."""
        from invarlock.observability.metrics import Timer

        timer = Timer("test_timer")

        with timer.time():
            time.sleep(0.1)

        stats = timer.get_stats()
        assert stats["count"] == 1
        assert stats["mean"] >= 0.1

    def test_context_manager_with_labels(self):
        """Test context manager with labels."""
        from invarlock.observability.metrics import Timer

        timer = Timer("test_timer")

        with timer.time(labels={"op": "test"}):
            time.sleep(0.05)

        stats = timer.get_stats(labels={"op": "test"})
        assert stats["count"] == 1


# =============================================================================
# MetricsRegistry Tests
# =============================================================================


@pytest.mark.unit
class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_register_counter(self):
        """Test registering a counter."""
        from invarlock.observability.metrics import Counter, MetricsRegistry

        registry = MetricsRegistry()

        counter = registry.register_counter("test_counter", "A test counter")

        assert isinstance(counter, Counter)
        assert counter.name == "test_counter"

    def test_register_gauge(self):
        """Test registering a gauge."""
        from invarlock.observability.metrics import Gauge, MetricsRegistry

        registry = MetricsRegistry()

        gauge = registry.register_gauge("test_gauge", "A test gauge")

        assert isinstance(gauge, Gauge)
        assert gauge.name == "test_gauge"

    def test_register_histogram(self):
        """Test registering a histogram."""
        from invarlock.observability.metrics import Histogram, MetricsRegistry

        registry = MetricsRegistry()

        histogram = registry.register_histogram("test_histogram")

        assert isinstance(histogram, Histogram)

    def test_register_histogram_custom_buckets(self):
        """Test registering histogram with custom buckets."""
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()

        buckets = [1.0, 5.0, 10.0]
        histogram = registry.register_histogram("test_histogram", buckets=buckets)

        assert histogram.buckets == buckets

    def test_register_timer(self):
        """Test registering a timer."""
        from invarlock.observability.metrics import MetricsRegistry, Timer

        registry = MetricsRegistry()

        timer = registry.register_timer("test_timer")

        assert isinstance(timer, Timer)

    def test_get_counter(self):
        """Test getting or creating counter."""
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()

        counter1 = registry.get_counter("test_counter")
        counter2 = registry.get_counter("test_counter")

        assert counter1 is counter2

    def test_get_gauge(self):
        """Test getting or creating gauge."""
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()

        gauge1 = registry.get_gauge("test_gauge")
        gauge2 = registry.get_gauge("test_gauge")

        assert gauge1 is gauge2

    def test_get_histogram(self):
        """Test getting or creating histogram."""
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()

        hist1 = registry.get_histogram("test_histogram")
        hist2 = registry.get_histogram("test_histogram")

        assert hist1 is hist2

    def test_get_timer(self):
        """Test getting or creating timer."""
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()

        timer1 = registry.get_timer("test_timer")
        timer2 = registry.get_timer("test_timer")

        assert timer1 is timer2

    def test_type_mismatch_error(self):
        """Test error when getting wrong metric type."""
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        registry.register_counter("test_metric")

        with pytest.raises(ValueError, match="not a gauge"):
            registry.get_gauge("test_metric")

    def test_register_duplicate_different_type(self):
        """Test registering same name with different type raises error."""
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        registry.register_counter("test_metric")

        with pytest.raises(ValueError, match="already exists"):
            registry.register_gauge("test_metric")

    def test_register_duplicate_same_type(self):
        """Test registering same name with same type returns existing."""
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()

        counter1 = registry.register_counter("test_counter")
        counter2 = registry.register_counter("test_counter")

        assert counter1 is counter2

    def test_get_all_metrics(self):
        """Test getting all metrics data."""
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()

        counter = registry.register_counter("test_counter")
        gauge = registry.register_gauge("test_gauge")

        counter.inc()
        gauge.set(42.0)

        all_metrics = registry.get_all_metrics()

        assert "test_counter" in all_metrics
        assert "test_gauge" in all_metrics
        assert all_metrics["test_counter"]["type"] == "counter"
        assert all_metrics["test_gauge"]["type"] == "gauge"

    def test_clear_all(self):
        """Test clearing all metrics."""
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        registry.register_counter("test_counter")
        registry.register_gauge("test_gauge")

        registry.clear_all()

        assert registry.list_metrics() == []

    def test_remove_metric(self):
        """Test removing a specific metric."""
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        registry.register_counter("test_counter")
        registry.register_gauge("test_gauge")

        registry.remove_metric("test_counter")

        metrics = registry.list_metrics()
        assert "test_counter" not in metrics
        assert "test_gauge" in metrics

    def test_list_metrics(self):
        """Test listing all metric names."""
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        registry.register_counter("counter1")
        registry.register_gauge("gauge1")
        registry.register_histogram("histogram1")

        metrics = registry.list_metrics()

        assert set(metrics) == {"counter1", "gauge1", "histogram1"}


# =============================================================================
# Utility Function Tests
# =============================================================================


@pytest.mark.unit
class TestMetricUtilities:
    """Tests for metric utility functions."""

    def test_create_operation_metrics(self):
        """Test creating standard operation metrics."""
        from invarlock.observability.metrics import (
            MetricsRegistry,
            create_operation_metrics,
        )

        registry = MetricsRegistry()

        metrics = create_operation_metrics(registry, "edit")

        assert "counter" in metrics
        assert "timer" in metrics
        assert "errors" in metrics
        assert "success_rate" in metrics

        # Check names are correct
        assert "invarlock.edit.total" in registry.list_metrics()
        assert "invarlock.edit.duration" in registry.list_metrics()
        assert "invarlock.edit.errors" in registry.list_metrics()

    def test_create_resource_metrics(self):
        """Test creating standard resource metrics."""
        from invarlock.observability.metrics import (
            MetricsRegistry,
            create_resource_metrics,
        )

        registry = MetricsRegistry()

        metrics = create_resource_metrics(registry)

        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert "gpu_memory" in metrics
        assert "disk_usage" in metrics


def test_summarize_memory_snapshots_peaks():
    from invarlock.observability.metrics import summarize_memory_snapshots

    snapshots = [
        {"phase": "prepare", "rss_mb": 10.0},
        {"phase": "eval", "rss_mb": 12.0, "gpu_peak_mb": 3.0},
        {"phase": "finalize", "gpu_reserved_mb": 4.0},
    ]

    summary = summarize_memory_snapshots(snapshots)
    assert summary["memory_mb_peak"] == 12.0
    assert summary["gpu_memory_mb_peak"] == 3.0
    assert summary["gpu_memory_reserved_mb_peak"] == 4.0
