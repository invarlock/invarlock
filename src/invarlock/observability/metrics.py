"""
Metrics collection and registry system.
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MetricType(Enum):
    """Types of metrics supported."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Container for metric values with metadata."""

    value: int | float
    labels: dict[str, str]
    timestamp: float

    def __post_init__(self):
        if not hasattr(self, "timestamp") or self.timestamp is None:
            self.timestamp = time.time()


class Counter:
    """Counter metric - monotonically increasing value."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._values: dict[str, float] = defaultdict(float)
        # Use RLock to allow nested acquisitions in helper methods
        self._lock = threading.RLock()

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None):
        """Increment counter by amount."""
        labels = labels or {}
        label_key = self._labels_to_key(labels)

        with self._lock:
            self._values[label_key] += amount

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Get current counter value."""
        labels = labels or {}
        label_key = self._labels_to_key(labels)

        with self._lock:
            return float(self._values[label_key])

    def get_all(self) -> list[MetricValue]:
        """Get all counter values with labels."""
        with self._lock:
            return [
                MetricValue(
                    value=value, labels=self._key_to_labels(key), timestamp=time.time()
                )
                for key, value in self._values.items()
            ]

    def reset(self, labels: dict[str, str] | None = None):
        """Reset counter to zero."""
        labels = labels or {}
        label_key = self._labels_to_key(labels)

        with self._lock:
            self._values[label_key] = 0.0

    @staticmethod
    def _labels_to_key(labels: dict[str, str]) -> str:
        """Convert labels dict to string key."""
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))

    @staticmethod
    def _key_to_labels(key: str) -> dict[str, str]:
        """Convert string key back to labels dict."""
        if not key:
            return {}

        labels = {}
        for pair in key.split("|"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                labels[k] = v
        return labels


class Gauge:
    """Gauge metric - value that can increase or decrease."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._values: dict[str, float] = defaultdict(float)
        # Use RLock to allow nested acquisitions in helper methods
        self._lock = threading.RLock()

    def set(self, value: float, labels: dict[str, str] | None = None):
        """Set gauge to specific value."""
        labels = labels or {}
        label_key = Counter._labels_to_key(labels)

        with self._lock:
            self._values[label_key] = value

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None):
        """Increment gauge by amount."""
        labels = labels or {}
        label_key = Counter._labels_to_key(labels)

        with self._lock:
            self._values[label_key] += amount

    def dec(self, amount: float = 1.0, labels: dict[str, str] | None = None):
        """Decrement gauge by amount."""
        self.inc(-amount, labels)

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Get current gauge value."""
        labels = labels or {}
        label_key = Counter._labels_to_key(labels)

        with self._lock:
            return float(self._values[label_key])

    def get_all(self) -> list[MetricValue]:
        """Get all gauge values with labels."""
        with self._lock:
            return [
                MetricValue(
                    value=value,
                    labels=Counter._key_to_labels(key),
                    timestamp=time.time(),
                )
                for key, value in self._values.items()
            ]


class Histogram:
    """Histogram metric - distribution of values."""

    def __init__(
        self, name: str, description: str = "", buckets: list[float] | None = None
    ):
        self.name = name
        self.description = description
        self.buckets = buckets or [
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

        # Store observations for each label set
        self._observations: dict[str, list[float]] = defaultdict(list)
        self._bucket_counts: dict[str, dict[float, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._sum: dict[str, float] = defaultdict(float)
        self._count: dict[str, int] = defaultdict(int)
        # Use RLock to allow nested acquisitions when get_stats calls get_percentile
        self._lock = threading.RLock()

    def observe(self, value: float, labels: dict[str, str] | None = None):
        """Observe a value."""
        labels = labels or {}
        label_key = Counter._labels_to_key(labels)

        with self._lock:
            # Store observation
            self._observations[label_key].append(value)

            # Keep only recent observations (last 10000)
            if len(self._observations[label_key]) > 10000:
                self._observations[label_key] = self._observations[label_key][-10000:]

            # Update bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[label_key][bucket] += 1

            # Update sum and count
            self._sum[label_key] += value
            self._count[label_key] += 1

    def get_percentile(
        self, percentile: float, labels: dict[str, str] | None = None
    ) -> float:
        """Get percentile value."""
        labels = labels or {}
        label_key = Counter._labels_to_key(labels)

        with self._lock:
            observations = self._observations[label_key]
            if not observations:
                return 0.0

            sorted_obs = sorted(observations)
            index = int(len(sorted_obs) * percentile / 100)
            return float(sorted_obs[min(index, len(sorted_obs) - 1)])

    def get_stats(self, labels: dict[str, str] | None = None) -> dict[str, float]:
        """Get histogram statistics."""
        labels = labels or {}
        label_key = Counter._labels_to_key(labels)

        with self._lock:
            observations = self._observations[label_key]
            if not observations:
                return {}

            count = self._count[label_key]
            total = self._sum[label_key]

            return {
                "count": count,
                "sum": total,
                "mean": total / count if count > 0 else 0,
                "min": min(observations),
                "max": max(observations),
                "p50": self.get_percentile(50, labels),
                "p90": self.get_percentile(90, labels),
                "p95": self.get_percentile(95, labels),
                "p99": self.get_percentile(99, labels),
            }

    def get_buckets(self, labels: dict[str, str] | None = None) -> dict[float, int]:
        """Get bucket counts."""
        labels = labels or {}
        label_key = Counter._labels_to_key(labels)

        with self._lock:
            return dict(self._bucket_counts[label_key])


class Timer:
    """Timer metric - specialized histogram for timing operations."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.histogram = Histogram(name, description)

    def time(self, labels: dict[str, str] | None = None):
        """Context manager for timing operations."""
        return TimerContext(self, labels)

    def record(self, duration: float, labels: dict[str, str] | None = None):
        """Record a duration."""
        self.histogram.observe(duration, labels)

    def get_stats(self, labels: dict[str, str] | None = None) -> dict[str, float]:
        """Get timing statistics."""
        return self.histogram.get_stats(labels)


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, timer: Timer, labels: dict[str, str] | None = None):
        self.timer = timer
        self.labels = labels
        self.start_time: float | None = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.timer.record(duration, self.labels)


class MetricsRegistry:
    """Central registry for all metrics."""

    def __init__(self):
        self._metrics: dict[str, Counter | Gauge | Histogram | Timer] = {}
        # Use RLock as registry may invoke metric methods that also lock internally
        self._lock = threading.RLock()

    def register_counter(self, name: str, description: str = "") -> Counter:
        """Register a counter metric."""
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Counter):
                    raise ValueError(f"Metric {name} already exists as different type")
                return metric

            counter = Counter(name, description)
            self._metrics[name] = counter
            return counter

    def register_gauge(self, name: str, description: str = "") -> Gauge:
        """Register a gauge metric."""
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Gauge):
                    raise ValueError(f"Metric {name} already exists as different type")
                return metric

            gauge = Gauge(name, description)
            self._metrics[name] = gauge
            return gauge

    def register_histogram(
        self, name: str, description: str = "", buckets: list[float] | None = None
    ) -> Histogram:
        """Register a histogram metric."""
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Histogram):
                    raise ValueError(f"Metric {name} already exists as different type")
                return metric

            histogram = Histogram(name, description, buckets)
            self._metrics[name] = histogram
            return histogram

    def register_timer(self, name: str, description: str = "") -> Timer:
        """Register a timer metric."""
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Timer):
                    raise ValueError(f"Metric {name} already exists as different type")
                return metric

            timer = Timer(name, description)
            self._metrics[name] = timer
            return timer

    def get_counter(self, name: str) -> Counter:
        """Get or create counter metric."""
        if name not in self._metrics:
            return self.register_counter(name)

        metric = self._metrics[name]
        if not isinstance(metric, Counter):
            raise ValueError(f"Metric {name} is not a counter")
        return metric

    def get_gauge(self, name: str) -> Gauge:
        """Get or create gauge metric."""
        if name not in self._metrics:
            return self.register_gauge(name)

        metric = self._metrics[name]
        if not isinstance(metric, Gauge):
            raise ValueError(f"Metric {name} is not a gauge")
        return metric

    def get_histogram(self, name: str) -> Histogram:
        """Get or create histogram metric."""
        if name not in self._metrics:
            return self.register_histogram(name)

        metric = self._metrics[name]
        if not isinstance(metric, Histogram):
            raise ValueError(f"Metric {name} is not a histogram")
        return metric

    def get_timer(self, name: str) -> Timer:
        """Get or create timer metric."""
        if name not in self._metrics:
            return self.register_timer(name)

        metric = self._metrics[name]
        if not isinstance(metric, Timer):
            raise ValueError(f"Metric {name} is not a timer")
        return metric

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics data."""
        with self._lock:
            result: dict[str, Any] = {}

            for name, metric in self._metrics.items():
                if isinstance(metric, Counter | Gauge):
                    result[name] = {
                        "type": type(metric).__name__.lower(),
                        "description": metric.description,
                        "values": [value.__dict__ for value in metric.get_all()],
                    }
                elif isinstance(metric, Histogram):
                    result[name] = {
                        "type": "histogram",
                        "description": metric.description,
                        "buckets": metric.buckets,
                        "stats": metric.get_stats(),
                    }
                elif isinstance(metric, Timer):
                    result[name] = {
                        "type": "timer",
                        "description": metric.histogram.description,
                        "stats": metric.get_stats(),
                    }

            return result

    def clear_all(self):
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()

    def remove_metric(self, name: str):
        """Remove a specific metric."""
        with self._lock:
            self._metrics.pop(name, None)

    def list_metrics(self) -> list[str]:
        """List all registered metric names."""
        with self._lock:
            return list(self._metrics.keys())


# Utility functions for common metric patterns
def create_operation_metrics(
    registry: MetricsRegistry, operation: str
) -> dict[str, Any]:
    """Create standard metrics for an operation."""
    return {
        "counter": registry.register_counter(f"invarlock.{operation}.total"),
        "timer": registry.register_timer(f"invarlock.{operation}.duration"),
        "errors": registry.register_counter(f"invarlock.{operation}.errors"),
        "success_rate": registry.register_gauge(f"invarlock.{operation}.success_rate"),
    }


def create_resource_metrics(registry: MetricsRegistry) -> dict[str, Any]:
    """Create standard resource monitoring metrics."""
    return {
        "cpu_usage": registry.register_gauge("invarlock.resource.cpu_percent"),
        "memory_usage": registry.register_gauge("invarlock.resource.memory_percent"),
        "gpu_memory": registry.register_gauge("invarlock.resource.gpu_memory_percent"),
        "disk_usage": registry.register_gauge("invarlock.resource.disk_percent"),
    }
