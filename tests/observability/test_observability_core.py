"""
Tests for observability core monitoring components.

This module tests:
- MonitoringConfig dataclass
- MonitoringManager lifecycle and operations
- TelemetryCollector operation tracking
- PerformanceMonitor statistics
- ResourceMonitor system metrics
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# MonitoringConfig Tests
# =============================================================================


@pytest.mark.unit
class TestMonitoringConfig:
    """Tests for MonitoringConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values are sensible."""
        from invarlock.observability.core import MonitoringConfig

        config = MonitoringConfig()

        # Collection intervals
        assert config.metrics_interval == 10.0
        assert config.health_check_interval == 30.0
        assert config.resource_check_interval == 5.0

        # Data retention
        assert config.metrics_retention_hours == 24
        assert config.max_events == 10000

        # Alerting
        assert config.enable_alerting is True
        assert config.alert_channels == []

        # Export settings
        assert config.prometheus_enabled is False
        assert config.prometheus_port == 9090
        assert config.json_export_enabled is True

        # Resource thresholds
        assert config.cpu_threshold == 80.0
        assert config.memory_threshold == 85.0
        assert config.gpu_memory_threshold == 90.0

        # Performance
        assert config.latency_percentiles == [50, 90, 95, 99]
        assert config.slow_request_threshold == 30.0

    def test_custom_values(self):
        """Test custom configuration values."""
        from invarlock.observability.core import MonitoringConfig

        config = MonitoringConfig(
            metrics_interval=5.0,
            cpu_threshold=90.0,
            enable_alerting=False,
            json_export_path="/custom/path",
        )

        assert config.metrics_interval == 5.0
        assert config.cpu_threshold == 90.0
        assert config.enable_alerting is False
        assert config.json_export_path == "/custom/path"


# =============================================================================
# MonitoringManager Tests
# =============================================================================


@pytest.mark.unit
class TestMonitoringManager:
    """Tests for MonitoringManager."""

    def test_initialization(self):
        """Test manager initializes correctly."""
        from invarlock.observability.core import MonitoringConfig, MonitoringManager

        config = MonitoringConfig(enable_alerting=False)
        manager = MonitoringManager(config)

        assert manager.config is config
        assert manager.metrics is not None
        assert manager.health_checker is not None
        assert manager.alert_manager is not None
        assert manager.performance_monitor is not None
        assert manager.resource_monitor is not None

    def test_initialization_without_config(self):
        """Test manager uses default config when none provided."""
        from invarlock.observability.core import MonitoringManager

        manager = MonitoringManager()
        assert manager.config is not None
        assert manager.config.metrics_interval == 10.0

    def test_default_metrics_registered(self):
        """Test default metrics are registered on init."""
        from invarlock.observability.core import MonitoringConfig, MonitoringManager

        config = MonitoringConfig(enable_alerting=False)
        manager = MonitoringManager(config)

        # Check some expected default metrics
        metrics_list = manager.metrics.list_metrics()
        assert "invarlock.operations.total" in metrics_list
        assert "invarlock.errors.total" in metrics_list
        assert "invarlock.operation.duration" in metrics_list

    def test_record_operation(self):
        """Test recording an operation."""
        from invarlock.observability.core import MonitoringConfig, MonitoringManager

        config = MonitoringConfig(enable_alerting=False)
        manager = MonitoringManager(config)

        manager.record_operation("test_op", 1.5, status="success")

        # Check performance monitor has the data
        stats = manager.performance_monitor.get_operation_stats("test_op")
        assert stats["count"] == 1
        assert stats["mean"] == 1.5

    def test_record_error(self):
        """Test recording an error."""
        from invarlock.observability.core import MonitoringConfig, MonitoringManager

        config = MonitoringConfig(enable_alerting=False)
        manager = MonitoringManager(config)

        manager.record_error("test_error", "Something went wrong", context_key="value")

        # Check error counter incremented
        counter = manager.metrics.get_counter("invarlock.errors.total")
        assert counter.get(labels={"type": "test_error"}) == 1.0

    def test_get_status(self):
        """Test getting monitoring status."""
        from invarlock.observability.core import MonitoringConfig, MonitoringManager

        config = MonitoringConfig(enable_alerting=False)
        manager = MonitoringManager(config)

        status = manager.get_status()

        assert "monitoring_active" in status
        assert "metrics_count" in status
        assert "health_status" in status
        assert "active_alerts" in status
        assert "resource_usage" in status
        assert "performance_stats" in status
        assert "uptime" in status

    def test_start_stop_lifecycle(self):
        """Test start and stop lifecycle."""
        from invarlock.observability.core import MonitoringConfig, MonitoringManager

        # Use very long intervals to avoid actual work
        config = MonitoringConfig(
            metrics_interval=3600,
            health_check_interval=3600,
            resource_check_interval=3600,
            enable_alerting=False,
            json_export_enabled=False,
        )
        manager = MonitoringManager(config)

        manager.start()
        assert len(manager._monitoring_threads) == 3
        assert not manager._stop_event.is_set()

        manager.stop()
        assert manager._stop_event.is_set()


# =============================================================================
# TelemetryCollector Tests
# =============================================================================


@pytest.mark.unit
class TestTelemetryCollector:
    """Tests for TelemetryCollector."""

    def test_start_operation(self):
        """Test starting an operation."""
        from invarlock.observability.core import (
            MonitoringConfig,
            MonitoringManager,
            TelemetryCollector,
        )

        config = MonitoringConfig(enable_alerting=False)
        manager = MonitoringManager(config)
        collector = TelemetryCollector(manager)

        op_id = collector.start_operation("op1", "test_type", key="value")

        assert op_id == "op1"
        assert "op1" in collector.active_operations
        assert collector.active_operations["op1"]["type"] == "test_type"
        assert collector.active_operations["op1"]["metadata"]["key"] == "value"

    def test_end_operation(self):
        """Test ending an operation."""
        from invarlock.observability.core import (
            MonitoringConfig,
            MonitoringManager,
            TelemetryCollector,
        )

        config = MonitoringConfig(enable_alerting=False)
        manager = MonitoringManager(config)
        collector = TelemetryCollector(manager)

        collector.start_operation("op1", "test_type")
        time.sleep(0.1)  # Small delay
        collector.end_operation("op1", status="success", result="done")

        assert "op1" not in collector.active_operations
        assert len(collector.operation_history) == 1

        record = collector.operation_history[0]
        assert record["status"] == "success"
        assert record["duration"] >= 0.1
        assert record["result_metadata"]["result"] == "done"

    def test_end_unknown_operation(self):
        """Test ending an unknown operation logs warning."""
        from invarlock.observability.core import (
            MonitoringConfig,
            MonitoringManager,
            TelemetryCollector,
        )

        config = MonitoringConfig(enable_alerting=False)
        manager = MonitoringManager(config)
        collector = TelemetryCollector(manager)

        # Should not raise, just log warning
        collector.end_operation("unknown_op")

    def test_get_operation_stats(self):
        """Test getting operation statistics."""
        from invarlock.observability.core import (
            MonitoringConfig,
            MonitoringManager,
            TelemetryCollector,
        )

        config = MonitoringConfig(enable_alerting=False)
        manager = MonitoringManager(config)
        collector = TelemetryCollector(manager)

        # Complete a few operations
        for i in range(3):
            collector.start_operation(f"op{i}", "test_type")
            time.sleep(0.01)
            collector.end_operation(f"op{i}", status="success" if i < 2 else "failure")

        stats = collector.get_operation_stats()

        assert stats["total_operations"] == 3
        assert stats["active_operations"] == 0
        assert stats["status_distribution"]["success"] == 2
        assert stats["status_distribution"]["failure"] == 1
        assert stats["type_distribution"]["test_type"] == 3

    def test_empty_operation_stats(self):
        """Test operation stats when no operations recorded."""
        from invarlock.observability.core import (
            MonitoringConfig,
            MonitoringManager,
            TelemetryCollector,
        )

        config = MonitoringConfig(enable_alerting=False)
        manager = MonitoringManager(config)
        collector = TelemetryCollector(manager)

        stats = collector.get_operation_stats()
        assert stats == {}


# =============================================================================
# PerformanceMonitor Tests
# =============================================================================


@pytest.mark.unit
class TestPerformanceMonitor:
    """Tests for PerformanceMonitor."""

    def test_record_operation(self):
        """Test recording operation performance."""
        from invarlock.observability.core import PerformanceMonitor
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        monitor = PerformanceMonitor(registry)

        monitor.record_operation("test_op", 1.5, key="value")

        assert "test_op" in monitor.operation_times
        assert monitor.operation_times["test_op"] == [1.5]
        assert monitor.performance_data["test_op"]["key"] == "value"

    def test_operation_times_limited(self):
        """Test operation times buffer is limited to 1000."""
        from invarlock.observability.core import PerformanceMonitor
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        monitor = PerformanceMonitor(registry)

        # Add more than 1000 measurements
        for i in range(1100):
            monitor.record_operation("test_op", float(i))

        assert len(monitor.operation_times["test_op"]) == 1000
        # Should keep the most recent
        assert monitor.operation_times["test_op"][0] == 100.0

    def test_get_operation_stats(self):
        """Test getting operation statistics."""
        from invarlock.observability.core import PerformanceMonitor
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        monitor = PerformanceMonitor(registry)

        # Record some operations
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        for t in times:
            monitor.record_operation("test_op", t)

        stats = monitor.get_operation_stats("test_op")

        assert stats["count"] == 5
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    def test_get_operation_stats_empty(self):
        """Test getting stats for non-existent operation."""
        from invarlock.observability.core import PerformanceMonitor
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        monitor = PerformanceMonitor(registry)

        stats = monitor.get_operation_stats("unknown")
        assert stats == {}

    def test_get_summary(self):
        """Test getting summary of all operations."""
        from invarlock.observability.core import PerformanceMonitor
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        monitor = PerformanceMonitor(registry)

        monitor.record_operation("op1", 1.0)
        monitor.record_operation("op2", 2.0)

        summary = monitor.get_summary()

        assert "op1" in summary
        assert "op2" in summary
        assert summary["op1"]["count"] == 1
        assert summary["op2"]["count"] == 1


# =============================================================================
# ResourceMonitor Tests
# =============================================================================


@pytest.mark.unit
class TestResourceMonitor:
    """Tests for ResourceMonitor."""

    def test_collect_usage(self):
        """Test collecting resource usage."""
        from invarlock.observability.core import MonitoringConfig, ResourceMonitor
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        config = MonitoringConfig()
        monitor = ResourceMonitor(registry, config)

        usage = monitor.collect_usage()

        # Should have CPU and memory at minimum
        assert "cpu_percent" in usage
        assert "memory_percent" in usage
        assert "memory_available_gb" in usage
        assert "disk_percent" in usage

    def test_update_metrics(self):
        """Test updating metrics gauges."""
        from invarlock.observability.core import MonitoringConfig, ResourceMonitor
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        config = MonitoringConfig()
        monitor = ResourceMonitor(registry, config)

        monitor.update_metrics()

        # Check that gauges were updated
        metrics_list = registry.list_metrics()
        assert any("resource" in m for m in metrics_list)

    def test_check_thresholds(self):
        """Test threshold checking."""
        from invarlock.observability.core import MonitoringConfig, ResourceMonitor
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        # Set very low thresholds to trigger warnings
        config = MonitoringConfig(cpu_threshold=0.0, memory_threshold=0.0)
        monitor = ResourceMonitor(registry, config)

        warnings = monitor.check_thresholds()

        # Should have warnings due to low thresholds
        assert len(warnings) >= 2  # CPU and memory at least

    def test_get_current_usage(self):
        """Test getting current usage is same as collect_usage."""
        from invarlock.observability.core import MonitoringConfig, ResourceMonitor
        from invarlock.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        config = MonitoringConfig()
        monitor = ResourceMonitor(registry, config)

        usage1 = monitor.collect_usage()
        usage2 = monitor.get_current_usage()

        # Keys should be the same
        assert set(usage1.keys()) == set(usage2.keys())
