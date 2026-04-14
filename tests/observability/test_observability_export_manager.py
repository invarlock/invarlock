"""
Tests for observability metrics exporters.

This module tests:
- ExportedMetric dataclass and format conversions
- MetricsExporter base class
- PrometheusExporter
- JSONExporter
- InfluxDBExporter
- StatsExporter
- ExportManager multi-exporter coordination
- export_or_raise error handling
- Utility functions for exporter setup
"""

from __future__ import annotations

import time

import pytest

# =============================================================================
# ExportedMetric Tests
# =============================================================================


# =============================================================================
# MetricsExporter Base Class Tests
# =============================================================================


# =============================================================================
# PrometheusExporter Tests
# =============================================================================


# =============================================================================
# JSONExporter Tests
# =============================================================================


# =============================================================================
# InfluxDBExporter Tests
# =============================================================================


# =============================================================================
# StatsExporter Tests
# =============================================================================


# =============================================================================
# ExportManager Tests
# =============================================================================


# =============================================================================
# export_or_raise Tests
# =============================================================================


# =============================================================================
# Utility Function Tests
# =============================================================================


@pytest.mark.unit
class TestExportManager:
    """Tests for ExportManager."""

    def test_initialization(self):
        """Test export manager initialization."""
        from invarlock.observability.exporters import ExportManager

        manager = ExportManager()

        assert manager.exporters == {}
        assert manager.export_interval == 10
        assert manager._running is False

    def test_add_exporter(self):
        """Test adding an exporter."""
        from invarlock.observability.exporters import ExportManager, JSONExporter

        manager = ExportManager()
        exporter = JSONExporter()

        manager.add_exporter(exporter)

        assert "json" in manager.exporters
        assert manager.exporters["json"] is exporter

    def test_remove_exporter(self):
        """Test removing an exporter."""
        from invarlock.observability.exporters import ExportManager, JSONExporter

        manager = ExportManager()
        manager.add_exporter(JSONExporter())

        manager.remove_exporter("json")

        assert "json" not in manager.exporters

    def test_queue_metrics(self):
        """Test queuing metrics for export."""
        from invarlock.observability.exporters import ExportedMetric, ExportManager

        manager = ExportManager()

        metrics = [ExportedMetric(name="test", value=1.0, timestamp=time.time())]
        manager.queue_metrics(metrics)

        assert len(manager._metrics_queue) == 1

    def test_export_now(self):
        """Test immediate export."""
        from invarlock.observability.exporters import (
            ExportedMetric,
            ExportManager,
            JSONExporter,
        )

        manager = ExportManager()
        exporter = JSONExporter()
        manager.add_exporter(exporter)

        metrics = [ExportedMetric(name="test", value=1.0, timestamp=time.time())]
        results = manager.export_now(metrics)

        assert results["json"] is True
        assert len(exporter._metrics_buffer) == 1

    def test_export_now_from_queue(self):
        """Test export_now drains queue when no metrics provided."""
        from invarlock.observability.exporters import (
            ExportedMetric,
            ExportManager,
            JSONExporter,
        )

        manager = ExportManager()
        manager.add_exporter(JSONExporter())

        # Queue metrics first
        metrics = [ExportedMetric(name="test", value=1.0, timestamp=time.time())]
        manager.queue_metrics(metrics)

        # Export without providing metrics
        results = manager.export_now()

        assert results["json"] is True
        assert len(manager._metrics_queue) == 0  # Queue should be drained

    def test_export_disabled_exporter(self):
        """Test disabled exporters return False."""
        from invarlock.observability.exporters import (
            ExportedMetric,
            ExportManager,
            JSONExporter,
        )

        manager = ExportManager()
        exporter = JSONExporter()
        exporter.enabled = False
        manager.add_exporter(exporter)

        metrics = [ExportedMetric(name="test", value=1.0, timestamp=time.time())]
        results = manager.export_now(metrics)

        assert results["json"] is False

    def test_export_now_handles_exporter_runtime_error(self):
        """Exporter runtime failures should be downgraded to False results."""
        from invarlock.observability.exporters import (
            ExportedMetric,
            ExportManager,
            MetricsExporter,
        )

        class FailingExporter(MetricsExporter):
            def export(self, metrics):  # noqa: ANN001
                raise RuntimeError("export boom")

        manager = ExportManager()
        manager.add_exporter(FailingExporter("failing"))

        metrics = [ExportedMetric(name="test", value=1.0, timestamp=time.time())]
        results = manager.export_now(metrics)

        assert results["failing"] is False

    def test_get_exporter_stats(self):
        """Test getting stats for all exporters."""
        from invarlock.observability.exporters import ExportManager, JSONExporter

        manager = ExportManager()
        manager.add_exporter(JSONExporter())

        stats = manager.get_exporter_stats()

        assert "json" in stats
        assert stats["json"]["name"] == "json"

    def test_get_summary(self):
        """Test getting export manager summary."""
        from invarlock.observability.exporters import (
            ExportedMetric,
            ExportManager,
            JSONExporter,
        )

        manager = ExportManager()
        exporter = JSONExporter()
        manager.add_exporter(exporter)

        # Do an export to have some stats
        metrics = [ExportedMetric(name="test", value=1.0, timestamp=time.time())]
        manager.export_now(metrics)

        summary = manager.get_summary()

        assert summary["total_exporters"] == 1
        assert summary["enabled_exporters"] == 1
        assert summary["total_exports"] == 1
        assert summary["total_errors"] == 0
        assert summary["success_rate"] == 1.0

    def test_start_stop_background_export(self):
        """Test starting and stopping background export."""
        from invarlock.observability.exporters import ExportManager

        manager = ExportManager()
        manager.export_interval = 100  # Long interval to avoid actual work

        manager.start_background_export()
        assert manager._running is True
        assert manager._export_thread is not None

        manager.stop_background_export()
        assert manager._running is False


@pytest.mark.unit
class TestExportOrRaise:
    """Tests for export_or_raise helper function."""

    def test_success(self):
        """Test successful export doesn't raise."""
        from invarlock.observability.exporters import (
            ExportedMetric,
            JSONExporter,
            export_or_raise,
        )

        exporter = JSONExporter()
        metrics = [ExportedMetric(name="test", value=1.0, timestamp=time.time())]

        # Should not raise
        export_or_raise(exporter, metrics)

    def test_raises_on_false_return(self):
        """Test raises ObservabilityError when export returns False."""
        from invarlock.core.exceptions import ObservabilityError
        from invarlock.observability.exporters import (
            ExportedMetric,
            MetricsExporter,
            export_or_raise,
        )

        class FailingExporter(MetricsExporter):
            def export(self, metrics):
                return False

        exporter = FailingExporter("failing")
        metrics = [ExportedMetric(name="test", value=1.0, timestamp=time.time())]

        with pytest.raises(ObservabilityError) as exc_info:
            export_or_raise(exporter, metrics)

        assert exc_info.value.code == "E801"
        assert exc_info.value.details["exporter"] == "failing"
        assert exc_info.value.details["reason"] == "returned_false"

    def test_raises_on_exception(self):
        """Test raises ObservabilityError when export raises exception."""
        from invarlock.core.exceptions import ObservabilityError
        from invarlock.observability.exporters import (
            ExportedMetric,
            MetricsExporter,
            export_or_raise,
        )

        class ExplodingExporter(MetricsExporter):
            def export(self, metrics):
                raise RuntimeError("Boom!")

        exporter = ExplodingExporter("exploding")
        metrics = [ExportedMetric(name="test", value=1.0, timestamp=time.time())]

        with pytest.raises(ObservabilityError) as exc_info:
            export_or_raise(exporter, metrics)

        assert exc_info.value.code == "E801"
        assert exc_info.value.details["exporter"] == "exploding"
        assert exc_info.value.details["reason"] == "RuntimeError"


@pytest.mark.unit
class TestExporterUtilities:
    """Tests for exporter utility functions."""

    def test_setup_prometheus_exporter(self):
        """Test setting up Prometheus exporter."""
        from invarlock.observability.exporters import setup_prometheus_exporter

        exporter = setup_prometheus_exporter(
            gateway_url="http://gateway:9091", job_name="test_job"
        )

        assert exporter.gateway_url == "http://gateway:9091"
        assert exporter.job_name == "test_job"

    def test_setup_json_file_exporter(self):
        """Test setting up JSON file exporter."""
        from invarlock.observability.exporters import setup_json_file_exporter

        exporter = setup_json_file_exporter("/tmp/metrics.json")

        assert exporter.output_file == "/tmp/metrics.json"

    def test_setup_influxdb_exporter(self):
        """Test setting up InfluxDB exporter."""
        from invarlock.observability.exporters import setup_influxdb_exporter

        exporter = setup_influxdb_exporter(
            url="http://influx:8086",
            database="metrics",
            username="user",
            password="pass",
        )

        assert exporter.url == "http://influx:8086"
        assert exporter.database == "metrics"
        assert exporter.username == "user"
        assert exporter.password == "pass"

    def test_setup_statsd_exporter(self):
        """Test setting up StatsD exporter."""
        from invarlock.observability.exporters import setup_statsd_exporter

        exporter = setup_statsd_exporter(host="statsd.local", port=9125, prefix="myapp")

        assert exporter.host == "statsd.local"
        assert exporter.port == 9125
        assert exporter.prefix == "myapp"
