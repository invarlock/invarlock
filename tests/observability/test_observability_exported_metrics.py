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

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

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
class TestExportedMetric:
    """Tests for ExportedMetric dataclass."""

    def test_creation(self):
        """Test creating an exported metric."""
        from invarlock.observability.exporters import ExportedMetric

        metric = ExportedMetric(
            name="test_metric",
            value=42.0,
            timestamp=1000.0,
            labels={"env": "test"},
            metric_type="gauge",
            help_text="A test metric",
        )

        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.timestamp == 1000.0
        assert metric.labels == {"env": "test"}
        assert metric.metric_type == "gauge"
        assert metric.help_text == "A test metric"

    def test_default_values(self):
        """Test default values for optional fields."""
        from invarlock.observability.exporters import ExportedMetric

        metric = ExportedMetric(name="test", value=1.0, timestamp=1000.0)

        assert metric.labels == {}
        assert metric.metric_type == "gauge"
        assert metric.help_text == ""

    def test_to_prometheus_format(self):
        """Test conversion to Prometheus exposition format."""
        from invarlock.observability.exporters import ExportedMetric

        metric = ExportedMetric(
            name="test_metric",
            value=42.0,
            timestamp=1000.0,
            labels={"env": "prod", "region": "us-east"},
            metric_type="gauge",
            help_text="Test metric description",
        )

        prometheus_text = metric.to_prometheus_format()

        assert "# HELP test_metric Test metric description" in prometheus_text
        assert "# TYPE test_metric gauge" in prometheus_text
        assert 'test_metric{env="prod",region="us-east"}' in prometheus_text
        assert "42.0" in prometheus_text

    def test_to_prometheus_format_no_labels(self):
        """Test Prometheus format without labels."""
        from invarlock.observability.exporters import ExportedMetric

        metric = ExportedMetric(
            name="test_metric",
            value=100.0,
            timestamp=1000.0,
            metric_type="counter",
        )

        prometheus_text = metric.to_prometheus_format()

        assert "test_metric 100.0" in prometheus_text
        assert "# TYPE test_metric counter" in prometheus_text

    def test_to_json_format(self):
        """Test conversion to JSON format."""
        from invarlock.observability.exporters import ExportedMetric

        metric = ExportedMetric(
            name="test_metric",
            value=42.0,
            timestamp=1000.0,
            labels={"env": "test"},
            metric_type="gauge",
            help_text="Test help",
        )

        json_dict = metric.to_json_format()

        assert json_dict["metric"] == "test_metric"
        assert json_dict["value"] == 42.0
        assert json_dict["timestamp"] == 1000.0
        assert json_dict["labels"] == {"env": "test"}
        assert json_dict["type"] == "gauge"
        assert json_dict["help"] == "Test help"


@pytest.mark.unit
class TestMetricsExporter:
    """Tests for MetricsExporter base class."""

    def test_initialization(self):
        """Test exporter initialization."""
        from invarlock.observability.exporters import JSONExporter

        exporter = JSONExporter()

        assert exporter.name == "json"
        assert exporter.enabled is True
        assert exporter.last_export_time == 0.0
        assert exporter.export_count == 0
        assert exporter.error_count == 0

    def test_get_stats(self):
        """Test getting exporter statistics."""
        from invarlock.observability.exporters import JSONExporter

        exporter = JSONExporter()
        exporter.export_count = 10
        exporter.error_count = 2

        stats = exporter.get_stats()

        assert stats["name"] == "json"
        assert stats["enabled"] is True
        assert stats["export_count"] == 10
        assert stats["error_count"] == 2
        assert stats["success_rate"] == 0.8


@pytest.mark.unit
class TestPrometheusExporter:
    """Tests for PrometheusExporter."""

    def test_initialization(self):
        """Test Prometheus exporter initialization."""
        from invarlock.observability.exporters import PrometheusExporter

        exporter = PrometheusExporter(
            gateway_url="http://prometheus-gateway:9091",
            job_name="test_job",
            push_interval=30,
            instance="test-instance",
        )

        assert exporter.gateway_url == "http://prometheus-gateway:9091"
        assert exporter.job_name == "test_job"
        assert exporter.push_interval == 30
        assert exporter.instance == "test-instance"

    def test_initialization_defaults(self):
        """Test default values."""
        from invarlock.observability.exporters import PrometheusExporter

        exporter = PrometheusExporter()

        assert exporter.gateway_url is None
        assert exporter.job_name == "invarlock"
        assert exporter.push_interval == 15
        assert exporter.instance == "localhost"

    def test_export_to_cache(self):
        """Test exporting to internal cache (no gateway)."""
        from invarlock.observability.exporters import ExportedMetric, PrometheusExporter

        exporter = PrometheusExporter()

        metrics = [
            ExportedMetric(name="metric1", value=1.0, timestamp=time.time()),
            ExportedMetric(name="metric2", value=2.0, timestamp=time.time()),
        ]

        result = exporter.export(metrics)

        assert result is True
        assert exporter.export_count == 1
        assert len(exporter._metrics_cache) == 2

    def test_get_metrics_text(self):
        """Test getting cached metrics as Prometheus text."""
        from invarlock.observability.exporters import ExportedMetric, PrometheusExporter

        exporter = PrometheusExporter()

        metrics = [
            ExportedMetric(
                name="test_metric",
                value=42.0,
                timestamp=time.time(),
                metric_type="gauge",
            )
        ]
        exporter.export(metrics)

        text = exporter.get_metrics_text()

        assert "test_metric" in text
        assert "42.0" in text

    @patch("requests.post")
    def test_push_to_gateway(self, mock_post):
        """Test pushing metrics to Prometheus gateway."""
        from invarlock.observability.exporters import ExportedMetric, PrometheusExporter

        mock_post.return_value.status_code = 200

        exporter = PrometheusExporter(gateway_url="http://gateway:9091")

        metrics = [ExportedMetric(name="test", value=1.0, timestamp=time.time())]
        result = exporter.export(metrics)

        assert result is True
        assert mock_post.called
        call_url = mock_post.call_args[0][0]
        assert "gateway:9091" in call_url
        assert "invarlock" in call_url

    @patch("requests.post")
    def test_push_to_gateway_failure(self, mock_post):
        """Test handling gateway push failure."""
        from invarlock.observability.exporters import ExportedMetric, PrometheusExporter

        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal Server Error"

        exporter = PrometheusExporter(gateway_url="http://gateway:9091")

        metrics = [ExportedMetric(name="test", value=1.0, timestamp=time.time())]
        result = exporter.export(metrics)

        assert result is False
        assert exporter.error_count == 1


@pytest.mark.unit
class TestJSONExporter:
    """Tests for JSONExporter."""

    def test_initialization(self):
        """Test JSON exporter initialization."""
        from invarlock.observability.exporters import JSONExporter

        exporter = JSONExporter(output_file="/tmp/metrics.json", pretty_print=False)

        assert exporter.output_file == "/tmp/metrics.json"
        assert exporter.pretty_print is False

    def test_export_to_buffer(self):
        """Test exporting to internal buffer."""
        from invarlock.observability.exporters import ExportedMetric, JSONExporter

        exporter = JSONExporter()  # No output file = buffer mode

        metrics = [
            ExportedMetric(name="metric1", value=1.0, timestamp=time.time()),
            ExportedMetric(name="metric2", value=2.0, timestamp=time.time()),
        ]

        result = exporter.export(metrics)

        assert result is True
        assert len(exporter._metrics_buffer) == 2

    def test_get_buffered_metrics(self):
        """Test getting buffered metrics."""
        from invarlock.observability.exporters import ExportedMetric, JSONExporter

        exporter = JSONExporter()

        metrics = [ExportedMetric(name="test", value=42.0, timestamp=time.time())]
        exporter.export(metrics)

        buffered = exporter.get_buffered_metrics()

        assert len(buffered) == 1
        assert buffered[0]["metric"] == "test"
        assert buffered[0]["value"] == 42.0

    def test_clear_buffer(self):
        """Test clearing metrics buffer."""
        from invarlock.observability.exporters import ExportedMetric, JSONExporter

        exporter = JSONExporter()

        metrics = [ExportedMetric(name="test", value=1.0, timestamp=time.time())]
        exporter.export(metrics)
        assert len(exporter._metrics_buffer) == 1

        exporter.clear_buffer()
        assert len(exporter._metrics_buffer) == 0

    def test_buffer_size_limited(self):
        """Test buffer is limited to 10000 entries."""
        from invarlock.observability.exporters import ExportedMetric, JSONExporter

        exporter = JSONExporter()

        # Add many metrics
        for i in range(11000):
            metrics = [
                ExportedMetric(
                    name=f"metric_{i}", value=float(i), timestamp=time.time()
                )
            ]
            exporter.export(metrics)

        assert len(exporter._metrics_buffer) == 10000

    def test_export_to_file(self):
        """Test exporting to JSON file."""
        from invarlock.observability.exporters import ExportedMetric, JSONExporter

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            exporter = JSONExporter(output_file=output_path, pretty_print=True)

            metrics = [ExportedMetric(name="test_metric", value=42.0, timestamp=1000.0)]
            result = exporter.export(metrics)

            assert result is True

            # Read back and verify
            with open(output_path) as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]["metric"] == "test_metric"
            assert data[0]["value"] == 42.0
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_to_file_failure_returns_false(self):
        """File write failures should be reported as exporter errors."""
        from invarlock.observability.exporters import ExportedMetric, JSONExporter

        exporter = JSONExporter(output_file="/tmp/metrics.json", pretty_print=True)
        metrics = [ExportedMetric(name="test_metric", value=42.0, timestamp=1000.0)]

        with patch("builtins.open", side_effect=OSError("disk full")):
            result = exporter.export(metrics)

        assert result is False
        assert exporter.error_count == 1


@pytest.mark.unit
class TestInfluxDBExporter:
    """Tests for InfluxDBExporter."""

    def test_initialization(self):
        """Test InfluxDB exporter initialization."""
        from invarlock.observability.exporters import InfluxDBExporter

        exporter = InfluxDBExporter(
            url="http://influxdb:8086",
            database="metrics",
            username="user",
            password="pass",
            retention_policy="default",
        )

        assert exporter.url == "http://influxdb:8086"
        assert exporter.database == "metrics"
        assert exporter.username == "user"
        assert exporter.password == "pass"
        assert exporter.retention_policy == "default"

    def test_to_line_protocol(self):
        """Test conversion to InfluxDB line protocol."""
        from invarlock.observability.exporters import ExportedMetric, InfluxDBExporter

        exporter = InfluxDBExporter(url="http://influx:8086", database="test")

        metric = ExportedMetric(
            name="test_metric",
            value=42.0,
            timestamp=1.0,
            labels={"env": "prod", "region": "us"},
        )

        line = exporter._to_line_protocol(metric)

        assert "test_metric" in line
        assert "env=prod" in line
        assert "region=us" in line
        assert "value=42.0" in line
        assert "1000" in line  # timestamp in ms

    @patch("requests.post")
    def test_export_success(self, mock_post):
        """Test successful export to InfluxDB."""
        from invarlock.observability.exporters import ExportedMetric, InfluxDBExporter

        mock_post.return_value.status_code = 204

        exporter = InfluxDBExporter(url="http://influx:8086", database="test")

        metrics = [ExportedMetric(name="test", value=1.0, timestamp=time.time())]
        result = exporter.export(metrics)

        assert result is True
        assert mock_post.called

    @patch("requests.post")
    def test_export_empty_metrics(self, mock_post):
        """Test export with empty metrics list."""
        from invarlock.observability.exporters import InfluxDBExporter

        exporter = InfluxDBExporter(url="http://influx:8086", database="test")

        result = exporter.export([])

        assert result is True
        assert not mock_post.called


@pytest.mark.unit
class TestStatsExporter:
    """Tests for StatsD exporter."""

    def test_initialization(self):
        """Test StatsD exporter initialization."""
        from invarlock.observability.exporters import StatsExporter

        exporter = StatsExporter(host="statsd.local", port=8125, prefix="myapp")

        assert exporter.host == "statsd.local"
        assert exporter.port == 8125
        assert exporter.prefix == "myapp"

    def test_to_statsd_format_gauge(self):
        """Test conversion to StatsD gauge format."""
        from invarlock.observability.exporters import ExportedMetric, StatsExporter

        exporter = StatsExporter(prefix="test")

        metric = ExportedMetric(
            name="cpu_usage", value=75.5, timestamp=time.time(), metric_type="gauge"
        )

        line = exporter._to_statsd_format(metric)

        assert "test.cpu_usage" in line
        assert "75.5" in line
        assert "|g" in line

    def test_to_statsd_format_counter(self):
        """Test conversion to StatsD counter format."""
        from invarlock.observability.exporters import ExportedMetric, StatsExporter

        exporter = StatsExporter(prefix="test")

        metric = ExportedMetric(
            name="requests", value=100, timestamp=time.time(), metric_type="counter"
        )

        line = exporter._to_statsd_format(metric)

        assert "|c" in line

    def test_to_statsd_format_histogram(self):
        """Test conversion to StatsD histogram format."""
        from invarlock.observability.exporters import ExportedMetric, StatsExporter

        exporter = StatsExporter(prefix="test")

        metric = ExportedMetric(
            name="latency", value=0.5, timestamp=time.time(), metric_type="histogram"
        )

        line = exporter._to_statsd_format(metric)

        assert "|h" in line
