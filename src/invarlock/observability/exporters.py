"""
Metrics exporters for various monitoring systems.
"""

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from invarlock.core.exceptions import ObservabilityError


@dataclass
class ExportedMetric:
    """Represents a metric for export."""

    name: str
    value: float | int
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram, summary
    help_text: str = ""

    def to_prometheus_format(self) -> str:
        """Convert to Prometheus exposition format."""
        lines = []

        # Add help text
        if self.help_text:
            lines.append(f"# HELP {self.name} {self.help_text}")

        # Add type
        lines.append(f"# TYPE {self.name} {self.metric_type}")

        # Format labels
        if self.labels:
            label_str = ",".join([f'{k}="{v}"' for k, v in self.labels.items()])
            metric_line = (
                f"{self.name}{{{label_str}}} {self.value} {int(self.timestamp * 1000)}"
            )
        else:
            metric_line = f"{self.name} {self.value} {int(self.timestamp * 1000)}"

        lines.append(metric_line)
        return "\n".join(lines)

    def to_json_format(self) -> dict[str, Any]:
        """Convert to JSON format."""
        return {
            "metric": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "type": self.metric_type,
            "help": self.help_text,
        }


class MetricsExporter(ABC):
    """Base class for metrics exporters."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.enabled = True
        self.last_export_time: float = 0.0
        self.export_count = 0
        self.error_count = 0

    @abstractmethod
    def export(self, metrics: list[ExportedMetric]) -> bool:
        """Export metrics. Returns True if successful."""
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get exporter statistics."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "last_export_time": self.last_export_time,
            "export_count": self.export_count,
            "error_count": self.error_count,
            "success_rate": (self.export_count - self.error_count)
            / max(1, self.export_count),
        }


class PrometheusExporter(MetricsExporter):
    """Exporter for Prometheus format."""

    def __init__(
        self,
        gateway_url: str | None = None,
        job_name: str = "invarlock",
        push_interval: int = 15,
        instance: str | None = None,
    ):
        super().__init__("prometheus")
        self.gateway_url = gateway_url
        self.job_name = job_name
        self.push_interval = push_interval
        self.instance = instance or "localhost"

        # For HTTP server mode
        self._metrics_cache: dict[str, ExportedMetric] = {}
        self._cache_lock = threading.Lock()

    def export(self, metrics: list[ExportedMetric]) -> bool:
        """Export metrics to Prometheus."""
        try:
            if self.gateway_url:
                return self._push_to_gateway(metrics)
            else:
                return self._update_cache(metrics)
        except Exception as e:
            self.logger.error(f"Failed to export to Prometheus: {e}")
            self.error_count += 1
            return False

    def _push_to_gateway(self, metrics: list[ExportedMetric]) -> bool:
        """Push metrics to Prometheus Gateway."""
        try:
            import requests
        except ImportError:
            self.logger.error("requests library required for Prometheus Gateway")
            return False

        # Convert metrics to Prometheus format
        prometheus_data = "\n".join([m.to_prometheus_format() for m in metrics])

        # Push to gateway
        url = f"{self.gateway_url}/metrics/job/{self.job_name}/instance/{self.instance}"

        response = requests.post(
            url,
            data=prometheus_data,
            headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"},
            timeout=10,
        )

        if response.status_code == 200:
            self.export_count += 1
            self.last_export_time = time.time()
            return True
        else:
            self.logger.error(
                f"Prometheus Gateway returned {response.status_code}: {response.text}"
            )
            self.error_count += 1
            return False

    def _update_cache(self, metrics: list[ExportedMetric]) -> bool:
        """Update internal cache for HTTP server mode."""
        with self._cache_lock:
            for metric in metrics:
                key = f"{metric.name}_{hash(str(sorted(metric.labels.items())))}"
                self._metrics_cache[key] = metric

        self.export_count += 1
        self.last_export_time = time.time()
        return True

    def get_metrics_text(self) -> str:
        """Get current metrics in Prometheus format (for HTTP server)."""
        with self._cache_lock:
            return "\n".join(
                [m.to_prometheus_format() for m in self._metrics_cache.values()]
            )


class JSONExporter(MetricsExporter):
    """Exporter for JSON format."""

    def __init__(self, output_file: str | None = None, pretty_print: bool = True):
        super().__init__("json")
        self.output_file = output_file
        self.pretty_print = pretty_print
        self._metrics_buffer: list[dict[str, Any]] = []

    def export(self, metrics: list[ExportedMetric]) -> bool:
        """Export metrics to JSON."""
        try:
            json_metrics = [m.to_json_format() for m in metrics]

            if self.output_file:
                return self._write_to_file(json_metrics)
            else:
                return self._buffer_metrics(json_metrics)

        except Exception as e:
            self.logger.error(f"Failed to export to JSON: {e}")
            self.error_count += 1
            return False

    def _write_to_file(self, json_metrics: list[dict[str, Any]]) -> bool:
        """Write metrics to JSON file."""
        if self.output_file is None:
            self.logger.error("No output file specified")
            self.error_count += 1
            return False

        try:
            with open(self.output_file, "w") as f:
                if self.pretty_print:
                    json.dump(json_metrics, f, indent=2, default=str)
                else:
                    json.dump(json_metrics, f, default=str)

            self.export_count += 1
            self.last_export_time = time.time()
            return True

        except Exception as e:
            self.logger.error(f"Failed to write JSON file: {e}")
            self.error_count += 1
            return False

    def _buffer_metrics(self, json_metrics: list[dict[str, Any]]) -> bool:
        """Buffer metrics in memory."""
        self._metrics_buffer.extend(json_metrics)

        # Keep buffer size limited
        if len(self._metrics_buffer) > 10000:
            self._metrics_buffer = self._metrics_buffer[-10000:]

        self.export_count += 1
        self.last_export_time = time.time()
        return True

    def get_buffered_metrics(self) -> list[dict[str, Any]]:
        """Get buffered metrics."""
        return self._metrics_buffer.copy()

    def clear_buffer(self):
        """Clear metrics buffer."""
        self._metrics_buffer.clear()


def export_or_raise(exporter: MetricsExporter, metrics: list[ExportedMetric]) -> None:
    """Export metrics via an exporter or raise a typed ObservabilityError.

    - Raises ObservabilityError(E801) when export returns False or raises.
    - Includes exporter name and reason in details for debugging.
    """
    try:
        ok = exporter.export(metrics)
    except (
        Exception
    ) as e:  # pragma: no cover - covered via tests using failing exporter
        raise ObservabilityError(
            code="E801",
            message="OBSERVABILITY-EXPORT-FAILED",
            details={"exporter": exporter.name, "reason": type(e).__name__},
        ) from e
    if not ok:
        raise ObservabilityError(
            code="E801",
            message="OBSERVABILITY-EXPORT-FAILED",
            details={"exporter": exporter.name, "reason": "returned_false"},
        )


class InfluxDBExporter(MetricsExporter):
    """Exporter for InfluxDB."""

    def __init__(
        self,
        url: str,
        database: str,
        username: str | None = None,
        password: str | None = None,
        retention_policy: str = "autogen",
    ):
        super().__init__("influxdb")
        self.url = url.rstrip("/")
        self.database = database
        self.username = username
        self.password = password
        self.retention_policy = retention_policy

    def export(self, metrics: list[ExportedMetric]) -> bool:
        """Export metrics to InfluxDB."""
        try:
            import requests
        except ImportError:
            self.logger.error("requests library required for InfluxDB")
            return False

        try:
            # Convert metrics to InfluxDB line protocol
            lines = []
            for metric in metrics:
                line = self._to_line_protocol(metric)
                if line:
                    lines.append(line)

            if not lines:
                return True

            # Write to InfluxDB
            write_url = f"{self.url}/write"
            params = {
                "db": self.database,
                "rp": self.retention_policy,
                "precision": "ms",
            }

            auth = None
            if self.username and self.password:
                auth = (self.username, self.password)

            response = requests.post(
                write_url,
                params=params,
                data="\n".join(lines),
                auth=auth,
                headers={"Content-Type": "text/plain"},
                timeout=10,
            )

            if response.status_code == 204:
                self.export_count += 1
                self.last_export_time = time.time()
                return True
            else:
                self.logger.error(
                    f"InfluxDB returned {response.status_code}: {response.text}"
                )
                self.error_count += 1
                return False

        except Exception as e:
            self.logger.error(f"Failed to export to InfluxDB: {e}")
            self.error_count += 1
            return False

    def _to_line_protocol(self, metric: ExportedMetric) -> str:
        """Convert metric to InfluxDB line protocol."""
        # Escape special characters in measurement name
        measurement = (
            metric.name.replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")
        )

        # Build tags
        tag_parts = []
        for key, value in metric.labels.items():
            escaped_key = (
                key.replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")
            )
            escaped_value = (
                str(value).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")
            )
            tag_parts.append(f"{escaped_key}={escaped_value}")

        tags = "," + ",".join(tag_parts) if tag_parts else ""

        # Build fields (for InfluxDB, we need at least one field)
        fields = f"value={metric.value}"

        # Timestamp in milliseconds
        timestamp = int(metric.timestamp * 1000)

        return f"{measurement}{tags} {fields} {timestamp}"


class StatsExporter(MetricsExporter):
    """Exporter for StatsD protocol."""

    def __init__(
        self, host: str = "localhost", port: int = 8125, prefix: str = "invarlock"
    ):
        super().__init__("statsd")
        self.host = host
        self.port = port
        self.prefix = prefix
        self._socket: Any | None = None

    def export(self, metrics: list[ExportedMetric]) -> bool:
        """Export metrics to StatsD."""
        try:
            import socket
        except ImportError:
            self.logger.error("socket library required for StatsD")
            return False

        try:
            if not self._socket:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            for metric in metrics:
                statsd_line = self._to_statsd_format(metric)
                if statsd_line and self._socket:
                    self._socket.sendto(
                        statsd_line.encode("utf-8"), (self.host, self.port)
                    )

            self.export_count += 1
            self.last_export_time = time.time()
            return True

        except Exception as e:
            self.logger.error(f"Failed to export to StatsD: {e}")
            self.error_count += 1
            return False

    def _to_statsd_format(self, metric: ExportedMetric) -> str:
        """Convert metric to StatsD format."""
        # Build metric name with prefix
        name_parts = [self.prefix] if self.prefix else []
        name_parts.append(metric.name.replace(".", "_").replace(" ", "_"))

        # Add labels as tags (if supported)
        if metric.labels:
            label_parts = [f"{k}:{v}" for k, v in metric.labels.items()]
            name_parts.extend(label_parts)

        metric_name = ".".join(name_parts)

        # Determine StatsD type
        if metric.metric_type == "counter":
            return f"{metric_name}:{metric.value}|c"
        elif metric.metric_type == "histogram":
            return f"{metric_name}:{metric.value}|h"
        else:  # gauge
            return f"{metric_name}:{metric.value}|g"


class ExportManager:
    """Manages multiple metrics exporters."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.exporters: dict[str, MetricsExporter] = {}
        self.export_interval = 10  # seconds
        self._running = False
        self._export_thread = None
        self._metrics_queue = []
        self._queue_lock = threading.Lock()

    def add_exporter(self, exporter: MetricsExporter):
        """Add a metrics exporter."""
        self.exporters[exporter.name] = exporter
        self.logger.info(f"Added exporter: {exporter.name}")

    def remove_exporter(self, name: str):
        """Remove a metrics exporter."""
        self.exporters.pop(name, None)
        self.logger.info(f"Removed exporter: {name}")

    def queue_metrics(self, metrics: list[ExportedMetric]):
        """Queue metrics for export."""
        with self._queue_lock:
            self._metrics_queue.extend(metrics)

    def export_now(
        self, metrics: list[ExportedMetric] | None = None
    ) -> dict[str, bool]:
        """Export metrics immediately."""
        if metrics is None:
            with self._queue_lock:
                metrics = self._metrics_queue.copy()
                self._metrics_queue.clear()

        results = {}
        for name, exporter in self.exporters.items():
            if exporter.enabled:
                try:
                    results[name] = exporter.export(metrics)
                except Exception as e:
                    self.logger.error(f"Exporter {name} failed: {e}")
                    results[name] = False
            else:
                results[name] = False  # Disabled

        return results

    def start_background_export(self):
        """Start background export thread."""
        if self._running:
            return

        self._running = True
        self._export_thread = threading.Thread(target=self._export_loop, daemon=True)
        self._export_thread.start()
        self.logger.info("Started background metrics export")

    def stop_background_export(self):
        """Stop background export thread."""
        self._running = False
        if self._export_thread:
            self._export_thread.join(timeout=5)
        self.logger.info("Stopped background metrics export")

    def _export_loop(self):
        """Background export loop."""
        while self._running:
            try:
                time.sleep(self.export_interval)

                # Get queued metrics
                with self._queue_lock:
                    if self._metrics_queue:
                        metrics = self._metrics_queue.copy()
                        self._metrics_queue.clear()

                        # Export to all enabled exporters
                        self.export_now(metrics)

            except Exception as e:
                self.logger.error(f"Error in export loop: {e}")

    def get_exporter_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all exporters."""
        return {name: exporter.get_stats() for name, exporter in self.exporters.items()}

    def get_summary(self) -> dict[str, Any]:
        """Get export manager summary."""
        total_exports = sum(e.export_count for e in self.exporters.values())
        total_errors = sum(e.error_count for e in self.exporters.values())

        with self._queue_lock:
            queue_size = len(self._metrics_queue)

        return {
            "total_exporters": len(self.exporters),
            "enabled_exporters": len([e for e in self.exporters.values() if e.enabled]),
            "total_exports": total_exports,
            "total_errors": total_errors,
            "success_rate": (total_exports - total_errors) / max(1, total_exports),
            "queue_size": queue_size,
            "background_running": self._running,
            "export_interval": self.export_interval,
        }


# Utility functions for common exporter setups
def setup_prometheus_exporter(
    gateway_url: str | None = None, job_name: str = "invarlock"
) -> PrometheusExporter:
    """Setup Prometheus exporter."""
    return PrometheusExporter(gateway_url=gateway_url, job_name=job_name)


def setup_json_file_exporter(output_file: str) -> JSONExporter:
    """Setup JSON file exporter."""
    return JSONExporter(output_file=output_file)


def setup_influxdb_exporter(
    url: str, database: str, username: str | None = None, password: str | None = None
) -> InfluxDBExporter:
    """Setup InfluxDB exporter."""
    return InfluxDBExporter(
        url=url, database=database, username=username, password=password
    )


def setup_statsd_exporter(
    host: str = "localhost", port: int = 8125, prefix: str = "invarlock"
) -> StatsExporter:
    """Setup StatsD exporter."""
    return StatsExporter(host=host, port=port, prefix=prefix)
