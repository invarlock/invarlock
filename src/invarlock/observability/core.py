"""
Core monitoring and telemetry infrastructure.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import psutil
import torch

from .alerting import AlertManager, AlertSeverity
from .health import HealthChecker
from .metrics import MetricsRegistry


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system."""

    # Collection intervals
    metrics_interval: float = 10.0  # seconds
    health_check_interval: float = 30.0  # seconds
    resource_check_interval: float = 5.0  # seconds

    # Data retention
    metrics_retention_hours: int = 24
    max_events: int = 10000

    # Alerting
    enable_alerting: bool = True
    alert_channels: list[str] = field(default_factory=list)

    # Export settings
    prometheus_enabled: bool = False
    prometheus_port: int = 9090
    json_export_enabled: bool = True
    json_export_path: str = "./monitoring"

    # Resource monitoring
    cpu_threshold: float = 80.0  # percent
    memory_threshold: float = 85.0  # percent
    gpu_memory_threshold: float = 90.0  # percent

    # Performance monitoring
    latency_percentiles: list[float] = field(default_factory=lambda: [50, 90, 95, 99])
    slow_request_threshold: float = 30.0  # seconds


class MonitoringManager:
    """Central monitoring manager for InvarLock operations."""

    def __init__(self, config: MonitoringConfig | None = None):
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(__name__)

        # Core components
        self.metrics = MetricsRegistry()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()

        # Monitoring threads
        self._monitoring_threads: list = []
        self._stop_event = threading.Event()

        # Performance tracking
        self.performance_monitor = PerformanceMonitor(self.metrics)
        self.resource_monitor = ResourceMonitor(self.metrics, self.config)

        # Initialize default metrics
        self._setup_default_metrics()

        # Setup alerting rules
        self._setup_default_alerts()

    def start(self):
        """Start all monitoring components."""
        self.logger.info("Starting InvarLock monitoring system")

        # Start metrics collection
        metrics_thread = threading.Thread(
            target=self._metrics_collection_loop, name="MetricsCollector"
        )
        metrics_thread.daemon = True
        metrics_thread.start()
        self._monitoring_threads.append(metrics_thread)

        # Start health checking
        health_thread = threading.Thread(
            target=self._health_check_loop, name="HealthChecker"
        )
        health_thread.daemon = True
        health_thread.start()
        self._monitoring_threads.append(health_thread)

        # Start resource monitoring
        resource_thread = threading.Thread(
            target=self._resource_monitoring_loop, name="ResourceMonitor"
        )
        resource_thread.daemon = True
        resource_thread.start()
        self._monitoring_threads.append(resource_thread)

        self.logger.info("Monitoring system started successfully")

    def stop(self):
        """Stop all monitoring components."""
        self.logger.info("Stopping InvarLock monitoring system")

        self._stop_event.set()

        # Wait for threads to finish
        for thread in self._monitoring_threads:
            thread.join(timeout=5.0)

        # Export final metrics
        self._export_metrics()

        self.logger.info("Monitoring system stopped")

    def record_operation(self, operation: str, duration: float, **metadata):
        """Record an operation with timing and metadata."""
        self.performance_monitor.record_operation(operation, duration, **metadata)

    def record_error(self, error_type: str, error_msg: str, **context):
        """Record an error event."""
        self.metrics.get_counter("invarlock.errors.total").inc(
            labels={"type": error_type}
        )

        # Log error with context
        self.logger.error(f"Error recorded: {error_type} - {error_msg}", extra=context)

        # Check if alert should be triggered
        self.alert_manager.check_error_alerts(error_type, error_msg, context)

    def get_status(self) -> dict[str, Any]:
        """Get current monitoring status."""
        return {
            "monitoring_active": not self._stop_event.is_set(),
            "metrics_count": len(self.metrics._metrics),
            "health_status": self.health_checker.get_overall_status(),
            "active_alerts": self.alert_manager.get_active_alerts(),
            "resource_usage": self.resource_monitor.get_current_usage(),
            "performance_stats": self.performance_monitor.get_summary(),
            "uptime": self._get_uptime(),
        }

    def _setup_default_metrics(self):
        """Setup default InvarLock metrics."""
        # Operation counters
        self.metrics.register_counter(
            "invarlock.operations.total", "Total InvarLock operations"
        )
        self.metrics.register_counter("invarlock.errors.total", "Total errors")
        self.metrics.register_counter("invarlock.edits.applied", "Total edits applied")
        self.metrics.register_counter("invarlock.guards.triggered", "Guard triggers")

        # Performance metrics
        self.metrics.register_histogram(
            "invarlock.operation.duration", "Operation duration"
        )
        self.metrics.register_histogram(
            "invarlock.edit.duration", "Edit operation duration"
        )
        self.metrics.register_histogram(
            "invarlock.guard.duration", "Guard execution duration"
        )

        # Resource metrics
        self.metrics.register_gauge("invarlock.memory.usage", "Memory usage")
        self.metrics.register_gauge("invarlock.gpu.memory.usage", "GPU memory usage")
        self.metrics.register_gauge("invarlock.cpu.usage", "CPU usage")

        # Model metrics
        self.metrics.register_gauge(
            "invarlock.model.parameters", "Model parameter count"
        )
        self.metrics.register_gauge("invarlock.model.size_mb", "Model size in MB")
        self.metrics.register_counter("invarlock.model.loads", "Model loads")

    def _setup_default_alerts(self):
        """Setup default alerting rules."""
        if not self.config.enable_alerting:
            return

        from .alerting import AlertRule

        # High error rate alert
        self.alert_manager.add_rule(
            AlertRule(
                name="high_error_rate",
                metric="invarlock.errors.total",
                threshold=10,
                window_minutes=5,
                severity=AlertSeverity.WARNING,
                message="High error rate detected",
            )
        )

        # Resource usage alerts
        self.alert_manager.add_rule(
            AlertRule(
                name="high_memory_usage",
                metric="invarlock.memory.usage",
                threshold=self.config.memory_threshold,
                severity=AlertSeverity.WARNING,
                message="High memory usage detected",
            )
        )

        # Performance alerts
        self.alert_manager.add_rule(
            AlertRule(
                name="slow_operations",
                metric="invarlock.operation.duration",
                threshold=self.config.slow_request_threshold,
                percentile=95,
                severity=AlertSeverity.WARNING,
                message="Slow operations detected",
            )
        )

    def _metrics_collection_loop(self):
        """Main metrics collection loop."""
        while not self._stop_event.is_set():
            try:
                # Update resource metrics
                self.resource_monitor.update_metrics()

                # Update performance metrics
                self.performance_monitor.update_metrics()

                # Export metrics if needed
                if self.config.json_export_enabled:
                    self._export_metrics()

            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")

            self._stop_event.wait(self.config.metrics_interval)

    def _health_check_loop(self):
        """Health check monitoring loop."""
        while not self._stop_event.is_set():
            try:
                # Run health checks
                health_status = self.health_checker.check_all()

                # Update health metrics
                for component, status in health_status.items():
                    self.metrics.get_gauge("invarlock.health.status").set(
                        1 if status.healthy else 0, labels={"component": component}
                    )

                # Check for health-based alerts
                self.alert_manager.check_health_alerts(health_status)

            except Exception as e:
                self.logger.error(f"Error in health checking: {e}")

            self._stop_event.wait(self.config.health_check_interval)

    def _resource_monitoring_loop(self):
        """Resource monitoring loop."""
        while not self._stop_event.is_set():
            try:
                # Monitor resource usage
                usage = self.resource_monitor.collect_usage()

                # Check resource-based alerts
                self.alert_manager.check_resource_alerts(usage)

            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")

            self._stop_event.wait(self.config.resource_check_interval)

    def _export_metrics(self):
        """Export metrics to configured outputs."""
        try:
            if self.config.json_export_enabled:
                from .exporters import JSONExporter

                exporter = JSONExporter(self.config.json_export_path)
                exporter.export(self.metrics.get_all_metrics())

        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")

    def _get_uptime(self) -> float:
        """Get monitoring system uptime in seconds."""
        return time.time() - getattr(self, "_start_time", time.time())


class TelemetryCollector:
    """Collects telemetry data for InvarLock operations."""

    def __init__(self, monitoring_manager: MonitoringManager):
        self.monitoring = monitoring_manager
        self.logger = logging.getLogger(__name__)

        # Operation tracking
        self.active_operations: dict = {}
        self.operation_history: deque = deque(maxlen=1000)

    def start_operation(
        self, operation_id: str, operation_type: str, **metadata
    ) -> str:
        """Start tracking an operation."""
        start_time = time.time()

        operation_data = {
            "id": operation_id,
            "type": operation_type,
            "start_time": start_time,
            "metadata": metadata,
        }

        self.active_operations[operation_id] = operation_data

        # Record operation start
        self.monitoring.metrics.get_counter("invarlock.operations.total").inc(
            labels={"type": operation_type, "status": "started"}
        )

        self.logger.info(f"Operation started: {operation_id} ({operation_type})")
        return operation_id

    def end_operation(
        self, operation_id: str, status: str = "success", **result_metadata
    ):
        """End tracking an operation."""
        if operation_id not in self.active_operations:
            self.logger.warning(f"Unknown operation ID: {operation_id}")
            return

        operation_data = self.active_operations.pop(operation_id)
        end_time = time.time()
        duration = end_time - operation_data["start_time"]

        # Complete operation record
        operation_record = {
            **operation_data,
            "end_time": end_time,
            "duration": duration,
            "status": status,
            "result_metadata": result_metadata,
        }

        self.operation_history.append(operation_record)

        # Record metrics
        self.monitoring.record_operation(
            operation_data["type"],
            duration,
            status=status,
            **operation_data["metadata"],
            **result_metadata,
        )

        self.logger.info(
            f"Operation completed: {operation_id} ({operation_data['type']}) "
            f"- {status} in {duration:.2f}s"
        )

    def get_operation_stats(self) -> dict[str, Any]:
        """Get operation statistics."""
        if not self.operation_history:
            return {}

        operations = list(self.operation_history)
        total_ops = len(operations)

        # Calculate statistics
        durations = [op["duration"] for op in operations]
        avg_duration = sum(durations) / len(durations)

        status_counts: dict = defaultdict(int)
        type_counts: dict = defaultdict(int)

        for op in operations:
            status_counts[op["status"]] += 1
            type_counts[op["type"]] += 1

        return {
            "total_operations": total_ops,
            "active_operations": len(self.active_operations),
            "average_duration": avg_duration,
            "status_distribution": dict(status_counts),
            "type_distribution": dict(type_counts),
            "success_rate": status_counts["success"] / total_ops
            if total_ops > 0
            else 0,
        }


class PerformanceMonitor:
    """Monitors InvarLock performance metrics."""

    def __init__(self, metrics_registry: MetricsRegistry):
        self.metrics = metrics_registry
        self.operation_times: dict = defaultdict(list)
        self.performance_data: dict = defaultdict(dict)

    def record_operation(self, operation: str, duration: float, **metadata):
        """Record an operation's performance."""
        # Store timing data
        self.operation_times[operation].append(duration)

        # Keep only recent measurements (last 1000)
        if len(self.operation_times[operation]) > 1000:
            self.operation_times[operation] = self.operation_times[operation][-1000:]

        # Update histogram metric
        self.metrics.get_histogram("invarlock.operation.duration").observe(
            duration, labels={"operation": operation}
        )

        # Store metadata
        if metadata:
            self.performance_data[operation].update(metadata)

    def get_operation_stats(self, operation: str) -> dict[str, float]:
        """Get statistics for a specific operation."""
        times = self.operation_times.get(operation, [])
        if not times:
            return {}

        times_sorted = sorted(times)
        count = len(times)

        return {
            "count": count,
            "mean": sum(times) / count,
            "min": min(times),
            "max": max(times),
            "p50": times_sorted[int(count * 0.5)],
            "p90": times_sorted[int(count * 0.9)],
            "p95": times_sorted[int(count * 0.95)],
            "p99": times_sorted[int(count * 0.99)],
        }

    def get_summary(self) -> dict[str, Any]:
        """Get performance summary for all operations."""
        summary = {}
        for operation in self.operation_times:
            summary[operation] = self.get_operation_stats(operation)
        return summary

    def update_metrics(self):
        """Update performance metrics."""
        # Update operation-specific metrics
        for operation, stats in self.get_summary().items():
            if stats:
                # Update gauge metrics for key percentiles
                self.metrics.get_gauge("invarlock.operation.p95_duration").set(
                    stats["p95"], labels={"operation": operation}
                )
                self.metrics.get_gauge("invarlock.operation.mean_duration").set(
                    stats["mean"], labels={"operation": operation}
                )


class ResourceMonitor:
    """Monitors system resource usage."""

    def __init__(self, metrics_registry: MetricsRegistry, config: MonitoringConfig):
        self.metrics = metrics_registry
        self.config = config
        self.logger = logging.getLogger(__name__)

    def collect_usage(self) -> dict[str, float]:
        """Collect current resource usage."""
        usage = {}

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            usage["cpu_percent"] = cpu_percent

            # Memory usage
            memory = psutil.virtual_memory()
            usage["memory_percent"] = memory.percent
            usage["memory_available_gb"] = memory.available / (1024**3)
            usage["memory_used_gb"] = memory.used / (1024**3)

            # GPU usage (if available)
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.memory_stats(i)
                    allocated = gpu_memory.get("allocated_bytes.all.current", 0)
                    reserved = gpu_memory.get("reserved_bytes.all.current", 0)

                    usage[f"gpu_{i}_memory_allocated_gb"] = allocated / (1024**3)
                    usage[f"gpu_{i}_memory_reserved_gb"] = reserved / (1024**3)

                    # Calculate percentage of total memory
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    usage[f"gpu_{i}_memory_percent"] = (allocated / total_memory) * 100

            # Disk usage
            disk = psutil.disk_usage("/")
            usage["disk_percent"] = (disk.used / disk.total) * 100
            usage["disk_free_gb"] = disk.free / (1024**3)

        except Exception as e:
            self.logger.error(f"Error collecting resource usage: {e}")

        return usage

    def update_metrics(self):
        """Update resource metrics."""
        usage = self.collect_usage()

        for metric_name, value in usage.items():
            metric_key = f"invarlock.resource.{metric_name}"
            self.metrics.get_gauge(metric_key).set(value)

    def get_current_usage(self) -> dict[str, float]:
        """Get current resource usage."""
        return self.collect_usage()

    def check_thresholds(self) -> list[str]:
        """Check if any resource usage exceeds thresholds."""
        usage = self.collect_usage()
        warnings = []

        if usage.get("cpu_percent", 0) > self.config.cpu_threshold:
            warnings.append(f"High CPU usage: {usage['cpu_percent']:.1f}%")

        if usage.get("memory_percent", 0) > self.config.memory_threshold:
            warnings.append(f"High memory usage: {usage['memory_percent']:.1f}%")

        # Check GPU memory
        for key, value in usage.items():
            if (
                key.endswith("_memory_percent")
                and value > self.config.gpu_memory_threshold
            ):
                warnings.append(f"High GPU memory usage: {key} = {value:.1f}%")

        return warnings
