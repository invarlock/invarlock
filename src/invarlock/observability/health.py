"""
Health checking and status monitoring.
"""

import logging
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psutil
import torch


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a component."""

    name: str
    status: HealthStatus
    message: str
    details: dict[str, Any]
    timestamp: float

    @property
    def healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == HealthStatus.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "healthy": self.healthy,
        }


class HealthChecker:
    """System health monitoring."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.health_checks: dict[str, Callable[[], ComponentHealth]] = {}
        self.last_results: dict[str, ComponentHealth] = {}

        # Register default health checks
        self._register_default_checks()

    def register_check(self, name: str, check_func: Callable[[], ComponentHealth]):
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")

    def check_component(self, name: str) -> ComponentHealth:
        """Check health of a specific component."""
        if name not in self.health_checks:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"No health check registered for {name}",
                details={},
                timestamp=time.time(),
            )

        try:
            result = self.health_checks[name]()
            self.last_results[name] = result
            return result
        except Exception as e:
            error_result = ComponentHealth(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e), "traceback": traceback.format_exc()},
                timestamp=time.time(),
            )
            self.last_results[name] = error_result
            return error_result

    def check_all(self) -> dict[str, ComponentHealth]:
        """Check health of all registered components."""
        results = {}
        for name in self.health_checks:
            results[name] = self.check_component(name)
        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.last_results:
            return HealthStatus.UNKNOWN

        statuses = [result.status for result in self.last_results.values()]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def get_summary(self) -> dict[str, Any]:
        """Get health summary."""
        overall_status = self.get_overall_status()

        status_counts = {status.value: 0 for status in HealthStatus}
        for result in self.last_results.values():
            status_counts[result.status.value] += 1

        return {
            "overall_status": overall_status.value,
            "total_components": len(self.health_checks),
            "status_counts": status_counts,
            "last_check": max([r.timestamp for r in self.last_results.values()])
            if self.last_results
            else 0,
            "components": {
                name: result.to_dict() for name, result in self.last_results.items()
            },
        }

    def _register_default_checks(self):
        """Register default system health checks."""

        def check_memory():
            """Check system memory usage."""
            try:
                memory = psutil.virtual_memory()
                percent = memory.percent

                if percent > 90:
                    status = HealthStatus.CRITICAL
                    message = f"Critical memory usage: {percent:.1f}%"
                elif percent > 80:
                    status = HealthStatus.WARNING
                    message = f"High memory usage: {percent:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Memory usage normal: {percent:.1f}%"

                return ComponentHealth(
                    name="memory",
                    status=status,
                    message=message,
                    details={
                        "percent": percent,
                        "available_gb": memory.available / (1024**3),
                        "used_gb": memory.used / (1024**3),
                        "total_gb": memory.total / (1024**3),
                    },
                    timestamp=time.time(),
                )
            except Exception as e:
                return ComponentHealth(
                    name="memory",
                    status=HealthStatus.CRITICAL,
                    message=f"Failed to check memory: {e}",
                    details={"error": str(e)},
                    timestamp=time.time(),
                )

        def check_cpu():
            """Check CPU usage."""
            try:
                cpu_percent = psutil.cpu_percent(interval=1)

                if cpu_percent > 95:
                    status = HealthStatus.CRITICAL
                    message = f"Critical CPU usage: {cpu_percent:.1f}%"
                elif cpu_percent > 85:
                    status = HealthStatus.WARNING
                    message = f"High CPU usage: {cpu_percent:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"CPU usage normal: {cpu_percent:.1f}%"

                return ComponentHealth(
                    name="cpu",
                    status=status,
                    message=message,
                    details={
                        "percent": cpu_percent,
                        "core_count": psutil.cpu_count(),
                        "load_avg": psutil.getloadavg()
                        if hasattr(psutil, "getloadavg")
                        else None,
                    },
                    timestamp=time.time(),
                )
            except Exception as e:
                return ComponentHealth(
                    name="cpu",
                    status=HealthStatus.CRITICAL,
                    message=f"Failed to check CPU: {e}",
                    details={"error": str(e)},
                    timestamp=time.time(),
                )

        def check_disk():
            """Check disk space."""
            try:
                disk = psutil.disk_usage("/")
                percent = (disk.used / disk.total) * 100

                if percent > 95:
                    status = HealthStatus.CRITICAL
                    message = f"Critical disk usage: {percent:.1f}%"
                elif percent > 85:
                    status = HealthStatus.WARNING
                    message = f"High disk usage: {percent:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Disk usage normal: {percent:.1f}%"

                return ComponentHealth(
                    name="disk",
                    status=status,
                    message=message,
                    details={
                        "percent": percent,
                        "free_gb": disk.free / (1024**3),
                        "used_gb": disk.used / (1024**3),
                        "total_gb": disk.total / (1024**3),
                    },
                    timestamp=time.time(),
                )
            except Exception as e:
                return ComponentHealth(
                    name="disk",
                    status=HealthStatus.CRITICAL,
                    message=f"Failed to check disk: {e}",
                    details={"error": str(e)},
                    timestamp=time.time(),
                )

        def check_gpu():
            """Check GPU status."""
            try:
                if not torch.cuda.is_available():
                    return ComponentHealth(
                        name="gpu",
                        status=HealthStatus.HEALTHY,
                        message="GPU not available (CPU-only mode)",
                        details={"cuda_available": False},
                        timestamp=time.time(),
                    )

                gpu_count = torch.cuda.device_count()
                gpu_details = {}
                max_memory_percent = 0

                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_stats = torch.cuda.memory_stats(i)

                    allocated = memory_stats.get("allocated_bytes.all.current", 0)
                    total = props.total_memory
                    percent = (allocated / total) * 100
                    max_memory_percent = max(max_memory_percent, percent)

                    gpu_details[f"gpu_{i}"] = {
                        "name": props.name,
                        "memory_allocated_gb": allocated / (1024**3),
                        "memory_total_gb": total / (1024**3),
                        "memory_percent": percent,
                    }

                if max_memory_percent > 95:
                    status = HealthStatus.CRITICAL
                    message = f"Critical GPU memory usage: {max_memory_percent:.1f}%"
                elif max_memory_percent > 85:
                    status = HealthStatus.WARNING
                    message = f"High GPU memory usage: {max_memory_percent:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = (
                        f"GPU status normal: {max_memory_percent:.1f}% memory used"
                    )

                return ComponentHealth(
                    name="gpu",
                    status=status,
                    message=message,
                    details={
                        "cuda_available": True,
                        "device_count": gpu_count,
                        "max_memory_percent": max_memory_percent,
                        "devices": gpu_details,
                    },
                    timestamp=time.time(),
                )
            except Exception as e:
                return ComponentHealth(
                    name="gpu",
                    status=HealthStatus.WARNING,
                    message=f"Failed to check GPU: {e}",
                    details={"error": str(e)},
                    timestamp=time.time(),
                )

        def check_pytorch():
            """Check PyTorch availability and functionality."""
            try:
                # Basic PyTorch functionality test
                test_tensor = torch.randn(10, 10)
                torch.mm(test_tensor, test_tensor.t())

                details = {
                    "version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_version": torch.version.cuda
                    if torch.cuda.is_available()
                    else None,
                    "device_count": torch.cuda.device_count()
                    if torch.cuda.is_available()
                    else 0,
                }

                # Check for MPS (Apple Silicon) availability
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    details["mps_available"] = True

                return ComponentHealth(
                    name="pytorch",
                    status=HealthStatus.HEALTHY,
                    message="PyTorch working correctly",
                    details=details,
                    timestamp=time.time(),
                )
            except Exception as e:
                return ComponentHealth(
                    name="pytorch",
                    status=HealthStatus.CRITICAL,
                    message=f"PyTorch check failed: {e}",
                    details={"error": str(e)},
                    timestamp=time.time(),
                )

        # Register all default checks
        self.register_check("memory", check_memory)
        self.register_check("cpu", check_cpu)
        self.register_check("disk", check_disk)
        self.register_check("gpu", check_gpu)
        self.register_check("pytorch", check_pytorch)


class InvarLockHealthChecker(HealthChecker):
    """InvarLock-specific health checker with additional checks."""

    def __init__(self):
        super().__init__()
        self._register_invarlock_checks()

    def _register_invarlock_checks(self):
        """Register InvarLock-specific health checks."""

        def check_adapters():
            """Check adapter availability."""
            try:
                from invarlock.adapters import (
                    HF_BERT_Adapter,
                    HF_GPT2_Adapter,
                    HF_LLaMA_Adapter,
                )

                adapters = {
                    "hf_gpt2": HF_GPT2_Adapter,
                    "hf_llama": HF_LLaMA_Adapter,
                    "hf_bert": HF_BERT_Adapter,
                }

                available_adapters = []
                failed_adapters = []

                for name, adapter_class in adapters.items():
                    try:
                        adapter_class()
                        available_adapters.append(name)
                    except Exception as e:
                        failed_adapters.append({"name": name, "error": str(e)})

                if not available_adapters:
                    status = HealthStatus.CRITICAL
                    message = "No adapters available"
                elif failed_adapters:
                    status = HealthStatus.WARNING
                    message = (
                        f"Some adapters failed: {[f['name'] for f in failed_adapters]}"
                    )
                else:
                    status = HealthStatus.HEALTHY
                    message = f"All adapters available: {available_adapters}"

                return ComponentHealth(
                    name="adapters",
                    status=status,
                    message=message,
                    details={
                        "available": available_adapters,
                        "failed": failed_adapters,
                        "total_adapters": len(adapters),
                    },
                    timestamp=time.time(),
                )
            except Exception as e:
                return ComponentHealth(
                    name="adapters",
                    status=HealthStatus.CRITICAL,
                    message=f"Failed to check adapters: {e}",
                    details={"error": str(e)},
                    timestamp=time.time(),
                )

        def check_guards():
            """Check guard system availability."""
            try:
                from invarlock.guards import (
                    InvariantsGuard,
                    RMTGuard,
                    SpectralGuard,
                    VarianceGuard,
                )

                guards = {
                    "spectral": SpectralGuard,
                    "rmt": RMTGuard,
                    "invariants": InvariantsGuard,
                    "variance": VarianceGuard,
                }

                available_guards = []
                failed_guards = []

                for name, guard_class in guards.items():
                    try:
                        if name == "variance":
                            # Variance guard needs a policy
                            from invarlock.guards.policies import get_variance_policy

                            guard_class(get_variance_policy("balanced"))
                        else:
                            guard_class()
                        available_guards.append(name)
                    except Exception as e:
                        failed_guards.append({"name": name, "error": str(e)})

                if not available_guards:
                    status = HealthStatus.CRITICAL
                    message = "No guards available"
                elif failed_guards:
                    status = HealthStatus.WARNING
                    message = (
                        f"Some guards failed: {[f['name'] for f in failed_guards]}"
                    )
                else:
                    status = HealthStatus.HEALTHY
                    message = f"All guards available: {available_guards}"

                return ComponentHealth(
                    name="guards",
                    status=status,
                    message=message,
                    details={
                        "available": available_guards,
                        "failed": failed_guards,
                        "total_guards": len(guards),
                    },
                    timestamp=time.time(),
                )
            except Exception as e:
                return ComponentHealth(
                    name="guards",
                    status=HealthStatus.CRITICAL,
                    message=f"Failed to check guards: {e}",
                    details={"error": str(e)},
                    timestamp=time.time(),
                )

        def check_dependencies():
            """Check critical dependencies."""
            try:
                dependencies = {
                    "torch": "torch",
                    "transformers": "transformers",
                    "numpy": "numpy",
                    "psutil": "psutil",
                }

                available_deps = []
                missing_deps = []

                for name, module_name in dependencies.items():
                    try:
                        __import__(module_name)
                        available_deps.append(name)
                    except ImportError:
                        missing_deps.append(name)

                if missing_deps:
                    if "torch" in missing_deps:
                        status = HealthStatus.CRITICAL
                        message = f"Critical dependencies missing: {missing_deps}"
                    else:
                        status = HealthStatus.WARNING
                        message = f"Optional dependencies missing: {missing_deps}"
                else:
                    status = HealthStatus.HEALTHY
                    message = "All dependencies available"

                return ComponentHealth(
                    name="dependencies",
                    status=status,
                    message=message,
                    details={
                        "available": available_deps,
                        "missing": missing_deps,
                        "total_checked": len(dependencies),
                    },
                    timestamp=time.time(),
                )
            except Exception as e:
                return ComponentHealth(
                    name="dependencies",
                    status=HealthStatus.CRITICAL,
                    message=f"Failed to check dependencies: {e}",
                    details={"error": str(e)},
                    timestamp=time.time(),
                )

        # Register InvarLock-specific checks
        self.register_check("adapters", check_adapters)
        self.register_check("guards", check_guards)
        self.register_check("dependencies", check_dependencies)


def create_health_endpoint():
    """Create a simple HTTP health endpoint."""
    try:
        import json
        from http.server import BaseHTTPRequestHandler, HTTPServer

        health_checker = InvarLockHealthChecker()

        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    health_summary = health_checker.get_summary()

                    # Set response code based on overall status
                    if health_summary["overall_status"] == "healthy":
                        self.send_response(200)
                    elif health_summary["overall_status"] == "warning":
                        self.send_response(200)  # Still OK, just warnings
                    else:
                        self.send_response(503)  # Service unavailable

                    self.send_header("Content-type", "application/json")
                    self.end_headers()

                    response = json.dumps(health_summary, indent=2)
                    self.wfile.write(response.encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                # Suppress default logging
                pass

        return HTTPServer, HealthHandler
    except ImportError:
        return None, None
