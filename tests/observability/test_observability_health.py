"""
Tests for observability health checking system.

This module tests:
- HealthStatus enum
- ComponentHealth dataclass
- HealthChecker registration and execution
- Default system health checks
- InvarLockHealthChecker with additional checks
- Health endpoint creation
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# HealthStatus Tests
# =============================================================================


@pytest.mark.unit
class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_status_values(self):
        """Test HealthStatus enum values."""
        from invarlock.observability.health import HealthStatus

        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.CRITICAL.value == "critical"
        assert HealthStatus.UNKNOWN.value == "unknown"


# =============================================================================
# ComponentHealth Tests
# =============================================================================


@pytest.mark.unit
class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_component_health_creation(self):
        """Test creating component health."""
        from invarlock.observability.health import ComponentHealth, HealthStatus

        health = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            message="All good",
            details={"key": "value"},
            timestamp=1000.0,
        )

        assert health.name == "test_component"
        assert health.status == HealthStatus.HEALTHY
        assert health.message == "All good"
        assert health.details == {"key": "value"}
        assert health.timestamp == 1000.0

    def test_healthy_property_true(self):
        """Test healthy property returns True for HEALTHY status."""
        from invarlock.observability.health import ComponentHealth, HealthStatus

        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            message="OK",
            details={},
            timestamp=time.time(),
        )

        assert health.healthy is True

    def test_healthy_property_false_for_warning(self):
        """Test healthy property returns False for WARNING status."""
        from invarlock.observability.health import ComponentHealth, HealthStatus

        health = ComponentHealth(
            name="test",
            status=HealthStatus.WARNING,
            message="Warning",
            details={},
            timestamp=time.time(),
        )

        assert health.healthy is False

    def test_healthy_property_false_for_critical(self):
        """Test healthy property returns False for CRITICAL status."""
        from invarlock.observability.health import ComponentHealth, HealthStatus

        health = ComponentHealth(
            name="test",
            status=HealthStatus.CRITICAL,
            message="Critical",
            details={},
            timestamp=time.time(),
        )

        assert health.healthy is False

    def test_to_dict(self):
        """Test serialization to dict."""
        from invarlock.observability.health import ComponentHealth, HealthStatus

        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            message="OK",
            details={"cpu_percent": 50.0},
            timestamp=1000.0,
        )

        d = health.to_dict()

        assert d["name"] == "test"
        assert d["status"] == "healthy"
        assert d["message"] == "OK"
        assert d["details"]["cpu_percent"] == 50.0
        assert d["timestamp"] == 1000.0
        assert d["healthy"] is True


# =============================================================================
# HealthChecker Tests
# =============================================================================


@pytest.mark.unit
class TestHealthChecker:
    """Tests for HealthChecker."""

    def test_initialization(self):
        """Test health checker initializes with default checks."""
        from invarlock.observability.health import HealthChecker

        checker = HealthChecker()

        # Should have default checks registered
        assert "memory" in checker.health_checks
        assert "cpu" in checker.health_checks
        assert "disk" in checker.health_checks
        assert "gpu" in checker.health_checks
        assert "pytorch" in checker.health_checks

    def test_register_check(self):
        """Test registering a custom health check."""
        from invarlock.observability.health import (
            ComponentHealth,
            HealthChecker,
            HealthStatus,
        )

        checker = HealthChecker()

        def custom_check():
            return ComponentHealth(
                name="custom",
                status=HealthStatus.HEALTHY,
                message="Custom check passed",
                details={},
                timestamp=time.time(),
            )

        checker.register_check("custom", custom_check)

        assert "custom" in checker.health_checks

    def test_check_component(self):
        """Test checking a specific component."""
        from invarlock.observability.health import (
            ComponentHealth,
            HealthChecker,
            HealthStatus,
        )

        checker = HealthChecker()

        def test_check():
            return ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                message="Test OK",
                details={"value": 42},
                timestamp=time.time(),
            )

        checker.register_check("test", test_check)

        result = checker.check_component("test")

        assert result.name == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.details["value"] == 42

    def test_check_unknown_component(self):
        """Test checking unknown component returns UNKNOWN status."""
        from invarlock.observability.health import HealthChecker, HealthStatus

        checker = HealthChecker()

        result = checker.check_component("nonexistent")

        assert result.status == HealthStatus.UNKNOWN
        assert "No health check registered" in result.message

    def test_check_component_exception_handling(self):
        """Test exception during check is caught and returns CRITICAL."""
        from invarlock.observability.health import HealthChecker, HealthStatus

        checker = HealthChecker()

        def failing_check():
            raise RuntimeError("Check failed!")

        checker.register_check("failing", failing_check)

        result = checker.check_component("failing")

        assert result.status == HealthStatus.CRITICAL
        assert "Health check failed" in result.message
        assert "RuntimeError" in str(
            result.details.get("error", "")
        ) or "Check failed" in str(result.details.get("error", ""))

    def test_check_all(self):
        """Test checking all components."""
        from invarlock.observability.health import HealthChecker

        checker = HealthChecker()

        results = checker.check_all()

        # Should have results for all default checks
        assert "memory" in results
        assert "cpu" in results
        assert "disk" in results
        assert "gpu" in results
        assert "pytorch" in results

    def test_get_overall_status_healthy(self):
        """Test overall status is HEALTHY when all components are healthy."""
        from invarlock.observability.health import (
            ComponentHealth,
            HealthChecker,
            HealthStatus,
        )

        checker = HealthChecker()
        checker.health_checks.clear()  # Remove defaults

        def healthy_check():
            return ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK",
                details={},
                timestamp=time.time(),
            )

        checker.register_check("test1", healthy_check)
        checker.register_check("test2", healthy_check)

        checker.check_all()
        status = checker.get_overall_status()

        assert status == HealthStatus.HEALTHY

    def test_get_overall_status_critical(self):
        """Test overall status is CRITICAL when any component is critical."""
        from invarlock.observability.health import (
            ComponentHealth,
            HealthChecker,
            HealthStatus,
        )

        checker = HealthChecker()
        checker.health_checks.clear()

        def healthy_check():
            return ComponentHealth(
                name="healthy",
                status=HealthStatus.HEALTHY,
                message="OK",
                details={},
                timestamp=time.time(),
            )

        def critical_check():
            return ComponentHealth(
                name="critical",
                status=HealthStatus.CRITICAL,
                message="Critical!",
                details={},
                timestamp=time.time(),
            )

        checker.register_check("healthy", healthy_check)
        checker.register_check("critical", critical_check)

        checker.check_all()
        status = checker.get_overall_status()

        assert status == HealthStatus.CRITICAL

    def test_get_overall_status_warning(self):
        """Test overall status is WARNING when no critical but has warnings."""
        from invarlock.observability.health import (
            ComponentHealth,
            HealthChecker,
            HealthStatus,
        )

        checker = HealthChecker()
        checker.health_checks.clear()

        def healthy_check():
            return ComponentHealth(
                name="healthy",
                status=HealthStatus.HEALTHY,
                message="OK",
                details={},
                timestamp=time.time(),
            )

        def warning_check():
            return ComponentHealth(
                name="warning",
                status=HealthStatus.WARNING,
                message="Warning!",
                details={},
                timestamp=time.time(),
            )

        checker.register_check("healthy", healthy_check)
        checker.register_check("warning", warning_check)

        checker.check_all()
        status = checker.get_overall_status()

        assert status == HealthStatus.WARNING

    def test_get_overall_status_unknown_no_results(self):
        """Test overall status is UNKNOWN when no checks have run."""
        from invarlock.observability.health import HealthChecker, HealthStatus

        checker = HealthChecker()
        checker.health_checks.clear()
        checker.last_results.clear()

        status = checker.get_overall_status()

        assert status == HealthStatus.UNKNOWN

    def test_get_summary(self):
        """Test getting health summary."""
        from invarlock.observability.health import (
            ComponentHealth,
            HealthChecker,
            HealthStatus,
        )

        checker = HealthChecker()
        checker.health_checks.clear()

        def healthy_check():
            return ComponentHealth(
                name="healthy",
                status=HealthStatus.HEALTHY,
                message="OK",
                details={},
                timestamp=time.time(),
            )

        def warning_check():
            return ComponentHealth(
                name="warning",
                status=HealthStatus.WARNING,
                message="Warning!",
                details={},
                timestamp=time.time(),
            )

        checker.register_check("healthy", healthy_check)
        checker.register_check("warning", warning_check)

        checker.check_all()
        summary = checker.get_summary()

        assert summary["overall_status"] == "warning"
        assert summary["total_components"] == 2
        assert summary["status_counts"]["healthy"] == 1
        assert summary["status_counts"]["warning"] == 1
        assert "healthy" in summary["components"]
        assert "warning" in summary["components"]


# =============================================================================
# Default Health Check Tests
# =============================================================================


@pytest.mark.unit
class TestDefaultHealthChecks:
    """Tests for default system health checks."""

    def test_memory_check_returns_component_health(self):
        """Test memory check returns valid ComponentHealth."""
        from invarlock.observability.health import ComponentHealth, HealthChecker

        checker = HealthChecker()
        result = checker.check_component("memory")

        assert isinstance(result, ComponentHealth)
        assert result.name == "memory"
        assert "percent" in result.details

    def test_cpu_check_returns_component_health(self):
        """Test CPU check returns valid ComponentHealth."""
        from invarlock.observability.health import ComponentHealth, HealthChecker

        checker = HealthChecker()
        result = checker.check_component("cpu")

        assert isinstance(result, ComponentHealth)
        assert result.name == "cpu"
        assert "percent" in result.details

    def test_disk_check_returns_component_health(self):
        """Test disk check returns valid ComponentHealth."""
        from invarlock.observability.health import ComponentHealth, HealthChecker

        checker = HealthChecker()
        result = checker.check_component("disk")

        assert isinstance(result, ComponentHealth)
        assert result.name == "disk"
        assert "percent" in result.details

    def test_gpu_check_returns_component_health(self):
        """Test GPU check returns valid ComponentHealth."""
        from invarlock.observability.health import ComponentHealth, HealthChecker

        checker = HealthChecker()
        result = checker.check_component("gpu")

        assert isinstance(result, ComponentHealth)
        assert result.name == "gpu"
        # GPU might not be available, but should still return valid health

    def test_pytorch_check_returns_component_health(self):
        """Test PyTorch check returns valid ComponentHealth."""
        from invarlock.observability.health import ComponentHealth, HealthChecker

        checker = HealthChecker()
        result = checker.check_component("pytorch")

        assert isinstance(result, ComponentHealth)
        assert result.name == "pytorch"
        assert "version" in result.details


# =============================================================================
# InvarLockHealthChecker Tests
# =============================================================================


@pytest.mark.unit
class TestInvarLockHealthChecker:
    """Tests for InvarLock-specific health checker."""

    def test_initialization(self):
        """Test InvarLock health checker has additional checks."""
        from invarlock.observability.health import InvarLockHealthChecker

        checker = InvarLockHealthChecker()

        # Should have base checks plus InvarLock-specific
        assert "adapters" in checker.health_checks
        assert "guards" in checker.health_checks
        assert "dependencies" in checker.health_checks

    def test_adapters_check(self):
        """Test adapters health check."""
        from invarlock.observability.health import (
            ComponentHealth,
            InvarLockHealthChecker,
        )

        checker = InvarLockHealthChecker()
        result = checker.check_component("adapters")

        assert isinstance(result, ComponentHealth)
        assert result.name == "adapters"
        assert "available" in result.details
        assert "total_adapters" in result.details

    def test_guards_check(self):
        """Test guards health check."""
        from invarlock.observability.health import (
            ComponentHealth,
            InvarLockHealthChecker,
        )

        checker = InvarLockHealthChecker()
        result = checker.check_component("guards")

        assert isinstance(result, ComponentHealth)
        assert result.name == "guards"
        assert "available" in result.details
        assert "total_guards" in result.details

    def test_dependencies_check(self):
        """Test dependencies health check."""
        from invarlock.observability.health import (
            ComponentHealth,
            HealthStatus,
            InvarLockHealthChecker,
        )

        checker = InvarLockHealthChecker()
        result = checker.check_component("dependencies")

        assert isinstance(result, ComponentHealth)
        assert result.name == "dependencies"
        assert "available" in result.details
        assert "torch" in result.details["available"]
        assert "numpy" in result.details["available"]
        # Should be healthy since we're running in a valid env
        assert result.status == HealthStatus.HEALTHY


# =============================================================================
# Health Endpoint Tests
# =============================================================================


@pytest.mark.unit
class TestHealthEndpoint:
    """Tests for health HTTP endpoint creation."""

    def test_create_health_endpoint_returns_server_classes(self):
        """Test health endpoint creation returns server classes."""
        from invarlock.observability.health import create_health_endpoint

        HTTPServer, HealthHandler = create_health_endpoint()

        # Should return server and handler classes
        assert HTTPServer is not None
        assert HealthHandler is not None
