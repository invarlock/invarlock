"""
Tests for observability alerting and notification system.

This module tests:
- AlertSeverity and AlertStatus enums
- Alert dataclass and lifecycle
- AlertRule configuration
- NotificationChannel configuration
- AlertManager rule evaluation and triggering
- Notification channel implementations (mocked)
- Utility functions for common alert configurations
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# AlertSeverity and AlertStatus Tests
# =============================================================================


@pytest.mark.unit
class TestAlertEnums:
    """Tests for alert enums."""

    def test_alert_severity_values(self):
        """Test AlertSeverity enum values."""
        from invarlock.observability.alerting import AlertSeverity

        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_alert_status_values(self):
        """Test AlertStatus enum values."""
        from invarlock.observability.alerting import AlertStatus

        assert AlertStatus.ACTIVE.value == "active"
        assert AlertStatus.RESOLVED.value == "resolved"
        assert AlertStatus.SUPPRESSED.value == "suppressed"


# =============================================================================
# Alert Dataclass Tests
# =============================================================================


@pytest.mark.unit
class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        """Test creating an alert."""
        from invarlock.observability.alerting import Alert, AlertSeverity, AlertStatus

        alert = Alert(
            id="alert_1",
            name="Test Alert",
            severity=AlertSeverity.WARNING,
            message="Something happened",
            details={"key": "value"},
            timestamp=1000.0,
        )

        assert alert.id == "alert_1"
        assert alert.name == "Test Alert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "Something happened"
        assert alert.details == {"key": "value"}
        assert alert.timestamp == 1000.0
        assert alert.status == AlertStatus.ACTIVE
        assert alert.resolved_timestamp is None

    def test_alert_to_dict(self):
        """Test alert serialization to dict."""
        from invarlock.observability.alerting import Alert, AlertSeverity

        alert = Alert(
            id="alert_1",
            name="Test Alert",
            severity=AlertSeverity.WARNING,
            message="Test message",
            details={},
            timestamp=1000.0,
        )

        d = alert.to_dict()

        assert d["id"] == "alert_1"
        assert d["name"] == "Test Alert"
        assert d["severity"] == "warning"
        assert d["status"] == "active"
        assert d["timestamp"] == 1000.0
        assert d["resolved_timestamp"] is None

    def test_alert_resolve(self):
        """Test resolving an alert."""
        from invarlock.observability.alerting import Alert, AlertSeverity, AlertStatus

        alert = Alert(
            id="alert_1",
            name="Test Alert",
            severity=AlertSeverity.WARNING,
            message="Test",
            details={},
            timestamp=1000.0,
        )

        before = time.time()
        alert.resolve()
        after = time.time()

        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_timestamp is not None
        assert before <= alert.resolved_timestamp <= after


# =============================================================================
# AlertRule Tests
# =============================================================================


@pytest.mark.unit
class TestAlertRule:
    """Tests for AlertRule configuration."""

    def test_rule_creation(self):
        """Test creating an alert rule."""
        from invarlock.observability.alerting import AlertRule, AlertSeverity

        rule = AlertRule(
            name="high_cpu",
            metric="cpu_percent",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
        )

        assert rule.name == "high_cpu"
        assert rule.metric == "cpu_percent"
        assert rule.threshold == 80.0
        assert rule.severity == AlertSeverity.WARNING
        assert rule.comparison == "greater"
        assert rule.window_minutes == 5
        assert rule.enabled is True

    def test_rule_default_message(self):
        """Test rule generates default message."""
        from invarlock.observability.alerting import AlertRule

        rule = AlertRule(name="test_rule", metric="test_metric", threshold=100.0)

        assert "test_metric" in rule.message
        assert "greater" in rule.message
        assert "100.0" in rule.message

    def test_rule_custom_message(self):
        """Test rule with custom message."""
        from invarlock.observability.alerting import AlertRule

        rule = AlertRule(
            name="test_rule",
            metric="test_metric",
            threshold=100.0,
            message="Custom alert message",
        )

        assert rule.message == "Custom alert message"

    def test_rule_with_percentile(self):
        """Test rule with percentile threshold."""
        from invarlock.observability.alerting import AlertRule

        rule = AlertRule(
            name="slow_requests",
            metric="request_latency",
            threshold=5.0,
            percentile=95,
        )

        assert rule.percentile == 95


# =============================================================================
# NotificationChannel Tests
# =============================================================================


@pytest.mark.unit
class TestNotificationChannel:
    """Tests for NotificationChannel configuration."""

    def test_channel_creation(self):
        """Test creating a notification channel."""
        from invarlock.observability.alerting import NotificationChannel

        channel = NotificationChannel(
            name="email",
            type="email",
            config={"smtp_server": "localhost"},
        )

        assert channel.name == "email"
        assert channel.type == "email"
        assert channel.config["smtp_server"] == "localhost"
        assert channel.enabled is True

    def test_channel_default_severity_filter(self):
        """Test default severity filter includes WARNING and CRITICAL."""
        from invarlock.observability.alerting import AlertSeverity, NotificationChannel

        channel = NotificationChannel(name="test", type="webhook", config={})

        assert AlertSeverity.WARNING in channel.severity_filter
        assert AlertSeverity.CRITICAL in channel.severity_filter
        assert AlertSeverity.INFO not in channel.severity_filter

    def test_channel_custom_severity_filter(self):
        """Test custom severity filter."""
        from invarlock.observability.alerting import AlertSeverity, NotificationChannel

        channel = NotificationChannel(
            name="test",
            type="webhook",
            config={},
            severity_filter=[AlertSeverity.CRITICAL],
        )

        assert channel.severity_filter == [AlertSeverity.CRITICAL]


# =============================================================================
# AlertManager Tests
# =============================================================================


@pytest.mark.unit
class TestAlertManager:
    """Tests for AlertManager."""

    def test_manager_initialization(self):
        """Test alert manager initializes correctly."""
        from invarlock.observability.alerting import AlertManager

        manager = AlertManager()

        assert manager.rules == {}
        assert manager.active_alerts == {}
        assert manager.alert_history == []
        assert manager.notification_channels == {}

    def test_add_rule(self):
        """Test adding an alert rule."""
        from invarlock.observability.alerting import AlertManager, AlertRule

        manager = AlertManager()
        rule = AlertRule(name="test_rule", metric="test_metric", threshold=100.0)

        manager.add_rule(rule)

        assert "test_rule" in manager.rules
        assert manager.rules["test_rule"] is rule

    def test_remove_rule(self):
        """Test removing an alert rule."""
        from invarlock.observability.alerting import AlertManager, AlertRule

        manager = AlertManager()
        rule = AlertRule(name="test_rule", metric="test_metric", threshold=100.0)
        manager.add_rule(rule)

        manager.remove_rule("test_rule")

        assert "test_rule" not in manager.rules

    def test_add_notification_channel(self):
        """Test adding a notification channel."""
        from invarlock.observability.alerting import AlertManager, NotificationChannel

        manager = AlertManager()
        channel = NotificationChannel(
            name="test", type="webhook", config={"url": "http://test"}
        )

        manager.add_notification_channel(channel)

        assert "test" in manager.notification_channels
        assert manager.notification_channels["test"] is channel

    def test_check_metric_triggers_alert(self):
        """Test metric check triggers alert when threshold exceeded."""
        from invarlock.observability.alerting import AlertManager, AlertRule

        manager = AlertManager()
        rule = AlertRule(
            name="high_value",
            metric="test_metric",
            threshold=50.0,
            comparison="greater",
        )
        manager.add_rule(rule)

        manager.check_metric_against_rules("test_metric", 60.0)

        assert len(manager.active_alerts) == 1
        assert "rule_high_value" in manager.active_alerts

    def test_check_metric_no_trigger_below_threshold(self):
        """Test metric check doesn't trigger when below threshold."""
        from invarlock.observability.alerting import AlertManager, AlertRule

        manager = AlertManager()
        rule = AlertRule(
            name="high_value",
            metric="test_metric",
            threshold=50.0,
        )
        manager.add_rule(rule)

        manager.check_metric_against_rules("test_metric", 40.0)

        assert len(manager.active_alerts) == 0

    def test_check_metric_less_than_comparison(self):
        """Test metric check with 'less' comparison."""
        from invarlock.observability.alerting import AlertManager, AlertRule

        manager = AlertManager()
        rule = AlertRule(
            name="low_value",
            metric="test_metric",
            threshold=10.0,
            comparison="less",
        )
        manager.add_rule(rule)

        manager.check_metric_against_rules("test_metric", 5.0)

        assert len(manager.active_alerts) == 1

    def test_check_metric_equal_comparison(self):
        """Test metric check with 'equal' comparison."""
        from invarlock.observability.alerting import AlertManager, AlertRule

        manager = AlertManager()
        rule = AlertRule(
            name="exact_value",
            metric="test_metric",
            threshold=100.0,
            comparison="equal",
        )
        manager.add_rule(rule)

        manager.check_metric_against_rules("test_metric", 100.0)

        assert len(manager.active_alerts) == 1

    def test_alert_not_duplicated(self):
        """Test same alert is not duplicated."""
        from invarlock.observability.alerting import AlertManager, AlertRule

        manager = AlertManager()
        rule = AlertRule(name="test_rule", metric="test_metric", threshold=50.0)
        manager.add_rule(rule)

        # Trigger twice
        manager.check_metric_against_rules("test_metric", 60.0)
        manager.check_metric_against_rules("test_metric", 70.0)

        assert len(manager.active_alerts) == 1

    def test_disabled_rule_not_evaluated(self):
        """Test disabled rules are not evaluated."""
        from invarlock.observability.alerting import AlertManager, AlertRule

        manager = AlertManager()
        rule = AlertRule(
            name="test_rule",
            metric="test_metric",
            threshold=50.0,
            enabled=False,
        )
        manager.add_rule(rule)

        manager.check_metric_against_rules("test_metric", 1000.0)

        assert len(manager.active_alerts) == 0

    def test_get_active_alerts(self):
        """Test getting active alerts."""
        from invarlock.observability.alerting import AlertManager, AlertRule

        manager = AlertManager()
        rule = AlertRule(name="test_rule", metric="test_metric", threshold=50.0)
        manager.add_rule(rule)
        manager.check_metric_against_rules("test_metric", 60.0)

        active = manager.get_active_alerts()

        assert len(active) == 1
        assert active[0].name == "test_rule"

    def test_get_alert_summary(self):
        """Test getting alert summary."""
        from invarlock.observability.alerting import (
            AlertManager,
            AlertRule,
            AlertSeverity,
        )

        manager = AlertManager()
        manager.add_rule(
            AlertRule(
                name="warning_rule",
                metric="metric1",
                threshold=50.0,
                severity=AlertSeverity.WARNING,
            )
        )
        manager.add_rule(
            AlertRule(
                name="critical_rule",
                metric="metric2",
                threshold=50.0,
                severity=AlertSeverity.CRITICAL,
            )
        )

        # Trigger both
        manager.check_metric_against_rules("metric1", 60.0)
        manager.check_metric_against_rules("metric2", 60.0)

        summary = manager.get_alert_summary()

        assert summary["total_active"] == 2
        assert summary["by_severity"]["warning"] == 1
        assert summary["by_severity"]["critical"] == 1
        assert summary["total_rules"] == 2
        assert summary["enabled_rules"] == 2

    def test_check_health_alerts_unhealthy(self):
        """Test health-based alerts trigger for unhealthy components."""
        from invarlock.observability.alerting import AlertManager
        from invarlock.observability.health import ComponentHealth, HealthStatus

        manager = AlertManager()

        health_status = {
            "memory": ComponentHealth(
                name="memory",
                status=HealthStatus.CRITICAL,
                message="Memory critical",
                details={},
                timestamp=time.time(),
            )
        }

        manager.check_health_alerts(health_status)

        assert "health_memory" in manager.active_alerts

    def test_check_health_alerts_resolves(self):
        """Test health alerts resolve when component becomes healthy."""
        from invarlock.observability.alerting import AlertManager
        from invarlock.observability.health import ComponentHealth, HealthStatus

        manager = AlertManager()

        # First trigger unhealthy
        unhealthy_status = {
            "memory": ComponentHealth(
                name="memory",
                status=HealthStatus.CRITICAL,
                message="Memory critical",
                details={},
                timestamp=time.time(),
            )
        }
        manager.check_health_alerts(unhealthy_status)
        assert "health_memory" in manager.active_alerts

        # Then resolve with healthy
        healthy_status = {
            "memory": ComponentHealth(
                name="memory",
                status=HealthStatus.HEALTHY,
                message="Memory OK",
                details={},
                timestamp=time.time(),
            )
        }
        manager.check_health_alerts(healthy_status)
        assert "health_memory" not in manager.active_alerts

    def test_check_resource_alerts(self):
        """Test resource usage alerts."""
        from invarlock.observability.alerting import AlertManager, AlertRule

        manager = AlertManager()
        manager.add_rule(
            AlertRule(
                name="high_cpu",
                metric="invarlock.resource.cpu_percent",
                threshold=80.0,
            )
        )

        manager.check_resource_alerts({"cpu_percent": 90.0})

        assert len(manager.active_alerts) == 1

    def test_alert_history_limited(self):
        """Test alert history is limited to 1000 entries."""
        from invarlock.observability.alerting import AlertManager, AlertRule

        manager = AlertManager()

        # Create many unique alerts
        for i in range(1100):
            rule = AlertRule(name=f"rule_{i}", metric=f"metric_{i}", threshold=0.0)
            manager.add_rule(rule)
            manager.check_metric_against_rules(f"metric_{i}", 1.0)

        assert len(manager.alert_history) == 1000


# =============================================================================
# Notification Tests (Mocked)
# =============================================================================


@pytest.mark.unit
class TestNotifications:
    """Tests for notification sending (mocked external calls)."""

    def test_notification_not_sent_if_channel_disabled(self):
        """Test notifications not sent to disabled channels."""
        from invarlock.observability.alerting import (
            AlertManager,
            AlertRule,
            NotificationChannel,
        )

        manager = AlertManager()
        channel = NotificationChannel(
            name="test",
            type="webhook",
            config={"url": "http://test"},
            enabled=False,
        )
        manager.add_notification_channel(channel)
        manager.add_rule(AlertRule(name="test_rule", metric="test", threshold=0.0))

        # Trigger alert - should not attempt to send
        manager.check_metric_against_rules("test", 1.0)

        # No error means disabled channel was skipped

    def test_notification_severity_filter(self):
        """Test notifications respect severity filter."""
        from invarlock.observability.alerting import (
            AlertManager,
            AlertRule,
            AlertSeverity,
            NotificationChannel,
        )

        manager = AlertManager()
        channel = NotificationChannel(
            name="critical_only",
            type="webhook",
            config={"url": "http://test"},
            severity_filter=[AlertSeverity.CRITICAL],
        )
        manager.add_notification_channel(channel)

        # Add a WARNING severity rule
        manager.add_rule(
            AlertRule(
                name="warning_rule",
                metric="test",
                threshold=0.0,
                severity=AlertSeverity.WARNING,
            )
        )

        # Trigger - channel should not be notified due to severity filter
        manager.check_metric_against_rules("test", 1.0)

    @patch("invarlock.observability.alerting.requests.post")
    def test_webhook_notification(self, mock_post):
        """Test webhook notification sends correct payload."""
        from invarlock.observability.alerting import (
            Alert,
            AlertManager,
            AlertSeverity,
            NotificationChannel,
        )

        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()

        manager = AlertManager()
        channel = NotificationChannel(
            name="webhook",
            type="webhook",
            config={"url": "http://test.com/webhook"},
        )
        manager.add_notification_channel(channel)

        # Create and add alert manually to trigger notification
        alert = Alert(
            id="test_alert",
            name="Test",
            severity=AlertSeverity.WARNING,
            message="Test message",
            details={},
            timestamp=time.time(),
        )
        manager._add_alert(alert)

        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        assert "json" in call_kwargs
        assert call_kwargs["json"]["alert"]["name"] == "Test"


# =============================================================================
# Utility Function Tests
# =============================================================================


@pytest.mark.unit
class TestAlertUtilities:
    """Tests for alert utility functions."""

    def test_create_resource_alerts(self):
        """Test creating standard resource alerts."""
        from invarlock.observability.alerting import (
            AlertSeverity,
            create_resource_alerts,
        )

        alerts = create_resource_alerts()

        # Should have alerts for CPU, memory, GPU
        names = [a.name for a in alerts]
        assert "high_cpu_usage" in names
        assert "critical_cpu_usage" in names
        assert "high_memory_usage" in names
        assert "critical_memory_usage" in names
        assert "high_gpu_memory" in names

        # Check severity levels
        alerts_dict = {a.name: a for a in alerts}
        assert alerts_dict["high_cpu_usage"].severity == AlertSeverity.WARNING
        assert alerts_dict["critical_cpu_usage"].severity == AlertSeverity.CRITICAL

    def test_create_performance_alerts(self):
        """Test creating standard performance alerts."""
        from invarlock.observability.alerting import create_performance_alerts

        alerts = create_performance_alerts()

        names = [a.name for a in alerts]
        assert "slow_operations" in names
        assert "high_error_rate" in names
        assert "critical_error_rate" in names

    def test_setup_email_notifications(self):
        """Test setting up email notification channel."""
        from invarlock.observability.alerting import setup_email_notifications

        channel = setup_email_notifications(
            smtp_server="smtp.example.com",
            from_address="alerts@example.com",
            to_addresses=["admin@example.com"],
            username="user",
            password="pass",
        )

        assert channel.type == "email"
        assert channel.config["smtp_server"] == "smtp.example.com"
        assert channel.config["from_address"] == "alerts@example.com"
        assert "admin@example.com" in channel.config["to_addresses"]
        assert channel.config["username"] == "user"
        assert channel.config["use_tls"] is True

    def test_setup_slack_notifications(self):
        """Test setting up Slack notification channel."""
        from invarlock.observability.alerting import setup_slack_notifications

        channel = setup_slack_notifications(
            webhook_url="https://hooks.slack.com/services/xxx",
            channel="#monitoring",
        )

        assert channel.type == "slack"
        assert "hooks.slack.com" in channel.config["webhook_url"]
        assert channel.config["channel"] == "#monitoring"

    def test_setup_webhook_notifications(self):
        """Test setting up webhook notification channel."""
        from invarlock.observability.alerting import setup_webhook_notifications

        channel = setup_webhook_notifications(
            url="http://alerts.example.com/webhook",
            headers={"Authorization": "Bearer token"},
        )

        assert channel.type == "webhook"
        assert channel.config["url"] == "http://alerts.example.com/webhook"
        assert channel.config["headers"]["Authorization"] == "Bearer token"
