"""
Alerting and notification system.
"""

import logging
import smtplib
import time
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

import requests


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Represents an alert."""

    id: str
    name: str
    severity: AlertSeverity
    message: str
    details: dict[str, Any]
    timestamp: float
    status: AlertStatus = AlertStatus.ACTIVE
    resolved_timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "resolved_timestamp": self.resolved_timestamp,
        }

    def resolve(self):
        """Mark alert as resolved."""
        self.status = AlertStatus.RESOLVED
        self.resolved_timestamp = time.time()


@dataclass
class AlertRule:
    """Defines conditions for triggering alerts."""

    name: str
    metric: str
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    comparison: str = "greater"  # greater, less, equal
    window_minutes: int = 5
    percentile: float | None = None
    message: str = ""
    enabled: bool = True

    def __post_init__(self):
        if not self.message:
            self.message = f"{self.metric} {self.comparison} {self.threshold}"


@dataclass
class NotificationChannel:
    """Configuration for notification channels."""

    name: str
    type: str  # email, webhook, slack, etc.
    config: dict[str, Any]
    enabled: bool = True
    severity_filter: list[AlertSeverity] = field(
        default_factory=lambda: [AlertSeverity.WARNING, AlertSeverity.CRITICAL]
    )


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.notification_channels: dict[str, NotificationChannel] = {}

        # Alert tracking
        self.last_check: dict[str, float] = {}
        self.alert_counts: dict[str, int] = {}

    def add_rule(self, rule: AlertRule):
        """Add an alerting rule."""
        self.rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_name: str):
        """Remove an alerting rule."""
        self.rules.pop(rule_name, None)
        self.logger.info(f"Removed alert rule: {rule_name}")

    def add_notification_channel(self, channel: NotificationChannel):
        """Add a notification channel."""
        self.notification_channels[channel.name] = channel
        self.logger.info(f"Added notification channel: {channel.name} ({channel.type})")

    def check_metric_against_rules(self, metric_name: str, value: float, **context):
        """Check a metric value against all applicable rules."""
        for rule in self.rules.values():
            if not rule.enabled or rule.metric != metric_name:
                continue

            if self._evaluate_rule(rule, value, **context):
                self._trigger_alert(rule, value, **context)

    def check_error_alerts(
        self, error_type: str, error_msg: str, context: dict[str, Any]
    ):
        """Check for error-based alerts."""
        # Count errors by type
        error_key = f"error_{error_type}"
        self.alert_counts[error_key] = self.alert_counts.get(error_key, 0) + 1

        # Check error rate rules
        for rule in self.rules.values():
            if rule.metric.endswith("errors.total") and rule.enabled:
                error_count = self.alert_counts.get(error_key, 0)
                if self._evaluate_rule(rule, error_count, error_type=error_type):
                    self._trigger_alert(
                        rule,
                        error_count,
                        error_type=error_type,
                        error_message=error_msg,
                        **context,
                    )

    def check_health_alerts(self, health_status: dict[str, Any]):
        """Check for health-based alerts."""
        for component, health in health_status.items():
            if not health.healthy:
                alert_id = f"health_{component}"

                if alert_id not in self.active_alerts:
                    alert = Alert(
                        id=alert_id,
                        name=f"Component Health: {component}",
                        severity=AlertSeverity.CRITICAL
                        if health.status.value == "critical"
                        else AlertSeverity.WARNING,
                        message=f"{component} health check failed: {health.message}",
                        details={
                            "component": component,
                            "health_status": health.status.value,
                            "health_details": health.details,
                        },
                        timestamp=time.time(),
                    )

                    self._add_alert(alert)
            else:
                # Resolve health alert if component is now healthy
                alert_id = f"health_{component}"
                if alert_id in self.active_alerts:
                    self.active_alerts[alert_id].resolve()
                    self._remove_alert(alert_id)

    def check_resource_alerts(self, resource_usage: dict[str, float]):
        """Check for resource usage alerts."""
        for rule in self.rules.values():
            if not rule.enabled or not rule.metric.startswith("invarlock.resource"):
                continue

            # Extract resource type from metric name
            resource_type = rule.metric.replace("invarlock.resource.", "")
            if resource_type in resource_usage:
                value = resource_usage[resource_type]
                if self._evaluate_rule(rule, value):
                    self._trigger_alert(rule, value, resource_usage=resource_usage)

    def get_active_alerts(self) -> list[Alert]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())

    def get_alert_summary(self) -> dict[str, Any]:
        """Get alert summary statistics."""
        active_by_severity = {severity.value: 0 for severity in AlertSeverity}
        for alert in self.active_alerts.values():
            active_by_severity[alert.severity.value] += 1

        return {
            "total_active": len(self.active_alerts),
            "by_severity": active_by_severity,
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "notification_channels": len(self.notification_channels),
            "recent_alerts": len(
                [a for a in self.alert_history if time.time() - a.timestamp < 3600]
            ),
        }

    def _evaluate_rule(self, rule: AlertRule, value: float, **context) -> bool:
        """Evaluate if a rule should trigger."""
        if rule.comparison == "greater":
            return value > rule.threshold
        elif rule.comparison == "less":
            return value < rule.threshold
        elif rule.comparison == "equal":
            return abs(value - rule.threshold) < 0.001
        else:
            return False

    def _trigger_alert(self, rule: AlertRule, value: float, **context):
        """Trigger an alert."""
        alert_id = f"rule_{rule.name}"

        # Check if alert is already active
        if alert_id in self.active_alerts:
            return

        alert = Alert(
            id=alert_id,
            name=rule.name,
            severity=rule.severity,
            message=f"{rule.message} (current: {value})",
            details={
                "rule": rule.name,
                "metric": rule.metric,
                "threshold": rule.threshold,
                "current_value": value,
                "context": context,
            },
            timestamp=time.time(),
        )

        self._add_alert(alert)

    def _add_alert(self, alert: Alert):
        """Add an alert and send notifications."""
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)

        # Keep history limited
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

        self.logger.warning(f"Alert triggered: {alert.name} - {alert.message}")

        # Send notifications
        self._send_notifications(alert)

    def _remove_alert(self, alert_id: str):
        """Remove resolved alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            self.logger.info(f"Alert resolved: {alert.name}")

    def _send_notifications(self, alert: Alert):
        """Send alert notifications to configured channels."""
        for channel in self.notification_channels.values():
            if not channel.enabled:
                continue

            if alert.severity not in channel.severity_filter:
                continue

            try:
                if channel.type == "email":
                    self._send_email_notification(alert, channel)
                elif channel.type == "webhook":
                    self._send_webhook_notification(alert, channel)
                elif channel.type == "slack":
                    self._send_slack_notification(alert, channel)
                else:
                    self.logger.warning(
                        f"Unknown notification channel type: {channel.type}"
                    )
            except Exception as e:
                self.logger.error(
                    f"Failed to send notification via {channel.name}: {e}"
                )

    def _send_email_notification(self, alert: Alert, channel: NotificationChannel):
        """Send email notification."""
        config = channel.config

        # Create message
        msg = MIMEMultipart()
        msg["From"] = config["from_address"]
        msg["To"] = ", ".join(config["to_addresses"])
        msg["Subject"] = (
            f"InvarLock Alert: {alert.name} [{alert.severity.value.upper()}]"
        )

        # Email body
        body = f"""
InvarLock Alert Notification

Alert: {alert.name}
Severity: {alert.severity.value.upper()}
Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alert.timestamp))}

Message: {alert.message}

Details:
{self._format_alert_details(alert.details)}
        """

        msg.attach(MIMEText(body, "plain"))

        # Send email
        with smtplib.SMTP(
            config["smtp_server"], config.get("smtp_port", 587)
        ) as server:
            if config.get("use_tls", True):
                server.starttls()
            if "username" in config and "password" in config:
                server.login(config["username"], config["password"])
            server.send_message(msg)

    def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel):
        """Send webhook notification."""
        config = channel.config

        payload = {
            "alert": alert.to_dict(),
            "timestamp": time.time(),
            "source": "invarlock-monitoring",
        }

        headers = config.get("headers", {})
        headers.setdefault("Content-Type", "application/json")

        response = requests.post(
            config["url"],
            json=payload,
            headers=headers,
            timeout=config.get("timeout", 10),
        )
        response.raise_for_status()

    def _send_slack_notification(self, alert: Alert, channel: NotificationChannel):
        """Send Slack notification."""
        config = channel.config

        # Determine color based on severity
        color_map = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.CRITICAL: "danger",
        }

        payload = {
            "channel": config.get("channel", "#alerts"),
            "username": config.get("username", "InvarLock Monitor"),
            "icon_emoji": config.get("icon_emoji", ":warning:"),
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "warning"),
                    "title": f"InvarLock Alert: {alert.name}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True,
                        },
                        {
                            "title": "Time",
                            "value": time.strftime(
                                "%Y-%m-%d %H:%M:%S", time.localtime(alert.timestamp)
                            ),
                            "short": True,
                        },
                    ],
                    "footer": "InvarLock Monitoring",
                    "ts": int(alert.timestamp),
                }
            ],
        }

        response = requests.post(
            config["webhook_url"], json=payload, timeout=config.get("timeout", 10)
        )
        response.raise_for_status()

    def _format_alert_details(self, details: dict[str, Any]) -> str:
        """Format alert details for display."""
        lines = []
        for key, value in details.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  {sub_key}: {sub_value}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)


# Utility functions for common alert configurations
def create_resource_alerts() -> list[AlertRule]:
    """Create standard resource monitoring alerts."""
    return [
        AlertRule(
            name="high_cpu_usage",
            metric="invarlock.resource.cpu_percent",
            threshold=85.0,
            severity=AlertSeverity.WARNING,
            message="High CPU usage detected",
        ),
        AlertRule(
            name="critical_cpu_usage",
            metric="invarlock.resource.cpu_percent",
            threshold=95.0,
            severity=AlertSeverity.CRITICAL,
            message="Critical CPU usage detected",
        ),
        AlertRule(
            name="high_memory_usage",
            metric="invarlock.resource.memory_percent",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            message="High memory usage detected",
        ),
        AlertRule(
            name="critical_memory_usage",
            metric="invarlock.resource.memory_percent",
            threshold=90.0,
            severity=AlertSeverity.CRITICAL,
            message="Critical memory usage detected",
        ),
        AlertRule(
            name="high_gpu_memory",
            metric="invarlock.resource.gpu_memory_percent",
            threshold=85.0,
            severity=AlertSeverity.WARNING,
            message="High GPU memory usage detected",
        ),
    ]


def create_performance_alerts() -> list[AlertRule]:
    """Create standard performance monitoring alerts."""
    return [
        AlertRule(
            name="slow_operations",
            metric="invarlock.operation.duration",
            threshold=30.0,
            percentile=95,
            severity=AlertSeverity.WARNING,
            message="Slow operations detected (P95 > 30s)",
        ),
        AlertRule(
            name="high_error_rate",
            metric="invarlock.errors.total",
            threshold=10,
            window_minutes=5,
            severity=AlertSeverity.WARNING,
            message="High error rate detected",
        ),
        AlertRule(
            name="critical_error_rate",
            metric="invarlock.errors.total",
            threshold=50,
            window_minutes=5,
            severity=AlertSeverity.CRITICAL,
            message="Critical error rate detected",
        ),
    ]


def setup_email_notifications(
    smtp_server: str,
    from_address: str,
    to_addresses: list[str],
    username: str | None = None,
    password: str | None = None,
) -> NotificationChannel:
    """Setup email notification channel."""
    config = {
        "smtp_server": smtp_server,
        "from_address": from_address,
        "to_addresses": to_addresses,
        "use_tls": True,
    }

    if username and password:
        config.update({"username": username, "password": password})

    return NotificationChannel(name="email", type="email", config=config)


def setup_slack_notifications(
    webhook_url: str, channel: str = "#alerts"
) -> NotificationChannel:
    """Setup Slack notification channel."""
    return NotificationChannel(
        name="slack",
        type="slack",
        config={
            "webhook_url": webhook_url,
            "channel": channel,
            "username": "InvarLock Monitor",
            "icon_emoji": ":warning:",
        },
    )


def setup_webhook_notifications(
    url: str, headers: dict[str, str] | None = None
) -> NotificationChannel:
    """Setup webhook notification channel."""
    return NotificationChannel(
        name="webhook",
        type="webhook",
        config={"url": url, "headers": headers or {}, "timeout": 10},
    )
