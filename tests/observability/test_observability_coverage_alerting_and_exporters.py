from __future__ import annotations

import builtins
import sys
import types

import pytest


class _OneShotEvent:
    def __init__(self) -> None:
        self._set = False

    def is_set(self) -> bool:
        return self._set

    def wait(self, timeout: float) -> bool:
        self._set = True
        return True

    def set(self) -> None:
        self._set = True


class _MemoryInfo:
    def __init__(self, percent: float, available: int, used: int, total: int) -> None:
        self.percent = percent
        self.available = available
        self.used = used
        self.total = total


class _DiskInfo:
    def __init__(self, used: int, total: int, free: int | None = None) -> None:
        self.used = used
        self.total = total
        self.free = total - used if free is None else free


class _Response:
    def __init__(self, status_code: int = 200, text: str = "ok") -> None:
        self.status_code = status_code
        self.text = text
        self.raise_calls = 0

    def raise_for_status(self) -> None:
        self.raise_calls += 1
        if self.status_code >= 400:
            raise OSError(self.text)


class _SMTPRecorder:
    started_tls = False
    logged_in: tuple[str, str] | None = None
    sent_subjects: list[str] = []

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def __enter__(self) -> _SMTPRecorder:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def starttls(self) -> None:
        type(self).started_tls = True

    def login(self, username: str, password: str) -> None:
        type(self).logged_in = (username, password)

    def send_message(self, message) -> None:
        type(self).sent_subjects.append(message["Subject"])


def _make_alert(alerting_module, *, severity=None):
    severity = severity or alerting_module.AlertSeverity.WARNING
    return alerting_module.Alert(
        id="alert-1",
        name="Edge Alert",
        severity=severity,
        message="edge-case",
        details={"outer": {"inner": "value"}, "plain": 1},
        timestamp=1_700_000_000.0,
    )


def _fake_import_without(missing_name: str):
    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == missing_name:
            raise ImportError(missing_name)
        return real_import(name, globals, locals, fromlist, level)

    return _import


@pytest.mark.unit
def test_alerting_error_resource_and_notification_edges(monkeypatch, caplog):
    import invarlock.observability.alerting as alerting

    manager = alerting.AlertManager()
    captured: list[tuple[str, float, dict[str, object]]] = []
    monkeypatch.setattr(
        manager,
        "_trigger_alert",
        lambda rule, value, **context: captured.append((rule.name, value, context)),
    )

    manager.add_rule(
        alerting.AlertRule(
            name="error-equal",
            metric="invarlock.errors.total",
            threshold=1,
            comparison="equal",
        )
    )
    manager.add_rule(
        alerting.AlertRule(
            name="resource-cpu",
            metric="invarlock.resource.cpu_percent",
            threshold=50.0,
        )
    )
    manager.add_rule(
        alerting.AlertRule(
            name="resource-disabled",
            metric="invarlock.resource.memory_percent",
            threshold=10.0,
            enabled=False,
        )
    )

    manager.check_error_alerts("runtime", "boom", {"phase": "collect"})
    manager.check_resource_alerts({"cpu_percent": 90.0})

    assert (
        "error-equal",
        1,
        {"error_type": "runtime", "error_message": "boom", "phase": "collect"},
    ) in captured
    assert any(name == "resource-cpu" and value == 90.0 for name, value, _ in captured)
    assert (
        manager._evaluate_rule(
            alerting.AlertRule(
                name="unsupported",
                metric="metric",
                threshold=1.0,
                comparison="not-a-real-op",
            ),
            5.0,
        )
        is False
    )

    posted: list[tuple[str, dict[str, object]]] = []

    def fake_post(url: str, **kwargs):
        posted.append((url, kwargs))
        return _Response()

    monkeypatch.setattr(alerting.requests, "post", fake_post)
    monkeypatch.setattr(alerting.smtplib, "SMTP", _SMTPRecorder)

    manager = alerting.AlertManager()
    manager.add_notification_channel(
        alerting.NotificationChannel(
            name="email",
            type="email",
            config={
                "smtp_server": "smtp.example.com",
                "smtp_port": 2525,
                "from_address": "alerts@example.com",
                "to_addresses": ["ops@example.com"],
                "username": "user",
                "password": "pass",
            },
        )
    )
    manager.add_notification_channel(
        alerting.NotificationChannel(
            name="slack",
            type="slack",
            config={"webhook_url": "https://hooks.slack.test/1"},
        )
    )
    manager.add_notification_channel(
        alerting.NotificationChannel(
            name="pager",
            type="pagerduty",
            config={},
        )
    )

    with caplog.at_level("WARNING"):
        manager._send_notifications(_make_alert(alerting))

    assert _SMTPRecorder.started_tls is True
    assert _SMTPRecorder.logged_in == ("user", "pass")
    assert _SMTPRecorder.sent_subjects
    assert any(url == "https://hooks.slack.test/1" for url, _ in posted)
    assert "Unknown notification channel type" in caplog.text
    assert "outer:\n  inner: value" in manager._format_alert_details(
        {"outer": {"inner": "value"}, "plain": 1}
    )

    channel = alerting.setup_email_notifications(
        smtp_server="smtp.example.com",
        from_address="alerts@example.com",
        to_addresses=["ops@example.com"],
    )
    assert "username" not in channel.config


@pytest.mark.unit
def test_core_monitoring_loops_export_and_gpu_thresholds(monkeypatch, caplog):
    import invarlock.observability.core as core
    from invarlock.observability.metrics import MetricsRegistry

    manager = core.MonitoringManager()

    manager._stop_event = _OneShotEvent()
    monkeypatch.setattr(
        manager.resource_monitor,
        "update_metrics",
        lambda: (_ for _ in ()).throw(OSError("metrics failed")),
    )
    with caplog.at_level("ERROR"):
        manager._metrics_collection_loop()
    assert "Error in metrics collection" in caplog.text

    manager._stop_event = _OneShotEvent()
    monkeypatch.setattr(
        manager.health_checker,
        "check_all",
        lambda: (_ for _ in ()).throw(OSError("health failed")),
    )
    with caplog.at_level("ERROR"):
        manager._health_check_loop()
    assert "Error in health checking" in caplog.text

    manager._stop_event = _OneShotEvent()
    monkeypatch.setattr(
        manager.resource_monitor,
        "collect_usage",
        lambda: (_ for _ in ()).throw(OSError("resource failed")),
    )
    with caplog.at_level("ERROR"):
        manager._resource_monitoring_loop()
    assert "Error in resource monitoring" in caplog.text

    class BrokenJSONExporter:
        def __init__(self, path: str) -> None:
            self.path = path

        def export(self, metrics) -> None:
            raise OSError("disk full")

    import invarlock.observability.exporters as exporters

    monkeypatch.setattr(exporters, "JSONExporter", BrokenJSONExporter)
    with caplog.at_level("ERROR"):
        manager._export_metrics()
    assert "Error exporting metrics" in caplog.text

    registry = MetricsRegistry()
    perf = core.PerformanceMonitor(registry)
    perf.record_operation("verify", 1.0)
    perf.record_operation("verify", 2.0)
    perf.update_metrics()
    assert (
        registry.get_gauge("invarlock.operation.p95_duration").get(
            labels={"operation": "verify"}
        )
        == 2.0
    )
    perf.operation_times["empty"] = []
    perf.update_metrics()

    resource = core.ResourceMonitor(
        registry,
        core.MonitoringConfig(
            cpu_threshold=80.0, memory_threshold=80.0, gpu_memory_threshold=40.0
        ),
    )
    monkeypatch.setattr(core.psutil, "cpu_percent", lambda interval=1: 10.0)
    monkeypatch.setattr(
        core.psutil,
        "virtual_memory",
        lambda: _MemoryInfo(
            percent=20.0, available=8 * 1024**3, used=2 * 1024**3, total=10 * 1024**3
        ),
    )
    monkeypatch.setattr(
        core.psutil, "disk_usage", lambda path: _DiskInfo(used=20, total=100, free=80)
    )
    monkeypatch.setattr(core.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(core.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        core.torch.cuda,
        "memory_stats",
        lambda index: {
            "allocated_bytes.all.current": 6 * 1024**3,
            "reserved_bytes.all.current": 7 * 1024**3,
        },
    )
    monkeypatch.setattr(
        core.torch.cuda,
        "get_device_properties",
        lambda index: types.SimpleNamespace(total_memory=10 * 1024**3),
    )

    usage = resource.collect_usage()
    warnings = resource.check_thresholds()

    assert usage["gpu_0_memory_percent"] == 60.0
    assert any("High GPU memory usage" in warning for warning in warnings)


@pytest.mark.unit
def test_alerting_resolution_and_noop_branches(monkeypatch):
    import invarlock.observability.alerting as alerting

    _SMTPRecorder.started_tls = False
    _SMTPRecorder.logged_in = None
    _SMTPRecorder.sent_subjects = []

    manager = alerting.AlertManager()
    triggered: list[str] = []
    monkeypatch.setattr(
        manager,
        "_trigger_alert",
        lambda rule, value, **context: triggered.append(rule.name),
    )

    manager.add_rule(
        alerting.AlertRule(
            name="error-too-high",
            metric="invarlock.errors.total",
            threshold=5,
            comparison="greater",
        )
    )
    manager.add_rule(
        alerting.AlertRule(
            name="resource-high",
            metric="invarlock.resource.cpu_percent",
            threshold=95.0,
        )
    )

    manager.check_error_alerts("runtime", "boom", {"phase": "collect"})
    manager.check_resource_alerts({})
    manager.check_resource_alerts({"cpu_percent": 50.0})
    assert triggered == []

    active_health = alerting.Alert(
        id="health_db",
        name="db",
        severity=alerting.AlertSeverity.WARNING,
        message="still bad",
        details={},
        timestamp=1.0,
    )
    manager.active_alerts[active_health.id] = active_health
    manager.check_health_alerts(
        {
            "db": types.SimpleNamespace(
                healthy=False,
                status=types.SimpleNamespace(value="warning"),
                message="down",
                details={},
            )
        }
    )
    assert manager.active_alerts["health_db"] is active_health

    manager.check_health_alerts(
        {
            "db": types.SimpleNamespace(
                healthy=True,
                status=types.SimpleNamespace(value="healthy"),
                message="ok",
                details={},
            )
        }
    )
    assert "health_db" not in manager.active_alerts

    manager._remove_alert("missing-alert")

    monkeypatch.setattr(alerting.smtplib, "SMTP", _SMTPRecorder)
    manager._send_email_notification(
        _make_alert(alerting),
        alerting.NotificationChannel(
            name="email-no-auth",
            type="email",
            config={
                "smtp_server": "smtp.example.com",
                "smtp_port": 2525,
                "from_address": "alerts@example.com",
                "to_addresses": ["ops@example.com"],
                "use_tls": False,
            },
        ),
    )
    assert _SMTPRecorder.started_tls is False
    assert _SMTPRecorder.logged_in is None
    assert _SMTPRecorder.sent_subjects


@pytest.mark.unit
def test_exporter_error_paths_and_background_loop(monkeypatch, caplog, tmp_path):
    import invarlock.observability.exporters as exporters

    metric = exporters.ExportedMetric(
        name="metric.name", value=3.0, timestamp=1.0, labels={"env": "dev"}
    )
    real_import = builtins.__import__

    monkeypatch.setattr(builtins, "__import__", _fake_import_without("requests"))
    gateway_exporter = exporters.PrometheusExporter(
        gateway_url="https://push.example.test"
    )
    assert gateway_exporter._push_to_gateway([metric]) is False

    monkeypatch.setattr(builtins, "__import__", real_import)
    cache_exporter = exporters.PrometheusExporter()
    monkeypatch.setattr(
        cache_exporter,
        "_update_cache",
        lambda metrics: (_ for _ in ()).throw(OSError("cache failed")),
    )
    assert cache_exporter.export([metric]) is False

    file_exporter = exporters.JSONExporter(
        output_file=str(tmp_path / "metrics.json"), pretty_print=False
    )
    assert file_exporter.export([metric]) is True
    assert (
        (tmp_path / "metrics.json").read_text()
        == '[{"metric": "metric.name", "value": 3.0, "timestamp": 1.0, "labels": {"env": "dev"}, "type": "gauge", "help": ""}]'
    )

    no_file_exporter = exporters.JSONExporter()
    assert no_file_exporter._write_to_file([]) is False

    class BrokenMetric:
        def to_json_format(self):
            raise ValueError("bad json")

    assert exporters.JSONExporter().export([BrokenMetric()]) is False

    request_module = types.SimpleNamespace()
    request_module.post = lambda *args, **kwargs: _Response(
        status_code=500, text="bad gateway"
    )
    monkeypatch.setitem(sys.modules, "requests", request_module)
    influx = exporters.InfluxDBExporter(
        url="https://influx.example.test",
        database="metrics",
        username="user",
        password="pass",
    )
    assert influx.export([metric]) is False

    request_module.post = lambda *args, **kwargs: (_ for _ in ()).throw(
        OSError("network down")
    )
    assert influx.export([metric]) is False

    monkeypatch.setattr(builtins, "__import__", _fake_import_without("socket"))
    stats = exporters.StatsExporter()
    assert stats.export([metric]) is False
    monkeypatch.setattr(builtins, "__import__", real_import)

    class BrokenSocket:
        def sendto(self, payload: bytes, address) -> None:
            raise OSError("udp blocked")

    socket_module = types.SimpleNamespace(
        AF_INET=1,
        SOCK_DGRAM=2,
        socket=lambda *args: BrokenSocket(),
    )
    monkeypatch.setitem(sys.modules, "socket", socket_module)
    assert stats.export([metric]) is False
    assert "env:dev" in stats._to_statsd_format(metric)

    manager = exporters.ExportManager()
    dummy_exporter = exporters.JSONExporter()
    manager.add_exporter(dummy_exporter)
    manager._running = True
    manager.start_background_export()
    assert manager._export_thread is None

    manager._running = True
    manager._metrics_queue = [metric]
    exports: list[list[exporters.ExportedMetric]] = []
    monkeypatch.setattr(
        manager,
        "export_now",
        lambda metrics=None: exports.append(metrics or []) or {"json": True},
    )

    def stop_after_sleep(interval: float) -> None:
        manager._running = False

    monkeypatch.setattr(exporters.time, "sleep", stop_after_sleep)
    manager._export_loop()
    assert exports and exports[0][0].name == "metric.name"

    manager._running = True

    def fail_once(interval: float) -> None:
        manager._running = False
        raise OSError("loop failed")

    monkeypatch.setattr(exporters.time, "sleep", fail_once)
    with caplog.at_level("ERROR"):
        manager._export_loop()
    assert "Error in export loop" in caplog.text

    class Joiner:
        def __init__(self) -> None:
            self.joined = False

        def join(self, timeout: float) -> None:
            self.joined = True

    joiner = Joiner()
    manager._export_thread = joiner
    manager.stop_background_export()
    assert joiner.joined is True


@pytest.mark.unit
def test_exporter_noop_and_existing_socket_branches(monkeypatch):
    import invarlock.observability.exporters as exporters

    metric = exporters.ExportedMetric(name="metric.name", value=3.0, timestamp=1.0)
    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(post=lambda *args, **kwargs: _Response(status_code=204)),
    )

    influx = exporters.InfluxDBExporter(
        url="https://influx.example.test",
        database="metrics",
    )
    monkeypatch.setattr(influx, "_to_line_protocol", lambda current: "")
    assert influx.export([metric]) is True

    stats = exporters.StatsExporter()
    sent_payloads: list[tuple[bytes, tuple[str, int]]] = []
    stats._socket = types.SimpleNamespace(
        sendto=lambda payload, address: sent_payloads.append((payload, address))
    )
    assert stats.export([]) is True
    monkeypatch.setattr(stats, "_to_statsd_format", lambda current: "")
    assert stats.export([metric]) is True
    assert sent_payloads == []

    manager = exporters.ExportManager()
    manager.stop_background_export()

    manager._running = True
    manager._metrics_queue = []
    export_calls: list[list[exporters.ExportedMetric] | None] = []
    monkeypatch.setattr(
        manager,
        "export_now",
        lambda metrics=None: export_calls.append(metrics) or {},
    )
    monkeypatch.setattr(
        exporters.time,
        "sleep",
        lambda interval: setattr(manager, "_running", False),
    )
    manager._export_loop()
    assert export_calls == []


@pytest.mark.unit
def test_prometheus_name_fallback_and_influx_missing_requests(monkeypatch, caplog):
    import invarlock.observability.exporters as exporters

    assert exporters._prometheus_name(" !!! ", fallback="fallback_metric") == "___"
    assert exporters._prometheus_name("", fallback="fallback_metric") == (
        "fallback_metric"
    )

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "requests":
            raise ImportError("requests missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    influx = exporters.InfluxDBExporter(
        url="https://influx.example.test",
        database="metrics",
    )

    with caplog.at_level("ERROR"):
        assert influx.export([]) is False
    assert "requests library required" in caplog.text
