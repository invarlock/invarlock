from __future__ import annotations

import builtins
import io
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
def test_health_checker_branch_coverage(monkeypatch):
    import invarlock.observability.health as health

    checker = health.HealthChecker()

    checker.last_results = {
        "unknown": health.ComponentHealth(
            name="unknown",
            status=health.HealthStatus.UNKNOWN,
            message="unknown",
            details={},
            timestamp=1.0,
        )
    }
    assert checker.get_overall_status() == health.HealthStatus.UNKNOWN

    monkeypatch.setattr(
        health.psutil,
        "virtual_memory",
        lambda: _MemoryInfo(
            percent=85.0, available=8 * 1024**3, used=2 * 1024**3, total=10 * 1024**3
        ),
    )
    assert checker.check_component("memory").status == health.HealthStatus.WARNING

    monkeypatch.setattr(health.psutil, "cpu_percent", lambda interval=1: 97.0)
    monkeypatch.setattr(
        health.psutil,
        "cpu_count",
        lambda: (_ for _ in ()).throw(OSError("cpu count failed")),
    )
    monkeypatch.setattr(health.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        health.psutil,
        "getloadavg",
        lambda: (_ for _ in ()).throw(OSError("load failed")),
    )
    cpu_result = checker.check_component("cpu")
    assert cpu_result.status == health.HealthStatus.CRITICAL
    assert "warnings" in cpu_result.details

    monkeypatch.setattr(health.psutil, "cpu_percent", lambda interval=1: 90.0)
    monkeypatch.setattr(health.psutil, "cpu_count", lambda: None)
    monkeypatch.delattr(health.psutil, "getloadavg", raising=False)
    monkeypatch.setattr(health.os, "getloadavg", lambda: (1.0, 2.0, 3.0), raising=False)
    cpu_warning = checker.check_component("cpu")
    assert cpu_warning.status == health.HealthStatus.WARNING
    assert cpu_warning.details["core_count"] == 8

    monkeypatch.setattr(
        health.psutil, "disk_usage", lambda path: _DiskInfo(used=96, total=100)
    )
    assert checker.check_component("disk").status == health.HealthStatus.CRITICAL
    monkeypatch.setattr(
        health.psutil, "disk_usage", lambda path: _DiskInfo(used=86, total=100)
    )
    assert checker.check_component("disk").status == health.HealthStatus.WARNING
    monkeypatch.setattr(
        health.psutil,
        "disk_usage",
        lambda path: (_ for _ in ()).throw(OSError("disk failed")),
    )
    assert checker.check_component("disk").status == health.HealthStatus.CRITICAL

    monkeypatch.setattr(health.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(health.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        health.torch.cuda,
        "get_device_properties",
        lambda index: types.SimpleNamespace(name="Fake GPU", total_memory=100),
    )
    monkeypatch.setattr(
        health.torch.cuda,
        "memory_stats",
        lambda index: {"allocated_bytes.all.current": 90},
    )
    assert checker.check_component("gpu").status == health.HealthStatus.WARNING

    monkeypatch.setattr(
        health.torch.cuda,
        "memory_stats",
        lambda index: {"allocated_bytes.all.current": 96},
    )
    assert checker.check_component("gpu").status == health.HealthStatus.CRITICAL

    monkeypatch.setattr(
        health.torch.cuda,
        "get_device_properties",
        lambda index: (_ for _ in ()).throw(OSError("gpu failed")),
    )
    assert checker.check_component("gpu").status == health.HealthStatus.WARNING

    monkeypatch.setattr(
        health.torch,
        "randn",
        lambda *shape: (_ for _ in ()).throw(OSError("torch failed")),
    )
    assert checker.check_component("pytorch").status == health.HealthStatus.CRITICAL

    monkeypatch.setattr(
        health.torch,
        "randn",
        lambda *shape: types.SimpleNamespace(t=lambda: "transpose"),
    )
    monkeypatch.setattr(health.torch, "mm", lambda left, right: None)
    monkeypatch.setattr(health.torch.backends.mps, "is_available", lambda: True)
    pytorch_result = checker.check_component("pytorch")
    assert pytorch_result.status == health.HealthStatus.HEALTHY
    assert pytorch_result.details["mps_available"] is True


@pytest.mark.unit
def test_invarlock_health_checker_partial_missing_and_endpoint(monkeypatch):
    import invarlock.observability.health as health

    fake_hf_causal = sys.modules["invarlock.adapters.hf_causal"]
    fake_hf_mlm = sys.modules["invarlock.adapters.hf_mlm"]
    fake_hf_multimodal = sys.modules["invarlock.adapters.hf_multimodal"]
    fake_hf_seq2seq = sys.modules["invarlock.adapters.hf_seq2seq"]
    fake_invariants = sys.modules["invarlock.guards.invariants"]
    fake_rmt = sys.modules["invarlock.guards.rmt"]
    fake_spectral = sys.modules["invarlock.guards.spectral"]
    fake_variance = sys.modules["invarlock.guards.variance"]

    monkeypatch.setattr(fake_hf_causal, "HF_Causal_Adapter", lambda: None)
    monkeypatch.setattr(
        fake_hf_mlm,
        "HF_MLM_Adapter",
        lambda: (_ for _ in ()).throw(OSError("mlm failed")),
    )
    monkeypatch.setattr(fake_hf_multimodal, "HF_Multimodal_Adapter", lambda: None)
    monkeypatch.setattr(
        fake_hf_seq2seq,
        "HF_Seq2Seq_Adapter",
        lambda: (_ for _ in ()).throw(OSError("seq2seq failed")),
    )

    checker = health.InvarLockHealthChecker()
    adapters_result = checker.check_component("adapters")
    assert adapters_result.status == health.HealthStatus.WARNING
    assert len(adapters_result.details["failed"]) == 2

    monkeypatch.setattr(
        fake_hf_causal,
        "HF_Causal_Adapter",
        lambda: (_ for _ in ()).throw(OSError("all fail")),
    )
    monkeypatch.setattr(
        fake_hf_multimodal,
        "HF_Multimodal_Adapter",
        lambda: (_ for _ in ()).throw(OSError("all fail")),
    )
    checker = health.InvarLockHealthChecker()
    no_adapters = checker.check_component("adapters")
    assert no_adapters.status == health.HealthStatus.CRITICAL

    monkeypatch.setattr(fake_invariants, "InvariantsGuard", lambda: None)
    monkeypatch.setattr(
        fake_rmt, "RMTGuard", lambda: (_ for _ in ()).throw(OSError("rmt failed"))
    )
    monkeypatch.setattr(fake_spectral, "SpectralGuard", lambda: None)
    monkeypatch.setattr(
        fake_variance,
        "VarianceGuard",
        lambda policy=None: (_ for _ in ()).throw(OSError("variance failed")),
    )
    checker = health.InvarLockHealthChecker()
    guards_result = checker.check_component("guards")
    assert guards_result.status == health.HealthStatus.WARNING
    assert len(guards_result.details["failed"]) == 2

    monkeypatch.setattr(
        fake_invariants,
        "InvariantsGuard",
        lambda: (_ for _ in ()).throw(OSError("all fail")),
    )
    monkeypatch.setattr(
        fake_spectral,
        "SpectralGuard",
        lambda: (_ for _ in ()).throw(OSError("all fail")),
    )
    checker = health.InvarLockHealthChecker()
    no_guards = checker.check_component("guards")
    assert no_guards.status == health.HealthStatus.CRITICAL

    real_import = builtins.__import__

    def import_without_numpy(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy":
            raise ImportError("numpy")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_numpy)
    checker = health.InvarLockHealthChecker()
    missing_optional = checker.check_component("dependencies")
    assert missing_optional.status == health.HealthStatus.WARNING
    assert "numpy" in missing_optional.details["missing"]

    def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("torch")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    checker = health.InvarLockHealthChecker()
    missing_torch = checker.check_component("dependencies")
    assert missing_torch.status == health.HealthStatus.CRITICAL
    assert "torch" in missing_torch.details["missing"]
    monkeypatch.setattr(builtins, "__import__", real_import)

    class FakeHealthChecker:
        def __init__(self, summary):
            self._summary = summary

        def get_summary(self):
            return self._summary

    monkeypatch.setattr(
        health,
        "InvarLockHealthChecker",
        lambda: FakeHealthChecker(
            {
                "overall_status": "healthy",
                "components": {},
                "status_counts": {},
                "total_components": 0,
                "last_check": 0,
            }
        ),
    )
    _, handler_class = health.create_health_endpoint()
    handler = handler_class.__new__(handler_class)
    handler.path = "/health"
    handler.wfile = io.BytesIO()
    responses: list[int] = []
    headers: list[tuple[str, str]] = []
    handler.send_response = responses.append
    handler.send_header = lambda key, value: headers.append((key, value))
    handler.end_headers = lambda: None
    handler.do_GET()
    assert responses == [200]
    assert headers == [("Content-type", "application/json")]
    assert b'"overall_status": "healthy"' in handler.wfile.getvalue()

    missing_handler = handler_class.__new__(handler_class)
    missing_handler.path = "/missing"
    missing_handler.wfile = io.BytesIO()
    not_found: list[int] = []
    missing_handler.send_response = not_found.append
    missing_handler.send_header = lambda key, value: None
    missing_handler.end_headers = lambda: None
    missing_handler.do_GET()
    missing_handler.log_message("ignored")
    assert not_found == [404]


@pytest.mark.unit
def test_metrics_edge_branches(monkeypatch):
    import invarlock.observability.metrics as metrics

    assert metrics.Histogram._percentile_from_sorted([], 95) == 0.0

    timer = metrics.Timer("timed")
    context = metrics.TimerContext(timer)
    calls: list[tuple[float, dict[str, str] | None]] = []
    monkeypatch.setattr(
        timer, "record", lambda duration, labels=None: calls.append((duration, labels))
    )
    context.__exit__(None, None, None)
    assert calls == []

    registry = metrics.MetricsRegistry()
    registry.register_gauge("shared")
    with pytest.raises(ValueError):
        registry.register_counter("shared")
    with pytest.raises(ValueError):
        registry.register_histogram("shared")
    with pytest.raises(ValueError):
        registry.register_timer("shared")
    with pytest.raises(ValueError):
        registry.get_counter("shared")

    registry = metrics.MetricsRegistry()
    registry.register_counter("as_counter")
    with pytest.raises(ValueError):
        registry.get_gauge("as_counter")
    with pytest.raises(ValueError):
        registry.get_histogram("as_counter")
    with pytest.raises(ValueError):
        registry.get_timer("as_counter")

    histogram = registry.register_histogram("hist")
    histogram.observe(1.0)
    timer_metric = registry.register_timer("timer")
    timer_metric.record(2.0)
    all_metrics = registry.get_all_metrics()
    assert all_metrics["hist"]["type"] == "histogram"
    assert all_metrics["timer"]["type"] == "timer"

    reset_calls: list[str] = []
    monkeypatch.setattr(metrics, "torch", sys.modules["torch"], raising=False)
    monkeypatch.setattr(metrics.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        metrics.torch.cuda,
        "reset_peak_memory_stats",
        lambda: reset_calls.append("cuda"),
        raising=False,
    )
    metrics.torch.mps = types.SimpleNamespace(
        reset_peak_memory_stats=lambda: reset_calls.append("mps")
    )
    monkeypatch.setattr(metrics.torch.backends.mps, "is_available", lambda: True)
    metrics.reset_peak_memory_stats()
    assert reset_calls == ["cuda", "mps"]

    monkeypatch.setattr(metrics.time, "time", lambda: 123.0)
    process = types.SimpleNamespace(
        memory_info=lambda: types.SimpleNamespace(rss=64 * 1024 * 1024)
    )
    monkeypatch.setattr(sys.modules["psutil"], "Process", lambda pid: process)
    monkeypatch.setattr(metrics.torch.cuda, "current_device", lambda: 0, raising=False)
    monkeypatch.setattr(
        metrics.torch.cuda,
        "memory_allocated",
        lambda device: 2 * 1024 * 1024,
        raising=False,
    )
    monkeypatch.setattr(
        metrics.torch.cuda,
        "memory_reserved",
        lambda device: 3 * 1024 * 1024,
        raising=False,
    )
    monkeypatch.setattr(
        metrics.torch.cuda,
        "max_memory_allocated",
        lambda device: 4 * 1024 * 1024,
        raising=False,
    )
    monkeypatch.setattr(
        metrics.torch.cuda,
        "max_memory_reserved",
        lambda device: 5 * 1024 * 1024,
        raising=False,
    )
    snapshot = metrics.capture_memory_snapshot("cuda")
    assert snapshot["gpu_device"] == "cuda:0"
    assert snapshot["gpu_peak_reserved_mb"] == 5.0

    monkeypatch.setattr(metrics.torch.cuda, "is_available", lambda: False)
    metrics.torch.mps = types.SimpleNamespace(
        current_allocated_memory=lambda: 6 * 1024 * 1024,
        driver_allocated_memory=lambda: 7 * 1024 * 1024,
    )
    monkeypatch.setattr(metrics.torch.backends.mps, "is_available", lambda: True)
    mps_snapshot = metrics.capture_memory_snapshot("mps")
    assert mps_snapshot["gpu_device"] == "mps"
    assert mps_snapshot["gpu_reserved_mb"] == 7.0

    summary = metrics.summarize_memory_snapshots(
        [
            {"rss_mb": 1.0, "gpu_mb": 2.0, "gpu_reserved_mb": 3.0},
            {"rss_mb": 4.0, "gpu_mb": 5.0, "gpu_reserved_mb": 6.0},
        ]
    )
    assert summary == {
        "memory_mb_peak": 4.0,
        "gpu_memory_mb_peak": 5.0,
        "gpu_memory_reserved_mb_peak": 6.0,
    }


@pytest.mark.unit
def test_metrics_existing_registry_and_snapshot_edge_branches(monkeypatch):
    import invarlock.observability.metrics as metrics

    assert metrics.Counter._key_to_labels("novalue") == {}

    registry = metrics.MetricsRegistry()
    registry.register_counter("counter")
    gauge = registry.register_gauge("gauge")
    histogram = registry.register_histogram("hist")
    timer = registry.register_timer("timer")
    assert registry.register_gauge("gauge") is gauge
    assert registry.register_histogram("hist") is histogram
    assert registry.register_timer("timer") is timer

    metric_dump = registry.get_all_metrics()
    assert metric_dump["counter"]["type"] == "counter"
    assert metric_dump["gauge"]["type"] == "gauge"
    assert metric_dump["hist"]["type"] == "histogram"
    assert metric_dump["timer"]["type"] == "timer"

    monkeypatch.setattr(metrics.time, "time", lambda: 456.0)
    monkeypatch.setattr(metrics, "torch", sys.modules["torch"], raising=False)
    monkeypatch.setattr(metrics.torch.cuda, "is_available", lambda: False)
    metrics.torch.mps = types.SimpleNamespace()
    monkeypatch.setattr(metrics.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(
        sys.modules["psutil"],
        "Process",
        lambda pid: (_ for _ in ()).throw(OSError("process missing")),
    )
    snapshot = metrics.capture_memory_snapshot("mps-lite", timestamp=456.0)
    assert snapshot == {"phase": "mps-lite", "ts": 456.0, "gpu_device": "mps"}

    metrics.torch.mps = None
    assert metrics.capture_memory_snapshot("cpu-only", timestamp=789.0) == {}

    summary = metrics.summarize_memory_snapshots(
        [{"gpu_peak_reserved_mb": 9.0, "gpu_peak_mb": 4.0}]
    )
    assert summary == {
        "gpu_memory_mb_peak": 4.0,
        "gpu_memory_reserved_mb_peak": 9.0,
    }


@pytest.mark.unit
def test_utils_false_paths_and_custom_logger(monkeypatch):
    import invarlock.observability.utils as utils

    callback_names: list[str] = []

    @utils.timing_decorator(
        auto_log=False,
        callback=lambda context: callback_names.append(context.operation),
    )
    def decorated() -> str:
        return "ok"

    assert decorated() == "ok"
    assert callback_names

    moving = utils.MovingAverage(window_size=2)
    assert moving.get_stats() == {"average": 0, "min": 0, "max": 0, "count": 0}

    percentiles = utils.PercentileCalculator()
    assert percentiles.get_percentiles([50, 95]) == {50: 0, 95: 0}

    monkeypatch.setattr(
        utils.psutil,
        "cpu_count",
        lambda logical=False: 4 if not logical else 8,
    )
    monkeypatch.setattr(
        utils.psutil,
        "virtual_memory",
        lambda: _MemoryInfo(percent=10.0, available=8, used=2, total=10),
    )
    monkeypatch.setattr(
        utils.psutil,
        "disk_usage",
        lambda path: _DiskInfo(used=20, total=100, free=80),
    )
    monkeypatch.setattr(utils.psutil, "cpu_freq", lambda: None)
    monkeypatch.setattr(utils.psutil, "sys", {"version": "3.12-test"}, raising=False)
    monkeypatch.setattr(utils.psutil, "os", {"name": "darwin"}, raising=False)
    monkeypatch.setattr(utils, "torch", sys.modules["torch"], raising=False)
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    info = utils.get_system_info()
    assert info["gpu"]["gpu_available"] is False

    assert utils.format_bytes(512) == "512.0 B"

    sleeps: list[float] = []

    @utils.retry_with_backoff(
        max_attempts=1,
        base_delay=0.01,
        exceptions=(ValueError,),
    )
    def fail_once():
        raise ValueError("boom")

    monkeypatch.setattr(utils.time, "sleep", lambda delay: sleeps.append(delay))
    with pytest.raises(ValueError):
        fail_once()
    assert sleeps == []

    class Recorder:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def log(self, level: int, message: str, exc_info: bool = False) -> None:
            self.messages.append(message)

    recorder = Recorder()

    @utils.log_exceptions(logger=recorder, reraise=False)
    def swallowed() -> None:
        raise RuntimeError("suppressed")

    assert swallowed() is None
    assert recorder.messages == ["Exception in swallowed: suppressed"]
