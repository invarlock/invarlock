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
            percent=95.0, available=5 * 1024**3, used=5 * 1024**3, total=10 * 1024**3
        ),
    )
    assert checker.check_component("memory").status == health.HealthStatus.CRITICAL

    monkeypatch.setattr(
        health.psutil,
        "virtual_memory",
        lambda: _MemoryInfo(
            percent=85.0, available=8 * 1024**3, used=2 * 1024**3, total=10 * 1024**3
        ),
    )
    assert checker.check_component("memory").status == health.HealthStatus.WARNING

    monkeypatch.setattr(
        health.psutil,
        "virtual_memory",
        lambda: _MemoryInfo(
            percent=50.0, available=9 * 1024**3, used=1 * 1024**3, total=10 * 1024**3
        ),
    )
    assert checker.check_component("memory").status == health.HealthStatus.HEALTHY

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
        health.psutil,
        "cpu_percent",
        lambda interval=1: (_ for _ in ()).throw(OSError("cpu unavailable")),
    )
    cpu_failure = checker.check_component("cpu")
    assert cpu_failure.status == health.HealthStatus.CRITICAL
    assert cpu_failure.details["error"] == "cpu unavailable"

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
        lambda index: {"allocated_bytes.all.current": 80},
    )
    assert checker.check_component("gpu").status == health.HealthStatus.HEALTHY

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

    def import_with_adapter_setup_failure(
        name, globals=None, locals=None, fromlist=(), level=0
    ):
        if name == "invarlock.adapters.hf_causal":
            raise OSError("adapter import failed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_adapter_setup_failure)
    checker = health.InvarLockHealthChecker()
    adapter_import_failure = checker.check_component("adapters")
    assert adapter_import_failure.status == health.HealthStatus.CRITICAL
    assert adapter_import_failure.details["error"] == "adapter import failed"

    def import_with_guard_setup_failure(
        name, globals=None, locals=None, fromlist=(), level=0
    ):
        if name == "invarlock.guards.invariants":
            raise OSError("guard import failed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_guard_setup_failure)
    checker = health.InvarLockHealthChecker()
    guard_import_failure = checker.check_component("guards")
    assert guard_import_failure.status == health.HealthStatus.CRITICAL
    assert guard_import_failure.details["error"] == "guard import failed"

    def import_with_dependency_probe_failure(
        name, globals=None, locals=None, fromlist=(), level=0
    ):
        if name == "numpy":
            raise OSError("dependency probe failed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_dependency_probe_failure)
    checker = health.InvarLockHealthChecker()
    dependency_probe_failure = checker.check_component("dependencies")
    assert dependency_probe_failure.status == health.HealthStatus.CRITICAL
    assert dependency_probe_failure.details["error"] == "dependency probe failed"

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

    monkeypatch.setattr(
        health,
        "InvarLockHealthChecker",
        lambda: FakeHealthChecker(
            {
                "overall_status": "warning",
                "components": {},
                "status_counts": {},
                "total_components": 0,
                "last_check": 0,
            }
        ),
    )
    _, warning_handler_class = health.create_health_endpoint()
    warning_handler = warning_handler_class.__new__(warning_handler_class)
    warning_handler.path = "/health"
    warning_handler.wfile = io.BytesIO()
    warning_responses: list[int] = []
    warning_handler.send_response = warning_responses.append
    warning_handler.send_header = lambda key, value: None
    warning_handler.end_headers = lambda: None
    warning_handler.do_GET()
    assert warning_responses == [200]

    monkeypatch.setattr(
        health,
        "InvarLockHealthChecker",
        lambda: FakeHealthChecker(
            {
                "overall_status": "critical",
                "components": {},
                "status_counts": {},
                "total_components": 0,
                "last_check": 0,
            }
        ),
    )
    _, critical_handler_class = health.create_health_endpoint()
    critical_handler = critical_handler_class.__new__(critical_handler_class)
    critical_handler.path = "/health"
    critical_handler.wfile = io.BytesIO()
    critical_responses: list[int] = []
    critical_handler.send_response = critical_responses.append
    critical_handler.send_header = lambda key, value: None
    critical_handler.end_headers = lambda: None
    critical_handler.do_GET()
    assert critical_responses == [503]

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

    def import_without_http_server(
        name, globals=None, locals=None, fromlist=(), level=0
    ):
        if name == "http.server":
            raise ImportError("http.server")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_http_server)
    assert health.create_health_endpoint() == (None, None)
    monkeypatch.setattr(builtins, "__import__", real_import)


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
def test_metrics_exception_and_fallback_memory_branches(monkeypatch):
    import invarlock.observability.metrics as metrics

    monkeypatch.setattr(metrics, "torch", sys.modules["torch"], raising=False)
    monkeypatch.setattr(metrics.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        metrics.torch.cuda,
        "reset_peak_memory_stats",
        lambda: (_ for _ in ()).throw(RuntimeError("cuda reset failed")),
        raising=False,
    )
    metrics.reset_peak_memory_stats()

    process = types.SimpleNamespace(
        memory_info=lambda: types.SimpleNamespace(rss=32 * 1024 * 1024)
    )
    monkeypatch.setattr(sys.modules["psutil"], "Process", lambda pid: process)
    monkeypatch.setattr(
        metrics.torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(RuntimeError("cuda probe failed")),
    )
    snapshot = metrics.capture_memory_snapshot("torch-error", timestamp=1.0)
    assert snapshot == {"phase": "torch-error", "ts": 1.0, "rss_mb": 32.0}

    summary = metrics.summarize_memory_snapshots(
        [{"gpu_mb": 2.0, "gpu_reserved_mb": 3.0}]
    )
    assert summary == {
        "gpu_memory_mb_peak": 2.0,
        "gpu_memory_reserved_mb_peak": 3.0,
    }


@pytest.mark.unit
def test_health_cpu_loadavg_os_fallback(monkeypatch):
    import invarlock.observability.health as health

    class PsutilWithoutLoadavg:
        Error = health.psutil.Error

        @staticmethod
        def cpu_percent(interval=1):
            return 1.0

        @staticmethod
        def cpu_count():
            return 8

    monkeypatch.setattr(health, "psutil", PsutilWithoutLoadavg)
    monkeypatch.setattr(health.os, "getloadavg", lambda: (1.0, 2.0, 3.0))

    checker = health.HealthChecker()
    result = checker.check_component("cpu")

    assert result.details["load_avg"] == (1.0, 2.0, 3.0)
