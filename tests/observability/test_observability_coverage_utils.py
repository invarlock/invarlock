from __future__ import annotations

import builtins
import sys

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
def test_utils_false_paths_and_custom_logger(monkeypatch):
    import invarlock.observability.utils as utils

    with pytest.raises(ValueError, match="max_calls must be positive"):
        utils.RateLimiter(max_calls=0, window_seconds=60)
    with pytest.raises(ValueError, match="window_seconds must be positive"):
        utils.RateLimiter(max_calls=1, window_seconds=0)
    with pytest.raises(ValueError, match="size must be positive"):
        utils.CircularBuffer(size=0)

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
    for value in [10.0, 20.0, 30.0]:
        percentiles.add(value)
    assert percentiles.get_percentile(-10) == 10.0
    assert percentiles.get_percentile(110) == 30.0
    assert percentiles.get_percentiles([-1, 101]) == {-1: 10.0, 101: 30.0}

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

    callback_contexts = []

    @utils.timing_decorator(auto_log=True, callback=callback_contexts.append)
    def timed_value():
        return "ok"

    assert timed_value() == "ok"
    assert callback_contexts

    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(utils.torch.cuda, "get_device_name", lambda index: "cuda-test")
    utils.torch.version.cuda = "12.8"
    info_cuda = utils.get_system_info()
    assert info_cuda["gpu"]["gpu_available"] is True
    assert info_cuda["gpu"]["gpu_names"] == ["cuda-test"]

    assert utils.format_bytes(1024**6) == "1024.0 PB"
    assert utils.safe_divide("bad", 2, default=-1) == -1

    @utils.retry_with_backoff(max_attempts=0, base_delay=0.01)
    def never_called():
        return "unreachable"

    with pytest.raises(RuntimeError, match="Failed after all retry attempts"):
        never_called()

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
