from __future__ import annotations

import builtins


class OneShotEvent:
    def __init__(self) -> None:
        self._set = False

    def is_set(self) -> bool:
        return self._set

    def wait(self, timeout: float) -> bool:
        self._set = True
        return True

    def set(self) -> None:
        self._set = True


class MemoryInfo:
    def __init__(self, percent: float, available: int, used: int, total: int) -> None:
        self.percent = percent
        self.available = available
        self.used = used
        self.total = total


class DiskInfo:
    def __init__(self, used: int, total: int, free: int | None = None) -> None:
        self.used = used
        self.total = total
        self.free = total - used if free is None else free


class Response:
    def __init__(self, status_code: int = 200, text: str = "ok") -> None:
        self.status_code = status_code
        self.text = text
        self.raise_calls = 0

    def raise_for_status(self) -> None:
        self.raise_calls += 1
        if self.status_code >= 400:
            raise OSError(self.text)


class SMTPRecorder:
    started_tls = False
    logged_in: tuple[str, str] | None = None
    sent_subjects: list[str] = []

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def __enter__(self) -> SMTPRecorder:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def starttls(self) -> None:
        type(self).started_tls = True

    def login(self, username: str, password: str) -> None:
        type(self).logged_in = (username, password)

    def send_message(self, message) -> None:
        type(self).sent_subjects.append(message["Subject"])


def make_alert(alerting_module, *, severity=None):
    severity = severity or alerting_module.AlertSeverity.WARNING
    return alerting_module.Alert(
        id="alert-1",
        name="Edge Alert",
        severity=severity,
        message="edge-case",
        details={"outer": {"inner": "value"}, "plain": 1},
        timestamp=1_700_000_000.0,
    )


def fake_import_without(missing_name: str):
    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == missing_name:
            raise ImportError(missing_name)
        return real_import(name, globals, locals, fromlist, level)

    return _import
