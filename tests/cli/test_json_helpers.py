from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
import typer

from invarlock.cli import _json


@dataclass
class _Payload:
    message: str


def test_emit_adds_ts_and_component_for_dict_payload(monkeypatch, capsys) -> None:
    calls: dict[str, int] = {"ts": 0}

    def _fake_ts() -> str:
        calls["ts"] += 1
        return "2025-01-01T00:00:00+00:00"

    monkeypatch.setattr(_json, "_ts", _fake_ts)

    with pytest.raises(typer.Exit) as ei:
        _json.emit({"ok": True}, exit_code=5)

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["ts"] == "2025-01-01T00:00:00+00:00"
    assert payload["component"] == "cli"
    assert ei.value.exit_code == 5


def test_emit_accepts_dataclass_payload(monkeypatch, capsys) -> None:
    monkeypatch.setattr(_json, "_ts", lambda: "X")

    with pytest.raises(typer.Exit):
        _json.emit(_Payload("hello"), exit_code=0)

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["message"] == "hello"
    assert payload["ts"] == "X"
    assert payload["component"] == "cli"


def test_emit_passes_through_non_mapping_payload(monkeypatch, capsys) -> None:
    monkeypatch.setattr(_json, "_ts", lambda: "IGNORED")

    with pytest.raises(typer.Exit):
        _json.emit(["not-a-dict"], exit_code=3)

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload == ["not-a-dict"]


def test_encode_error_for_generic_exception() -> None:
    exc = RuntimeError("boom")
    encoded = _json.encode_error(exc)
    assert encoded["code"] == "E_GENERIC"
    assert encoded["category"] == "RuntimeError"
    assert encoded["recoverable"] is False
    assert encoded["context"] == {}


def test_encode_error_for_schema_like_errors() -> None:
    class ValidationError(Exception): ...

    err = ValidationError("bad schema")
    encoded = _json.encode_error(err)
    assert encoded["code"] == "E_SCHEMA"
    assert encoded["category"] == "ValidationError"


def test_encode_error_handles_invarlock_error(monkeypatch) -> None:
    class FakeInvarlockError(Exception):
        def __init__(self) -> None:
            self.code = "E_CUSTOM"
            self.recoverable = True
            self.details = {"reason": "details"}

    monkeypatch.setattr(_json, "InvarlockError", FakeInvarlockError)

    err = FakeInvarlockError()
    encoded = _json.encode_error(err)
    assert encoded["code"] == "E_CUSTOM"
    assert encoded["recoverable"] is True
    assert encoded["context"] == {"reason": "details"}


def test_encode_error_invarlock_branch_handles_non_dict_details(monkeypatch) -> None:
    class FakeInvarlockError(Exception):
        def __init__(self) -> None:
            self.code = "E_CUSTOM"
            self.recoverable = False
            self.details = "not-a-dict"

    monkeypatch.setattr(_json, "InvarlockError", FakeInvarlockError)

    err = FakeInvarlockError()
    encoded = _json.encode_error(err)
    assert encoded["code"] == "E_CUSTOM"
    assert encoded["recoverable"] is False
    assert encoded["context"] == {}


def test_encode_error_handles_category_introspection_failure() -> None:
    class _Meta(type):
        def __getattribute__(cls, name: str):  # type: ignore[override]
            if name == "__name__":
                raise RuntimeError("boom")
            return super().__getattribute__(name)

    class BrokenExc(Exception, metaclass=_Meta): ...

    encoded = _json.encode_error(BrokenExc())
    assert encoded["category"] == "Exception"
