from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer


def _touch_min_cert(tmp_path: Path) -> Path:
    p = tmp_path / "c.json"
    p.write_text("{}", encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "exc_factory,profile,expected_exit",
    [
        (
            # InvarlockError → exit 3 in CI/Release; include structured error object
            lambda: __import__(
                "invarlock.cli.errors", fromlist=["InvarlockError"]
            ).InvarlockError(
                code="E005", message="boom", details={"x": 1}, recoverable=True
            ),
            "ci",
            3,
        ),
        (
            # ConfigError (schema/validation) → exit 2
            lambda: __import__(
                "invarlock.core.exceptions", fromlist=["ConfigError"]
            ).ConfigError(code="E201", message="bad cfg"),
            "dev",
            2,
        ),
        (
            # DataError (schema/validation) → exit 2
            lambda: __import__(
                "invarlock.core.exceptions", fromlist=["DataError"]
            ).DataError(code="E301", message="bad data"),
            "release",
            2,
        ),
        (
            # Generic unexpected → exit 1
            lambda: RuntimeError("unexpected"),
            "dev",
            1,
        ),
    ],
)
def test_verify_json_structured_error_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    exc_factory,
    profile,
    expected_exit,
) -> None:
    # Force verify_command to raise our exception before processing
    from invarlock.cli.commands import verify as v

    def _boom(*a, **k):
        raise exc_factory()

    monkeypatch.setattr(v, "_load_certificate", _boom)

    cert_path = _touch_min_cert(tmp_path)
    with pytest.raises(typer.Exit) as ei:
        v.verify_command([cert_path], baseline=None, profile=profile, json_out=True)
    # exit codes mapped as expected
    assert (
        getattr(ei.value, "exit_code", getattr(ei.value, "code", None)) == expected_exit
    )
    # JSON payload includes structured error envelope
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload.get("resolution"), dict)
    assert payload["resolution"].get("exit_code") == expected_exit
    err = payload.get("error")
    assert isinstance(err, dict)
    assert (
        "code" in err
        and "category" in err
        and "recoverable" in err
        and "context" in err
    )
