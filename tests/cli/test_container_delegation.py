from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import typer

from invarlock.cli import security_helpers


def test_maybe_delegate_model_command_delegates_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_HOST_EXECUTION", "0")
    monkeypatch.delenv("INVARLOCK_CONTAINER_EXECUTION", raising=False)
    calls: list[str] = []

    def _delegate() -> int:
        calls.append("delegated")
        return 0

    monkeypatch.setattr(
        security_helpers,
        "build_current_process_container_launch_plan",
        lambda: "plan",
        raising=True,
    )
    monkeypatch.setattr(
        security_helpers,
        "delegate_container_command",
        lambda plan: _delegate() if plan == "plan" else 1,
        raising=True,
    )

    with pytest.raises(typer.Exit) as exc:
        security_helpers.maybe_delegate_model_command()

    assert exc.value.exit_code == 0
    assert calls == ["delegated"]


def test_maybe_delegate_model_command_respects_shell_host_exec_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_HOST_EXECUTION", "1")
    monkeypatch.delenv("INVARLOCK_CONTAINER_EXECUTION", raising=False)

    monkeypatch.setattr(
        security_helpers,
        "delegate_container_command",
        lambda plan: pytest.fail("should not delegate when shell env allows host"),
        raising=True,
    )

    security_helpers.maybe_delegate_model_command()
    assert os.environ["INVARLOCK_ALLOW_HOST_EXECUTION"] == "1"


def test_maybe_delegate_model_command_reports_container_launch_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("INVARLOCK_ALLOW_HOST_EXECUTION", "0")
    monkeypatch.delenv("INVARLOCK_CONTAINER_EXECUTION", raising=False)
    monkeypatch.setattr(
        security_helpers,
        "build_current_process_container_launch_plan",
        lambda: "plan",
    )
    monkeypatch.setattr(
        security_helpers,
        "delegate_container_command",
        lambda _plan: (_ for _ in ()).throw(RuntimeError("container unavailable")),
    )

    with pytest.raises(typer.Exit) as exc:
        security_helpers.maybe_delegate_model_command()

    assert exc.value.exit_code == 1
    assert "container unavailable" in capsys.readouterr().err


def test_runtime_security_scope_treats_explicit_host_mode_as_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _Scope:
        def __enter__(self) -> None:
            observed["entered"] = True

        def __exit__(self, *_args: object) -> None:
            observed["exited"] = True

    def configure(**kwargs: object) -> _Scope:
        observed["policy"] = kwargs
        return _Scope()

    monkeypatch.setattr(security_helpers, "configure_runtime_security", configure)

    @security_helpers.runtime_security_scoped
    def command(**kwargs: object) -> str:
        observed["command"] = kwargs
        return "ok"

    assert command(execution_mode=" HOST ") == "ok"
    assert observed["policy"] == {
        "allow_network": False,
        "allow_host_execution": True,
        "allow_third_party_plugins": False,
        "allow_remote_code": False,
        "allow_unverified_provenance": True,
    }
    assert observed["entered"] is True
    assert observed["exited"] is True


def test_emit_runtime_manifest_skips_absent_reports_and_forwards_existing_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    assert security_helpers.emit_runtime_manifest(None) is None
    assert security_helpers.emit_runtime_manifest(tmp_path / "missing.json") is None

    report = tmp_path / "evaluation.report.json"
    report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        security_helpers,
        "write_runtime_manifest",
        lambda path, **kwargs: (path, kwargs),
    )

    result = security_helpers.emit_runtime_manifest(
        report,
        config_payload={"command": "evaluate"},
        extra={"profile": "release"},
    )

    assert result == (
        report,
        {
            "config_path": None,
            "config_payload": {"command": "evaluate"},
            "extra": {"profile": "release"},
            "execution": None,
        },
    )


def test_runtime_security_policy_reuses_active_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = object()
    monkeypatch.setattr(
        security_helpers, "current_runtime_security_policy", lambda: active
    )

    assert security_helpers.resolve_shell_runtime_security_policy() is active


def test_configure_runtime_security_forwards_resolved_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, bool]] = []
    policy = SimpleNamespace(
        allow_network=True,
        allow_host_execution=False,
        allow_third_party_plugins=True,
        allow_remote_code=False,
        allow_unverified_provenance=True,
    )
    monkeypatch.setattr(
        security_helpers,
        "resolve_shell_runtime_security_policy",
        lambda **_kwargs: policy,
    )

    from invarlock import runtime_provenance

    @contextmanager
    def configured(**kwargs: bool):
        observed.append(kwargs)
        yield

    monkeypatch.setattr(runtime_provenance, "configure_runtime_security", configured)

    with security_helpers.configure_runtime_security(allow_network=False):
        observed.append({"inside": True})

    assert observed == [
        {
            "allow_network": True,
            "allow_host_execution": False,
            "allow_third_party_plugins": True,
            "allow_remote_code": False,
            "allow_unverified_provenance": True,
        },
        {"inside": True},
    ]


def test_security_helper_delegates_launch_plan_and_provenance_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from invarlock import runtime_provenance, runtime_security

    monkeypatch.setattr(
        runtime_security,
        "build_current_process_container_launch_plan",
        lambda argv: ("plan", argv),
    )
    monkeypatch.setattr(
        runtime_provenance,
        "verify_runtime_provenance",
        lambda report_path, *, allow_unverified: [f"{report_path}:{allow_unverified}"],
    )

    assert security_helpers.build_current_process_container_launch_plan(["run"]) == (
        "plan",
        ["run"],
    )
    assert security_helpers.verify_runtime_provenance(
        "evaluation.report.json", allow_unverified=True
    ) == ["evaluation.report.json:True"]
