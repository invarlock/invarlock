from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

import invarlock.cli.config_execution as config_execution
import invarlock.runtime_security as runtime_security


def test_run_from_config_executes_concrete_run_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    report_path = Path("reports/demo.report.json")
    monkeypatch.setattr(
        config_execution,
        "resolve_shell_runtime_security_policy",
        lambda **kwargs: runtime_security.build_runtime_security_policy(
            **kwargs,
        ),
        raising=True,
    )

    @contextmanager
    def _scope(*, policy):
        seen["policy"] = policy
        yield

    monkeypatch.setattr(
        config_execution,
        "runtime_allowances_scope",
        _scope,
        raising=True,
    )
    monkeypatch.setattr(
        config_execution,
        "execute_config_run_request",
        lambda request: (seen.__setitem__("request", request), str(report_path))[1],
        raising=True,
    )
    monkeypatch.setattr(
        config_execution,
        "write_runtime_manifest",
        lambda *args, **kwargs: None,
        raising=True,
    )

    out = config_execution.run_from_config(config="configs/demo.yaml", delegate=False)

    assert out == report_path.resolve()
    assert seen["request"] == config_execution.ConfigExecutionRequest(
        config="configs/demo.yaml"
    )
    assert seen["policy"] == runtime_security.build_runtime_security_policy()


def test_run_from_config_executes_without_delegation_and_writes_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}
    report_path = tmp_path / "report.json"
    report_path.write_text('{"ok": true}\n', encoding="utf-8")

    monkeypatch.setattr(
        config_execution,
        "resolve_shell_runtime_security_policy",
        lambda **kwargs: runtime_security.build_runtime_security_policy(
            **kwargs,
        ),
        raising=True,
    )

    @contextmanager
    def _scope(*, policy):
        seen["policy"] = policy
        yield

    monkeypatch.setattr(
        config_execution,
        "runtime_allowances_scope",
        _scope,
        raising=True,
    )
    monkeypatch.setattr(
        config_execution,
        "write_runtime_manifest",
        lambda report, config_path=None, extra=None: seen.setdefault(
            "manifest",
            (Path(report), config_path, extra),
        ),
        raising=True,
    )

    monkeypatch.setattr(
        config_execution,
        "execute_config_run_request",
        lambda request: (seen.__setitem__("request", request), str(report_path))[1],
        raising=True,
    )

    out = config_execution.run_from_config(
        config="configs/demo.yaml",
        profile="ci",
        out="runs/demo",
        tier="balanced",
        allow_network=True,
        allow_remote_code=True,
        allow_third_party_plugins=True,
        command_name="proof-pack-run",
        delegate=False,
    )

    assert out == report_path.resolve()
    assert seen["policy"] == runtime_security.build_runtime_security_policy(
        allow_network=True,
        allow_third_party_plugins=True,
        allow_remote_code=True,
    )
    assert seen["request"] == config_execution.ConfigExecutionRequest(
        config="configs/demo.yaml",
        profile="ci",
        out="runs/demo",
        tier="balanced",
    )
    manifest_report, manifest_config, manifest_extra = seen["manifest"]
    assert manifest_report == report_path
    assert manifest_config == "configs/demo.yaml"
    assert manifest_extra == {
        "command": "proof-pack-run",
        "profile": "ci",
        "allow_network": True,
        "allow_remote_code": True,
        "allow_third_party_plugins": True,
    }


def test_run_from_config_delegates_when_secure_default_requires_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        config_execution,
        "resolve_shell_runtime_security_policy",
        lambda **kwargs: runtime_security.build_runtime_security_policy(),
        raising=True,
    )

    @contextmanager
    def _scope(*, policy):
        seen["policy"] = policy
        yield

    monkeypatch.setattr(
        config_execution,
        "runtime_allowances_scope",
        _scope,
        raising=True,
    )
    monkeypatch.setattr(
        config_execution,
        "running_inside_container",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        config_execution,
        "host_execution_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        config_execution,
        "build_request_container_launch_plan",
        lambda command_name, request: (
            seen.__setitem__("plan", (command_name, request)),
            "plan",
        )[1],
        raising=True,
    )
    monkeypatch.setattr(
        config_execution,
        "delegate_container_command",
        lambda plan: 7 if plan == "plan" else 99,
        raising=True,
    )

    with pytest.raises(SystemExit) as excinfo:
        config_execution.run_from_config(config="configs/demo.yaml")

    assert excinfo.value.code == 7
    assert seen["policy"] == runtime_security.build_runtime_security_policy()
    assert seen["plan"] == (
        "run",
        config_execution.ConfigExecutionRequest(config="configs/demo.yaml"),
    )


def test_run_from_config_wraps_runtime_delegation_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_execution,
        "resolve_shell_runtime_security_policy",
        lambda **kwargs: runtime_security.build_runtime_security_policy(),
        raising=True,
    )

    @contextmanager
    def _scope(*, policy):
        yield

    monkeypatch.setattr(
        config_execution,
        "runtime_allowances_scope",
        _scope,
        raising=True,
    )
    monkeypatch.setattr(
        config_execution,
        "running_inside_container",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        config_execution,
        "host_execution_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        config_execution,
        "build_request_container_launch_plan",
        lambda command_name, request: "plan",
        raising=True,
    )
    monkeypatch.setattr(
        config_execution,
        "delegate_container_command",
        lambda plan: (_ for _ in ()).throw(RuntimeError("no runtime image")),
        raising=True,
    )

    with pytest.raises(
        config_execution.RuntimeDelegationError, match="no runtime image"
    ):
        config_execution.run_from_config(config="configs/demo.yaml")


def test_run_from_config_skips_manifest_for_missing_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}
    missing_report = tmp_path / "missing.report.json"

    monkeypatch.setattr(
        config_execution,
        "resolve_shell_runtime_security_policy",
        lambda **kwargs: runtime_security.build_runtime_security_policy(
            **kwargs,
        ),
        raising=True,
    )

    @contextmanager
    def _scope(*, policy):
        seen["policy"] = policy
        yield

    monkeypatch.setattr(
        config_execution,
        "runtime_allowances_scope",
        _scope,
        raising=True,
    )

    monkeypatch.setattr(
        config_execution,
        "execute_config_run_request",
        lambda request: (seen.__setitem__("request", request), str(missing_report))[1],
        raising=True,
    )

    monkeypatch.setattr(
        config_execution,
        "write_runtime_manifest",
        lambda *args, **kwargs: pytest.fail("manifest should not be written"),
        raising=True,
    )

    out = config_execution.run_from_config(
        config="configs/demo.yaml",
        profile="ci",
        delegate=False,
    )

    assert out == missing_report.resolve()

    assert seen["policy"] == runtime_security.build_runtime_security_policy()
    assert seen["request"] == config_execution.ConfigExecutionRequest(
        config="configs/demo.yaml",
        profile="ci",
    )


def test_run_from_config_wraps_explicit_cuda_visibility_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_execution,
        "resolve_shell_runtime_security_policy",
        lambda **kwargs: runtime_security.build_runtime_security_policy(**kwargs),
        raising=True,
    )

    @contextmanager
    def _scope(*, policy):
        yield

    monkeypatch.setattr(
        config_execution,
        "runtime_allowances_scope",
        _scope,
        raising=True,
    )
    monkeypatch.setattr(
        config_execution,
        "running_inside_container",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        config_execution,
        "host_execution_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        config_execution,
        "build_request_container_launch_plan",
        lambda command_name, request: (_ for _ in ()).throw(
            RuntimeError("Requested --device cuda, but no NVIDIA runtime is visible")
        ),
        raising=True,
    )

    with pytest.raises(
        config_execution.RuntimeDelegationError,
        match="Requested --device cuda",
    ):
        config_execution.run_from_config(
            config="configs/demo.yaml",
            device="cuda",
        )


def test_run_from_config_raises_when_run_execution_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_execution,
        "resolve_shell_runtime_security_policy",
        lambda **kwargs: runtime_security.build_runtime_security_policy(),
        raising=True,
    )

    @contextmanager
    def _scope(*, policy):
        yield

    monkeypatch.setattr(
        config_execution,
        "runtime_allowances_scope",
        _scope,
        raising=True,
    )
    monkeypatch.setattr(
        config_execution,
        "write_runtime_manifest",
        lambda *args, **kwargs: pytest.fail("manifest should not be written"),
        raising=True,
    )

    monkeypatch.setattr(
        config_execution,
        "execute_config_run_request",
        lambda request: None,
        raising=True,
    )

    with pytest.raises(
        RuntimeError, match="run execution did not return a report path"
    ):
        config_execution.run_from_config(
            config="configs/demo.yaml",
            delegate=False,
        )
