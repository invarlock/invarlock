from __future__ import annotations

from pathlib import Path

import pytest

import invarlock.core.config_execution as config_execution


def test_run_from_config_requires_explicit_impls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_execution,
        "apply_runtime_allowances",
        lambda **kwargs: None,
        raising=True,
    )
    with pytest.raises(
        RuntimeError,
        match="requires an explicit executor callable",
    ):
        config_execution.run_from_config(config="configs/demo.yaml", delegate=False)


def test_run_from_config_executes_without_delegation_and_writes_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}
    report_path = tmp_path / "report.json"
    report_path.write_text('{"ok": true}\n', encoding="utf-8")

    monkeypatch.setattr(
        config_execution,
        "apply_runtime_allowances",
        lambda **kwargs: seen.setdefault("allowances", kwargs),
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

    def _executor(request: config_execution.ConfigExecutionRequest):
        seen["request"] = request
        return str(report_path)

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
        executor=_executor,
    )

    assert out == report_path.resolve()
    assert seen["allowances"] == {
        "allow_network": True,
        "allow_host_execution": False,
        "allow_third_party_plugins": True,
        "allow_remote_code": True,
    }
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
    monkeypatch.setattr(
        config_execution,
        "apply_runtime_allowances",
        lambda **kwargs: None,
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
        "delegate_current_process_to_container",
        lambda: 7,
        raising=True,
    )

    with pytest.raises(SystemExit) as excinfo:
        config_execution.run_from_config(config="configs/demo.yaml")

    assert excinfo.value.code == 7


def test_run_from_config_wraps_runtime_delegation_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_execution,
        "apply_runtime_allowances",
        lambda **kwargs: None,
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
        "delegate_current_process_to_container",
        lambda: (_ for _ in ()).throw(RuntimeError("no runtime image")),
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
        "apply_runtime_allowances",
        lambda **kwargs: seen.setdefault("allowances", kwargs),
        raising=True,
    )

    def _executor(request: config_execution.ConfigExecutionRequest) -> str:
        seen["request"] = request
        return str(missing_report)

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
        executor=_executor,
    )

    assert out == missing_report.resolve()

    assert seen["allowances"] == {
        "allow_network": False,
        "allow_host_execution": False,
        "allow_third_party_plugins": False,
        "allow_remote_code": False,
    }
    assert seen["request"] == config_execution.ConfigExecutionRequest(
        config="configs/demo.yaml",
        profile="ci",
    )


def test_run_from_config_raises_when_executor_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_execution,
        "apply_runtime_allowances",
        lambda **kwargs: None,
        raising=True,
    )
    monkeypatch.setattr(
        config_execution,
        "write_runtime_manifest",
        lambda *args, **kwargs: pytest.fail("manifest should not be written"),
        raising=True,
    )

    with pytest.raises(RuntimeError, match="executor did not return a report path"):
        config_execution.run_from_config(
            config="configs/demo.yaml",
            delegate=False,
            executor=lambda request: None,
        )
