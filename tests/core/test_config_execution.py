from __future__ import annotations

from pathlib import Path

import pytest

import invarlock.core.config_execution as config_execution


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

    def _run_impl(**kwargs):
        seen["run_kwargs"] = kwargs
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
        run_impl=_run_impl,
        deps_builder=lambda: {"deps": "ok"},
    )

    assert out == str(report_path)
    assert seen["allowances"] == {
        "allow_network": True,
        "allow_host_execution": False,
        "allow_third_party_plugins": True,
        "allow_remote_code": True,
    }
    assert seen["run_kwargs"] == {
        "config": "configs/demo.yaml",
        "device": None,
        "profile": "ci",
        "out": "runs/demo",
        "edit": None,
        "edit_label": None,
        "tier": "balanced",
        "metric_kind": None,
        "probes": None,
        "until_pass": False,
        "max_attempts": 3,
        "timeout": None,
        "baseline": None,
        "no_cleanup": False,
        "style": None,
        "progress": False,
        "timing": False,
        "telemetry": False,
        "no_color": False,
        "deps": {"deps": "ok"},
    }
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

    with pytest.raises(config_execution.RuntimeDelegationError, match="no runtime image"):
        config_execution.run_from_config(config="configs/demo.yaml")
