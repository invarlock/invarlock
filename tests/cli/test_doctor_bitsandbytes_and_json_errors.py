from __future__ import annotations

import builtins
import json
from types import SimpleNamespace

import pytest
import typer

from invarlock.cli.commands import doctor as doctor_mod
from invarlock.core.doctor_findings import build_cross_check_findings
from tests.cli.test_doctor_preflight_config_paths import (
    DummyConsole,
    _install_fake_torch,
    _patch_minimal_doctor_env,
)


def _setup_bitsandbytes_env(
    monkeypatch: pytest.MonkeyPatch, *, cuda_available: bool, bitsandbytes_present: bool
) -> DummyConsole:
    """Prepare a minimal environment so doctor_command reaches the optional deps block."""
    _install_fake_torch(monkeypatch, cuda_available=cuda_available)
    _patch_minimal_doctor_env(monkeypatch)
    dummy_console = DummyConsole()
    monkeypatch.setattr(doctor_mod, "console", dummy_console, raising=False)

    def fake_find_spec(name: str):
        key = name.replace("-", "_")
        if key == "bitsandbytes":
            return SimpleNamespace(name=name) if bitsandbytes_present else None
        return SimpleNamespace(name=name)

    monkeypatch.setattr(
        doctor_mod.importlib.util, "find_spec", fake_find_spec, raising=False
    )
    return dummy_console


def _run_doctor_and_capture(dummy_console: DummyConsole) -> list[str]:
    with pytest.raises((SystemExit, typer.Exit)) as exc:
        doctor_mod.doctor_command()
    code = getattr(exc.value, "exit_code", getattr(exc.value, "code", None))
    assert code == 0
    return dummy_console.lines


def test_doctor_bitsandbytes_warns_when_gpu_missing(monkeypatch: pytest.MonkeyPatch):
    dummy = _setup_bitsandbytes_env(
        monkeypatch, cuda_available=False, bitsandbytes_present=True
    )
    monkeypatch.setattr(
        doctor_mod, "bitsandbytes_runtime_available", lambda: False, raising=False
    )
    lines = _run_doctor_and_capture(dummy)
    assert any("GPU not detected" in line for line in lines)


def test_doctor_bitsandbytes_install_hint_without_cuda(monkeypatch: pytest.MonkeyPatch):
    dummy = _setup_bitsandbytes_env(
        monkeypatch, cuda_available=False, bitsandbytes_present=False
    )
    lines = _run_doctor_and_capture(dummy)
    assert any("not installed" in line for line in lines)
    assert any("invarlock[gpu]" in line for line in lines)


def test_doctor_bitsandbytes_cuda_ready_marker(monkeypatch: pytest.MonkeyPatch):
    dummy = _setup_bitsandbytes_env(
        monkeypatch, cuda_available=True, bitsandbytes_present=True
    )
    monkeypatch.setattr(
        doctor_mod, "bitsandbytes_runtime_available", lambda: True, raising=False
    )
    lines = _run_doctor_and_capture(dummy)
    assert any("✅ bitsandbytes" in line for line in lines)


def test_doctor_bitsandbytes_host_runtime_ready_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
):
    dummy = _setup_bitsandbytes_env(
        monkeypatch, cuda_available=False, bitsandbytes_present=True
    )
    monkeypatch.setattr(
        doctor_mod,
        "bitsandbytes_runtime_available",
        lambda: True,
        raising=False,
    )
    lines = _run_doctor_and_capture(dummy)
    assert any("runtime available on this host" in line for line in lines)


def test_doctor_bitsandbytes_runtime_unavailable_with_cuda(
    monkeypatch: pytest.MonkeyPatch,
):
    dummy = _setup_bitsandbytes_env(
        monkeypatch, cuda_available=True, bitsandbytes_present=True
    )
    monkeypatch.setattr(
        doctor_mod, "bitsandbytes_runtime_available", lambda: False, raising=False
    )
    lines = _run_doctor_and_capture(dummy)
    assert any("runtime unavailable on this host" in line for line in lines)


def test_cross_check_reports_report_missing_paths_as_errors(tmp_path):
    findings, had_error = build_cross_check_findings(
        str(tmp_path / "missing_baseline.json"),
        str(tmp_path / "missing_subject.json"),
        cfg_metric_kind=None,
        strict=False,
        profile=None,
    )
    assert had_error is True
    assert len(findings) == 2
    assert all(finding.code == "D014" for finding in findings)


def test_cross_check_reports_reject_invalid_json_inputs(tmp_path):
    baseline = tmp_path / "baseline.json"
    subject = tmp_path / "subject.json"
    baseline.write_text("{not-json", encoding="utf-8")
    subject.write_text(json.dumps({}), encoding="utf-8")

    findings, had_error = build_cross_check_findings(
        str(baseline),
        str(subject),
        cfg_metric_kind=None,
        strict=False,
        profile=None,
    )

    assert had_error is True
    assert any(finding.code == "D014" for finding in findings)


def test_cross_check_reports_accepts_canonical_directories(tmp_path):
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    subject_dir = tmp_path / "subject"
    subject_dir.mkdir()
    (baseline_dir / "report.json").write_text(json.dumps({}), encoding="utf-8")
    (subject_dir / "evaluation.report.json").write_text(
        json.dumps({}),
        encoding="utf-8",
    )

    findings, had_error = build_cross_check_findings(
        str(baseline_dir),
        str(subject_dir),
        cfg_metric_kind=None,
        strict=False,
        profile=None,
    )

    assert had_error is False
    assert not findings


def test_cross_check_reports_reject_noncanonical_directory_inputs(tmp_path):
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    subject = tmp_path / "subject.json"
    subject.write_text(json.dumps({}), encoding="utf-8")
    (baseline_dir / "my_report.json").write_text(json.dumps({}), encoding="utf-8")

    findings, had_error = build_cross_check_findings(
        str(baseline_dir),
        str(subject),
        cfg_metric_kind=None,
        strict=False,
        profile=None,
    )

    assert had_error is True
    assert any(finding.code == "D014" for finding in findings)


def test_cross_check_reports_reject_ambiguous_directory_inputs(tmp_path):
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    subject_dir = tmp_path / "subject"
    subject_dir.mkdir()
    (baseline_dir / "report.json").write_text(json.dumps({}), encoding="utf-8")
    (baseline_dir / "evaluation.report.json").write_text(
        json.dumps({}),
        encoding="utf-8",
    )
    (subject_dir / "evaluation.report.json").write_text(
        json.dumps({}),
        encoding="utf-8",
    )

    findings, had_error = build_cross_check_findings(
        str(baseline_dir),
        str(subject_dir),
        cfg_metric_kind=None,
        strict=False,
        profile=None,
    )

    assert had_error is True
    assert any(finding.code == "D014" for finding in findings)


def test_doctor_json_mode_emits_findings(monkeypatch, capsys):
    _install_fake_torch(monkeypatch, cuda_available=False)
    _patch_minimal_doctor_env(monkeypatch)
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")
    with pytest.raises((SystemExit, typer.Exit)) as exc:
        doctor_mod.doctor_command(json_out=True)
    assert getattr(exc.value, "exit_code", getattr(exc.value, "code", None)) == 0
    output = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(output[-1])
    codes = {item["code"] for item in payload.get("findings", [])}
    assert "D013" in codes


def test_doctor_json_core_import_failure(monkeypatch, capsys):
    _install_fake_torch(monkeypatch, cuda_available=False)
    _patch_minimal_doctor_env(monkeypatch)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "invarlock.core.registry":
            raise ImportError("missing core")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises((SystemExit, typer.Exit)) as exc:
        doctor_mod.doctor_command(json_out=True)
    assert getattr(exc.value, "exit_code", getattr(exc.value, "code", None)) == 1
    capsys.readouterr()


def test_doctor_json_torch_import_failure(monkeypatch, capsys):
    _patch_minimal_doctor_env(monkeypatch)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("torch missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises((SystemExit, typer.Exit)) as exc:
        doctor_mod.doctor_command(json_out=True)
    assert getattr(exc.value, "exit_code", getattr(exc.value, "code", None)) == 1
    capsys.readouterr()


def test_doctor_torch_import_failure_console(monkeypatch):
    _patch_minimal_doctor_env(monkeypatch)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("torch missing")
        return real_import(name, globals, locals, fromlist, level)

    class CaptureConsole:
        def __init__(self):
            self.lines: list[str] = []

        def print(self, *args, **kwargs):
            self.lines.append(" ".join(str(arg) for arg in args))

    dummy_console = CaptureConsole()
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(doctor_mod, "console", dummy_console, raising=False)
    with pytest.raises((SystemExit, typer.Exit)) as exc:
        doctor_mod.doctor_command(json_out=False)
    assert getattr(exc.value, "exit_code", getattr(exc.value, "code", None)) == 1
    assert any("PyTorch not available" in line for line in dummy_console.lines)
