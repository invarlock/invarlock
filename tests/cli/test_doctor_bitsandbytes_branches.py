from __future__ import annotations

import builtins
import io
import json
from types import SimpleNamespace

import pytest
import typer
from rich.console import Console

from invarlock.cli.commands import doctor as doctor_mod
from tests.cli.test_doctor_additional import (
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
    lines = _run_doctor_and_capture(dummy)
    assert any("✅ bitsandbytes" in line for line in lines)


def test_doctor_bitsandbytes_cpu_only_warning(monkeypatch: pytest.MonkeyPatch):
    dummy = _setup_bitsandbytes_env(
        monkeypatch, cuda_available=True, bitsandbytes_present=True
    )

    class ExplodingCtx:
        def __enter__(self):
            raise RuntimeError("cpu-only build detected")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        doctor_mod,
        "warnings",
        SimpleNamespace(
            catch_warnings=lambda: ExplodingCtx(), simplefilter=lambda *a, **k: None
        ),
        raising=False,
    )
    lines = _run_doctor_and_capture(dummy)
    assert any("CPU-only build detected" in line for line in lines)
    assert any("Reinstall with: pip install 'invarlock[gpu]'" in line for line in lines)


def test_cross_check_reports_ignore_missing_paths(tmp_path):
    adds: list[tuple] = []
    console = Console(file=io.StringIO())
    result = doctor_mod._cross_check_reports(
        str(tmp_path / "missing_baseline.json"),
        str(tmp_path / "missing_subject.json"),
        cfg_metric_kind=None,
        strict=False,
        profile=None,
        json_out=True,
        console=console,
        add_fn=lambda *args, **kwargs: adds.append((args, kwargs)),
    )
    assert result is False
    assert not adds


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
    with pytest.raises(SystemExit) as exc:
        doctor_mod.doctor_command(json_out=False)
    assert exc.value.code == 1
    assert any("PyTorch not available" in line for line in dummy_console.lines)
