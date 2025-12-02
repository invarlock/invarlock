from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest

import invarlock.cli.device as device_mod
from invarlock.cli.device import is_device_available, resolve_device


def test_is_device_available_cpu_true():
    assert is_device_available("cpu") is True


def test_resolve_device_invalid_raises():
    with pytest.raises(RuntimeError):
        resolve_device("notadev")


def test_resolve_device_auto_returns_supported_choice():
    # Should resolve to one of the supported identifiers depending on host
    choice = resolve_device("auto")
    assert choice in {"cpu", "mps", "cuda:0"}


def test_resolve_device_explicit_not_available(monkeypatch):
    monkeypatch.setattr(device_mod, "is_device_available", lambda d: False)
    with pytest.raises(RuntimeError):
        device_mod.resolve_device("cuda")


def test_resolve_device_auto_prefers_cuda_then_mps_then_cpu(monkeypatch):
    calls = {"cuda": True, "mps": False}
    monkeypatch.setattr(
        device_mod, "is_device_available", lambda d: calls.get(d, False)
    )
    assert device_mod.resolve_device("auto") == "cuda:0"
    calls["cuda"] = False
    calls["mps"] = True
    assert device_mod.resolve_device("auto") == "mps"
    calls["mps"] = False
    assert device_mod.resolve_device("auto") == "cpu"


def test_get_device_info_cpu_only(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            current_device=lambda: 0,
            get_device_name=lambda idx: "GPU-0",
            get_device_properties=lambda idx: SimpleNamespace(total_memory=0),
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(device_mod, "torch", fake_torch, raising=False)
    monkeypatch.setattr(device_mod, "resolve_device", lambda _: "cpu")
    info = device_mod.get_device_info()
    assert info["cpu"]["available"] is True
    assert info["auto_selected"] == "cpu"


def test_validate_device_for_config(monkeypatch):
    monkeypatch.setattr(device_mod, "is_device_available", lambda d: True)
    ok, msg = device_mod.validate_device_for_config("cpu", {"required_device": "cuda"})
    assert ok is False and "requires device" in msg
    ok2, msg2 = device_mod.validate_device_for_config("cuda", None)
    assert ok2 is True and msg2 == ""


def _install_torch_stub(monkeypatch, *, cuda_available=True, mps_available=True):
    torch_stub = ModuleType("torch")
    cuda_props = SimpleNamespace(name="StubGPU", total_memory=8 * 1e9)
    torch_stub.cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: 2,
        get_device_properties=lambda _idx: cuda_props,
    )
    torch_stub.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: mps_available)
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    return torch_stub


def test_is_device_available_handles_import_failure(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise RuntimeError("torch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert device_mod.is_device_available("cuda") is False


def test_is_device_available_cuda_and_mps_paths(monkeypatch):
    torch_stub = _install_torch_stub(monkeypatch)
    assert device_mod.is_device_available("cuda:3") is True
    assert device_mod.is_device_available("mps") is True
    torch_stub.cuda.is_available = lambda: False  # type: ignore[attr-defined]
    assert device_mod.is_device_available("cuda") is False


def test_validate_device_for_config_rejects_unknown():
    ok, message = device_mod.validate_device_for_config("tpu")
    assert ok is False
    assert "Unsupported device" in message


def test_get_device_info_populates_cuda_details(monkeypatch):
    _install_torch_stub(monkeypatch)
    monkeypatch.setattr(device_mod, "resolve_device", lambda _: "cuda:0")
    info = device_mod.get_device_info()
    assert info["cuda"]["available"] is True
    assert info["cuda"]["device_name"] == "StubGPU"
    assert info["cuda"]["device_count"] == 2
    assert info["mps"]["available"] is True
    assert info["auto_selected"] == "cuda:0"
