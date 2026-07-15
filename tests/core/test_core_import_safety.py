from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def _block_torch_import(monkeypatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("INVARLOCK_LIGHT_IMPORT", raising=False)


def _import_root_without_torch(monkeypatch):
    _block_torch_import(monkeypatch)

    for mod in ["invarlock", "invarlock.adapters"]:
        monkeypatch.delitem(sys.modules, mod, raising=False)

    return importlib.import_module("invarlock")


def test_root_import_does_not_require_torch(monkeypatch):
    mod = _import_root_without_torch(monkeypatch)

    assert hasattr(mod, "__version__")
    assert hasattr(mod, "CFG")
    assert hasattr(mod, "Defaults")
    assert hasattr(mod, "get_default_config")
    assert not hasattr(mod, "adapters")


def test_root_import_safety_restores_preloaded_adapter_imports():
    root = importlib.import_module("invarlock")
    adapters = importlib.import_module("invarlock.adapters")
    capabilities = importlib.import_module("invarlock.adapters.capabilities")

    with pytest.MonkeyPatch.context() as monkeypatch:
        isolated_root = _import_root_without_torch(monkeypatch)
        assert isolated_root is not root
        assert not hasattr(isolated_root, "adapters")

    assert sys.modules["invarlock"] is root
    assert sys.modules["invarlock.adapters"] is adapters
    assert importlib.import_module("invarlock.adapters.capabilities") is capabilities
    assert adapters.capabilities is capabilities


def test_utils_import_and_memory_probe_do_not_require_torch(monkeypatch):
    _block_torch_import(monkeypatch)

    monkeypatch.delitem(sys.modules, "invarlock.utils", raising=False)

    mod = importlib.import_module("invarlock.utils")

    assert hasattr(mod, "get_memory_usage")
    memory = mod.get_memory_usage()
    assert "rss_mb" in memory
