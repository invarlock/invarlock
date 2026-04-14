from __future__ import annotations

import builtins
import importlib
import sys


def _block_torch_import(monkeypatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("INVARLOCK_LIGHT_IMPORT", raising=False)


def test_root_import_does_not_require_torch(monkeypatch):
    _block_torch_import(monkeypatch)

    for mod in ["invarlock", "invarlock.adapters"]:
        sys.modules.pop(mod, None)

    mod = importlib.import_module("invarlock")

    assert hasattr(mod, "__version__")
    assert hasattr(mod, "CFG")
    assert hasattr(mod, "Defaults")
    assert hasattr(mod, "get_default_config")
    assert not hasattr(mod, "adapters")


def test_utils_import_and_memory_probe_do_not_require_torch(monkeypatch):
    _block_torch_import(monkeypatch)

    sys.modules.pop("invarlock.utils", None)

    mod = importlib.import_module("invarlock.utils")

    assert hasattr(mod, "get_memory_usage")
    memory = mod.get_memory_usage()
    assert "rss_mb" in memory
