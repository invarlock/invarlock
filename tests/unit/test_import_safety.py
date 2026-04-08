from __future__ import annotations

import builtins
import importlib
import sys


def test_root_import_does_not_require_torch(monkeypatch):
    # Simulate an environment where torch is not installed.
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("INVARLOCK_LIGHT_IMPORT", raising=False)

    # Ensure a clean import state for the package root.
    for mod in ["invarlock", "invarlock.adapters"]:
        sys.modules.pop(mod, None)

    mod = importlib.import_module("invarlock")

    # Core public API should be available without importing adapters/torch.
    assert hasattr(mod, "__version__")
    assert hasattr(mod, "CFG")
    assert hasattr(mod, "Defaults")
    assert hasattr(mod, "get_default_config")

    # Top-level package should not auto-expose adapters.
    assert not hasattr(mod, "adapters")


def test_guard_helper_import_does_not_require_torch(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("INVARLOCK_LIGHT_IMPORT", raising=False)

    for mod in [
        "invarlock.guards",
        "invarlock.guards.invariants",
        "invarlock.guards.rmt",
        "invarlock.guards.spectral",
        "invarlock.guards.variance",
        "invarlock.guards.spectral_results",
    ]:
        sys.modules.pop(mod, None)

    mod = importlib.import_module("invarlock.guards.spectral_results")

    assert hasattr(mod, "build_spectral_finalize_metrics")


def test_utils_import_and_memory_probe_do_not_require_torch(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("INVARLOCK_LIGHT_IMPORT", raising=False)

    sys.modules.pop("invarlock.utils", None)

    mod = importlib.import_module("invarlock.utils")

    assert hasattr(mod, "get_memory_usage")
    memory = mod.get_memory_usage()
    assert "rss_mb" in memory
