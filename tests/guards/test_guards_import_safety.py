from __future__ import annotations

import builtins
import importlib
import sys


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
