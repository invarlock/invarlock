from __future__ import annotations

import builtins
import importlib
import sys


def test_alerting_import_survives_missing_requests(monkeypatch):
    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "requests":
            raise ModuleNotFoundError("No module named 'requests'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    sys.modules.pop("invarlock.observability.alerting", None)

    alerting = importlib.import_module("invarlock.observability.alerting")
    assert hasattr(alerting, "AlertManager")
    assert hasattr(alerting.requests, "post")

    # Restore the normal import behavior and reload for test isolation.
    monkeypatch.setattr(builtins, "__import__", original_import)
    importlib.reload(alerting)
