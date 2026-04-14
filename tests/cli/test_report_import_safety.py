from __future__ import annotations

import builtins
import importlib
import sys


def test_report_command_module_import_is_light_without_numpy(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "numpy":
            raise ModuleNotFoundError("numpy not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("INVARLOCK_LIGHT_IMPORT", raising=False)

    for mod in [
        "invarlock.cli.commands.report",
        "invarlock.reporting.report_contract",
        "invarlock.reporting.report_make",
        "invarlock.eval.primary_metric",
    ]:
        sys.modules.pop(mod, None)

    mod = importlib.import_module("invarlock.cli.commands.report")

    assert hasattr(mod, "report_app")
    assert "invarlock.reporting.report_contract" not in sys.modules
    assert "invarlock.reporting.report_make" not in sys.modules
    assert "invarlock.eval.primary_metric" not in sys.modules
