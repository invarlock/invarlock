import runpy
import sys
from types import ModuleType


def test_app_main_guard_runs_without_invoking_real_typer(monkeypatch):
    # Provide a fake invarlock.security.enforce_default_security to avoid side effects
    invarlock = ModuleType("invarlock")
    security = ModuleType("invarlock.security")

    def nosec():
        return None

    security.enforce_default_security = nosec
    security.enforce_network_policy = nosec
    security.network_policy_allows = lambda: True
    monkeypatch.setitem(sys.modules, "invarlock", invarlock)
    monkeypatch.setitem(sys.modules, "invarlock.security", security)

    # Monkeypatch Typer.__call__ to a no-op to avoid CLI exit/parse
    import typer

    def noop_call(self, *args, **kwargs):  # noqa: D401
        """Do nothing when the app is invoked."""
        return None

    monkeypatch.setattr(typer.Typer, "__call__", noop_call)

    module_globals = runpy.run_module("invarlock.cli.app", run_name="__main__")
    assert "app" in module_globals
