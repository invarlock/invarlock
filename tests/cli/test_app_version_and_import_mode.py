from __future__ import annotations

import importlib
import os

from typer.testing import CliRunner


def test_version_outputs_schema(monkeypatch):
    # Force package metadata path
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    # Fresh import to apply LIGHT_IMPORT side-effect
    app_mod = importlib.import_module("invarlock.cli.app")
    importlib.reload(app_mod)

    # Mock package version
    import importlib.metadata as im

    monkeypatch.setattr(im, "version", lambda _: "0.0.0-test")

    runner = CliRunner()
    result = runner.invoke(app_mod.app, ["version"])  # type: ignore[attr-defined]
    assert result.exit_code == 0
    assert "InvarLock 0.0.0-test" in result.output
    # Should include schema version when available
    assert "schema=" in result.output


def test_light_import_sets_disable_discovery(monkeypatch):
    """Test that LIGHT_IMPORT mode no longer sets DISABLE_PLUGIN_DISCOVERY.

    The behavior was intentionally changed so that INVARLOCK_LIGHT_IMPORT=1
    no longer auto-sets INVARLOCK_DISABLE_PLUGIN_DISCOVERY. Individual commands
    may gate discovery based on their own flags.
    """
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    # Ensure var not already set
    monkeypatch.delenv("INVARLOCK_DISABLE_PLUGIN_DISCOVERY", raising=False)
    app_mod = importlib.import_module("invarlock.cli.app")
    importlib.reload(app_mod)
    # LIGHT_IMPORT no longer forces DISABLE_PLUGIN_DISCOVERY
    assert os.getenv("INVARLOCK_DISABLE_PLUGIN_DISCOVERY") is None


def test_version_fallbacks(monkeypatch):
    # Cause package metadata to raise so we hit fallback to __version__
    import importlib
    import importlib.metadata as im

    import invarlock

    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    app_mod = importlib.import_module("invarlock.cli.app")
    importlib.reload(app_mod)

    monkeypatch.setattr(
        im, "version", lambda *_: (_ for _ in ()).throw(Exception("boom"))
    )
    runner = CliRunner()
    result = runner.invoke(app_mod.app, ["version"])  # type: ignore[attr-defined]
    assert result.exit_code == 0
    assert "InvarLock" in result.output  # fallback ok

    # Remove __version__ and ensure we print unknown
    monkeypatch.delattr(invarlock, "__version__", raising=False)
    result2 = runner.invoke(app_mod.app, ["version"])  # type: ignore[attr-defined]
    assert result2.exit_code == 0
    assert "unknown" in result2.output.lower()
