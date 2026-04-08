from __future__ import annotations

import importlib
import os
from importlib.metadata import PackageNotFoundError
from typing import Any, cast

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
    result = runner.invoke(cast(Any, app_mod).app, ["version"])
    assert result.exit_code == 0
    assert "InvarLock 0.0.0-test" in result.output
    # Should include schema version when available
    assert "schema=" in result.output


def test_light_import_does_not_force_plugin_policy_env(monkeypatch):
    """Test that LIGHT_IMPORT mode stays out of plugin policy env handling."""
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    monkeypatch.delenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", raising=False)
    app_mod = importlib.import_module("invarlock.cli.app")
    importlib.reload(app_mod)
    assert os.getenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS") is None


def test_version_fallbacks(monkeypatch):
    # Cause package metadata to raise so we hit fallback to __version__
    import importlib
    import importlib.metadata as im

    import invarlock

    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    app_mod = importlib.import_module("invarlock.cli.app")
    importlib.reload(app_mod)

    monkeypatch.setattr(
        im, "version", lambda *_: (_ for _ in ()).throw(PackageNotFoundError("boom"))
    )
    runner = CliRunner()
    result = runner.invoke(cast(Any, app_mod).app, ["version"])
    assert result.exit_code == 0
    assert "InvarLock" in result.output  # fallback ok

    # Remove __version__ and ensure we print unknown
    monkeypatch.delattr(invarlock, "__version__", raising=False)
    result2 = runner.invoke(cast(Any, app_mod).app, ["version"])
    assert result2.exit_code == 0
    assert "unknown" in result2.output.lower()
