from __future__ import annotations

import types

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_plugins_adapters_compact_and_verbose_tables(monkeypatch):
    # No CUDA environment to exercise unsupported display
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    r = CliRunner().invoke(app, ["plugins", "adapters", "--show-unsupported"])
    assert r.exit_code == 0
    # Title of compact table should appear
    assert "Adapters  ready:" in r.stdout or "Adapters — ready:" in r.stdout

    rv = CliRunner().invoke(
        app, ["plugins", "adapters", "--verbose", "--show-unsupported"]
    )
    assert rv.exit_code == 0
    assert "Adapters (verbose)" in rv.stdout


def test_plugins_adapter_explain(monkeypatch):
    # Explain a core adapter
    r = CliRunner().invoke(app, ["plugins", "adapters", "--explain", "hf_gpt2"])
    assert r.exit_code == 0
    assert "Support" in r.stdout and "Module" in r.stdout
