from __future__ import annotations

from typer.testing import CliRunner


def _load_app(monkeypatch):
    # Ensure lightweight import path and skip heavy discovery
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    monkeypatch.setenv("INVARLOCK_DISABLE_PLUGIN_DISCOVERY", "1")
    from invarlock.cli.app import app

    return app


def test_run_help_exposes_typed_options(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    res = runner.invoke(app, ["run", "--help"])
    assert res.exit_code == 0, res.output
    out = res.stdout
    # Regression guard: no raw ARGS/KWARGS placeholder
    assert "ARGS KWARGS" not in out
    # Must expose config and common options
    assert "--config" in out or "-c" in out
    assert "--profile" in out
    assert "--out" in out


def test_certify_help_exposes_baseline_and_subject(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    res = runner.invoke(app, ["certify", "--help"])
    assert res.exit_code == 0, res.output
    out = res.stdout
    assert "--baseline" in out or "--source" in out
    assert "--subject" in out or "--edited" in out
    assert "--preset" in out


def test_doctor_help_is_typed(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    res = runner.invoke(app, ["doctor", "--help"])
    assert res.exit_code == 0, res.output
    out = res.stdout
    assert "ARGS KWARGS" not in out
    # Provide a config option
    assert "--config" in out


def test_groups_help_list_subcommands(monkeypatch):
    app = _load_app(monkeypatch)
    runner = CliRunner()
    for cmd, expected in (
        ("report", ["verify", "explain", "html", "validate"]),
        ("plugins", ["list", "guards", "edits", "install", "uninstall"]),
    ):
        res = runner.invoke(app, [cmd, "--help"])
        assert res.exit_code == 0, f"help failed for {cmd}: {res.output}"
        out = res.stdout
        for token in expected:
            assert token in out
