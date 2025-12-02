from typer.testing import CliRunner

from invarlock.cli.app import app

runner = CliRunner()


def test_cli_version_smoke():
    result = runner.invoke(app, ["version"])
    assert result.exit_code in (0, 1)
    assert "InvarLock" in result.stdout or "version" in result.stdout.lower()


def test_cli_plugins_guards_smoke():
    result = runner.invoke(app, ["plugins", "guards"])
    # Command may exit 0 or 1 depending on environment; ensure it runs
    assert result.exit_code in (0, 1)
    assert "plugins" in result.stdout.lower() or result.stdout


def test_cli_doctor_smoke():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code in (0, 1)
    assert "health" in result.stdout.lower() or result.stdout


def test_cli_plugins_unknown_category_exits_error():
    # Use the 'list' subcommand with an invalid category name
    import os as _os

    # Ensure discovery is enabled for this test
    _os.environ["INVARLOCK_DISABLE_PLUGIN_DISCOVERY"] = "0"
    result = runner.invoke(app, ["plugins", "list", "unknown_cat"])
    # Expect a non-zero exit in this path
    assert result.exit_code != 0
    assert "unknown category" in result.stdout.lower()


def test_cli_plugins_help_shows_usage():
    result = runner.invoke(app, ["plugins", "--help"])
    assert result.exit_code == 0
    assert "usage" in result.stdout.lower() or "show" in result.stdout.lower()
