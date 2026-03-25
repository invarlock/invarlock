import os
import re
from unittest.mock import patch

import click
import typer
from click.termui import strip_ansi
from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import _load_lazy_subapp, app, version


def test_app_initialization():
    assert app.info.name == "invarlock"
    assert "evaluate model changes" in app.info.help.lower()
    assert app.info.no_args_is_help


def test_version_command_with_version():
    # Patch package metadata path to return a known version and assert
    # the console message includes it (schema suffix may be present).
    with patch("invarlock.cli.app.console") as mock_console:
        with patch("importlib.metadata.version", return_value="1.2.3"):
            version()
            assert mock_console.print.called
            args, _ = mock_console.print.call_args
            assert isinstance(args[0], str)
            assert args[0].startswith("InvarLock 1.2.3")


def test_version_command_no_version():
    with patch("invarlock.cli.app.console") as mock_console:
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: (
                ImportError("No module named 'invarlock'")
                if name == "invarlock"
                else __import__(name, *args, **kwargs)
            ),
        ):
            version()
            mock_console.print.assert_called()


def test_cli_help_lists_core_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    output = strip_ansi(result.stdout)
    assert "evaluate model changes" in output.lower()
    for command in ("evaluate", "report", "verify", "doctor", "advanced", "version"):
        assert re.search(rf"^\s*│\s+{re.escape(command)}\s", output, re.MULTILINE)
    for removed in ("run", "proof-pack", "policy", "plugins", "calibrate"):
        assert not re.search(rf"^\s*│\s+{re.escape(removed)}\s", output, re.MULTILINE)


def test_cli_version_flag_exits_through_root_callback():
    runner = CliRunner()
    with (
        patch("invarlock.cli.app.network_policy_allows", return_value=False),
        patch("invarlock.cli.app.enforce_default_security"),
        patch("invarlock.cli.app.enforce_network_policy"),
        patch("invarlock.cli.app._emit_version") as emit_version,
    ):
        result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    emit_version.assert_called_once_with()


def test_ordered_group_handles_advanced_and_unknown_lazy_subapps():
    command = typer.main.get_command(app)
    assert isinstance(command, click.Group)
    ctx = click.Context(command)

    assert _load_lazy_subapp(command, "advanced") is True
    assert command.get_command(ctx, "advanced") is not None
    assert _load_lazy_subapp(command, "_run") is False
    assert command.get_command(ctx, "_run") is None
    assert command.get_command(ctx, "definitely-missing-command") is None
