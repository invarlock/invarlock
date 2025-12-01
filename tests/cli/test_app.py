import os
from unittest.mock import patch

from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import app, version


def test_app_initialization():
    assert app.info.name == "invarlock"
    assert "certify model changes" in app.info.help.lower()
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
            side_effect=lambda name, *args, **kwargs: ImportError(
                "No module named 'invarlock'"
            )
            if name == "invarlock"
            else __import__(name, *args, **kwargs),
        ):
            version()
            mock_console.print.assert_called()


def test_cli_help_lists_core_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    output = result.stdout
    assert "certify model changes" in output.lower()
    # Verify the new grouped layout mentions key groups
    for command in ("certify", "report", "run", "plugins", "doctor", "version"):
        assert command in output


def test_run_help_mentions_profile_and_retry_options():
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    output = result.stdout
    for option in ("--profile", "--until-pass", "--baseline"):
        assert option in output
