import builtins
import importlib
import importlib.metadata
import os
import re
from unittest.mock import patch

import click
import typer
from click.termui import strip_ansi
from click.testing import CliRunner as ClickRunner
from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import _load_advanced_subapp, _load_lazy_subapp, app, version

app_mod = importlib.import_module("invarlock.cli.app")


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
    original_import = builtins.__import__

    with patch("invarlock.cli.app.console") as mock_console:
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: (
                ImportError("No module named 'invarlock'")
                if name == "invarlock"
                else original_import(name, *args, **kwargs)
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
    for removed in ("run", "evidence-pack", "policy", "plugins", "calibrate"):
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


def test_evaluate_cli_forwards_execution_mode(monkeypatch):
    seen: dict[str, object] = {}

    def fake_evaluate_command(**kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(
        "invarlock.cli.commands.evaluate.evaluate_command",
        fake_evaluate_command,
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--baseline",
            "baseline",
            "--subject",
            "subject",
            "--execution-mode",
            "host",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["execution_mode"] == "host"


def test_ordered_group_handles_advanced_and_unknown_lazy_subapps():
    command = typer.main.get_command(app)
    assert isinstance(command, click.Group)
    ctx = click.Context(command)

    assert _load_lazy_subapp(command, "advanced") is True
    assert command.get_command(ctx, "advanced") is not None
    assert _load_lazy_subapp(command, "_run") is False
    assert command.get_command(ctx, "_run") is None
    assert command.get_command(ctx, "definitely-missing-command") is None


def test_advanced_group_handles_registered_runtime_and_unknown_commands():
    command = typer.main.get_command(app)
    assert isinstance(command, click.Group)
    root_ctx = click.Context(command)
    assert _load_lazy_subapp(command, "advanced") is True
    advanced = command.get_command(root_ctx, "advanced")
    assert isinstance(advanced, click.Group)
    advanced_ctx = click.Context(advanced)

    assert _load_advanced_subapp(advanced, "runtime-verify") is True
    assert advanced.get_command(advanced_ctx, "runtime-verify") is not None
    assert _load_advanced_subapp(advanced, "definitely-missing") is False
    assert advanced.get_command(advanced_ctx, "definitely-missing") is None


def test_missing_dependency_subapp_raises_usage_error():
    missing = app_mod._missing_dependency_subapp("calibrate", "demo-lib")

    result = ClickRunner().invoke(typer.main.get_command(missing), [])

    assert result.exit_code != 0
    assert "demo-lib" in result.output


def test_package_version_import_failure_and_module_version_failure(monkeypatch):
    original_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):  # noqa: ANN001
        fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist", ())
        if name == "importlib.metadata":
            raise ImportError("metadata unavailable")
        if name == "invarlock" and fromlist:
            raise ImportError("version unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    assert app_mod._resolve_package_version() is None
    assert app_mod._resolve_module_version() is None


def test_package_version_not_found_falls_back_to_module_version(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(
            importlib.metadata.PackageNotFoundError("missing")
        ),
    )

    assert app_mod._resolve_package_version() is None
    with patch("invarlock.cli.app.console") as mock_console:
        app_mod._emit_version()

    args, _ = mock_console.print.call_args
    assert isinstance(args[0], str)
    assert args[0].startswith("InvarLock ")
