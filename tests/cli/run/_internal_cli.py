from __future__ import annotations

import typer

from invarlock.cli.commands.run import run_command

internal_run_app = typer.Typer(add_completion=False)


@internal_run_app.callback()
def _internal_root() -> None:
    """Internal run test harness."""


internal_run_app.command(name="run")(run_command)
