from __future__ import annotations

import click
import typer
from typer.core import TyperGroup


class AdvancedGroup(TyperGroup):
    def list_commands(self, ctx: click.Context) -> list[str]:
        return ["proof-pack", "policy", "plugins", "calibrate"]

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        if _load_advanced_subapp(self, cmd_name):
            return super().get_command(ctx, cmd_name)
        return None


advanced_app = typer.Typer(
    help=(
        "Advanced and maintenance workflows. "
        "These commands are intentionally outside the core evaluate/verify/report path."
    ),
    no_args_is_help=True,
    cls=AdvancedGroup,
)


@advanced_app.callback(invoke_without_command=True)
def _advanced_root() -> None:
    """Advanced command namespace."""


def _missing_dependency_subapp(name: str, missing: str) -> typer.Typer:
    subapp = typer.Typer(help=f"{name} requires optional dependency {missing!r}.")

    @subapp.callback(invoke_without_command=True)
    def _missing() -> None:
        raise click.UsageError(
            f"`invarlock advanced {name}` requires optional dependency {missing!r}."
        )

    return subapp


def _load_advanced_subapp(group: TyperGroup, name: str) -> bool:
    def _register(sub_name: str, subapp: typer.Typer) -> bool:
        command = typer.main.get_command(subapp)
        command.name = sub_name
        group.add_command(command, name=sub_name)
        return True

    if name == "proof-pack":
        from .proof_pack import proof_pack_app

        return _register(name, proof_pack_app)
    if name == "policy":
        from .policy import policy_app

        return _register(name, policy_app)
    if name == "plugins":
        from .plugins import plugins_app

        return _register(name, plugins_app)
    if name == "calibrate":
        try:
            from .calibrate import calibrate_app
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised in venvs
            missing = getattr(exc, "name", "") or "optional runtime"
            return _register(name, _missing_dependency_subapp(name, missing))
        return _register(name, calibrate_app)
    return False
