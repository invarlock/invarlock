"""Separate CLI for qualifying external evaluator exports."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from invarlock.evaluator_qualification import (
    EvaluatorQualificationError,
    qualify_evaluator_export,
)
from invarlock.security import enforce_default_security

app = typer.Typer(
    name="invarlock-qualify-evaluator",
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Qualify one digest-bound evaluator export for InvarLock runtime import, "
        "or retain it explicitly as observation-only."
    ),
)
console = Console()


@app.callback()
def _root() -> None:
    enforce_default_security()


@app.command(name="qualify")
def qualify(
    profile: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    schedule: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    export: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    raw_output: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Write the canonical qualification result without replacing a file.",
    ),
    require_verdict_authority: bool = typer.Option(
        False,
        "--require-verdict-authority",
        help="Exit nonzero when the valid result is observation-only.",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit the canonical machine-readable result.",
    ),
) -> None:
    """Qualify PROFILE + SCHEDULE + EXPORT + RAW_OUTPUT."""

    try:
        result = qualify_evaluator_export(
            profile_path=profile,
            schedule_path=schedule,
            export_path=export,
            raw_output_path=raw_output,
        )
        if output is not None:
            result.write(output)
    except EvaluatorQualificationError as exc:
        console.print(f"FAIL {exc}")
        raise typer.Exit(2) from exc
    if json_out:
        typer.echo(result.as_json(), nl=False)
    elif result.authority == "verdict_authority":
        console.print(
            f"PASS {result.profile_id}: {result.record_count} per-record results "
            "qualified for runtime import"
        )
    else:
        console.print(
            f"PASS {result.profile_id}: retained as observation-only "
            f"({', '.join(result.reason_codes)})"
        )
    if require_verdict_authority and result.authority != "verdict_authority":
        console.print("FAIL valid result is observation-only")
        raise typer.Exit(3)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["app", "main"]
