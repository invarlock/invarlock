from __future__ import annotations

import json
from pathlib import Path

import typer

from invarlock.catalog_inputs import (
    build_evaluation_input_binding,
    materialize_catalog_input,
    prepare_catalog_preset,
)
from invarlock.cli.constants import (
    CATALOG_INPUT_MATERIALIZE_FORMAT_VERSION,
    CATALOG_INPUT_PREPARE_FORMAT_VERSION,
)
from invarlock.evidence_catalog import EvidenceCatalogError

inputs_app = typer.Typer(
    help="Materialize and prepare catalog-pinned evaluation inputs."
)


@inputs_app.command(
    "binding", help="Build a closed catalog binding for one evaluation run."
)
def binding_command(
    catalog: str = typer.Option(..., "--catalog"),
    lane: str = typer.Option(..., "--lane"),
    resolved_inputs: str = typer.Option(..., "--resolved-inputs"),
    preset: str = typer.Option(..., "--preset"),
    input_materialization: str | None = typer.Option(None, "--input-materialization"),
    out: str = typer.Option(..., "--out"),
) -> None:
    try:
        payload = build_evaluation_input_binding(
            catalog_path=Path(catalog),
            lane_id=lane,
            resolved_inputs_path=Path(resolved_inputs),
            preset_path=Path(preset),
            input_materialization_path=(
                Path(input_materialization) if input_materialization else None
            ),
        )
        output = Path(out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", errors="strict") as handle:
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    except (EvidenceCatalogError, FileExistsError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@inputs_app.command(
    "materialize", help="Materialize one pinned vision-text catalog input."
)
def materialize_command(
    catalog: str = typer.Option(..., "--catalog"),
    lane: str = typer.Option(..., "--lane"),
    out: str = typer.Option(..., "--out"),
    allow_network: bool = typer.Option(False, "--allow-network"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    if not allow_network:
        raise typer.BadParameter("--allow-network is required to fetch a catalog input")
    try:
        payload = materialize_catalog_input(
            catalog_path=Path(catalog), lane_id=lane, output_dir=Path(out)
        )
    except EvidenceCatalogError as exc:
        if json_out:
            typer.echo(
                json.dumps(
                    {
                        "format_version": CATALOG_INPUT_MATERIALIZE_FORMAT_VERSION,
                        "ok": False,
                        "errors": [str(exc)],
                    }
                )
            )
        raise typer.Exit(2) from exc
    if json_out:
        typer.echo(
            json.dumps(
                {"format_version": CATALOG_INPUT_MATERIALIZE_FORMAT_VERSION, **payload},
                allow_nan=False,
            )
        )


@inputs_app.command(
    "prepare", help="Write the resolved evaluator preset for one catalog lane."
)
def prepare_command(
    catalog: str = typer.Option(..., "--catalog"),
    lane: str = typer.Option(..., "--lane"),
    resolved_inputs: str = typer.Option(..., "--resolved-inputs"),
    preset: str = typer.Option(..., "--preset"),
    materialization_dir: str | None = typer.Option(None, "--materialization-dir"),
    out: str = typer.Option(..., "--out"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    try:
        payload = prepare_catalog_preset(
            catalog_path=Path(catalog),
            lane_id=lane,
            resolved_inputs_path=Path(resolved_inputs),
            preset_path=Path(preset),
            output_path=Path(out),
            materialization_dir=(
                Path(materialization_dir) if materialization_dir else None
            ),
        )
    except EvidenceCatalogError as exc:
        if json_out:
            typer.echo(
                json.dumps(
                    {
                        "format_version": CATALOG_INPUT_PREPARE_FORMAT_VERSION,
                        "ok": False,
                        "errors": [str(exc)],
                    }
                )
            )
        raise typer.Exit(2) from exc
    if json_out:
        typer.echo(
            json.dumps(
                {"format_version": CATALOG_INPUT_PREPARE_FORMAT_VERSION, **payload},
                allow_nan=False,
            )
        )
