from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from invarlock.cli.constants import PROOF_PACK_VERIFY_FORMAT_VERSION
from invarlock.proof_pack import verify_proof_pack

console = Console()
proof_pack_app = typer.Typer(
    help="Build and verify proof-pack evidence artifacts.",
    no_args_is_help=True,
)


@proof_pack_app.callback()
def proof_pack_callback() -> None:
    """Proof-pack operations."""


@proof_pack_app.command(
    "verify",
    help="Verify a proof-pack manifest, checksums, attestation refs, and bundled certs.",
)
def verify_command(
    pack: str = typer.Argument(..., help="Path to the proof-pack directory."),
    json_out: bool = typer.Option(
        False, "--json", help="Emit machine-readable verification JSON."
    ),
    json_file: str | None = typer.Option(
        None,
        "--json-out",
        help="Write nested cert verification JSON to FILE (must be outside the pack).",
    ),
    skip_verify: bool = typer.Option(
        False, "--skip-verify", help="Skip bundled evaluation.report.json verification."
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail closed on missing/invalid signatures and extra unhashed files.",
    ),
    profile: str = typer.Option(
        "dev",
        "--profile",
        help="Execution profile to use for bundled cert verification (dev|ci|release).",
    ),
) -> None:
    payload, exit_code = verify_proof_pack(
        Path(pack),
        json_out_path=Path(json_file) if json_file else None,
        skip_verify=skip_verify,
        strict=strict,
        profile=profile,
    )
    payload = {
        "format_version": PROOF_PACK_VERIFY_FORMAT_VERSION,
        **payload,
    }

    if json_out:
        typer.echo(json.dumps(payload))
    else:
        for warning in payload["warnings"]:
            console.print(f"[yellow]WARNING:[/yellow] {warning}")
        if payload["ok"]:
            console.print("[green]Proof pack verified[/green]")
        else:
            for error in payload["errors"]:
                console.print(f"[red]ERROR:[/red] {error}")
    raise typer.Exit(exit_code)
