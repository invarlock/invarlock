from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from invarlock.cli.constants import (
    PROOF_PACK_BUILD_FORMAT_VERSION,
    PROOF_PACK_INSPECT_FORMAT_VERSION,
    PROOF_PACK_VERIFY_FORMAT_VERSION,
)
from invarlock.proof_pack import (
    _material_spec,
    build_proof_pack,
    inspect_proof_pack,
    verify_proof_pack,
)

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
    help="Verify a proof-pack manifest, checksums, attestation refs, and bundled reports.",
)
def verify_command(
    pack: str = typer.Argument(..., help="Path to the proof-pack directory."),
    json_out: bool = typer.Option(
        False, "--json", help="Emit machine-readable verification JSON."
    ),
    json_file: str | None = typer.Option(
        None,
        "--json-out",
        help="Write nested report verification JSON to FILE (must be outside the pack).",
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
        help="Execution profile to use for bundled report verification (dev|ci|release).",
    ),
) -> None:
    result = verify_proof_pack(
        Path(pack),
        json_out_path=Path(json_file) if json_file else None,
        skip_verify=skip_verify,
        strict=strict,
        profile=profile,
    )
    payload = {
        "format_version": PROOF_PACK_VERIFY_FORMAT_VERSION,
        **result.payload,
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
    raise typer.Exit(result.status.value)


@proof_pack_app.command(
    "inspect",
    help="Inspect a proof-pack summary without running nested report verification.",
)
def inspect_command(
    pack: str = typer.Argument(..., help="Path to the proof-pack directory."),
    json_out: bool = typer.Option(
        False, "--json", help="Emit machine-readable inspection JSON."
    ),
) -> None:
    result = inspect_proof_pack(Path(pack))
    payload = {
        "format_version": PROOF_PACK_INSPECT_FORMAT_VERSION,
        **result.payload,
    }

    if json_out:
        typer.echo(json.dumps(payload))
    else:
        if payload["ok"]:
            console.print("[green]Proof pack inspected[/green]")
            for issue in payload["issues"]:
                console.print(f"[yellow]ISSUE:[/yellow] {issue}")
        else:
            for issue in payload["issues"]:
                console.print(f"[red]ERROR:[/red] {issue}")
    raise typer.Exit(result.status.value)


@proof_pack_app.command(
    "build",
    help="Assemble a proof pack from existing verdict, metadata, and report artifacts.",
)
def build_command(
    out: str = typer.Argument(..., help="Output directory for the proof-pack."),
    final_verdict: str = typer.Option(
        ...,
        "--final-verdict",
        help="Path to the final verdict JSON to package as the proof-pack subject.",
    ),
    reports: list[str] = typer.Option(
        [],
        "--report",
        help=(
            "Path to an explicit evaluation.report.json file to verify and "
            "package (requires adjacent runtime.manifest.json)."
        ),
    ),
    source_repo: str | None = typer.Option(
        None,
        "--source-repo",
        help="Optional source_repo.json provenance sidecar.",
    ),
    environment: str | None = typer.Option(
        None,
        "--environment",
        help="Optional environment.json provenance sidecar.",
    ),
    materials: list[str] = typer.Option(
        [],
        "--material",
        help="Optional metadata sidecar as NAME=PATH. Repeat to include multiple.",
    ),
    readme: str | None = typer.Option(
        None,
        "--readme",
        help="Optional README markdown file to copy into the pack root.",
    ),
    profile: str = typer.Option(
        "dev",
        "--profile",
        help="Execution profile to use for report pre-verification (dev|ci|release).",
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit machine-readable build JSON."
    ),
) -> None:
    errors: list[str] = []
    material_specs: list[tuple[str, Path]] = []
    for material in materials:
        spec = _material_spec(material)
        if spec is None:
            errors.append(f"Invalid --material value {material!r}; expected NAME=PATH.")
            continue
        material_specs.append(spec)

    if errors:
        payload = {
            "format_version": PROOF_PACK_BUILD_FORMAT_VERSION,
            "pack": str(Path(out)),
            "ok": False,
            "warnings": [],
            "errors": errors,
            "reports": {"total": 0},
            "verify": None,
            "files": None,
        }
        if json_out:
            typer.echo(json.dumps(payload))
        else:
            for error in errors:
                console.print(f"[red]ERROR:[/red] {error}")
        raise typer.Exit(2)

    result = build_proof_pack(
        Path(out),
        final_verdict_path=Path(final_verdict),
        report_paths=[Path(path) for path in reports],
        source_repo_path=Path(source_repo) if source_repo else None,
        environment_path=Path(environment) if environment else None,
        material_specs=material_specs,
        readme_path=Path(readme) if readme else None,
        profile=profile,
    )
    payload = {
        "format_version": PROOF_PACK_BUILD_FORMAT_VERSION,
        **result.payload,
    }

    if json_out:
        typer.echo(json.dumps(payload))
    else:
        for warning in payload["warnings"]:
            console.print(f"[yellow]WARNING:[/yellow] {warning}")
        if payload["ok"]:
            console.print("[green]Proof pack built[/green]")
        else:
            for error in payload["errors"]:
                console.print(f"[red]ERROR:[/red] {error}")
    raise typer.Exit(result.status.value)
