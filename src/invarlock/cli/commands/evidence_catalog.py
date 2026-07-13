from __future__ import annotations

import json
from pathlib import Path

import typer

from invarlock.cli.constants import EVIDENCE_CATALOG_VALIDATE_FORMAT_VERSION
from invarlock.evidence_catalog import EvidenceCatalogError, load_evidence_catalog

evidence_catalog_app = typer.Typer(
    help="Validate versioned public evidence catalogs.", no_args_is_help=True
)


@evidence_catalog_app.callback()
def evidence_catalog_callback() -> None:
    """Evidence-catalog operations."""


@evidence_catalog_app.command(
    "validate",
    help="Validate one catalog and emit its immutable digest and exact entry set.",
)
def validate_command(
    catalog: str = typer.Argument(..., help="Path to the versioned catalog JSON file."),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Validate closed catalog structure without scheduling or materializing work."""

    try:
        loaded = load_evidence_catalog(Path(catalog))
        payload: dict[str, object] = {
            "format_version": EVIDENCE_CATALOG_VALIDATE_FORMAT_VERSION,
            "ok": True,
            "catalog_digest": loaded.digest,
            "entry_count": len(loaded.entries),
            "entry_ids": sorted(loaded.entries),
            "errors": [],
        }
        status = 0
    except EvidenceCatalogError:
        payload = {
            "format_version": EVIDENCE_CATALOG_VALIDATE_FORMAT_VERSION,
            "ok": False,
            "catalog_digest": None,
            "entry_count": 0,
            "entry_ids": [],
            "errors": ["catalog_invalid"],
        }
        status = 2

    if json_out:
        typer.echo(json.dumps(payload, allow_nan=False))
    elif payload["ok"]:
        typer.echo(
            "Evidence catalog valid: "
            f"{payload['entry_count']} entries, {payload['catalog_digest']}"
        )
    else:
        typer.echo("Evidence catalog is invalid", err=True)
    raise typer.Exit(status)


__all__ = ["evidence_catalog_app"]
