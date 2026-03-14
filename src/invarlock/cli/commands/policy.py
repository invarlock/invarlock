from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console

from invarlock.cli.constants import POLICY_PACK_VERIFY_FORMAT_VERSION
from invarlock.policy_pack import (
    build_policy_pack,
    load_policy_pack,
    verify_policy_pack,
    write_policy_pack,
)

console = Console()
policy_app = typer.Typer(help="Build and verify policy-pack artifacts.")


def _load_structured_input(path: str | None) -> Any:
    if not path:
        return None
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(raw)
    return json.loads(raw)


@policy_app.command(
    "build",
    help="Build a policy-pack artifact from resolved policy and ordered overrides.",
)
def build_command(
    resolved_policy: str = typer.Option(
        ..., "--resolved-policy", help="Path to resolved policy JSON/YAML."
    ),
    out: str = typer.Option(..., "--out", help="Output path for policy-pack JSON."),
    tier: str = typer.Option("balanced", "--tier", help="Policy tier label."),
    overrides: str | None = typer.Option(
        None, "--overrides", help="Optional overrides JSON/YAML."
    ),
    compatibility: str | None = typer.Option(
        None, "--compatibility", help="Optional compatibility JSON/YAML."
    ),
    owner: str | None = typer.Option(None, "--owner", help="Approval owner."),
    change_ticket: str | None = typer.Option(
        None, "--change-ticket", help="Change ticket or PR id."
    ),
    rationale: str | None = typer.Option(
        None, "--rationale", help="Short change rationale."
    ),
    effective_date: str | None = typer.Option(
        None, "--effective-date", help="Effective date string."
    ),
    signature: str | None = typer.Option(
        None, "--signature", help="Optional approval signature reference."
    ),
) -> None:
    resolved_payload = _load_structured_input(resolved_policy)
    if not isinstance(resolved_payload, dict):
        raise typer.BadParameter("--resolved-policy must decode to an object")

    overrides_payload = _load_structured_input(overrides)
    compatibility_payload = _load_structured_input(compatibility)
    approval = {
        key: value
        for key, value in {
            "owner": owner,
            "change_ticket": change_ticket,
            "rationale": rationale,
            "effective_date": effective_date,
            "signature": signature,
        }.items()
        if value
    }
    pack = build_policy_pack(
        tier=tier,
        resolved_policy=resolved_payload,
        overrides=overrides_payload if overrides_payload is not None else [],
        compatibility=compatibility_payload
        if isinstance(compatibility_payload, dict)
        else None,
        approval=approval,
    )
    out_path = Path(out)
    write_policy_pack(out_path, pack)
    console.print(f"Wrote policy pack: {out_path}")


@policy_app.command(
    "verify",
    help="Verify a policy-pack artifact schema and policy_digest.",
)
def verify_command(
    pack: str = typer.Argument(..., help="Policy-pack JSON/YAML file."),
    json_out: bool = typer.Option(
        False, "--json", help="Emit machine-readable verification JSON."
    ),
) -> None:
    payload = load_policy_pack(Path(pack))
    errors = verify_policy_pack(payload)
    exit_code = 0 if not errors else 2
    if json_out:
        result = {
            "format_version": POLICY_PACK_VERIFY_FORMAT_VERSION,
            "pack": pack,
            "ok": not errors,
            "errors": errors,
            "resolution": {"exit_code": exit_code},
        }
        typer.echo(json.dumps(result))
    else:
        if errors:
            for error in errors:
                console.print(f"[red]ERROR:[/red] {error}")
        else:
            console.print("[green]Policy pack verified[/green]")
    raise typer.Exit(exit_code)
