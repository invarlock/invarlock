from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from invarlock.cli import output as cli_output
from invarlock.cli.constants import POLICY_PACK_VERIFY_FORMAT_VERSION
from invarlock.policy_pack import (
    build_policy_pack,
    load_policy_input,
    load_policy_pack,
    verify_policy_pack,
    write_policy_pack,
)

console = cli_output.make_console()
policy_app = typer.Typer(help="Build and verify policy-pack artifacts.")


def _load_structured_input(path: str | None) -> Any:
    if not path:
        return None
    return load_policy_input(Path(path))


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
    emit = cli_output.make_command_event_emitter(console)
    try:
        resolved_payload = _load_structured_input(resolved_policy)
        overrides_payload = _load_structured_input(overrides)
        compatibility_payload = _load_structured_input(compatibility)
    except (OSError, UnicodeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not isinstance(resolved_payload, dict):
        raise typer.BadParameter("--resolved-policy must decode to an object")
    if compatibility_payload is not None and not isinstance(
        compatibility_payload, dict
    ):
        raise typer.BadParameter("--compatibility must decode to an object")
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
    try:
        pack = build_policy_pack(
            tier=tier,
            resolved_policy=resolved_payload,
            overrides=overrides_payload if overrides_payload is not None else [],
            compatibility=compatibility_payload,
            approval=approval,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    out_path = Path(out)
    write_policy_pack(out_path, pack)
    emit("PASS", "Wrote policy pack")
    cli_output.print_command_detail(console, f"Output: {out_path}")


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
    emit = cli_output.make_command_event_emitter(console)
    try:
        payload = load_policy_pack(Path(pack))
    except (OSError, UnicodeError, ValueError) as exc:
        errors = [f"policy pack is not valid JSON/YAML: {exc}"]
    else:
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
        typer.echo(json.dumps(result, allow_nan=False))
    else:
        if errors:
            emit("FAIL", "Policy pack verification failed")
            for error in errors:
                cli_output.print_command_detail(console, error, console_style="red")
        else:
            emit("PASS", "Policy pack verified")
    raise typer.Exit(exit_code)
