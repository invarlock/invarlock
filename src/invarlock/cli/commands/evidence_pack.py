from __future__ import annotations

import json
from pathlib import Path

import typer

from invarlock.cli import output as cli_output
from invarlock.cli.constants import (
    EVIDENCE_PACK_INSPECT_FORMAT_VERSION,
    EVIDENCE_PACK_SET_VERIFY_FORMAT_VERSION,
    EVIDENCE_PACK_VERIFY_FORMAT_VERSION,
)
from invarlock.evidence_catalog import EvidenceCatalogError
from invarlock.evidence_catalog_contracts.set_verifier import (
    verify_evidence_pack_set,
)
from invarlock.evidence_pack import (
    inspect_evidence_pack,
    verify_evidence_pack,
)

console = cli_output.make_console()
evidence_pack_app = typer.Typer(
    help="Inspect and verify evidence-pack artifacts.",
    no_args_is_help=True,
)


@evidence_pack_app.callback()
def evidence_pack_callback() -> None:
    """Evidence-pack operations."""


@evidence_pack_app.command(
    "verify",
    help="Verify an evidence-pack manifest, checksums, signed provenance refs, and bundled reports.",
)
def verify_command(
    pack: str = typer.Argument(..., help="Path to the evidence-pack directory."),
    json_out: bool = typer.Option(
        False, "--json", help="Emit machine-readable verification JSON."
    ),
    json_file: str | None = typer.Option(
        None,
        "--json-out",
        help="Write nested report verification JSON to FILE (must be outside the pack).",
    ),
    skip_verify: bool = typer.Option(
        False,
        "--skip-verify",
        help=(
            "Run integrity-only diagnostics without bundled report verification "
            "(non-assurance; exits with a distinct nonzero status)."
        ),
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail closed on missing/invalid signatures and extra unhashed files.",
    ),
    expected_fingerprint: str | None = typer.Option(
        None,
        "--expected-fingerprint",
        help="Require the manifest signer to match this sha256:... key fingerprint.",
    ),
    trust_store: str | None = typer.Option(
        None,
        "--trust-store",
        help=(
            "JSON trust store of accepted signer fingerprints "
            "(defaults to ~/.config/invarlock/trusted-signers.json when present)."
        ),
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help=(
            "Optional execution profile for bundled report verification. "
            "Catalog-bound packs derive it from the authenticated catalog entry."
        ),
    ),
    report_assurance: str = typer.Option(
        "report",
        "--report-assurance",
        help=(
            "Nested report assurance mode: report honors each report, strict "
            "requires strict assurance, off verifies reports with assurance disabled."
        ),
    ),
    expected_runtime_image_digest: str | None = typer.Option(
        None,
        "--expected-runtime-image-digest",
        help=(
            "Independent sha256:... trust anchor for bundled reports' runtime image."
        ),
    ),
    expected_catalog_digest: str | None = typer.Option(
        None,
        "--expected-catalog-digest",
        help="Independent sha256:... trust anchor required for catalog-bound packs.",
    ),
    policy_pack: str | None = typer.Option(
        None,
        "--policy-pack",
        help=(
            "External policy pack; required for strict nested report "
            "verification and matched to the signed pack material."
        ),
    ),
) -> None:
    emit = cli_output.make_command_event_emitter(console)
    result = verify_evidence_pack(
        Path(pack),
        json_out_path=Path(json_file) if json_file else None,
        skip_verify=skip_verify,
        strict=strict,
        profile=profile,
        report_assurance=report_assurance,
        expected_fingerprint=expected_fingerprint,
        trust_store_path=Path(trust_store) if trust_store else None,
        expected_catalog_digest=expected_catalog_digest,
        expected_runtime_image_digest=expected_runtime_image_digest,
        policy_pack_path=Path(policy_pack) if policy_pack else None,
    )
    payload = {
        "format_version": EVIDENCE_PACK_VERIFY_FORMAT_VERSION,
        **result.payload,
    }

    if json_out:
        typer.echo(json.dumps(payload, allow_nan=False))
    else:
        warnings = payload["warnings"]
        if warnings:
            emit(
                "WARN",
                f"Evidence pack verification reported {len(warnings)} warning(s)",
            )
            for warning in warnings:
                cli_output.print_command_detail(
                    console, warning, console_style="yellow"
                )
        if payload.get("reports_verified", payload["ok"]):
            emit("PASS", "Evidence pack verified")
        elif payload.get("integrity_ok"):
            emit(
                "WARN",
                "Evidence pack integrity inspection completed; bundled reports were not verified",
            )
            cli_output.print_command_detail(
                console,
                "This is not report assurance and cannot be used as a verification success.",
                console_style="yellow",
            )
        else:
            emit("FAIL", "Evidence pack verification failed")
            for error in payload["errors"]:
                cli_output.print_command_detail(console, error, console_style="red")
    raise typer.Exit(result.status.value)


@evidence_pack_app.command(
    "inspect",
    help="Inspect an evidence-pack summary without running nested report verification.",
)
def inspect_command(
    pack: str = typer.Argument(..., help="Path to the evidence-pack directory."),
    json_out: bool = typer.Option(
        False, "--json", help="Emit machine-readable inspection JSON."
    ),
) -> None:
    emit = cli_output.make_command_event_emitter(console)
    result = inspect_evidence_pack(Path(pack))
    payload = {
        "format_version": EVIDENCE_PACK_INSPECT_FORMAT_VERSION,
        **result.payload,
    }

    if json_out:
        typer.echo(json.dumps(payload, allow_nan=False))
    else:
        if payload["ok"]:
            emit("PASS", "Evidence pack inspected")
            for issue in payload["issues"]:
                cli_output.print_command_detail(
                    console, issue, prefix="  !", console_style="yellow"
                )
        else:
            emit("FAIL", "Evidence pack inspection failed")
            for issue in payload["issues"]:
                cli_output.print_command_detail(console, issue, console_style="red")
    raise typer.Exit(result.status.value)


@evidence_pack_app.command(
    "verify-set",
    help="Strictly verify an exact set of catalog-bound evidence packs.",
)
def verify_set_command(
    catalog: str = typer.Option(..., "--catalog"),
    expected_catalog_digest: str = typer.Option(..., "--expected-catalog-digest"),
    expected_source_commit: str = typer.Option(..., "--expected-source-commit"),
    expected_source_bundle_digest: str = typer.Option(
        ..., "--expected-source-bundle-digest"
    ),
    packs: list[str] = typer.Option([], "--pack"),
    receipt: str = typer.Option(..., "--receipt"),
    expected_fingerprint: str | None = typer.Option(None, "--expected-fingerprint"),
    trust_store: str | None = typer.Option(None, "--trust-store"),
    expected_runtime_image_digest: str = typer.Option(
        ..., "--expected-runtime-image-digest"
    ),
    policy_pack: str | None = typer.Option(None, "--policy-pack"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    if not packs:
        raise typer.BadParameter("at least one --pack is required")
    try:
        result = verify_evidence_pack_set(
            catalog_path=Path(catalog),
            pack_dirs=[Path(pack) for pack in packs],
            receipt_path=Path(receipt),
            expected_catalog_digest=expected_catalog_digest,
            expected_source_commit=expected_source_commit,
            expected_source_bundle_digest=expected_source_bundle_digest,
            expected_fingerprint=expected_fingerprint,
            trust_store_path=Path(trust_store) if trust_store else None,
            expected_runtime_image_digest=expected_runtime_image_digest,
            policy_pack_path=Path(policy_pack) if policy_pack else None,
        )
    except EvidenceCatalogError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = {
        "format_version": EVIDENCE_PACK_SET_VERIFY_FORMAT_VERSION,
        **result.payload,
    }
    if json_out:
        typer.echo(json.dumps(payload, allow_nan=False))
    raise typer.Exit(result.status.value)
