from __future__ import annotations

import json
from pathlib import Path

import typer

from invarlock.cli import output as cli_output
from invarlock.cli.constants import (
    EVIDENCE_PACK_BUILD_FORMAT_VERSION,
    EVIDENCE_PACK_INSPECT_FORMAT_VERSION,
    EVIDENCE_PACK_KEYGEN_FORMAT_VERSION,
    EVIDENCE_PACK_VERIFY_FORMAT_VERSION,
)
from invarlock.evidence_pack import (
    _generate_signing_keypair,
    _material_spec,
    build_evidence_pack,
    inspect_evidence_pack,
    verify_evidence_pack,
)

console = cli_output.make_console()
evidence_pack_app = typer.Typer(
    help="Build, inspect, and verify evidence-pack artifacts.",
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
        False, "--skip-verify", help="Skip bundled evaluation.report.json verification."
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
    profile: str = typer.Option(
        "dev",
        "--profile",
        help="Execution profile to use for bundled report verification (dev|ci|release).",
    ),
    report_assurance: str = typer.Option(
        "report",
        "--report-assurance",
        help=(
            "Nested report assurance mode: report honors each report, strict "
            "requires strict assurance, off verifies reports with assurance disabled."
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
    )
    payload = {
        "format_version": EVIDENCE_PACK_VERIFY_FORMAT_VERSION,
        **result.payload,
    }

    if json_out:
        typer.echo(json.dumps(payload))
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
        if payload["ok"]:
            emit("PASS", "Evidence pack verified")
        else:
            emit("FAIL", "Evidence pack verification failed")
            for error in payload["errors"]:
                cli_output.print_command_detail(console, error, console_style="red")
    raise typer.Exit(result.status.value)


@evidence_pack_app.command(
    "keygen",
    help="Generate an Ed25519 signing keypair for package-native evidence-pack manifests.",
)
def keygen_command(
    private_key_out: str = typer.Argument(
        ...,
        help="Output path for the private signing key PEM.",
    ),
    public_key_out: str | None = typer.Option(
        None,
        "--public-key-out",
        help="Optional output path for the public verification key PEM.",
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit machine-readable keygen JSON."
    ),
) -> None:
    emit = cli_output.make_command_event_emitter(console)
    private_key_path = Path(private_key_out)
    public_key_path = (
        Path(public_key_out)
        if public_key_out
        else private_key_path.with_name(f"{private_key_path.stem}.pub.pem")
    )
    try:
        fingerprint = _generate_signing_keypair(
            private_key_path,
            public_key_path=public_key_path,
        )
        payload = {
            "format_version": EVIDENCE_PACK_KEYGEN_FORMAT_VERSION,
            "ok": True,
            "algorithm": "ed25519",
            "private_key": str(private_key_path),
            "public_key": str(public_key_path),
            "signing_key_fingerprint": fingerprint,
        }
        exit_code = 0
    except FileExistsError as exc:
        payload = {
            "format_version": EVIDENCE_PACK_KEYGEN_FORMAT_VERSION,
            "ok": False,
            "errors": [str(exc)],
        }
        exit_code = 2

    if json_out:
        typer.echo(json.dumps(payload))
    else:
        if payload["ok"]:
            emit("PASS", "Evidence pack signing keypair created")
            cli_output.print_command_detail(
                console, f"Private key: {payload['private_key']}"
            )
            cli_output.print_command_detail(
                console, f"Public key: {payload['public_key']}"
            )
            cli_output.print_command_detail(
                console, f"Fingerprint: {payload['signing_key_fingerprint']}"
            )
        else:
            emit("FAIL", "Evidence pack key generation failed")
            for error in payload["errors"]:
                cli_output.print_command_detail(console, error, console_style="red")
    raise typer.Exit(exit_code)


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
        typer.echo(json.dumps(payload))
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
    "build",
    help="Assemble an evidence pack from existing verdict, metadata, and report artifacts.",
)
def build_command(
    out: str = typer.Argument(..., help="Output directory for the evidence-pack."),
    final_verdict: str = typer.Option(
        ...,
        "--final-verdict",
        help="Path to the final verdict JSON to package as the evidence-pack subject.",
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
    signing_key: str | None = typer.Option(
        None,
        "--signing-key",
        help="Optional Ed25519 private key PEM used to sign manifest.json.",
    ),
    profile: str = typer.Option(
        "dev",
        "--profile",
        help="Execution profile to use for report pre-verification (dev|ci|release).",
    ),
    report_assurance: str = typer.Option(
        "report",
        "--report-assurance",
        help="Report assurance mode for report pre-verification (report|strict|off).",
    ),
    release_review: bool = typer.Option(
        False,
        "--release-review",
        help="Require PASS verdict, signing key, runtime sidecars, and strict report assurance.",
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit machine-readable build JSON."
    ),
) -> None:
    emit = cli_output.make_command_event_emitter(console)
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
            "format_version": EVIDENCE_PACK_BUILD_FORMAT_VERSION,
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
            emit("FAIL", "Evidence pack build request is invalid")
            for error in errors:
                cli_output.print_command_detail(console, error, console_style="red")
        raise typer.Exit(2)

    result = build_evidence_pack(
        Path(out),
        final_verdict_path=Path(final_verdict),
        report_paths=[Path(path) for path in reports],
        source_repo_path=Path(source_repo) if source_repo else None,
        environment_path=Path(environment) if environment else None,
        material_specs=material_specs,
        readme_path=Path(readme) if readme else None,
        signing_key_path=Path(signing_key) if signing_key else None,
        profile=profile,
        report_assurance=report_assurance,
        release_review=release_review,
    )
    payload = {
        "format_version": EVIDENCE_PACK_BUILD_FORMAT_VERSION,
        **result.payload,
    }

    if json_out:
        typer.echo(json.dumps(payload))
    else:
        warnings = payload["warnings"]
        if warnings:
            emit("WARN", f"Evidence pack build reported {len(warnings)} warning(s)")
            for warning in warnings:
                cli_output.print_command_detail(
                    console, warning, console_style="yellow"
                )
        if payload["ok"]:
            emit("PASS", "Evidence pack built")
        else:
            emit("FAIL", "Evidence pack build failed")
            for error in payload["errors"]:
                cli_output.print_command_detail(console, error, console_style="red")
    raise typer.Exit(result.status.value)
