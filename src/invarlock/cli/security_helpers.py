from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import typer

from invarlock.runtime_security import (
    apply_runtime_allowances,
    delegate_current_process_to_container,
    host_execution_allowed,
    load_runtime_manifest,
    running_inside_container,
    runtime_verifier_binary,
    unattested_artifacts_allowed,
    write_runtime_manifest,
)


def configure_runtime_security(
    *,
    allow_network: bool = False,
    allow_host_execution: bool = False,
    allow_third_party_plugins: bool = False,
    allow_remote_code: bool = False,
    allow_unattested_artifacts: bool = False,
) -> None:
    apply_runtime_allowances(
        allow_network=allow_network,
        allow_host_execution=allow_host_execution,
        allow_third_party_plugins=allow_third_party_plugins,
        allow_remote_code=allow_remote_code,
        allow_unattested_artifacts=allow_unattested_artifacts,
    )


def maybe_delegate_model_command() -> None:
    if running_inside_container() or host_execution_allowed():
        return
    try:
        code = delegate_current_process_to_container()
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    raise typer.Exit(code)


def emit_runtime_manifest(
    report_path: str | Path | None,
    *,
    config_path: str | Path | None = None,
    config_payload: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> Path | None:
    if not report_path:
        return None
    path = Path(report_path)
    if not path.exists():
        return None
    return write_runtime_manifest(
        path,
        config_path=config_path,
        config_payload=config_payload,
        extra=extra,
    )


def verify_runtime_attestation(
    report_path: str | Path,
    *,
    allow_unattested: bool = False,
) -> list[str]:
    if allow_unattested or unattested_artifacts_allowed():
        return []

    report = Path(report_path)
    manifest_path, manifest = load_runtime_manifest(report)
    if manifest is None:
        return [
            f"{manifest_path.name} missing or unreadable for {report.name}; "
            "pass --allow-unattested-artifacts to override."
        ]

    if manifest.get("execution_mode") != "container":
        return [
            f"{manifest_path.name} marks {report.name} as "
            f"{manifest.get('execution_mode')!r}; pass "
            "--allow-unattested-artifacts to override."
        ]

    binary = runtime_verifier_binary()
    if shutil.which(binary) is None:
        return [
            f"Runtime verifier '{binary}' is not installed; cannot verify {report.name}."
        ]

    completed = subprocess.run(
        [
            binary,
            "--report",
            str(report),
            "--manifest",
            str(manifest_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return []

    message = (completed.stdout or completed.stderr or "").strip()
    if message:
        try:
            payload = json.loads(message)
        except Exception:
            pass
        else:
            errors = payload.get("errors")
            if isinstance(errors, list) and errors:
                return [str(item) for item in errors]
    return [message or f"Runtime verifier failed for {report.name}."]
