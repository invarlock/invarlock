from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from invarlock.core.runtime_attestation import (
    configure_runtime_security as _configure_runtime_security_core,
)
from invarlock.core.runtime_attestation import (
    verify_runtime_attestation as _verify_runtime_attestation_core,
)
from invarlock.runtime_security import (
    delegate_current_process_to_container,
    host_execution_allowed,
    running_inside_container,
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
    _configure_runtime_security_core(
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
    return _verify_runtime_attestation_core(
        report_path,
        allow_unattested=allow_unattested,
    )
