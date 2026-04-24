"""Thin internal helper for config-driven run execution."""

from __future__ import annotations

import typer

from invarlock.cli.config_execution import RuntimeDelegationError, run_from_config


def run_command(
    config: str,
    device: str | None = None,
    profile: str | None = None,
    out: str | None = None,
    edit: str | None = None,
    edit_label: str | None = None,
    tier: str | None = None,
    metric_kind: str | None = None,
    probes: int | None = None,
    until_pass: bool = False,
    max_attempts: int = 3,
    timeout: int | None = None,
    baseline: str | None = None,
    no_cleanup: bool = False,
    style: str | None = None,
    progress: bool = False,
    timing: bool = False,
    telemetry: bool = False,
    allow_network: bool = False,
    allow_host_execution: bool = False,
    allow_third_party_plugins: bool = False,
    allow_remote_code: bool = False,
    allow_unverified_provenance: bool = False,
    prefer_local_files_only: bool = False,
    no_color: bool = False,
):
    """Run a config-driven InvarLock pipeline and return the report path."""
    allow_network = bool(allow_network)
    allow_host_execution = bool(allow_host_execution)
    allow_third_party_plugins = bool(allow_third_party_plugins)
    allow_remote_code = bool(allow_remote_code)
    allow_unverified_provenance = bool(allow_unverified_provenance)
    prefer_local_files_only = bool(prefer_local_files_only)
    try:
        return run_from_config(
            config=config,
            device=device,
            profile=profile,
            out=out,
            edit=edit,
            edit_label=edit_label,
            tier=tier,
            metric_kind=metric_kind,
            probes=probes,
            until_pass=until_pass,
            max_attempts=max_attempts,
            timeout=timeout,
            baseline=baseline,
            no_cleanup=no_cleanup,
            style=style,
            progress=progress,
            timing=timing,
            telemetry=telemetry,
            no_color=no_color,
            allow_network=allow_network,
            allow_host_execution=allow_host_execution,
            allow_third_party_plugins=allow_third_party_plugins,
            allow_remote_code=allow_remote_code,
            allow_unverified_provenance=allow_unverified_provenance,
            prefer_local_files_only=prefer_local_files_only,
            command_name="run",
        )
    except RuntimeDelegationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
