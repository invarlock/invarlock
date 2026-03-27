from __future__ import annotations

import typer

from invarlock.cli.commands.run import run_command

internal_run_app = typer.Typer(add_completion=False)


@internal_run_app.callback()
def _internal_root() -> None:
    """Internal run test harness."""


@internal_run_app.command(name="run")
def _run_typed(
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to YAML configuration file"
    ),
    device: str | None = typer.Option(
        None, "--device", help="Device override (auto|cuda|mps|cpu)"
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Profile to apply (e.g. ci, release, ci_cpu; dev is a no-op)",
    ),
    out: str | None = typer.Option(None, "--out", help="Output directory override"),
    edit: str | None = typer.Option(
        None,
        "--edit",
        help="Edit name override (canonical plugin name, e.g. quant_rtn)",
    ),
    edit_label: str | None = typer.Option(
        None,
        "--edit-label",
        help=(
            "Edit algorithm label for BYOE models. Use 'noop' for baseline, "
            "'quant_rtn' etc. for built-in edits, 'custom' for pre-edited models."
        ),
    ),
    tier: str | None = typer.Option(
        None,
        "--tier",
        help="Auto-tuning tier override (conservative|balanced|aggressive)",
    ),
    metric_kind: str | None = typer.Option(
        None,
        "--metric-kind",
        help="Primary metric kind override (ppl_causal|ppl_mlm|accuracy|etc.)",
    ),
    probes: int | None = typer.Option(
        None, "--probes", help="Number of micro-probes (0=deterministic, >0=adaptive)"
    ),
    until_pass: bool = typer.Option(
        False,
        "--until-pass",
        help="Retry until evaluation report passes gates (max 3 attempts)",
    ),
    max_attempts: int = typer.Option(
        3, "--max-attempts", help="Maximum retry attempts for --until-pass mode"
    ),
    timeout: int | None = typer.Option(
        None, "--timeout", help="Timeout in seconds for --until-pass mode"
    ),
    baseline: str | None = typer.Option(
        None,
        "--baseline",
        help="Path to baseline report.json for evaluation report validation",
    ),
    no_cleanup: bool = typer.Option(
        False, "--no-cleanup", help="Skip cleanup of temporary artifacts"
    ),
    style: str | None = typer.Option(
        None, "--style", help="Output style (audit|friendly)"
    ),
    progress: bool = typer.Option(
        False, "--progress", help="Show progress done messages"
    ),
    timing: bool = typer.Option(False, "--timing", help="Show timing summary"),
    telemetry: bool = typer.Option(
        False, "--telemetry", help="Write telemetry JSON alongside the report"
    ),
    allow_network: bool = typer.Option(
        False,
        "--allow-network",
        help="Explicitly allow outbound network access for this command.",
    ),
    allow_host_execution: bool = typer.Option(
        False,
        "--allow-host-execution",
        help="Run on the host instead of auto-delegating to the runtime container.",
    ),
    allow_third_party_plugins: bool = typer.Option(
        False,
        "--allow-third-party-plugins",
        help="Enable third-party entry-point plugin discovery for this command.",
    ),
    allow_remote_code: bool = typer.Option(
        False,
        "--allow-remote-code",
        help="Allow trust_remote_code-style model loading for this command.",
    ),
    prefer_local_files_only: bool = typer.Option(False, hidden=True),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable ANSI colors (respects NO_COLOR=1)"
    ),
) -> str | None:
    return run_command(
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
        allow_network=allow_network,
        allow_host_execution=allow_host_execution,
        allow_third_party_plugins=allow_third_party_plugins,
        allow_remote_code=allow_remote_code,
        prefer_local_files_only=prefer_local_files_only,
        no_color=no_color,
    )
