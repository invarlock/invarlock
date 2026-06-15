"""Evaluation-report phase helper for the evaluate CLI command."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from invarlock.cli import output as cli_output
from invarlock.core.exceptions import ConfigError, MetricsError, ValidationError


def emit_evaluation_report_phase(
    *,
    edited_report: Path,
    baseline_report_path: Path,
    report_out: str | Path,
    baseline: str,
    subject: str,
    baseline_eff_adapter: str,
    subject_eff_adapter: str,
    profile_name: str,
    tier_name: str,
    preset: str | None,
    out: str,
    edit_config: str | None,
    edit_label: str | None,
    allow_network: bool,
    allow_remote_code: bool,
    allow_third_party_plugins: bool,
    execution_mode: str,
    assurance_mode: str,
    defer_report_rendering: bool,
    console: Any,
    output_style: Any,
    timings: dict[str, float],
    info_fn: Any,
    fail_fn: Any,
    generate_reports_fn: Any,
    emit_runtime_manifest_fn: Any,
    manifest_execution_fn: Any,
) -> None:
    """Generate the paired evaluation report and emit runtime provenance."""
    info_fn("Emitting evaluation report", tag="EXEC", emoji="📜")
    with cli_output.timed_step(
        console=console,
        style=output_style,
        timings=timings,
        key="evaluation_report",
        tag="EXEC",
        message="Evaluation Report",
        emoji="📜",
    ):
        try:
            report_kwargs: dict[str, Any] = {
                "run": str(edited_report),
                "format": "report",
                "baseline": str(baseline_report_path),
                "output": str(report_out),
                "render_optional": not defer_report_rendering,
            }
            generate_reports_fn(**report_kwargs)
        except (ConfigError, MetricsError, ValidationError) as exc:
            fail_fn(str(getattr(exc, "message", exc)), exit_code=1)
            return

    emit_runtime_manifest_fn(
        Path(report_out) / "evaluation.report.json",
        config_payload={
            "command": "evaluate",
            "baseline": baseline,
            "subject": subject,
            "baseline_adapter": baseline_eff_adapter,
            "subject_adapter": subject_eff_adapter,
            "profile": profile_name,
            "tier": tier_name,
            "preset": preset,
            "out": out,
            "report_out": report_out,
            "edit_config": edit_config,
            "edit_label": edit_label,
            "allow_network": allow_network,
            "allow_remote_code": allow_remote_code,
            "allow_third_party_plugins": allow_third_party_plugins,
            "execution_mode": execution_mode,
            "assurance": assurance_mode,
            "defer_report_rendering": bool(defer_report_rendering),
        },
        extra={
            "command": "evaluate",
            "profile": profile_name,
            "tier": tier_name,
            "execution_mode": execution_mode,
            "assurance": assurance_mode,
        },
        execution=manifest_execution_fn(
            execution_mode=execution_mode,
            allow_network=allow_network,
            allow_remote_code=allow_remote_code,
            allow_third_party_plugins=allow_third_party_plugins,
        ),
    )


__all__ = ["emit_evaluation_report_phase"]
