"""Evaluation-report phase helper for the evaluate CLI command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invarlock.cli import output as cli_output
from invarlock.core.exceptions import ConfigError, MetricsError, ValidationError


@dataclass(frozen=True)
class EvaluationReportRequest:
    edited_report: Path
    baseline_report_path: Path
    report_out: str | Path
    baseline: str
    subject: str
    baseline_eff_adapter: str
    subject_eff_adapter: str
    profile_name: str
    tier_name: str
    preset: str | None
    out: str
    edit_config: str | None
    edit_label: str | None
    allow_network: bool
    allow_remote_code: bool
    allow_third_party_plugins: bool
    execution_mode: str
    assurance_mode: str
    defer_report_rendering: bool


@dataclass(frozen=True)
class EvaluationReportRuntime:
    console: Any
    output_style: Any
    timings: dict[str, float]
    info_fn: Any
    fail_fn: Any
    generate_reports_fn: Any
    emit_runtime_manifest_fn: Any
    manifest_execution_fn: Any


def emit_evaluation_report_phase(
    request: EvaluationReportRequest,
    runtime: EvaluationReportRuntime,
) -> None:
    """Generate the paired evaluation report and emit runtime provenance."""
    runtime.info_fn("Emitting evaluation report", tag="EXEC", emoji="📜")
    with cli_output.timed_step(
        console=runtime.console,
        style=runtime.output_style,
        timings=runtime.timings,
        key="evaluation_report",
        tag="EXEC",
        message="Evaluation Report",
        emoji="📜",
    ):
        try:
            report_kwargs: dict[str, Any] = {
                "run": str(request.edited_report),
                "format": "report",
                "baseline": str(request.baseline_report_path),
                "output": str(request.report_out),
                "render_optional": not request.defer_report_rendering,
            }
            runtime.generate_reports_fn(**report_kwargs)
        except (ConfigError, MetricsError, ValidationError) as exc:
            runtime.fail_fn(str(getattr(exc, "message", exc)), exit_code=1)
            return

    runtime.emit_runtime_manifest_fn(
        Path(request.report_out) / "evaluation.report.json",
        config_payload={
            "command": "evaluate",
            "baseline": request.baseline,
            "subject": request.subject,
            "baseline_adapter": request.baseline_eff_adapter,
            "subject_adapter": request.subject_eff_adapter,
            "profile": request.profile_name,
            "tier": request.tier_name,
            "preset": request.preset,
            "out": request.out,
            "report_out": request.report_out,
            "edit_config": request.edit_config,
            "edit_label": request.edit_label,
            "allow_network": request.allow_network,
            "allow_remote_code": request.allow_remote_code,
            "allow_third_party_plugins": request.allow_third_party_plugins,
            "execution_mode": request.execution_mode,
            "assurance": request.assurance_mode,
            "defer_report_rendering": bool(request.defer_report_rendering),
        },
        extra={
            "command": "evaluate",
            "profile": request.profile_name,
            "tier": request.tier_name,
            "execution_mode": request.execution_mode,
            "assurance": request.assurance_mode,
        },
        execution=runtime.manifest_execution_fn(
            execution_mode=request.execution_mode,
            allow_network=request.allow_network,
            allow_remote_code=request.allow_remote_code,
            allow_third_party_plugins=request.allow_third_party_plugins,
        ),
    )


__all__ = [
    "EvaluationReportRequest",
    "EvaluationReportRuntime",
    "emit_evaluation_report_phase",
]
