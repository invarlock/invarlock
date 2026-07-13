"""Evaluation-report phase helper for the evaluate CLI command."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invarlock.cli import output as cli_output
from invarlock.core.exceptions import ConfigError, MetricsError, ValidationError
from invarlock.evidence_pack_json import (
    StrictJsonError,
    read_json_object_snapshot,
    read_regular_file_bytes,
)


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
    resolved_subject_config: Path | None = None
    clean_selection_context: Any = None
    clean_pruning_selection_context: Any = None


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
    finalize_clean_selection_report_fn: Any = None
    finalize_clean_pruning_selection_report_fn: Any = None


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

    report_root = Path(request.report_out)
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / "evaluation.report.json"
    resolved_config_path = report_root / "resolved-config.yaml"
    manifest_config_path: Path | None = None
    if request.resolved_subject_config is not None:
        try:
            resolved_config_bytes = read_regular_file_bytes(
                request.resolved_subject_config,
                label="resolved subject config",
            )
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(resolved_config_path, flags, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(resolved_config_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                if resolved_config_path.exists():
                    resolved_config_path.unlink()
                raise
            manifest_config_path = resolved_config_path
        except (OSError, StrictJsonError) as exc:
            if request.assurance_mode == "strict":
                runtime.fail_fn(
                    f"Could not preserve resolved evaluation config: {exc}",
                    exit_code=1,
                )
                return
    elif request.assurance_mode == "strict":
        runtime.fail_fn(
            "Strict evaluation requires the exact resolved subject config.",
            exit_code=1,
        )
        return
    manifest_extra: dict[str, Any] = {
        "command": "evaluate",
        "profile": request.profile_name,
        "tier": request.tier_name,
        "execution_mode": request.execution_mode,
        "assurance": request.assurance_mode,
    }
    try:
        _report_raw, report_payload = read_json_object_snapshot(
            report_path, label="evaluation report"
        )
    except StrictJsonError as exc:
        if request.assurance_mode == "strict":
            runtime.fail_fn(
                f"Could not load evaluation report provenance: {exc}", exit_code=1
            )
            return
        report_payload = {}
    report_context = report_payload.get("context")
    evaluation_inputs = (
        report_context.get("evaluation_inputs")
        if isinstance(report_context, dict)
        else None
    )
    if isinstance(evaluation_inputs, dict):
        manifest_extra["evaluation_inputs"] = evaluation_inputs
    if (
        request.clean_selection_context is not None
        and request.clean_pruning_selection_context is not None
    ):
        runtime.fail_fn(
            "Generic clean selection and clean pruning selection cannot both finalize one report.",
            exit_code=1,
        )
        return
    if request.clean_selection_context is not None:
        if not callable(runtime.finalize_clean_selection_report_fn):
            runtime.fail_fn(
                "Clean-selection evaluator finalizer is unavailable.", exit_code=1
            )
            return
        try:
            selection_link = runtime.finalize_clean_selection_report_fn(
                report_path,
                context=request.clean_selection_context,
            )
        except Exception as exc:
            runtime.fail_fn(
                f"Could not finalize receipt-bound clean-selection evidence: {exc}",
                exit_code=1,
            )
            return
        if not isinstance(selection_link, dict):
            runtime.fail_fn(
                "Clean-selection evaluator finalizer did not return a manifest link.",
                exit_code=1,
            )
            return
        manifest_extra["clean_selection_execution"] = selection_link
    if request.clean_pruning_selection_context is not None:
        if not callable(runtime.finalize_clean_pruning_selection_report_fn):
            runtime.fail_fn(
                "Clean-pruning-selection evaluator finalizer is unavailable.",
                exit_code=1,
            )
            return
        try:
            pruning_selection_link = runtime.finalize_clean_pruning_selection_report_fn(
                report_path,
                context=request.clean_pruning_selection_context,
            )
        except Exception as exc:
            runtime.fail_fn(
                f"Could not finalize receipt-bound clean-pruning-selection evidence: {exc}",
                exit_code=1,
            )
            return
        if not isinstance(pruning_selection_link, dict):
            runtime.fail_fn(
                "Clean-pruning-selection evaluator finalizer did not return a manifest link.",
                exit_code=1,
            )
            return
        manifest_extra["clean_pruning_selection_execution"] = pruning_selection_link

    runtime.emit_runtime_manifest_fn(
        report_path,
        config_path=manifest_config_path,
        extra=manifest_extra,
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
