"""Shell-facing output helpers for config-driven run execution."""

from __future__ import annotations

from typing import Any

from invarlock.cli import output as output_mod
from invarlock.cli.output import resolve_output_style
from invarlock.cli.run_shell_output import (
    _device_resolution_note,
    _event,
    _format_kv_line,
    _print_guard_overhead_summary,
)
from invarlock.core.run_orchestrator import (
    RunAdapterSelectedEvent,
    RunAttemptStartedEvent,
    RunAutoTuneAdjustmentEvent,
    RunBaselineScheduleLoadedEvent,
    RunCalibrationBatchSizesDebugEvent,
    RunCleanupStatusEvent,
    RunConfigLoadedEvent,
    RunConfigLoadingEvent,
    RunDatasetLoadingEvent,
    RunDeterministicSeedsEvent,
    RunDeviceResolvedEvent,
    RunDiagnosticEvent,
    RunEditSelectedEvent,
    RunEvaluationReportFailedEvent,
    RunEvaluationReportPassedEvent,
    RunEvaluationReportStartedEvent,
    RunExecutePipelineEvent,
    RunExecutionEvent,
    RunFailureEvent,
    RunGuardChainResolvedEvent,
    RunGuardOverheadSummaryEvent,
    RunLoadModelOnceEvent,
    RunMaskedTokensDebugEvent,
    RunOutputDirectoryReadyEvent,
    RunPipelineStartedEvent,
    RunPreviewLabelsDebugEvent,
    RunPrimaryMetricSummaryEvent,
    RunRetryAttemptStartedEvent,
    RunRetryExhaustedEvent,
    RunRetrySummaryEvent,
    RunRetryValidationErrorEvent,
    RunSnapshotModeEvent,
    RunTelemetryFailedEvent,
    RunTelemetrySavedEvent,
)


def resolve_shell_output_style(console: Any, request: Any) -> Any:
    output_style = resolve_output_style(
        style=str(request.style) if request.style is not None else None,
        profile=str(request.profile) if request.profile is not None else None,
        progress=bool(request.progress),
        timing=bool(request.timing),
        no_color=bool(request.no_color),
    )
    console._invarlock_output_style = output_style
    console.no_color = not bool(output_style.color)
    return output_style


def emit_console_line(console: Any, line: str, *, markup: bool = False) -> None:
    if markup:
        console.print(line)
        return
    try:
        console.print(line, markup=False)
    except TypeError:
        console.print(line)


def emit_console_blank_line(console: Any) -> None:
    console.print("")


def begin_progress_step(console: Any, key: str) -> None:
    output_style = getattr(console, "_invarlock_output_style", None)
    if output_style is None or not bool(getattr(output_style, "progress", False)):
        return
    progress_steps = getattr(console, "_invarlock_progress_steps", None)
    if not isinstance(progress_steps, dict):
        progress_steps = {}
        console._invarlock_progress_steps = progress_steps
    progress_steps[key] = float(output_mod.perf_counter())


def complete_progress_step(
    console: Any,
    key: str,
    *,
    tag: str,
    message: str,
    emoji: str | None = None,
) -> None:
    output_style = getattr(console, "_invarlock_output_style", None)
    if output_style is None or not bool(getattr(output_style, "progress", False)):
        return
    progress_steps = getattr(console, "_invarlock_progress_steps", None)
    if not isinstance(progress_steps, dict):
        return
    start = progress_steps.pop(key, None)
    if not isinstance(start, int | float):
        return
    completed_steps = getattr(console, "_invarlock_progress_completed", None)
    if not isinstance(completed_steps, set):
        completed_steps = set()
        console._invarlock_progress_completed = completed_steps
    completed_steps.add(key)
    elapsed = max(0.0, float(output_mod.perf_counter() - float(start)))
    _event(console, tag, f"{message} done ({elapsed:.2f}s)", emoji=emoji)


def transition_progress_step(
    console: Any,
    from_key: str,
    *,
    from_tag: str,
    from_message: str,
    to_key: str,
    from_emoji: str | None = None,
) -> None:
    output_style = getattr(console, "_invarlock_output_style", None)
    if output_style is None or not bool(getattr(output_style, "progress", False)):
        return
    progress_steps = getattr(console, "_invarlock_progress_steps", None)
    if not isinstance(progress_steps, dict):
        progress_steps = {}
        console._invarlock_progress_steps = progress_steps
    now = float(output_mod.perf_counter())
    start = progress_steps.pop(from_key, None)
    if isinstance(start, int | float):
        completed_steps = getattr(console, "_invarlock_progress_completed", None)
        if not isinstance(completed_steps, set):
            completed_steps = set()
            console._invarlock_progress_completed = completed_steps
        completed_steps.add(from_key)
        elapsed = max(0.0, now - float(start))
        _event(
            console,
            from_tag,
            f"{from_message} done ({elapsed:.2f}s)",
            emoji=from_emoji,
        )
    progress_steps[to_key] = now


def render_run_execution_event(console: Any, event: RunExecutionEvent) -> None:
    def _emit_status_line(tag: str, message: str, *, emoji: str | None = None) -> None:
        _event(console, tag, message, emoji=emoji)

    if isinstance(event, RunDeviceResolvedEvent):
        resolution_note = _device_resolution_note(
            event.requested_device,
            event.resolved_device,
        )
        emit_console_line(
            console,
            _format_kv_line(
                "Device",
                f"{event.resolved_device} ({resolution_note})",
            ),
            markup=False,
        )
        return

    if isinstance(event, RunOutputDirectoryReadyEvent):
        emit_console_line(
            console, _format_kv_line("Output", event.run_dir), markup=False
        )
        emit_console_line(
            console, _format_kv_line("Run ID", event.run_id), markup=False
        )
        return

    if isinstance(event, RunEditSelectedEvent):
        emit_console_line(
            console, _format_kv_line("Edit", event.edit_name), markup=False
        )
        return

    if isinstance(event, RunGuardChainResolvedEvent):
        emit_console_line(
            console,
            _format_kv_line("Guards", " → ".join(event.guard_names)),
            markup=False,
        )
        return

    if isinstance(event, RunDiagnosticEvent):
        code = event.code or ""
        context = event.context
        if code == "guard_missing":
            _event(
                console,
                "WARN",
                f"Guard '{context.get('guard_name', '')}' not found, skipping",
                emoji="⚠️",
            )
            return
        if code == "export_tokenizer_missing":
            _event(
                console,
                "WARN",
                "Exported model checkpoint without tokenizer artifacts; local tokenizer reload may fail.",
                emoji="⚠️",
            )
            return
        if code == "export_adapter_directory_missing":
            _event(
                console,
                "WARN",
                "Model export requested but adapter did not save a HF directory.",
                emoji="⚠️",
            )
            return
        if code == "export_failed":
            _event(
                console,
                "WARN",
                "Model export requested but failed due to an unexpected error.",
                emoji="⚠️",
            )
            return
        if code == "snapshot_restore_fallback":
            _event(
                console,
                "WARN",
                "Snapshot restore failed; switching to reload-per-attempt.",
                emoji="⚠️",
            )
            error = context.get("error")
            if error:
                _event(console, "WARN", f"↳ {error}")
            return
        if code == "retry_validation_telemetry_summary":
            emit_console_line(console, str(context.get("summary", "")), markup=False)
            return
        if code == "metric_diffs_debug":
            emit_console_line(
                console,
                f"[debug] DEBUG_METRIC_DIFFS: {context.get('summary', '')}",
                markup=False,
            )
            return

        if isinstance(event.summary, str) and event.summary:
            tag = str(event.level or context.get("tag") or "INFO").upper()
            emoji = context.get("emoji")
            if not isinstance(emoji, str):
                emoji = None
            _event(console, tag, event.summary, emoji=emoji)
            return

    if isinstance(event, RunGuardOverheadSummaryEvent):
        _print_guard_overhead_summary(
            console,
            event.guard_overhead_info or {},
            default_threshold=float(event.default_threshold or 0.01),
        )
        return

    if isinstance(event, RunRetrySummaryEvent):
        summary = event.summary
        emit_console_blank_line(console)
        _event(
            console,
            "METRIC",
            f"Retry Summary: {summary.get('total_attempts', 0)} attempts in {float(summary.get('elapsed_time', 0.0) or 0.0):.1f}s",
            emoji="📊",
        )
        return

    if isinstance(event, RunPrimaryMetricSummaryEvent):
        _emit_status_line(
            "METRIC",
            f"Primary Metric [{event.metric_kind}] — preview: {event.preview:.3f}, final: {event.final:.3f}",
            emoji="📌",
        )
        if event.ratio_vs_baseline is not None:
            _emit_status_line(
                "METRIC",
                f"Ratio vs baseline [{event.metric_kind}]: {event.ratio_vs_baseline:.3f}",
                emoji="🔗",
            )
        return

    if isinstance(event, RunFailureEvent):
        failure = event.failure
        code = failure.code
        context = failure.context
        if code == "torch_missing":
            _emit_status_line(
                "FAIL",
                'Torch is required for this command. Install extras with: pip install "invarlock[hf]" or "invarlock[adapters]".',
                emoji="❌",
            )
            return
        if code == "edit_name_missing":
            _emit_status_line(
                "FAIL",
                "Edit configuration must specify a non-empty `edit.name`.",
                emoji="❌",
            )
            return
        if code == "unknown_edit":
            _emit_status_line(
                "FAIL",
                f"Unknown edit '{context.get('edit_name', '')}'.",
                emoji="❌",
            )
            return
        if code == "baseline_windows_missing":
            _emit_status_line("FAIL", str(failure.summary or ""), emoji="❌")
            return
        if code == "guard_overhead_budget_exceeded":
            threshold_fraction = float(context.get("threshold_fraction", 0.01) or 0.01)
            _emit_status_line(
                "FAIL",
                "Guard overhead gate exceeded the configured budget "
                f"(>{threshold_fraction * 100:.1f}% increase)",
                emoji="❌",
            )
            return
        if code == "config_file_missing":
            _emit_status_line(
                "FAIL",
                f"Configuration file not found: {context.get('path', '')}",
                emoji="❌",
            )
            return
        if code == "schema_invalid_run_report":
            _emit_status_line(
                "FAIL",
                "Schema invalid: run report structure failed validation",
                emoji="❌",
            )
            return
        if code == "pipeline_failed":
            _emit_status_line(
                "FAIL",
                f"Pipeline execution failed: {failure.summary or ''}",
                emoji="❌",
            )
            return
        _emit_status_line("FAIL", str(failure.summary or code), emoji="❌")
        return

    if isinstance(event, RunDeterministicSeedsEvent):
        torch_display = event.torch_seed if event.torch_seed is not None else "N/A"
        _emit_status_line(
            "INIT",
            "Deterministic seeds → "
            f"python={event.python_seed}, numpy={event.numpy_seed}, torch={torch_display}",
            emoji="🎲",
        )
        return
    if isinstance(event, RunBaselineScheduleLoadedEvent):
        _emit_status_line(
            "DATA",
            "Loaded baseline evaluation schedule for pairing",
            emoji="🧬",
        )
        return
    if isinstance(event, RunPipelineStartedEvent):
        _emit_status_line("INIT", "Starting InvarLock pipeline...", emoji="🚀")
        return
    if isinstance(event, RunConfigLoadingEvent):
        _emit_status_line(
            "INIT",
            f"Loading configuration: {event.config_path}",
            emoji="📋",
        )
        return
    if isinstance(event, RunConfigLoadedEvent):
        return
    if isinstance(event, RunAdapterSelectedEvent):
        _emit_status_line("DATA", f"Adapter: {event.adapter_name}", emoji="🔌")
        return
    if isinstance(event, RunDatasetLoadingEvent):
        _emit_status_line(
            "DATA",
            f"Loading dataset: {event.provider}",
            emoji="📊",
        )
        return
    if isinstance(event, RunCalibrationBatchSizesDebugEvent):
        emit_console_line(
            console,
            "[debug] calibration batch size => preview="
            f"{event.preview_count} final={event.final_count} total={event.total_count}",
            markup=False,
        )
        return
    if isinstance(event, RunMaskedTokensDebugEvent):
        emit_console_line(
            console,
            f"[debug] masked tokens (preview/final) = {event.preview_masked}/{event.final_masked}",
            markup=False,
        )
        return
    if isinstance(event, RunPreviewLabelsDebugEvent):
        emit_console_line(
            console,
            f"[debug] sample labels first preview entry (first 10) = {list(event.labels)}",
            markup=False,
        )
        return
    if isinstance(event, RunExecutePipelineEvent):
        transition_progress_step(
            console,
            "load_model",
            from_tag="INIT",
            from_message="Loading model",
            to_key="execute",
            from_emoji="🔧",
        )
        _emit_status_line(
            "EXEC",
            f"Executing pipeline with {event.guard_count} guards...",
            emoji="⚙️",
        )
        return
    if isinstance(event, RunLoadModelOnceEvent):
        begin_progress_step(console, "load_model")
        _emit_status_line(
            "INIT",
            f"Loading model once: {event.model_id}",
            emoji="🔧",
        )
        return
    if isinstance(event, RunSnapshotModeEvent):
        state = "enabled" if event.enabled else "disabled"
        _emit_status_line("INIT", f"Snapshot mode: {state}", emoji="💾")
        return
    if isinstance(event, RunAttemptStartedEvent):
        message = (
            f"Attempt {event.attempt}/{event.max_attempts}"
            if event.max_attempts is not None
            else f"Attempt {event.attempt}"
        )
        _emit_status_line("EXEC", message, emoji="🚀")
        return
    if isinstance(event, RunRetryAttemptStartedEvent):
        _emit_status_line(
            "EXEC",
            f"Retry attempt {event.attempt}/{event.max_attempts}",
            emoji="🔄",
        )
        return
    if isinstance(event, RunTelemetrySavedEvent):
        _emit_status_line("DATA", f"Telemetry: {event.path}", emoji="📈")
        return
    if isinstance(event, RunTelemetryFailedEvent):
        _emit_status_line(
            "WARN",
            f"Telemetry export failed: {event.error}",
            emoji="⚠️",
        )
        return
    if isinstance(event, RunEvaluationReportStartedEvent):
        _emit_status_line("EXEC", "Generating evaluation report...", emoji="📜")
        return
    if isinstance(event, RunEvaluationReportPassedEvent):
        _emit_status_line("PASS", "Evaluation report PASSED all gates!", emoji="✅")
        return
    if isinstance(event, RunEvaluationReportFailedEvent):
        _emit_status_line(
            "FAIL",
            "Evaluation report FAILED gates: " + ", ".join(event.gate_codes),
            emoji="⚠️",
        )
        return
    if isinstance(event, RunAutoTuneAdjustmentEvent):
        _emit_status_line(
            "INIT",
            "Auto-tune adjust: global_k → "
            f"{event.global_k} (bounds {event.keep_low}-{event.keep_high})",
            emoji="🔧",
        )
        return
    if isinstance(event, RunRetryExhaustedEvent):
        _emit_status_line(
            "FAIL",
            f"Exhausted retry budget after {event.attempt} attempts",
            emoji="❌",
        )
        return
    if isinstance(event, RunRetryValidationErrorEvent):
        _emit_status_line(
            "WARN",
            f"Evaluation report validation failed: {event.summary}",
            emoji="⚠️",
        )
        return
    if isinstance(event, RunCleanupStatusEvent):
        status = "removed" if event.removed else "skipped"
        _emit_status_line("INFO", f"Cleanup: {status}", emoji="🧹")
        return
