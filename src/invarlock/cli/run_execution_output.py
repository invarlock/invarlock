"""Shell-facing output helpers for config-driven run execution."""

from __future__ import annotations

from typing import Any

from invarlock.cli import output as output_mod
from invarlock.cli.output import resolve_output_style
from invarlock.cli.run_shell_output import (
    _device_resolution_note,
    _event,
    _format_kv_line,
    _print_guard_metric_impact_summary,
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
    RunGuardMetricImpactSummaryEvent,
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


def _emit_status_line(
    console: Any,
    tag: str,
    message: str,
    *,
    emoji: str | None = None,
) -> None:
    _event(console, tag, message, emoji=emoji)


def _render_setup_event(console: Any, event: RunExecutionEvent) -> bool:
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
        return True
    if isinstance(event, RunOutputDirectoryReadyEvent):
        emit_console_line(
            console, _format_kv_line("Output", event.run_dir), markup=False
        )
        emit_console_line(
            console, _format_kv_line("Run ID", event.run_id), markup=False
        )
        return True
    if isinstance(event, RunEditSelectedEvent):
        emit_console_line(
            console, _format_kv_line("Edit", event.edit_name), markup=False
        )
        return True
    if isinstance(event, RunGuardChainResolvedEvent):
        emit_console_line(
            console,
            _format_kv_line("Guards", " → ".join(event.guard_names)),
            markup=False,
        )
        return True
    return False


def _render_diagnostic_event(console: Any, event: RunExecutionEvent) -> bool:
    if not isinstance(event, RunDiagnosticEvent):
        return False
    code = event.code or ""
    context = event.context
    diagnostic_messages = {
        "export_tokenizer_missing": (
            "WARN",
            "Exported model checkpoint without tokenizer artifacts; local tokenizer reload may fail.",
            "⚠️",
        ),
        "export_adapter_directory_missing": (
            "WARN",
            "Model export requested but adapter did not save a HF directory.",
            "⚠️",
        ),
        "export_failed": (
            "WARN",
            "Model export requested but failed due to an unexpected error.",
            "⚠️",
        ),
    }
    if code == "guard_missing":
        _event(
            console,
            "WARN",
            f"Guard '{context.get('guard_name', '')}' not found, skipping",
            emoji="⚠️",
        )
        return True
    if code in diagnostic_messages:
        tag, message, emoji = diagnostic_messages[code]
        _event(console, tag, message, emoji=emoji)
        return True
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
        return True
    if code == "retry_validation_telemetry_summary":
        emit_console_line(console, str(context.get("summary", "")), markup=False)
        return True
    if code == "metric_diffs_debug":
        emit_console_line(
            console,
            f"[debug] DEBUG_METRIC_DIFFS: {context.get('summary', '')}",
            markup=False,
        )
        return True
    if isinstance(event.summary, str) and event.summary:
        tag = str(event.level or context.get("tag") or "INFO").upper()
        emoji = context.get("emoji")
        if not isinstance(emoji, str):
            emoji = None
        _event(console, tag, event.summary, emoji=emoji)
        return True
    return False


def _render_metric_or_failure_event(console: Any, event: RunExecutionEvent) -> bool:
    if isinstance(event, RunGuardMetricImpactSummaryEvent):
        _print_guard_metric_impact_summary(
            console,
            event.guard_metric_impact_info or {},
            default_limit=float(event.default_limit or 0.01),
        )
        return True
    if isinstance(event, RunRetrySummaryEvent):
        summary = event.summary
        emit_console_blank_line(console)
        _event(
            console,
            "METRIC",
            f"Retry Summary: {summary.get('total_attempts', 0)} attempts in {float(summary.get('elapsed_time', 0.0) or 0.0):.1f}s",
            emoji="📊",
        )
        return True
    if isinstance(event, RunPrimaryMetricSummaryEvent):
        _emit_status_line(
            console,
            "METRIC",
            f"Primary Metric [{event.metric_kind}] — preview: {event.preview:.3f}, final: {event.final:.3f}",
            emoji="📌",
        )
        if event.ratio_vs_baseline is not None:
            _emit_status_line(
                console,
                "METRIC",
                f"Ratio vs baseline [{event.metric_kind}]: {event.ratio_vs_baseline:.3f}",
                emoji="🔗",
            )
        return True
    if not isinstance(event, RunFailureEvent):
        return False
    failure = event.failure
    code = failure.code
    context = failure.context
    if code == "torch_missing":
        _emit_status_line(
            console,
            "FAIL",
            'Torch is required for this command. Install extras with: pip install "invarlock[hf]" or "invarlock[adapters]".',
            emoji="❌",
        )
        return True
    if code == "edit_name_missing":
        _emit_status_line(
            console,
            "FAIL",
            "Edit configuration must specify a non-empty `edit.name`.",
            emoji="❌",
        )
        return True
    if code == "unknown_edit":
        _emit_status_line(
            console,
            "FAIL",
            f"Unknown edit '{context.get('edit_name', '')}'.",
            emoji="❌",
        )
        return True
    if code == "baseline_windows_missing":
        _emit_status_line(console, "FAIL", str(failure.summary or ""), emoji="❌")
        return True
    if code == "guard_metric_impact_budget_exceeded":
        degradation_limit = float(context.get("degradation_limit", 0.01) or 0.01)
        degradation_basis = context.get("degradation_basis")
        if degradation_basis == "absolute_drop":
            budget = f">{degradation_limit * 100:.1f} pp drop"
        elif degradation_basis == "relative_increase":
            budget = f">{degradation_limit * 100:.1f}% increase"
        else:
            budget = f">{degradation_limit:.4g} degradation"
        _emit_status_line(
            console,
            "FAIL",
            f"Guard metric impact gate exceeded the configured budget ({budget})",
            emoji="❌",
        )
        return True
    if code == "guard_metric_impact_unavailable":
        reason = str(context.get("reason", "required evidence is unavailable"))
        _emit_status_line(
            console,
            "FAIL",
            f"Guard metric impact gate could not be evaluated: {reason}.",
            emoji="❌",
        )
        return True
    if code == "config_file_missing":
        _emit_status_line(
            console,
            "FAIL",
            f"Configuration file not found: {context.get('path', '')}",
            emoji="❌",
        )
        return True
    if code == "schema_invalid_run_report":
        _emit_status_line(
            console,
            "FAIL",
            "Schema invalid: run report structure failed validation",
            emoji="❌",
        )
        return True
    if code == "pipeline_failed":
        _emit_status_line(
            console,
            "FAIL",
            f"Pipeline execution failed: {failure.summary or ''}",
            emoji="❌",
        )
        return True
    _emit_status_line(console, "FAIL", str(failure.summary or code), emoji="❌")
    return True


def _render_progress_or_debug_event(console: Any, event: RunExecutionEvent) -> bool:
    if isinstance(event, RunCalibrationBatchSizesDebugEvent):
        emit_console_line(
            console,
            "[debug] calibration batch size => preview="
            f"{event.preview_count} final={event.final_count} total={event.total_count}",
            markup=False,
        )
        return True
    if isinstance(event, RunMaskedTokensDebugEvent):
        emit_console_line(
            console,
            f"[debug] masked tokens (preview/final) = {event.preview_masked}/{event.final_masked}",
            markup=False,
        )
        return True
    if isinstance(event, RunPreviewLabelsDebugEvent):
        emit_console_line(
            console,
            f"[debug] sample labels first preview entry (first 10) = {list(event.labels)}",
            markup=False,
        )
        return True
    lifecycle_messages = {
        RunDeterministicSeedsEvent: (
            "INIT",
            lambda e: (
                "Deterministic seeds → "
                f"python={e.python_seed}, numpy={e.numpy_seed}, torch={e.torch_seed if e.torch_seed is not None else 'N/A'}"
            ),
            "🎲",
        ),
        RunBaselineScheduleLoadedEvent: (
            "DATA",
            lambda _e: "Loaded baseline evaluation schedule for pairing",
            "🧬",
        ),
        RunPipelineStartedEvent: (
            "INIT",
            lambda _e: "Starting InvarLock pipeline...",
            "🚀",
        ),
        RunConfigLoadingEvent: (
            "INIT",
            lambda e: f"Loading configuration: {e.config_path}",
            "📋",
        ),
        RunAdapterSelectedEvent: (
            "DATA",
            lambda e: f"Adapter: {e.adapter_name}",
            "🔌",
        ),
        RunDatasetLoadingEvent: (
            "DATA",
            lambda e: f"Loading dataset: {e.provider}",
            "📊",
        ),
        RunTelemetrySavedEvent: (
            "DATA",
            lambda e: f"Telemetry: {e.path}",
            "📈",
        ),
        RunTelemetryFailedEvent: (
            "WARN",
            lambda e: f"Telemetry export failed: {e.error}",
            "⚠️",
        ),
        RunEvaluationReportStartedEvent: (
            "EXEC",
            lambda _e: "Generating evaluation report...",
            "📜",
        ),
        RunEvaluationReportPassedEvent: (
            "PASS",
            lambda _e: "Evaluation report PASSED all gates!",
            "✅",
        ),
        RunEvaluationReportFailedEvent: (
            "FAIL",
            lambda e: "Evaluation report FAILED gates: " + ", ".join(e.gate_codes),
            "⚠️",
        ),
        RunAutoTuneAdjustmentEvent: (
            "INIT",
            lambda e: (
                "Auto-tune adjust: global_k → "
                f"{e.global_k} (bounds {e.keep_low}-{e.keep_high})"
            ),
            "🔧",
        ),
        RunRetryExhaustedEvent: (
            "FAIL",
            lambda e: f"Exhausted retry budget after {e.attempt} attempts",
            "❌",
        ),
        RunRetryValidationErrorEvent: (
            "WARN",
            lambda e: f"Evaluation report validation failed: {e.summary}",
            "⚠️",
        ),
        RunCleanupStatusEvent: (
            "INFO",
            lambda e: f"Cleanup: {'removed' if e.removed else 'skipped'}",
            "🧹",
        ),
    }
    for event_type, (tag, message_fn, emoji) in lifecycle_messages.items():
        if isinstance(event, event_type):
            _emit_status_line(console, tag, message_fn(event), emoji=emoji)
            return True
    return False


def _render_execution_progress_event(console: Any, event: RunExecutionEvent) -> bool:
    if isinstance(event, RunConfigLoadedEvent):
        return True
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
            console,
            "EXEC",
            f"Executing pipeline with {event.guard_count} guards...",
            emoji="⚙️",
        )
        return True
    if isinstance(event, RunLoadModelOnceEvent):
        begin_progress_step(console, "load_model")
        _emit_status_line(
            console,
            "INIT",
            f"Loading model once: {event.model_id}",
            emoji="🔧",
        )
        return True
    if isinstance(event, RunSnapshotModeEvent):
        state = "enabled" if event.enabled else "disabled"
        _emit_status_line(console, "INIT", f"Snapshot mode: {state}", emoji="💾")
        return True
    if isinstance(event, RunAttemptStartedEvent):
        message = (
            f"Attempt {event.attempt}/{event.max_attempts}"
            if event.max_attempts is not None
            else f"Attempt {event.attempt}"
        )
        _emit_status_line(console, "EXEC", message, emoji="🚀")
        return True
    if isinstance(event, RunRetryAttemptStartedEvent):
        _emit_status_line(
            console,
            "EXEC",
            f"Retry attempt {event.attempt}/{event.max_attempts}",
            emoji="🔄",
        )
        return True
    return False


def render_run_execution_event(console: Any, event: RunExecutionEvent) -> None:
    if _render_setup_event(console, event):
        return
    if _render_diagnostic_event(console, event):
        return
    if _render_metric_or_failure_event(console, event):
        return
    if _render_progress_or_debug_event(console, event):
        return
    if _render_execution_progress_event(console, event):
        return
