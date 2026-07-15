"""CLI-only output helpers for run command execution."""

from __future__ import annotations

from typing import Any

from rich.console import Console

from invarlock.cli.output import OutputStyle, print_event, resolve_output_style
from invarlock.reporting.report_metric_impact import (
    build_guard_metric_impact_summary as _build_guard_metric_impact_summary_impl,
)

KV_LABEL_WIDTH = 10
_RETRY_SUMMARY_ERRORS = (AttributeError, KeyError, RuntimeError, TypeError, ValueError)


def _style_from_console(console: Console, profile: str | None = None) -> OutputStyle:
    style = getattr(console, "_invarlock_output_style", None)
    if isinstance(style, OutputStyle):
        return style
    return resolve_output_style(
        style=None,
        profile=profile,
        progress=False,
        timing=False,
        no_color=False,
    )


def _event(
    console: Console,
    tag: str,
    message: str,
    *,
    emoji: str | None = None,
    console_style: str | None = None,
    profile: str | None = None,
) -> None:
    style = _style_from_console(console, profile=profile)
    print_event(
        console,
        tag,
        message,
        style=style,
        emoji=emoji,
        console_style=console_style,
    )


def _format_kv_line(label: str, value: str, *, width: int = KV_LABEL_WIDTH) -> str:
    return f"  {label:<{width}}: {value}"


def _device_resolution_note(target_device: str, resolved_device: str) -> str:
    target_norm = str(target_device or "").strip().lower()
    resolved_norm = str(resolved_device or "").strip().lower()
    if not target_norm or target_norm == "auto":
        return "auto-resolved"
    if target_norm == resolved_norm:
        return "requested"
    return f"resolved from {target_device}"


def _format_guard_chain(guards: list[Any]) -> str:
    names = [str(getattr(guard, "name", "unknown")) for guard in guards]
    return " → ".join(names)


def _print_pipeline_start(console: Console) -> None:
    _event(console, "INIT", "Starting InvarLock pipeline...", emoji="🚀")


def _print_guard_metric_impact_summary(
    console: Console,
    guard_metric_impact_info: dict[str, Any],
    *,
    default_limit: float = 0.01,
) -> float:
    """Print a concise guard-metric-impact summary and return its degradation limit."""

    summary = _build_guard_metric_impact_summary_impl(
        guard_metric_impact_info,
        default_limit=default_limit,
    )
    if not summary.evaluated:
        _event(console, "METRIC", "Guard Metric Impact: not evaluated", emoji="🛡️")
        return summary.degradation_limit
    status = "PASS" if summary.passed else "FAIL"
    if summary.display_value is not None and summary.display_unit == "percent":
        impact_display = f"{summary.display_value:+.2f}%"
        limit_display = f"≤ +{summary.degradation_limit * 100:.1f}%"
    elif (
        summary.display_value is not None
        and summary.display_unit == "percentage_points"
    ):
        impact_display = f"{summary.display_value:+.2f} pp"
        limit_display = f"≤ +{summary.degradation_limit * 100:.1f} pp"
    else:
        impact_display = "display unavailable"
        limit_display = "limit unavailable"
    _event(
        console,
        "METRIC",
        f"Guard Metric Impact: {status} {impact_display} ({limit_display})",
        emoji="🛡️",
    )
    return summary.degradation_limit


def _print_retry_summary(console: Console, retry_controller: Any | None) -> None:
    """Print a one-line retry summary when retries were attempted."""

    try:
        if retry_controller and getattr(retry_controller, "attempt_history", None):
            summary = retry_controller.get_attempt_summary()
            console.print("\n")
            _event(
                console,
                "METRIC",
                f"Retry Summary: {summary['total_attempts']} attempts in {summary['elapsed_time']:.1f}s",
                emoji="📊",
            )
    except _RETRY_SUMMARY_ERRORS:
        return
