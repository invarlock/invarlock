"""CLI-only output helpers for run command execution."""

from __future__ import annotations

from typing import Any

from rich.console import Console

from invarlock.cli.output import OutputStyle, print_event, resolve_output_style
from invarlock.core.run_guard_overhead_policy import (
    build_guard_overhead_summary as _build_guard_overhead_summary_impl,
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


def _print_guard_overhead_summary(
    console: Console,
    guard_overhead_info: dict[str, Any],
    *,
    default_threshold: float = 0.01,
) -> float:
    """Print a concise guard-overhead console summary. Returns threshold fraction used."""

    summary = _build_guard_overhead_summary_impl(
        guard_overhead_info,
        default_threshold=default_threshold,
    )
    if not summary.evaluated:
        _event(console, "METRIC", "Guard Overhead: not evaluated", emoji="🛡️")
        return summary.threshold_fraction
    status = "PASS" if summary.passed else "FAIL"
    if summary.overhead_percent is not None:
        overhead_display = f"{summary.overhead_percent:+.2f}%"
    elif summary.overhead_ratio is not None:
        overhead_display = f"{summary.overhead_ratio:.3f}x"
    else:
        overhead_display = "not evaluated"
    threshold_display = f"≤ +{summary.threshold_fraction * 100:.1f}%"
    _event(
        console,
        "METRIC",
        f"Guard Overhead: {status} {overhead_display} ({threshold_display})",
        emoji="🛡️",
    )
    return summary.threshold_fraction


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
