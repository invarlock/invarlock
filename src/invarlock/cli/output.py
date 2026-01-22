from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TextIO

from rich.console import Console

_STYLE_AUDIT = "audit"
_STYLE_FRIENDLY = "friendly"
_VALID_STYLES = {_STYLE_AUDIT, _STYLE_FRIENDLY}


def _safe_console_print(console: Console, *args: object, **kwargs: object) -> None:
    try:
        console.print(*args, **kwargs)
    except TypeError:
        console.print(*args)


def env_no_color() -> bool:
    """Return True when NO_COLOR is set (value-agnostic)."""
    return bool(str(os.environ.get("NO_COLOR", "")).strip())


def perf_counter() -> float:
    return time.perf_counter()


@dataclass(frozen=True, slots=True)
class OutputStyle:
    name: str
    progress: bool = False
    timing: bool = False
    color: bool = True

    @property
    def emojis(self) -> bool:
        return self.name != _STYLE_AUDIT

    @property
    def audit(self) -> bool:
        return self.name == _STYLE_AUDIT


def normalize_style(style: str | None) -> str | None:
    if style is None:
        return None
    value = str(style).strip().lower()
    if not value:
        return None
    return value if value in _VALID_STYLES else None


def resolve_style_name(style: str | None, profile: str | None) -> str:
    normalized = normalize_style(style)
    if normalized is not None:
        return normalized
    profile_norm = str(profile or "").strip().lower()
    if profile_norm in {"ci", "ci_cpu", "release"}:
        return _STYLE_AUDIT
    return _STYLE_FRIENDLY


def resolve_output_style(
    *,
    style: str | None,
    profile: str | None,
    progress: bool = False,
    timing: bool = False,
    no_color: bool = False,
) -> OutputStyle:
    name = resolve_style_name(style, profile)
    color_enabled = not (bool(no_color) or env_no_color())
    return OutputStyle(
        name=name,
        progress=bool(progress),
        timing=bool(timing),
        color=color_enabled,
    )


def make_console(
    *,
    file: TextIO | None = None,
    force_terminal: bool | None = None,
    no_color: bool | None = None,
) -> Console:
    if no_color is None:
        no_color = env_no_color()
    if no_color:
        color_system = None
    else:
        color_system = "standard" if force_terminal else "auto"
    return Console(
        file=file,
        force_terminal=force_terminal,
        no_color=bool(no_color),
        color_system=color_system,
    )


def format_event_line(
    tag: str,
    message: str,
    *,
    style: OutputStyle,
    emoji: str | None = None,
) -> str:
    tag_norm = str(tag or "").strip().upper() or "INFO"
    if style.emojis and emoji:
        prefix = emoji
    else:
        prefix = f"[{tag_norm}]"
    msg = str(message or "").rstrip()
    return f"{prefix} {msg}".rstrip()


def print_event(
    console: Console,
    tag: str,
    message: str,
    *,
    style: OutputStyle,
    emoji: str | None = None,
    console_style: str | None = None,
) -> None:
    line = format_event_line(tag, message, style=style, emoji=emoji)
    if console_style is None and style.color:
        tag_norm = str(tag or "").strip().upper()
        if tag_norm in {"PASS"}:
            console_style = "green"
        elif tag_norm in {"FAIL", "ERROR"}:
            console_style = "red"
        elif tag_norm in {"WARN", "WARNING"}:
            console_style = "yellow"
        elif tag_norm in {"METRIC"}:
            console_style = "cyan"
    _safe_console_print(console, line, style=console_style, markup=False)


@contextmanager
def timed_step(
    *,
    console: Console,
    style: OutputStyle,
    timings: dict[str, float] | None,
    key: str,
    tag: str,
    message: str,
    emoji: str | None = None,
) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = max(0.0, float(perf_counter() - start))
        if timings is not None:
            timings[key] = elapsed
        if style.progress:
            print_event(
                console,
                tag,
                f"{message} done ({elapsed:.2f}s)",
                style=style,
                emoji=emoji,
            )


def print_timing_summary(
    console: Console,
    timings: dict[str, float],
    *,
    style: OutputStyle,
    order: list[tuple[str, str]],
    extra_lines: list[str] | None = None,
) -> None:
    if not style.timing:
        return
    _safe_console_print(console, "", markup=False)
    _safe_console_print(console, "TIMING SUMMARY", markup=False)
    for label, key in order:
        if key not in timings:
            continue
        _safe_console_print(
            console, f"  {label:<11}: {timings[key]:.2f}s", markup=False
        )
    if extra_lines:
        for line in extra_lines:
            _safe_console_print(console, line, markup=False)
