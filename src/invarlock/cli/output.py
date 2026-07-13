from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from typing import Any, TextIO

import typer
from rich.console import Console

_STYLE_AUDIT = "audit"
_STYLE_FRIENDLY = "friendly"
_VALID_STYLES = {_STYLE_AUDIT, _STYLE_FRIENDLY}
_DEFAULT_HUMAN_PROFILE = "ci"


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


def _ts() -> str:
    return datetime.now(UTC).isoformat()


def emit(payload: Any, exit_code: int) -> None:
    """Emit a JSON payload with a stable envelope and exit."""
    payload_obj: Any = (
        asdict(payload)
        if is_dataclass(payload) and not isinstance(payload, type)
        else payload
    )
    if isinstance(payload_obj, dict):
        payload_obj.setdefault("ts", _ts())
        payload_obj.setdefault("component", "cli")
    typer.echo(json.dumps(payload_obj, sort_keys=True, allow_nan=False))
    raise typer.Exit(exit_code)


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
    return OutputStyle(
        name=name,
        progress=bool(progress),
        timing=bool(timing),
        color=not (bool(no_color) or env_no_color()),
    )


def resolve_human_output_style(*, no_color: bool = False) -> OutputStyle:
    return resolve_output_style(
        style=_STYLE_AUDIT,
        profile=_DEFAULT_HUMAN_PROFILE,
        progress=False,
        timing=False,
        no_color=no_color,
    )


def make_console(
    *,
    file: TextIO | None = None,
    force_terminal: bool | None = None,
    no_color: bool | None = None,
) -> Console:
    if no_color is None:
        no_color = env_no_color()
    color_system: str | None = None
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


def print_command_event(
    console: Console,
    tag: str,
    message: str,
    *,
    no_color: bool = False,
    emoji: str | None = None,
    console_style: str | None = None,
) -> None:
    print_event(
        console,
        tag,
        message,
        style=resolve_human_output_style(no_color=no_color),
        emoji=emoji,
        console_style=console_style,
    )


def print_command_detail(
    console: Console,
    message: str,
    *,
    prefix: str = "  ↳",
    console_style: str | None = "dim",
) -> None:
    _safe_console_print(
        console,
        f"{prefix} {str(message or '').rstrip()}".rstrip(),
        style=console_style,
        markup=False,
        soft_wrap=True,
    )


def make_command_event_emitter(
    console: Console,
    *,
    no_color: bool = False,
) -> Callable[..., None]:
    def _emit(
        tag: str,
        message: str,
        *,
        emoji: str | None = None,
        console_style: str | None = None,
    ) -> None:
        print_command_event(
            console,
            tag,
            message,
            no_color=no_color,
            emoji=emoji,
            console_style=console_style,
        )

    return _emit


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
