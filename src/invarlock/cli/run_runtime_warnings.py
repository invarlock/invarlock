"""Warning filtering helpers for runtime execution."""

from __future__ import annotations

import json
import logging
import os
import re
import sys as _sys
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

_NOISY_WARNING_PATTERNS = (r".*loss_type=None.*unrecognized.*",)
_LOG_MESSAGE_ERRORS = (RuntimeError, TypeError, ValueError)


def _resolve_warning_suppression(profile: str | None) -> tuple[bool, bool]:
    suppress_all = os.getenv("INVARLOCK_SUPPRESS_WARNINGS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    profile_norm = (profile or "").strip().lower()
    enabled = bool(suppress_all) or profile_norm in {"ci", "ci_cpu", "release"}
    return enabled, suppress_all


def _apply_warning_filters(profile: str | None) -> bool:
    enabled, suppress_all = _resolve_warning_suppression(profile)
    if not enabled:
        return False
    if suppress_all:
        warnings.simplefilter("ignore")
    else:
        for pattern in _NOISY_WARNING_PATTERNS:
            warnings.filterwarnings("ignore", message=pattern)
    return True


class FilteredWarningStream:
    """File-like wrapper that suppresses matched warning chunks."""

    def __init__(self, raw: Any, patterns: Sequence[re.Pattern[str]], sink: list[str]):
        self._raw = raw
        self._patterns = patterns
        self._sink = sink

    @property
    def encoding(self) -> str | None:
        return getattr(self._raw, "encoding", None)

    @property
    def errors(self) -> str | None:
        return getattr(self._raw, "errors", None)

    @property
    def buffer(self) -> Any:
        return self._raw.buffer

    @property
    def closed(self) -> bool:
        return bool(getattr(self._raw, "closed", False))

    def fileno(self) -> int:
        return int(self._raw.fileno())

    def isatty(self) -> bool:
        return bool(self._raw.isatty())

    def writable(self) -> bool:
        return bool(getattr(self._raw, "writable", lambda: True)())

    def write(self, s: object) -> int:
        try:
            if isinstance(s, bytes):
                text = s.decode("utf-8", errors="replace")
            else:
                text = str(s)
        except (TypeError, ValueError, UnicodeDecodeError):
            return int(self._raw.write(s))

        pieces = text.splitlines(keepends=True)
        for piece in pieces:
            if any(pattern.search(piece) for pattern in self._patterns):
                self._sink.append(piece.rstrip("\n"))
                continue
            self._raw.write(piece)
        return len(text)

    def flush(self) -> None:
        try:
            self._raw.flush()
        except (AttributeError, OSError, ValueError):
            pass


@contextmanager
def suppress_noisy_warnings(
    profile: str | None,
    *,
    event_path: Path | None = None,
    context: dict[str, Any] | None = None,
) -> Iterator[None]:
    enabled, suppress_all = _resolve_warning_suppression(profile)
    if not enabled:
        yield
        return

    prev_tf_verbosity = os.environ.get("TRANSFORMERS_VERBOSITY")
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    transformers_logger = logging.getLogger("transformers")
    prev_tf_level = transformers_logger.level
    transformers_logger.setLevel(logging.ERROR)

    patterns = [re.compile(p) for p in _NOISY_WARNING_PATTERNS]
    suppressed: list[str] = []

    class _NoisyLogFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
            try:
                message = record.getMessage()
            except _LOG_MESSAGE_ERRORS:
                return True
            if any(p.search(message) for p in patterns):
                suppressed.append(message)
                return False
            return True

    def _iter_handlers() -> list[logging.Handler]:
        handlers: list[logging.Handler] = []
        seen: set[int] = set()
        for logger in (
            logging.getLogger(),
            logging.getLogger("transformers"),
            logging.getLogger("huggingface_hub"),
            logging.getLogger("datasets"),
        ):
            for handler in getattr(logger, "handlers", []) or []:
                if id(handler) in seen:
                    continue
                seen.add(id(handler))
                handlers.append(handler)
        return handlers

    def _append_suppressed_warnings() -> None:
        if not suppressed or event_path is None:
            return
        try:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "component": "warnings",
                "operation": "suppressed",
                "level": "WARNING",
                "data": {
                    "count": len(suppressed),
                    "messages": suppressed[:50],
                    "profile": profile or "",
                    **(context or {}),
                },
            }
            event_path.parent.mkdir(parents=True, exist_ok=True)
            with event_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, allow_nan=False) + "\n")
        except (OSError, TypeError, ValueError):
            return

    log_filter = _NoisyLogFilter()
    handlers = _iter_handlers()
    for handler in handlers:
        handler.addFilter(log_filter)

    try:
        with warnings.catch_warnings():
            from contextlib import redirect_stderr, redirect_stdout

            stdout_proxy = FilteredWarningStream(_sys.stdout, patterns, suppressed)
            stderr_proxy = FilteredWarningStream(_sys.stderr, patterns, suppressed)

            with redirect_stdout(stdout_proxy), redirect_stderr(stderr_proxy):
                if suppress_all:
                    warnings.simplefilter("ignore")
                    yield
                else:
                    original_showwarning = warnings.showwarning

                    def _showwarning(
                        message: Warning | str,
                        category: type[Warning],
                        filename: str,
                        lineno: int,
                        file: TextIO | None = None,
                        line: str | None = None,
                    ) -> None:
                        try:
                            rendered = warnings.formatwarning(
                                message, category, filename, lineno, line
                            )
                        except (TypeError, ValueError):
                            rendered = str(message)
                        if any(p.search(rendered) for p in patterns):
                            suppressed.append(str(message))
                            return
                        original_showwarning(
                            message,
                            category,
                            filename,
                            lineno,
                            file=file,
                            line=line,
                        )

                    showwarning_fn: Any = _showwarning
                    warnings.showwarning = showwarning_fn
                    try:
                        yield
                    finally:
                        warnings.showwarning = original_showwarning
    finally:
        for handler in handlers:
            try:
                handler.removeFilter(log_filter)
            except ValueError:
                pass
        try:
            transformers_logger.setLevel(prev_tf_level)
        except (TypeError, ValueError):
            pass
        if prev_tf_verbosity is None:
            os.environ.pop("TRANSFORMERS_VERBOSITY", None)
        else:
            os.environ["TRANSFORMERS_VERBOSITY"] = prev_tf_verbosity
        _append_suppressed_warnings()


__all__ = [
    "FilteredWarningStream",
    "_NOISY_WARNING_PATTERNS",
    "_apply_warning_filters",
    "_resolve_warning_suppression",
    "suppress_noisy_warnings",
]
