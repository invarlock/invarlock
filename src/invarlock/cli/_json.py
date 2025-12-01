from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import Any

import typer

try:
    from invarlock.core.exceptions import InvarlockError
except Exception:  # pragma: no cover - optional in minimal environments
    InvarlockError = None  # type: ignore[assignment]


def _ts() -> str:
    return datetime.now(UTC).isoformat()


def emit(payload: Any, exit_code: int) -> None:
    """Emit a JSON payload with a stable envelope and exit.

    - Adds `ts` (UTC ISO) and `component=cli` if absent
    - Accepts dicts or dataclasses
    - Exits with provided code via Typer
    """
    if is_dataclass(payload):
        payload = asdict(payload)  # type: ignore[assignment]
    if isinstance(payload, dict):
        payload.setdefault("ts", _ts())
        payload.setdefault("component", "cli")
    typer.echo(json.dumps(payload, sort_keys=True))
    raise typer.Exit(exit_code)


def encode_error(exc: Exception) -> dict[str, Any]:
    """Encode an exception as a structured error object for JSON envelopes.

    Fields:
      - code: error code if available (InvarlockError), else a generic tag
      - category: exception class name
      - recoverable: bool when available, else False
      - context: attached details when available, else {}
    """
    try:
        category = type(exc).__name__
    except Exception:
        category = "Exception"

    # Default shape
    out: dict[str, Any] = {
        "code": "E_GENERIC",
        "category": category,
        "recoverable": False,
        "context": {},
    }

    # InvarlockError dataclass provides code/details/recoverable
    try:
        if InvarlockError is not None and isinstance(exc, InvarlockError):
            out["code"] = getattr(exc, "code", out["code"]) or out["code"]
            out["recoverable"] = bool(getattr(exc, "recoverable", False))
            details = getattr(exc, "details", None)
            if isinstance(details, dict):
                out["context"] = details
            return out
    except Exception:
        # Fall back to generic encoding when anything unexpected happens
        pass

    # Heuristic: common schema/validation categories → generic schema code
    if category in {"ValidationError", "ConfigError", "DataError"}:
        out["code"] = "E_SCHEMA"

    return out
