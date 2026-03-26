from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import Any

import typer


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
