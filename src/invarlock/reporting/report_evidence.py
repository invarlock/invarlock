"""Owner helpers for report-side evidence payload construction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .report_types import RunReport

_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)
_EVIDENCE_DUMP_EXCEPTIONS = (AttributeError, OSError, TypeError, ValueError)


def maybe_dump_guard_evidence(
    target_dir: str | Path, payload: dict[str, Any]
) -> Path | None:
    """Dump a small JSON blob of guard decision inputs when enabled."""

    if os.getenv("INVARLOCK_EVIDENCE_DEBUG", "0").strip() != "1":
        return None
    try:
        path = Path(target_dir)
        path.mkdir(parents=True, exist_ok=True)
        out = path / "guards_evidence.json"
        out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out
    except _EVIDENCE_DUMP_EXCEPTIONS:
        return None


def build_guard_evidence_payload(report: RunReport) -> dict[str, Any]:
    """Build the compact guard evidence payload persisted with report bundles."""
    try:
        guard_ctx = report.get("guards") or []
    except _NON_FATAL_EXCEPTIONS:
        guard_ctx = []

    if not isinstance(guard_ctx, list) or not guard_ctx:
        return {"guards_decisions": []}

    tiny: list[dict[str, object]] = []
    guard_items: list[Any] = list(guard_ctx)
    for guard in guard_items:
        if not isinstance(guard, dict):
            continue
        entry: dict[str, object] = {}
        policy = guard.get("policy") or {}
        if isinstance(policy, dict):
            for key in (
                "deadband",
                "min_effect_lognll",
                "max_caps",
                "sigma_quantile",
            ):
                if key in policy:
                    entry[key] = policy[key]
        if guard.get("name"):
            entry["name"] = guard.get("name")
        if entry:
            tiny.append(entry)
    return {"guards_decisions": tiny}


__all__ = ["build_guard_evidence_payload", "maybe_dump_guard_evidence"]
