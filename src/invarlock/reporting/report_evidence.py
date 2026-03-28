"""Owner helpers for report-side evidence payload construction."""

from __future__ import annotations

from typing import Any

from .report_types import RunReport


def build_guard_evidence_payload(report: RunReport) -> dict[str, Any]:
    """Build the compact guard evidence payload persisted with report bundles."""
    try:
        guard_ctx = report.get("guards") or []
    except Exception:
        guard_ctx = []

    if not isinstance(guard_ctx, list) or not guard_ctx:
        return {"guards_decisions": []}

    tiny: list[dict[str, object]] = []
    for guard in guard_ctx:
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


__all__ = ["build_guard_evidence_payload"]
