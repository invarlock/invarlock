from __future__ import annotations

import hashlib
import json
from typing import Any


def _measurement_contract_digest(contract: Any) -> str | None:
    if not isinstance(contract, dict) or not contract:
        return None
    try:
        canonical = json.dumps(contract, sort_keys=True, default=str)
    except (RuntimeError, TypeError, ValueError, OverflowError):
        return None
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _baseline_guard_payload(baseline: Any, guard_name: str) -> dict[str, Any]:
    """Return baseline guard payload from either an evaluation report or run report."""
    if not isinstance(baseline, dict):
        return {}
    try:
        block = baseline.get(guard_name)
        if isinstance(block, dict) and block:
            return block
        for guard in baseline.get("guards", []) or []:
            if not isinstance(guard, dict):
                continue
            if str(guard.get("name", "")).lower() != guard_name:
                continue
            metrics = guard.get("metrics")
            if isinstance(metrics, dict) and metrics:
                return metrics
            return {}
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return {}
    return {}
