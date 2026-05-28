from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_NON_FATAL_EXCEPTIONS = (AttributeError, OSError, TypeError, ValueError)


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
    except _NON_FATAL_EXCEPTIONS:
        # Never raise in evidence hooks.
        return None


__all__ = ["maybe_dump_guard_evidence"]
