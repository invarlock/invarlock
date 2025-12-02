from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def maybe_dump_guard_evidence(target_dir: str | Path, payload: dict[str, Any]) -> None:
    """Dump a small JSON blob of guard decision inputs when INVARLOCK_EVIDENCE_DEBUG=1.

    Keeps payload tiny; callers should pre-filter arrays and redact large fields.
    """
    if os.getenv("INVARLOCK_EVIDENCE_DEBUG", "0") != "1":
        return
    try:
        path = Path(target_dir)
        path.mkdir(parents=True, exist_ok=True)
        out = path / "guards_evidence.json"
        out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        # Never raise in evidence hook
        pass
