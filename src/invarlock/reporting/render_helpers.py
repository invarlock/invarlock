from __future__ import annotations

import math
from typing import Any


def _short_digest(v: str) -> str:
    v = str(v)
    return v if len(v) <= 16 else (v[:8] + "…" + v[-8:])


def _fmt_by_kind(x: Any, k: str) -> str:
    try:
        xv = float(x)
    except (TypeError, ValueError):
        return "N/A"
    k = str(k).lower()
    if k in {"accuracy", "vqa_accuracy"}:
        return f"{xv * 100.0:.1f}"
    if k.startswith("ppl"):
        return f"{xv:.3g}"
    return f"{xv:.3f}"


def _fmtv(key: str, v: Any) -> str:
    if not (isinstance(v, int | float) and math.isfinite(float(v))):
        return "-"
    if key.startswith("latency_ms_"):
        return f"{float(v):.0f}"
    if key.startswith("throughput_"):
        return f"{float(v):.1f}"
    return f"{float(v):.3f}"


def _p(x: Any) -> str:
    try:
        return f"{float(x) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return "N/A"
