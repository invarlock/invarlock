from __future__ import annotations

import os
from typing import Any


def telemetry_summary_line(evaluation_report: dict[str, Any]) -> str | None:
    telemetry = evaluation_report.get("telemetry")
    if not isinstance(telemetry, dict):
        return None
    summary = telemetry.get("summary_line")
    if isinstance(summary, str) and summary.strip():
        return summary
    return None


def telemetry_output_enabled() -> bool:
    return str(os.environ.get("INVARLOCK_TELEMETRY", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


__all__ = ["telemetry_output_enabled", "telemetry_summary_line"]
