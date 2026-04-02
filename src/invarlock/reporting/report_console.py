from __future__ import annotations

import json
from typing import Any

from invarlock.public_contracts import load_json_contract

_CONSOLE_LABELS_DEFAULT = [
    "Primary Metric Acceptable",
    "Preview Final Drift Acceptable",
    "Guard Overhead Acceptable",
    "Invariants Pass",
    "Spectral Stable",
    "Rmt Stable",
]


def load_console_labels() -> list[str]:
    """Load console labels allow-list from contracts with a safe fallback."""
    try:
        data = load_json_contract("console_labels.json")
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return list(data)
    except (OSError, TypeError, ValueError):
        pass
    return list(_CONSOLE_LABELS_DEFAULT)


def compute_console_validation_block(
    evaluation_report: dict[str, Any],
) -> dict[str, Any]:
    """Produce a normalized console validation block from an evaluation report."""
    labels = load_console_labels()
    validation = evaluation_report.get("validation", {}) or {}
    guard_ctx = evaluation_report.get("guard_overhead", {}) or {}
    guard_evaluated = (
        bool(guard_ctx.get("evaluated")) if isinstance(guard_ctx, dict) else False
    )

    def _to_key(label: str) -> str:
        return label.strip().lower().replace(" ", "_")

    rows: list[dict[str, Any]] = []
    ok_map: dict[str, bool] = {}
    effective_labels: list[str] = []
    for label in labels:
        key = _to_key(label)
        ok = bool(validation.get(key, False))
        if key == "guard_overhead_acceptable" and not guard_evaluated:
            continue
        rows.append(
            {
                "label": label,
                "status": "✅ PASS" if ok else "❌ FAIL",
                "evaluated": key != "guard_overhead_acceptable" or guard_evaluated,
                "ok": ok,
            }
        )
        effective_labels.append(label)
        ok_map[key] = ok

    keys_for_overall = [
        "primary_metric_acceptable",
        "preview_final_drift_acceptable",
        "invariants_pass",
        "spectral_stable",
        "rmt_stable",
    ]
    if guard_evaluated:
        keys_for_overall.append("guard_overhead_acceptable")

    overall_pass = all(ok_map.get(key, False) for key in keys_for_overall)
    return {"labels": effective_labels, "rows": rows, "overall_pass": overall_pass}


def compute_report_hash(evaluation_report: dict[str, Any]) -> str:
    """Compute a stable integrity hash for an evaluation report."""
    cert_copy = dict(evaluation_report or {})
    cert_copy.pop("artifacts", None)
    cert_str = json.dumps(cert_copy, sort_keys=True)
    import hashlib as _hash

    return _hash.sha256(cert_str.encode()).hexdigest()[:16]


def build_console_summary_pack(evaluation_report: dict[str, Any]) -> dict[str, Any]:
    """Build a reusable console summary from an evaluation report."""
    block = compute_console_validation_block(evaluation_report)
    overall_pass = bool(block.get("overall_pass"))
    emoji = "✅" if overall_pass else "❌"
    overall_line = f"Overall Status: {emoji} {'PASS' if overall_pass else 'FAIL'}"

    gate_lines: list[str] = []
    for row in block.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        label = row.get("label", "Gate")
        status = row.get("status", "")
        gate_lines.append(f"{label}: {status}")

    return {
        "overall_pass": overall_pass,
        "overall_line": overall_line,
        "gate_lines": gate_lines,
        "labels": block.get("labels", []),
    }


__all__ = [
    "load_console_labels",
    "compute_console_validation_block",
    "compute_report_hash",
    "build_console_summary_pack",
]
