from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ExtractPairingScheduleFn = Callable[[dict[str, Any] | None], dict[str, Any] | None]


@dataclass(frozen=True)
class BaselineEvidenceLoadResult:
    report_data: dict[str, Any] | None
    pairing_schedule: dict[str, Any] | None
    tokenizer_hash: str | None
    status: str
    message: str | None = None


def _merge_pairing_schedule(
    report_data: dict[str, Any], pairing_schedule: dict[str, Any]
) -> None:
    evaluation_windows = report_data.get("evaluation_windows")
    if not isinstance(evaluation_windows, dict):
        evaluation_windows = {}
        report_data["evaluation_windows"] = evaluation_windows

    for arm in ("preview", "final"):
        source = pairing_schedule.get(arm)
        if not isinstance(source, dict):
            continue
        target = evaluation_windows.get(arm)
        if not isinstance(target, dict):
            evaluation_windows[arm] = dict(source)
            continue
        for key, value in source.items():
            target[key] = value


def _harvest_tokenizer_hash(
    report_data: dict[str, Any], tokenizer_hash: str | None
) -> str | None:
    if tokenizer_hash:
        return tokenizer_hash
    meta = report_data.get("meta") if isinstance(report_data.get("meta"), dict) else {}
    data = report_data.get("data") if isinstance(report_data.get("data"), dict) else {}
    candidate = meta.get("tokenizer_hash") or data.get("tokenizer_hash")
    if isinstance(candidate, str) and candidate:
        return candidate
    return tokenizer_hash


def load_baseline_pairing_evidence(
    *,
    baseline_path: Path,
    tokenizer_hash: str | None,
    extract_pairing_schedule_fn: ExtractPairingScheduleFn,
) -> BaselineEvidenceLoadResult:
    path_str = str(baseline_path)
    missing_path_message = (
        "PAIRING-EVIDENCE-MISSING: baseline report path does not exist "
        f"({path_str})"
    )
    if not baseline_path.exists():
        return BaselineEvidenceLoadResult(
            report_data=None,
            pairing_schedule=None,
            tokenizer_hash=tokenizer_hash,
            status="missing_path",
            message=missing_path_message,
        )

    try:
        loaded = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        return BaselineEvidenceLoadResult(
            report_data=None,
            pairing_schedule=None,
            tokenizer_hash=tokenizer_hash,
            status="parse_failed",
            message=f"PAIRING-EVIDENCE-MISSING: baseline report JSON parse failed ({exc})",
        )

    if not isinstance(loaded, dict):
        return BaselineEvidenceLoadResult(
            report_data=None,
            pairing_schedule=None,
            tokenizer_hash=tokenizer_hash,
            status="invalid_report",
            message=(
                "PAIRING-EVIDENCE-MISSING: baseline report missing or invalid "
                f"evaluation_windows ({path_str})"
            ),
        )

    pairing_schedule = extract_pairing_schedule_fn(loaded)
    if not pairing_schedule:
        return BaselineEvidenceLoadResult(
            report_data=None,
            pairing_schedule=None,
            tokenizer_hash=tokenizer_hash,
            status="missing_schedule",
            message=(
                "PAIRING-EVIDENCE-MISSING: baseline report missing or invalid "
                f"evaluation_windows ({path_str})"
            ),
        )

    _merge_pairing_schedule(loaded, pairing_schedule)
    resolved_tokenizer_hash = _harvest_tokenizer_hash(loaded, tokenizer_hash)
    return BaselineEvidenceLoadResult(
        report_data=loaded,
        pairing_schedule=pairing_schedule,
        tokenizer_hash=resolved_tokenizer_hash,
        status="loaded",
        message=None,
    )
