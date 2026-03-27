from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _window_payload(window: Mapping[str, Any] | None) -> dict[str, Any]:
    window_map = dict(window or {})
    return {
        "window_ids": list(window_map.get("window_ids", [])),
        "logloss": list(window_map.get("logloss", [])),
        "input_ids": [list(seq) for seq in window_map.get("input_ids", [])],
        "attention_masks": [
            list(mask) for mask in window_map.get("attention_masks", [])
        ],
        "token_counts": list(window_map.get("token_counts", [])),
        "masked_token_counts": list(window_map.get("masked_token_counts", [])),
        "actual_token_counts": list(window_map.get("actual_token_counts", [])),
        "labels": [list(seq) for seq in window_map.get("labels", [])],
    }


def serialize_evaluation_windows(
    evaluation_windows: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]] | None:
    """Serialize runner-provided evaluation windows into plain JSON-ready data."""
    if not isinstance(evaluation_windows, Mapping) or not evaluation_windows:
        return None
    return {
        "preview": _window_payload(
            evaluation_windows.get("preview")
            if isinstance(evaluation_windows.get("preview"), Mapping)
            else None
        ),
        "final": _window_payload(
            evaluation_windows.get("final")
            if isinstance(evaluation_windows.get("final"), Mapping)
            else None
        ),
    }


def _token_count(record: Mapping[str, Any]) -> int:
    try:
        return int(len(record.get("input_ids", []) or []))
    except (TypeError, ValueError, OverflowError):
        return 0


def _fallback_window_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    start_index: int,
    use_mlm: bool,
    mask_counts: Sequence[int] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "window_ids": list(range(start_index, start_index + len(records))),
        "input_ids": [list(record["input_ids"]) for record in records],
        "attention_masks": [list(record["attention_mask"]) for record in records],
        "token_counts": [_token_count(record) for record in records],
    }
    if use_mlm:
        payload["masked_token_counts"] = list(mask_counts or [])
        payload["labels"] = [
            record.get("labels", [-100] * len(record["input_ids"]))
            for record in records
        ]
    return payload


def build_fallback_evaluation_windows(
    preview_records: Sequence[Mapping[str, Any]],
    final_records: Sequence[Mapping[str, Any]],
    *,
    use_mlm: bool,
    preview_mask_counts: Sequence[int] | None = None,
    final_mask_counts: Sequence[int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build evaluation windows from assembled records when the runner omits them."""
    preview_count = len(preview_records)
    return {
        "preview": _fallback_window_payload(
            preview_records,
            start_index=0,
            use_mlm=use_mlm,
            mask_counts=preview_mask_counts,
        ),
        "final": _fallback_window_payload(
            final_records,
            start_index=preview_count,
            use_mlm=use_mlm,
            mask_counts=final_mask_counts,
        ),
    }
