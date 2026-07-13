"""Evaluation-window serialization helpers for run policy."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_sequence_payload(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _list_payload(value: Any) -> list[Any]:
    return list(value) if _is_sequence_payload(value) else []


def _nested_list_payload(value: Any) -> list[list[Any]]:
    if not _is_sequence_payload(value):
        return []
    payload: list[list[Any]] = []
    for item in value:
        if _is_sequence_payload(item):
            payload.append(list(item))
    return payload


def _portable_reference(value: Any) -> str | None:
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    if value.startswith("~") or "\\" in value:
        return None
    candidate = PurePosixPath(value)
    parts = candidate.parts
    if (
        candidate.is_absolute()
        or not parts
        or parts[0].endswith(":")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        return None
    return candidate.as_posix()


def _portable_multimodal_record(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    runtime_image_path = payload.pop("image_path", None)
    payload.pop("source_file", None)

    image_ref = _portable_reference(payload.get("image_ref"))
    if image_ref is None:
        image_ref = _portable_reference(runtime_image_path)
    image_sha256 = payload.get("image_sha256")
    if (
        image_ref is None
        and isinstance(image_sha256, str)
        and _SHA256_HEX_RE.fullmatch(image_sha256)
    ):
        image_ref = f"sha256:{image_sha256}"
    if image_ref is None:
        payload.pop("image_ref", None)
    else:
        payload["image_ref"] = image_ref

    source_ref = _portable_reference(payload.get("source_ref"))
    if source_ref is None:
        payload.pop("source_ref", None)
    else:
        payload["source_ref"] = source_ref
    return payload


def _window_payload(window: Mapping[str, Any] | None) -> dict[str, Any]:
    window_map = dict(window or {})
    payload: dict[str, Any] = {
        "window_ids": _list_payload(window_map.get("window_ids", [])),
        "example_ids": [
            str(value) for value in _list_payload(window_map.get("example_ids", []))
        ],
        "logloss": _list_payload(window_map.get("logloss", [])),
        "input_ids": _nested_list_payload(window_map.get("input_ids", [])),
        "attention_masks": _nested_list_payload(window_map.get("attention_masks", [])),
        "token_counts": _list_payload(window_map.get("token_counts", [])),
        "masked_token_counts": _list_payload(window_map.get("masked_token_counts", [])),
        "actual_token_counts": _list_payload(window_map.get("actual_token_counts", [])),
        "labels": _nested_list_payload(window_map.get("labels", [])),
    }
    records = window_map.get("records", [])
    if isinstance(records, list):
        payload["records"] = [
            _portable_multimodal_record(record)
            for record in records
            if isinstance(record, Mapping)
        ]
    input_records = window_map.get("input_records")
    if isinstance(input_records, list):
        payload["input_records"] = [
            _portable_multimodal_record(record)
            for record in input_records
            if isinstance(record, Mapping)
        ]
    processor_sha = window_map.get("processor_sha256")
    if isinstance(processor_sha, str) and processor_sha:
        payload["processor_sha256"] = processor_sha
    processor_identity = window_map.get("processor_identity")
    if isinstance(processor_identity, Mapping):
        payload["processor_identity"] = dict(processor_identity)
    return payload


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
    multimodal_records = [
        _portable_multimodal_record(record)
        for record in records
        if "image_path" in record or "example_id" in record or "answers" in record
    ]
    if multimodal_records:
        multimodal_payload: dict[str, Any] = {
            "example_ids": [
                str(record.get("example_id") or record.get("id") or "")
                for record in multimodal_records
            ],
            "records": multimodal_records,
        }
        processor_sha = next(
            (
                str(record.get("processor_sha256"))
                for record in multimodal_records
                if isinstance(record.get("processor_sha256"), str)
                and str(record.get("processor_sha256")).strip()
            ),
            None,
        )
        if processor_sha:
            multimodal_payload["processor_sha256"] = processor_sha
        processor_identity = next(
            (
                record.get("processor_identity")
                for record in multimodal_records
                if isinstance(record.get("processor_identity"), Mapping)
            ),
            None,
        )
        if isinstance(processor_identity, Mapping):
            multimodal_payload["processor_identity"] = dict(processor_identity)
        return multimodal_payload

    sequence_payload: dict[str, Any] = {
        "window_ids": list(range(start_index, start_index + len(records))),
        "input_ids": [list(record["input_ids"]) for record in records],
        "attention_masks": [list(record["attention_mask"]) for record in records],
        "token_counts": [_token_count(record) for record in records],
    }
    if use_mlm:
        sequence_payload["masked_token_counts"] = list(mask_counts or [])
        sequence_payload["labels"] = [
            record.get("labels", [-100] * len(record["input_ids"]))
            for record in records
        ]
    return sequence_payload


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
