"""Baseline pairing and provider parity helpers for run command."""

from __future__ import annotations

import hashlib
from array import array
from collections.abc import Iterable, Sequence
from typing import Any

import click
import numpy as np
import typer
from rich.console import Console

from invarlock.core.metric_provider_resolution import (
    resolve_metric_and_provider as _resolve_metric_and_provider_core,
)
from invarlock.core.run_policy import (
    enforce_provider_parity as _enforce_provider_parity_core,
)

_PAIRING_INT_ERRORS = (OverflowError, TypeError, ValueError)
_PAIRING_ASSIGNMENT_ERRORS = (KeyError, TypeError, ValueError)
_IMPORT_UNSET = object()
torch: Any = _IMPORT_UNSET


def _get_torch() -> Any:
    global torch
    if torch is _IMPORT_UNSET:
        try:
            import torch as _torch
        except ImportError:
            torch = None
        else:
            torch = _torch
    return None if torch is _IMPORT_UNSET else torch


def _to_int_list(values: Sequence[int] | Iterable[int]) -> list[int]:
    return [int(v) for v in values]


def _tensor_or_list_to_ints(values: Any) -> list[int]:
    """Coerce possible tensor/list-like inputs to a list[int]."""
    torch_mod = _get_torch()
    if torch_mod is not None and hasattr(values, "tolist"):
        raw = values.tolist()
        if isinstance(raw, list):
            return _to_int_list(raw)
        try:
            return _to_int_list(list(raw))
        except (typer.Exit, SystemExit, click.exceptions.Exit):
            raise
        except (TypeError, ValueError):
            return []
    if isinstance(values, np.ndarray | list | tuple):
        return _to_int_list(list(values))
    if isinstance(values, Iterable):
        return _to_int_list(values)
    return []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hash_sequences(seqs: Sequence[Sequence[int]] | Iterable[Sequence[int]]) -> str:
    """Compute a stable digest for a sequence of integer token sequences."""
    hasher = hashlib.blake2s(digest_size=16)
    for seq in seqs:
        try:
            seq_len = len(seq)
        except TypeError:
            seq = list(seq)
            seq_len = len(seq)
        hasher.update(seq_len.to_bytes(4, "little", signed=False))
        arr = array("I", (int(token) & 0xFFFFFFFF for token in seq))
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def _compute_mask_positions_digest(windows: dict[str, Any]) -> str | None:
    """Compute a rolled hash of MLM mask positions across windows."""
    hasher = hashlib.blake2s(digest_size=16)
    any_masked = False
    for arm in ("preview", "final"):
        sec = windows.get(arm)
        if not isinstance(sec, dict):
            continue
        labels = sec.get("labels")
        if not isinstance(labels, list) or not labels:
            continue
        hasher.update(arm.encode("utf-8"))
        for row in labels:
            row_list = _tensor_or_list_to_ints(row)
            if not row_list:
                continue
            found = False
            for idx, value in enumerate(row_list):
                if int(value) != -100:
                    hasher.update(b"1")
                    hasher.update(idx.to_bytes(4, "little", signed=False))
                    found = True
            if found:
                any_masked = True
            hasher.update(b"|")
    if not any_masked:
        return None
    digest = hasher.hexdigest()
    return digest if digest else None


def _canonical_dataset_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, dict):
        candidate = (
            value.get("kind")
            or value.get("dataset")
            or value.get("name")
            or value.get("id")
            or value.get("provider")
        )
        return _canonical_dataset_id(candidate)
    if hasattr(value, "items"):
        try:
            return _canonical_dataset_id(dict(value.items()))
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pass
    for attr in ("kind", "dataset", "name", "id", "provider"):
        try:
            candidate = getattr(value, attr)
        except AttributeError:
            continue
        normalized = _canonical_dataset_id(candidate)
        if normalized is not None:
            return normalized
    try:
        normalized = str(value).strip()
    except (RuntimeError, TypeError, ValueError):
        return None
    return normalized or None


def extract_pairing_schedule(
    report: dict[str, Any] | None,
    *,
    tensor_or_list_to_ints_fn: Any | None = None,
) -> dict[str, Any] | None:
    """Extract sanitized pairing schedule from baseline-like report data."""
    if tensor_or_list_to_ints_fn is None:
        tensor_or_list_to_ints_fn = _tensor_or_list_to_ints

    if not isinstance(report, dict):
        return None
    windows = report.get("evaluation_windows")
    if not isinstance(windows, dict):
        return None

    def _wrap_single_row(raw: list[Any], *, expected_rows: int) -> list:
        if expected_rows == 1 and raw and not isinstance(raw[0], list):
            return [raw]
        return raw

    def _sanitize_multimodal(section: dict[str, Any]) -> dict[str, Any] | None:
        records_raw = section.get("input_records")
        if not isinstance(records_raw, list):
            records_raw = section.get("records")
        example_ids_raw = section.get("example_ids")
        records: list[dict[str, Any]] = []
        if isinstance(records_raw, list):
            for record in records_raw:
                if isinstance(record, dict):
                    records.append(dict(record))
        example_ids: list[str] = []
        if isinstance(example_ids_raw, list):
            example_ids = [str(value) for value in example_ids_raw]
        elif records:
            example_ids = [
                str(record.get("id") or record.get("example_id") or "")
                for record in records
            ]
        if not example_ids:
            return None
        if records and len(records) != len(example_ids):
            return None
        payload: dict[str, Any] = {"example_ids": example_ids}
        if records:
            payload["records"] = records
        processor_sha = section.get("processor_sha256")
        if isinstance(processor_sha, str) and processor_sha:
            payload["processor_sha256"] = processor_sha
        return payload

    def _sanitize(section_key: str, *, start_id: int) -> dict[str, Any] | None:
        section = windows.get(section_key)
        if not isinstance(section, dict):
            return None
        multimodal = _sanitize_multimodal(section)
        if multimodal is not None:
            return multimodal
        input_ids_raw = section.get("input_ids")
        if not isinstance(input_ids_raw, list):
            return None
        input_ids = [tensor_or_list_to_ints_fn(seq) for seq in input_ids_raw]
        if not input_ids:
            return None

        window_ids_raw = section.get("window_ids")
        window_ids: list[int] = []
        if isinstance(window_ids_raw, list):
            if len(window_ids_raw) != len(input_ids):
                return None
            for wid in window_ids_raw:
                try:
                    window_ids.append(int(wid))
                except _PAIRING_INT_ERRORS:
                    return None
        else:
            window_ids = list(range(int(start_id), int(start_id) + len(input_ids)))

        attention_raw = section.get("attention_masks")
        attention_masks: list[list[int]]
        if isinstance(attention_raw, list):
            maybe = _wrap_single_row(attention_raw, expected_rows=len(input_ids))
            if isinstance(maybe, list) and all(
                isinstance(mask, list) for mask in maybe
            ):
                attention_masks = [tensor_or_list_to_ints_fn(mask) for mask in maybe]
            else:
                attention_masks = [
                    [1 if int(token) != 0 else 0 for token in seq] for seq in input_ids
                ]
        else:
            attention_masks = [
                [1 if int(token) != 0 else 0 for token in seq] for seq in input_ids
            ]
        if len(attention_masks) != len(input_ids):
            return None
        for seq, mask in zip(input_ids, attention_masks, strict=False):
            if len(mask) != len(seq):
                return None

        def _coerce_count_list(raw: Any) -> list[int] | None:
            if isinstance(raw, bool):
                return None
            if isinstance(raw, int) and len(input_ids) == 1:
                raw = [raw]
            if not isinstance(raw, list) or len(raw) != len(input_ids):
                return None
            counts: list[int] = []
            for value in raw:
                if isinstance(value, bool):
                    return None
                try:
                    count = int(value)
                except _PAIRING_INT_ERRORS:
                    return None
                if count < 0:
                    return None
                counts.append(count)
            return counts

        labels_raw = section.get("labels")
        labels: list[list[int]] | None = None
        if isinstance(labels_raw, list) and labels_raw:
            maybe_labels = _wrap_single_row(labels_raw, expected_rows=len(input_ids))
            if not isinstance(maybe_labels, list) or len(maybe_labels) != len(
                input_ids
            ):
                return None
            labels = []
            for idx, raw_label in enumerate(maybe_labels):
                label_list = tensor_or_list_to_ints_fn(raw_label)
                target_len = len(input_ids[idx])
                if len(label_list) < target_len:
                    label_list = label_list + [-100] * (target_len - len(label_list))
                elif len(label_list) > target_len:
                    label_list = label_list[:target_len]
                labels.append(label_list)

        masked_counts: list[int] | None = None
        if section.get("masked_token_counts") is not None:
            masked_counts = _coerce_count_list(section.get("masked_token_counts"))
            if masked_counts is None:
                return None

        actual_counts: list[int] | None = None
        if section.get("actual_token_counts") is not None:
            actual_counts = _coerce_count_list(section.get("actual_token_counts"))
            if actual_counts is None:
                return None

        payload: dict[str, Any] = {
            "window_ids": window_ids,
            "input_ids": input_ids,
            "attention_masks": attention_masks,
        }
        if labels is not None:
            payload["labels"] = labels
        if masked_counts is not None:
            payload["masked_token_counts"] = masked_counts
        if actual_counts is not None:
            payload["actual_token_counts"] = actual_counts
        return payload

    preview = _sanitize("preview", start_id=0)
    if not preview:
        return None
    final = _sanitize("final", start_id=len(preview.get("input_ids") or []))
    if preview and final:
        return {"preview": preview, "final": final}
    return None


def compute_provider_digest(
    report: dict[str, Any],
    *,
    compute_mask_positions_digest_fn: Any | None = None,
) -> dict[str, Any] | None:
    """Compute provider digest (ids/tokenizer/masking) from report context."""
    from invarlock.utils import hash_json

    if compute_mask_positions_digest_fn is None:
        compute_mask_positions_digest_fn = _compute_mask_positions_digest

    windows = report.get("evaluation_windows") if isinstance(report, dict) else None
    if not isinstance(windows, dict) or not windows:
        return None

    all_ids: list[Any] = []
    processor_sha = None
    for key in ("preview", "final"):
        section = windows.get(key)
        if not isinstance(section, dict):
            continue
        example_ids = section.get("example_ids")
        if isinstance(example_ids, list) and example_ids:
            all_ids.extend(str(value) for value in example_ids)
        else:
            window_ids = section.get("window_ids")
            if isinstance(window_ids, list):
                all_ids.extend(list(window_ids))
        if processor_sha is None:
            section_processor = section.get("processor_sha256")
            if isinstance(section_processor, str) and section_processor:
                processor_sha = section_processor

    ids_sha = None
    if all_ids:
        ids_int: list[int] = []
        use_ints = True
        for raw in all_ids:
            try:
                ids_int.append(int(raw))
            except _PAIRING_INT_ERRORS:
                use_ints = False
                break
        if use_ints:
            ids_sha = hash_json(sorted(ids_int))
        else:
            ids_sha = hash_json(sorted(str(value) for value in all_ids))

    tok_hash = None
    meta = report.get("meta") if isinstance(report.get("meta"), dict) else None
    if isinstance(meta, dict):
        tok_hash = meta.get("tokenizer_hash")
        if processor_sha is None:
            candidate = meta.get("processor_sha256")
            if isinstance(candidate, str) and candidate:
                processor_sha = candidate
    if not tok_hash and isinstance(report.get("data"), dict):
        tok_hash = report["data"].get("tokenizer_hash")
    if processor_sha is None and isinstance(report.get("data"), dict):
        candidate = report["data"].get("processor_sha256")
        if isinstance(candidate, str) and candidate:
            processor_sha = candidate

    masking = compute_mask_positions_digest_fn(windows)

    digest: dict[str, Any] = {}
    if isinstance(ids_sha, str) and ids_sha:
        digest["ids_sha256"] = ids_sha
    if isinstance(tok_hash, str) and tok_hash:
        digest["tokenizer_sha256"] = str(tok_hash)
    if isinstance(processor_sha, str) and processor_sha:
        digest["processor_sha256"] = str(processor_sha)
    if isinstance(masking, str) and masking:
        digest["masking_sha256"] = masking
    from invarlock.vision_dataset_evidence import build_report_evidence_from_run_report

    dataset_evidence = build_report_evidence_from_run_report(report)
    if dataset_evidence is not None:
        digest["dataset_evidence"] = dataset_evidence
    return digest or None


def validate_and_harvest_baseline_schedule(
    cfg: Any,
    pairing_schedule: dict[str, Any],
    baseline_report_data: dict[str, Any] | None,
    *,
    tokenizer_hash: str | None,
    resolved_loss_type: str,
    profile: str | None = None,
    baseline_path_str: str | None = None,
    console: Console | None = None,
    event_fn: Any | None = None,
    typed_failures: bool = False,
    canonical_dataset_id_fn: Any | None = None,
    tensor_or_list_to_ints_fn: Any | None = None,
    hash_sequences_fn: Any | None = None,
    invarlock_error_cls: type[BaseException] | None = None,
) -> dict[str, Any]:
    """Validate baseline pairing compatibility and harvest dataset metadata."""
    from invarlock.cli.run_pairing_baseline import (
        validate_and_harvest_baseline_schedule_impl,
    )

    if canonical_dataset_id_fn is None:
        canonical_dataset_id_fn = _canonical_dataset_id
    if tensor_or_list_to_ints_fn is None:
        tensor_or_list_to_ints_fn = _tensor_or_list_to_ints
    if hash_sequences_fn is None:
        hash_sequences_fn = _hash_sequences
    if invarlock_error_cls is None:
        from invarlock.core.exceptions import InvarlockError

        invarlock_error_cls = InvarlockError
    return validate_and_harvest_baseline_schedule_impl(
        cfg,
        pairing_schedule,
        baseline_report_data,
        tokenizer_hash=tokenizer_hash,
        resolved_loss_type=resolved_loss_type,
        profile=profile,
        baseline_path_str=baseline_path_str,
        console=console,
        event_fn=event_fn,
        typed_failures=typed_failures,
        canonical_dataset_id_fn=canonical_dataset_id_fn,
        tensor_or_list_to_ints_fn=tensor_or_list_to_ints_fn,
        hash_sequences_fn=hash_sequences_fn,
        invarlock_error_cls=invarlock_error_cls,
    )


def enforce_provider_parity(
    subject_digest: dict | None,
    baseline_digest: dict | None,
    *,
    profile: str | None,
    invarlock_error_cls: type[BaseException] | None = None,
) -> None:
    if invarlock_error_cls is None:
        from invarlock.core.exceptions import InvarlockError

        invarlock_error_cls = InvarlockError
    """Enforce tokenizer/masking parity rules for CI and release profiles."""
    _enforce_provider_parity_core(
        subject_digest,
        baseline_digest,
        profile=profile,
        invarlock_error_cls=invarlock_error_cls,
    )


def resolve_metric_and_provider(
    cfg: Any,
    model_profile: Any,
    *,
    resolved_loss_type: str | None = None,
    metric_kind_override: str | None = None,
) -> tuple[str, str, dict[str, float]]:
    """Resolve metric/provider policy via the core owner."""
    return _resolve_metric_and_provider_core(
        cfg,
        model_profile,
        resolved_loss_type=resolved_loss_type,
        metric_kind_override=metric_kind_override,
    )
