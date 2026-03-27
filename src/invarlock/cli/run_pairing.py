"""Baseline pairing and provider parity helpers for run command."""

from __future__ import annotations

import hashlib
from array import array
from typing import Any

import typer
from rich.console import Console

from invarlock.core.metric_provider_resolution import (
    resolve_metric_and_provider as _resolve_metric_and_provider_core,
)
from invarlock.core.provider_parity import (
    enforce_provider_parity as _enforce_provider_parity_core,
)


def extract_pairing_schedule(
    report: dict[str, Any] | None,
    *,
    tensor_or_list_to_ints_fn: Any,
) -> dict[str, Any] | None:
    """Extract sanitized pairing schedule from baseline-like report data."""
    if not isinstance(report, dict):
        return None
    windows = report.get("evaluation_windows")
    if not isinstance(windows, dict):
        return None

    def _wrap_single_row(raw: Any, *, expected_rows: int) -> list | None:
        if not isinstance(raw, list):
            return None
        if expected_rows == 1 and raw and not isinstance(raw[0], list):
            return [raw]
        return raw

    def _sanitize(section_key: str, *, start_id: int) -> dict[str, Any] | None:
        section = windows.get(section_key)
        if not isinstance(section, dict):
            return None
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
                except Exception:
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
                if idx < len(input_ids):
                    target_len = len(input_ids[idx])
                    if len(label_list) < target_len:
                        label_list = label_list + [-100] * (
                            target_len - len(label_list)
                        )
                    elif len(label_list) > target_len:
                        label_list = label_list[:target_len]
                labels.append(label_list)

        masked_counts: list[int] | None = None
        if section.get("masked_token_counts") is not None:
            raw = section.get("masked_token_counts")
            if isinstance(raw, int) and len(input_ids) == 1:
                raw = [raw]
            if not isinstance(raw, list) or len(raw) != len(input_ids):
                return None
            masked_counts = [int(value) for value in raw]

        actual_counts: list[int] | None = None
        if section.get("actual_token_counts") is not None:
            raw = section.get("actual_token_counts")
            if isinstance(raw, int) and len(input_ids) == 1:
                raw = [raw]
            if not isinstance(raw, list) or len(raw) != len(input_ids):
                return None
            actual_counts = [int(value) for value in raw]

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
    compute_mask_positions_digest_fn: Any,
) -> dict[str, str] | None:
    """Compute provider digest (ids/tokenizer/masking) from report context."""
    from invarlock.utils.digest import hash_json

    windows = report.get("evaluation_windows") if isinstance(report, dict) else None
    if not isinstance(windows, dict) or not windows:
        return None

    all_ids: list[Any] = []
    for key in ("preview", "final"):
        section = windows.get(key)
        if not isinstance(section, dict):
            continue
        window_ids = section.get("window_ids")
        if isinstance(window_ids, list):
            all_ids.extend(list(window_ids))

    ids_sha = None
    if all_ids:
        ids_int: list[int] = []
        use_ints = True
        for raw in all_ids:
            try:
                ids_int.append(int(raw))
            except Exception:
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
    if not tok_hash and isinstance(report.get("data"), dict):
        tok_hash = report["data"].get("tokenizer_hash")

    masking = compute_mask_positions_digest_fn(windows)

    digest: dict[str, str] = {}
    if isinstance(ids_sha, str) and ids_sha:
        digest["ids_sha256"] = ids_sha
    if isinstance(tok_hash, str) and tok_hash:
        digest["tokenizer_sha256"] = str(tok_hash)
    if isinstance(masking, str) and masking:
        digest["masking_sha256"] = masking
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
    canonical_dataset_id_fn: Any,
    tensor_or_list_to_ints_fn: Any,
    hash_sequences_fn: Any,
    invarlock_error_cls: type[BaseException],
) -> dict[str, Any]:
    """Validate baseline pairing compatibility and harvest dataset metadata."""

    def _emit(tag: str, message: str, emoji: str) -> None:
        if console is not None and event_fn is not None:
            event_fn(console, tag, message, emoji=emoji, profile=profile)

    def _fail_schedule(reason: str) -> None:
        path = baseline_path_str or "baseline"
        prof = (profile or "dev").strip().lower()
        message = f"PAIRING-EVIDENCE-MISSING: {path}: {reason}"
        if prof in {"ci", "release"}:
            raise invarlock_error_cls(code="E001", message=message)
        _emit(
            "FAIL",
            f"Baseline pairing schedule '{path}' is incompatible: {reason}",
            "❌",
        )
        raise typer.Exit(1)

    baseline_meta = (
        baseline_report_data.get("data")
        if isinstance(baseline_report_data, dict)
        else {}
    )
    if not isinstance(baseline_meta, dict):
        baseline_meta = {}

    def _extract_meta(field: str, default: Any = None) -> Any:
        value = baseline_meta.get(field)
        return value if value is not None else default

    try:
        preview = (
            pairing_schedule.get("preview")
            if isinstance(pairing_schedule, dict)
            else None
        )
        final = (
            pairing_schedule.get("final")
            if isinstance(pairing_schedule, dict)
            else None
        )
        if not isinstance(preview, dict) or not isinstance(final, dict):
            _fail_schedule("missing preview/final evaluation_windows sections")

        def _arm_check(
            label: str,
            section: dict[str, Any],
        ) -> tuple[list[int], list[list[int]]]:
            window_ids = section.get("window_ids")
            input_ids = section.get("input_ids")
            masks = section.get("attention_masks")
            if not isinstance(window_ids, list) or not isinstance(input_ids, list):
                _fail_schedule(f"invalid {label} section: missing window_ids/input_ids")
            if len(window_ids) != len(input_ids):
                _fail_schedule(
                    f"{label} coherence error: len(window_ids)={len(window_ids)} len(input_ids)={len(input_ids)}"
                )

            ids_int: list[int] = []
            seqs: list[list[int]] = []
            for idx, (wid, seq) in enumerate(zip(window_ids, input_ids, strict=False)):
                try:
                    wid_int = int(wid)
                except Exception:
                    _fail_schedule(
                        f"{label} window_ids contains non-int at index {idx}"
                    )
                ids_int.append(wid_int)
                seq_ints = tensor_or_list_to_ints_fn(seq)
                if not seq_ints:
                    _fail_schedule(f"{label} input_ids empty at index {idx}")
                seqs.append(seq_ints)

            masks_rows: list[list[int]] = []
            masks_missing = masks is None or masks == []
            if (
                isinstance(masks, list)
                and masks
                and len(seqs) == 1
                and not isinstance(masks[0], list)
            ):
                masks = [masks]

            if isinstance(masks, list) and masks:
                if len(masks) != len(seqs):
                    _fail_schedule(
                        f"{label} coherence error: len(attention_masks)={len(masks)} len(input_ids)={len(seqs)}"
                    )
                for idx, (seq_ints, mask) in enumerate(zip(seqs, masks, strict=False)):
                    if not isinstance(mask, list):
                        _fail_schedule(
                            f"{label} attention_masks row is not a list at index {idx}"
                        )
                    mask_ints = tensor_or_list_to_ints_fn(mask)
                    if len(mask_ints) != len(seq_ints):
                        _fail_schedule(
                            f"{label} attention_masks length mismatch at index {idx}"
                        )
                    masks_rows.append(mask_ints)
            else:
                masks_missing = True
                masks_rows = [[1] * len(seq) for seq in seqs]

            if masks_missing:
                try:
                    section["attention_masks"] = masks_rows
                except Exception:
                    pass

            labels = section.get("labels")
            if isinstance(labels, list) and labels:
                if len(labels) != len(seqs):
                    _fail_schedule(f"{label} labels length mismatch")
                for idx, row in enumerate(labels):
                    row_ints = tensor_or_list_to_ints_fn(row)
                    if len(row_ints) != len(seqs[idx]):
                        _fail_schedule(f"{label} labels length mismatch at index {idx}")

            for key in ("masked_token_counts", "actual_token_counts"):
                if section.get(key) is not None:
                    raw_counts = section.get(key)
                    if not isinstance(raw_counts, list) or len(raw_counts) != len(seqs):
                        _fail_schedule(f"{label} {key} length mismatch")

            return ids_int, seqs

        preview_ids, preview_seqs = _arm_check("preview", preview)
        final_ids, final_seqs = _arm_check("final", final)

        if len(set(preview_ids)) != len(preview_ids):
            _fail_schedule("duplicate window_ids detected in preview arm")
        if len(set(final_ids)) != len(final_ids):
            _fail_schedule("duplicate window_ids detected in final arm")
        if set(preview_ids) & set(final_ids):
            _fail_schedule("window_ids overlap between preview and final arms")

        def _hash_tokens(tokens: list[int]) -> bytes:
            if not tokens:
                return b""
            token_array = array("I", (int(token) & 0xFFFFFFFF for token in tokens))
            return hashlib.blake2b(token_array.tobytes(), digest_size=16).digest()

        preview_hashes = [_hash_tokens(seq) for seq in preview_seqs]
        final_hashes = [_hash_tokens(seq) for seq in final_seqs]
        if len(set(preview_hashes)) != len(preview_hashes):
            _fail_schedule("duplicate token sequences detected in preview arm")
        if len(set(final_hashes)) != len(final_hashes):
            _fail_schedule("duplicate token sequences detected in final arm")
        if set(preview_hashes) & set(final_hashes):
            _fail_schedule("preview/final token sequence overlap detected")

        expected_preview_hash = hash_sequences_fn(preview_seqs)
        expected_final_hash = hash_sequences_fn(final_seqs)
        expected_dataset_hash = hashlib.blake2s(
            (expected_preview_hash + expected_final_hash).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

        baseline_preview_hash = baseline_meta.get("preview_hash")
        baseline_final_hash = baseline_meta.get("final_hash")
        baseline_dataset_hash = baseline_meta.get("dataset_hash")

        if (
            isinstance(baseline_preview_hash, str)
            and baseline_preview_hash
            and baseline_preview_hash != expected_preview_hash
        ):
            prof = (profile or "dev").strip().lower()
            if prof in {"ci", "release"}:
                _fail_schedule("preview_hash mismatch vs baseline report data")
            if console is not None and event_fn is not None:
                event_fn(
                    console,
                    "WARN",
                    "Baseline preview_hash mismatch; continuing in dev profile.",
                    emoji="⚠️",
                    profile=prof,
                )

        if (
            isinstance(baseline_final_hash, str)
            and baseline_final_hash
            and baseline_final_hash != expected_final_hash
        ):
            prof = (profile or "dev").strip().lower()
            if prof in {"ci", "release"}:
                _fail_schedule("final_hash mismatch vs baseline report data")
            if console is not None and event_fn is not None:
                event_fn(
                    console,
                    "WARN",
                    "Baseline final_hash mismatch; continuing in dev profile.",
                    emoji="⚠️",
                    profile=prof,
                )

        if (
            isinstance(baseline_dataset_hash, str)
            and baseline_dataset_hash
            and baseline_dataset_hash != expected_dataset_hash
        ):
            prof = (profile or "dev").strip().lower()
            if prof in {"ci", "release"}:
                _fail_schedule("dataset_hash mismatch vs baseline report data")
            if console is not None and event_fn is not None:
                event_fn(
                    console,
                    "WARN",
                    "Baseline dataset_hash mismatch; continuing in dev profile.",
                    emoji="⚠️",
                    profile=prof,
                )
    except invarlock_error_cls:
        raise
    except typer.Exit:
        raise
    except Exception as exc:
        _fail_schedule(f"failed to validate baseline schedule integrity ({exc})")

    baseline_preview = len(pairing_schedule["preview"].get("input_ids") or [])
    baseline_final = len(pairing_schedule["final"].get("input_ids") or [])
    cfg_preview = getattr(cfg.dataset, "preview_n", None)
    cfg_final = getattr(cfg.dataset, "final_n", None)
    if (
        cfg_preview is not None
        and baseline_preview is not None
        and baseline_preview != cfg_preview
    ) or (
        cfg_final is not None
        and baseline_final is not None
        and baseline_final != cfg_final
    ):
        _emit(
            "WARN",
            (
                "Adjusting evaluation window counts to match baseline schedule "
                f"({baseline_preview}/{baseline_final})."
            ),
            "⚠️",
        )

    effective_preview = int(baseline_preview)
    effective_final = int(baseline_final)
    preview_count = effective_preview
    final_count = effective_final

    cfg_seq_len = getattr(cfg.dataset, "seq_len", None)
    baseline_seq_len = _extract_meta("seq_len")
    if (
        cfg_seq_len is not None
        and baseline_seq_len is not None
        and baseline_seq_len != cfg_seq_len
    ):
        _fail_schedule(
            f"sequence length mismatch (baseline {baseline_seq_len} vs config {cfg_seq_len})"
        )

    cfg_stride = getattr(cfg.dataset, "stride", getattr(cfg.dataset, "seq_len", None))
    baseline_stride = _extract_meta("stride")
    if (
        baseline_stride is not None
        and cfg_stride is not None
        and baseline_stride != cfg_stride
    ):
        _fail_schedule(
            f"stride mismatch (baseline {baseline_stride} vs config {cfg_stride})"
        )

    cfg_dataset = getattr(cfg.dataset, "provider", None)
    if cfg_dataset is None:
        cfg_dataset = getattr(cfg.dataset, "dataset", None)
    cfg_dataset = canonical_dataset_id_fn(cfg_dataset)
    baseline_dataset = canonical_dataset_id_fn(_extract_meta("dataset"))
    if (
        baseline_dataset is not None
        and cfg_dataset is not None
        and baseline_dataset != cfg_dataset
    ):
        _fail_schedule(
            f"dataset mismatch (baseline {baseline_dataset} vs config {cfg_dataset})"
        )

    cfg_split = getattr(cfg.dataset, "split", "validation")
    baseline_split = _extract_meta("split")
    if (
        baseline_split is not None
        and cfg_split is not None
        and baseline_split != cfg_split
    ):
        _fail_schedule(
            f"split mismatch (baseline {baseline_split} vs config {cfg_split})"
        )

    baseline_tokenizer_hash = baseline_meta.get("tokenizer_hash")
    if (
        baseline_tokenizer_hash
        and tokenizer_hash
        and baseline_tokenizer_hash != tokenizer_hash
    ):
        _fail_schedule(
            "tokenizer hash mismatch between baseline and current configuration"
        )

    dataset_meta = {
        key: baseline_meta.get(key)
        for key in (
            "tokenizer_hash",
            "tokenizer_name",
            "vocab_size",
            "bos_token",
            "eos_token",
            "pad_token",
            "add_prefix_space",
            "dataset_hash",
            "preview_hash",
            "final_hash",
            "preview_total_tokens",
            "final_total_tokens",
        )
        if baseline_meta.get(key) is not None
    }
    dataset_meta["loss_type"] = resolved_loss_type
    window_plan = baseline_meta.get("window_plan")
    calibration_data: list[Any] | None = []

    return {
        "effective_preview": effective_preview,
        "effective_final": effective_final,
        "preview_count": preview_count,
        "final_count": final_count,
        "dataset_meta": dataset_meta,
        "window_plan": window_plan,
        "calibration_data": calibration_data,
    }


def enforce_provider_parity(
    subject_digest: dict | None,
    baseline_digest: dict | None,
    *,
    profile: str | None,
    invarlock_error_cls: type[BaseException],
) -> None:
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
