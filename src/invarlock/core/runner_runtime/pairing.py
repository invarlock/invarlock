from __future__ import annotations

import hashlib
import math
from array import array
from collections.abc import Sequence
from typing import Any

BOOTSTRAP_COVERAGE_REQUIREMENTS = {
    "conservative": {"preview": 220, "final": 220, "replicates": 1500},
    "balanced": {"preview": 180, "final": 180, "replicates": 1200},
    "aggressive": {"preview": 140, "final": 140, "replicates": 800},
}
_PAIRING_COERCION_ERRORS = (OverflowError, TypeError, ValueError)


def _hash_tokens(tokens: Sequence[int]) -> bytes:
    if not tokens:
        return b""
    token_array = array("I", (int(token) & 0xFFFFFFFF for token in tokens))
    return hashlib.blake2b(token_array.tobytes(), digest_size=16).digest()


def _hash_window_evidence(
    tokens: Sequence[int],
    labels: Sequence[int] | None = None,
) -> bytes:
    if not tokens:
        return b""
    hasher = hashlib.blake2b(digest_size=16)
    token_array = array("I", (int(token) & 0xFFFFFFFF for token in tokens))
    hasher.update(token_array.tobytes())
    if labels is None:
        hasher.update(b"\x00")
        return hasher.digest()
    label_array = array("q", (int(label) for label in labels))
    hasher.update(b"\x01")
    hasher.update(label_array.tobytes())
    return hasher.digest()


def duplicate_fraction(
    seqs: Sequence[Sequence[int]],
    *,
    labels: Sequence[Sequence[int]] | None = None,
) -> float:
    if not seqs:
        return 0.0
    use_labels = labels is not None and len(labels) == len(seqs)
    labels_for_hash = labels if use_labels else None
    hashes = [
        _hash_window_evidence(
            seq,
            labels_for_hash[idx] if labels_for_hash is not None else None,
        )
        for idx, seq in enumerate(seqs)
    ]
    unique = len(set(hashes))
    return max(0.0, (len(hashes) - unique) / len(hashes))


def overlap_fraction_from_context(context: dict[str, Any] | None) -> float | None:
    if not isinstance(context, dict):
        return None
    dataset_cfg = context.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return None
    seq_len_val = dataset_cfg.get("seq_len")
    if seq_len_val is None:
        return None
    stride_raw = dataset_cfg.get("stride", seq_len_val)
    if stride_raw is None:
        return None
    try:
        seq_len_f = float(seq_len_val)
        stride_f = float(stride_raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seq_len_f) or seq_len_f <= 0:
        return None
    if not math.isfinite(stride_f) or stride_f < 0:
        return None
    overlap = (seq_len_f - stride_f) / seq_len_f
    return max(0.0, min(1.0, float(overlap)))


def compare_with_baseline(
    run_ids: Sequence[int],
    run_tokens: Sequence[Sequence[int]],
    baseline_section: dict[str, Any] | None,
    split_label: str,
    *,
    run_labels: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    stats = {
        "matched": 0,
        "expected": 0,
        "missing_ids": [],
        "mismatched_ids": [],
        "unexpected_ids": [],
        "reason": None,
    }

    if not baseline_section:
        stats["matched"] = len(run_tokens)
        stats["expected"] = len(run_tokens)
        stats["reason"] = "no_baseline_reference"
        return stats

    base_ids = baseline_section.get("window_ids") or []
    base_tokens = baseline_section.get("input_ids") or []
    if not isinstance(base_ids, list) or not isinstance(base_tokens, list):
        stats["matched"] = len(run_tokens)
        stats["expected"] = len(run_tokens)
        stats["reason"] = "invalid_baseline_reference"
        return stats

    base_labels_raw = baseline_section.get("labels")
    base_labels: list[list[int]] | None = None
    if isinstance(base_labels_raw, list) and len(base_labels_raw) == len(base_ids):
        base_labels = []
        for row in base_labels_raw:
            try:
                row_list = list(row) if not isinstance(row, list) else row
                base_labels.append([int(value) for value in row_list])
            except _PAIRING_COERCION_ERRORS:
                base_labels = None
                break

    use_label_hashes = (
        base_labels is not None
        and run_labels is not None
        and len(run_labels) >= len(run_tokens)
    )
    base_labels_for_hash = base_labels if use_label_hashes else None
    run_labels_for_hash = run_labels if use_label_hashes else None

    base_map: dict[int, bytes] = {}
    invalid_baseline_reference = False
    for index, (base_id, seq) in enumerate(zip(base_ids, base_tokens, strict=False)):
        try:
            base_id_int = int(base_id)
            seq_list = list(seq) if not isinstance(seq, list) else seq
            base_map[base_id_int] = _hash_window_evidence(
                seq_list,
                (
                    base_labels_for_hash[index]
                    if base_labels_for_hash is not None
                    else None
                ),
            )
        except _PAIRING_COERCION_ERRORS:
            invalid_baseline_reference = True
            break

    if invalid_baseline_reference or len(base_map) != len(base_ids):
        stats["expected"] = max(len(base_ids), len(base_tokens))
        stats["reason"] = "invalid_baseline_reference"
        return stats

    stats["expected"] = len(base_map)
    matched = 0
    seen_ids: set[int] = set()
    mismatched: list[int] = []
    unexpected: list[int] = []

    for index, (run_id, seq) in enumerate(zip(run_ids, run_tokens, strict=False)):
        try:
            run_id_int = int(run_id)
        except _PAIRING_COERCION_ERRORS:
            unexpected.append(run_id)
            continue

        hashed = _hash_window_evidence(
            seq,
            run_labels_for_hash[index] if run_labels_for_hash is not None else None,
        )
        if run_id_int not in base_map:
            unexpected.append(run_id_int)
            continue

        seen_ids.add(run_id_int)
        if hashed == base_map[run_id_int]:
            matched += 1
        else:
            mismatched.append(run_id_int)

    missing = [base_id for base_id in base_map if base_id not in seen_ids]
    stats.update(
        {
            "matched": matched,
            "missing_ids": missing,
            "mismatched_ids": mismatched,
            "unexpected_ids": unexpected,
        }
    )

    if missing:
        stats["reason"] = f"{split_label}_missing_ids:{missing[:3]}"
    elif mismatched:
        stats["reason"] = f"{split_label}_token_mismatch:{mismatched[:3]}"
    elif unexpected:
        stats["reason"] = f"{split_label}_unexpected_ids:{unexpected[:3]}"
    else:
        stats["reason"] = None

    return stats


def compute_window_pairing_metrics(
    *,
    preview_window_ids: Sequence[int],
    preview_tokens: Sequence[Sequence[int]],
    preview_labels: Sequence[Sequence[int]] | None = None,
    final_window_ids: Sequence[int],
    final_tokens: Sequence[Sequence[int]],
    final_labels: Sequence[Sequence[int]] | None = None,
    pairing_context: dict[str, Any] | None,
    config_context: dict[str, Any] | None,
    preview_batches: int,
    final_batches: int,
) -> dict[str, Any]:
    baseline_preview = (
        pairing_context.get("preview") if isinstance(pairing_context, dict) else {}
    )
    baseline_final = (
        pairing_context.get("final") if isinstance(pairing_context, dict) else {}
    )

    preview_pair_stats = compare_with_baseline(
        preview_window_ids,
        preview_tokens,
        baseline_preview,
        "preview",
        run_labels=preview_labels,
    )
    final_pair_stats = compare_with_baseline(
        final_window_ids,
        final_tokens,
        baseline_final,
        "final",
        run_labels=final_labels,
    )

    total_expected = preview_pair_stats["expected"] + final_pair_stats["expected"]
    total_matched = preview_pair_stats["matched"] + final_pair_stats["matched"]
    total_unexpected = len(preview_pair_stats["unexpected_ids"]) + len(
        final_pair_stats["unexpected_ids"]
    )
    match_denominator = total_expected + total_unexpected
    match_fraction = (
        float(total_matched / match_denominator) if match_denominator > 0 else 1.0
    )
    combined_labels: list[Sequence[int]] | None = None
    if (
        preview_labels is not None
        and final_labels is not None
        and len(preview_labels) == len(preview_tokens)
        and len(final_labels) == len(final_tokens)
    ):
        combined_labels = [*preview_labels, *final_labels]
    duplicate_fraction_value = duplicate_fraction(
        [*preview_tokens, *final_tokens],
        labels=combined_labels,
    )
    overlap_fraction = overlap_fraction_from_context(config_context)
    overlap_unknown = overlap_fraction is None
    overlap_fraction_value = overlap_fraction if overlap_fraction is not None else 1.0
    count_mismatch = preview_batches != final_batches

    pairing_reason = None
    if total_expected > 0:
        for stats_dict, label in (
            (preview_pair_stats, "preview"),
            (final_pair_stats, "final"),
        ):
            if (
                stats_dict["expected"]
                and stats_dict["matched"] < stats_dict["expected"]
            ):
                pairing_reason = stats_dict.get("reason") or f"{label}_mismatch"
                break
    if pairing_reason is None:
        if overlap_unknown:
            pairing_reason = "overlap_unknown"
        elif overlap_fraction_value > 0.0:
            pairing_reason = "overlapping_windows"
        elif duplicate_fraction_value > 0.0:
            pairing_reason = "duplicate_windows"
        elif count_mismatch:
            pairing_reason = "count_mismatch"
        else:
            pairing_reason = preview_pair_stats.get("reason") or final_pair_stats.get(
                "reason"
            )

    return {
        "preview": preview_pair_stats,
        "final": final_pair_stats,
        "match_fraction": match_fraction,
        "overlap_fraction": float(overlap_fraction_value),
        "overlap_unknown": overlap_unknown,
        "duplicate_fraction": duplicate_fraction_value,
        "count_mismatch": count_mismatch,
        "reason": pairing_reason,
    }


def _meets_requirement(actual: int, required: int) -> bool:
    if required <= 0:
        return True
    return actual >= required


def assess_bootstrap_coverage(
    *,
    tier: str,
    preview_batches: int,
    final_batches: int,
    bootstrap_enabled: bool,
    bootstrap_replicates: int,
    requirements: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    effective_requirements = requirements or BOOTSTRAP_COVERAGE_REQUIREMENTS
    balanced_fallback = effective_requirements.get(
        "balanced", {"preview": 0, "final": 0, "replicates": 0}
    )
    coverage_requirements = effective_requirements.get(tier, balanced_fallback)

    preview_required = int(coverage_requirements.get("preview", 0))
    final_required = int(coverage_requirements.get("final", 0))
    replicates_required = int(coverage_requirements.get("replicates", 0))

    preview_ok = _meets_requirement(preview_batches, preview_required)
    final_ok = _meets_requirement(final_batches, final_required)
    replicates_ok = (
        _meets_requirement(bootstrap_replicates, replicates_required)
        if bootstrap_enabled
        else True
    )

    coverage = {
        "tier": tier,
        "preview": {
            "used": int(preview_batches),
            "required": preview_required,
            "ok": bool(preview_ok),
        },
        "final": {
            "used": int(final_batches),
            "required": final_required,
            "ok": bool(final_ok),
        },
        "replicates": {
            "used": int(bootstrap_replicates),
            "required": replicates_required,
            "ok": bool(replicates_ok),
        },
    }

    return {
        "ok": bool(preview_ok and final_ok and replicates_ok),
        "preview_required": preview_required,
        "final_required": final_required,
        "replicates_required": replicates_required,
        "preview_ok": bool(preview_ok),
        "final_ok": bool(final_ok),
        "replicates_ok": bool(replicates_ok),
        "coverage": coverage,
    }
