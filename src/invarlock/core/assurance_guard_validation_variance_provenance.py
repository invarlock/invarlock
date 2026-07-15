"""Shared strict variance policy, details, and A/B provenance reconciliation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from .assurance_guard_validation_common import (
    _mapping,
    _nonnegative_int,
    _normalized_token,
)


def _variance_policy_calibration_errors(
    policy: Mapping[str, Any] | None,
    metrics: Mapping[str, Any],
    *,
    source: str,
) -> list[str]:
    if policy is None:
        return []
    policy_calibration = _mapping(policy.get("calibration"))
    calibration = _mapping(metrics.get("calibration"))
    if policy_calibration is None or calibration is None:
        return [
            f"{source}.policy.calibration and {source}.metrics.calibration are required."
        ]
    errors: list[str] = []
    for policy_key, evidence_key in (
        ("windows", "requested"),
        ("min_coverage", "min_coverage"),
        ("seed", "seed"),
    ):
        policy_value = _nonnegative_int(policy_calibration.get(policy_key))
        evidence_value = _nonnegative_int(calibration.get(evidence_key))
        if policy_value is None or policy_value != evidence_value:
            errors.append(
                f"{source}.policy.calibration.{policy_key} must match "
                f"{source}.metrics.calibration.{evidence_key}."
            )
    calibration_seed = _nonnegative_int(policy_calibration.get("seed"))
    if calibration_seed != _nonnegative_int(metrics.get("ab_seed_used")):
        errors.append(
            f"{source}.metrics.ab_seed_used must match policy.calibration.seed."
        )
    return errors


def _variance_details_mirror_errors(
    entry: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    source: str,
) -> tuple[list[str], Mapping[str, Any] | None]:
    details = _mapping(entry.get("details"))
    if details is None:
        return [f"{source}.details is required for strict variance assurance."], None
    errors: list[str] = []
    for detail_key, metric_key in (
        ("ve_tested", "ve_enabled_during_validation"),
        ("ve_applied", "ve_enabled"),
        ("subject_restored_after_ab", "subject_restored_after_ab"),
    ):
        if (
            detail_key not in details
            or metric_key not in metrics
            or details.get(detail_key) != metrics.get(metric_key)
        ):
            errors.append(
                f"{source}.details.{detail_key} must match "
                f"{source}.metrics.{metric_key} exactly."
            )
    if details.get("policy") != entry.get("policy"):
        errors.append(f"{source}.details.policy must match {source}.policy exactly.")

    stats = _mapping(details.get("stats"))
    if stats is None:
        errors.append(
            f"{source}.details.stats is required for strict variance assurance."
        )
        return errors, None
    for key in (
        "ab_provenance",
        "ab_point_estimates",
        "ab_measurements",
        "predictive_gate",
    ):
        if stats.get(key) != metrics.get(key):
            errors.append(
                f"{source}.details.stats.{key} must match "
                f"{source}.metrics.{key} exactly."
            )
    return errors, stats


def _variance_ab_condition_errors(
    block: Mapping[str, Any],
    *,
    source: str,
    condition: str,
    expected_mode: str,
    valid_statuses: frozenset[str],
    coverage: int | None,
) -> list[str]:
    errors: list[str] = []
    if _normalized_token(str(block.get("mode") or "")) != expected_mode:
        errors.append(f"{source}.metrics.ab_provenance.{condition}.mode is invalid.")
    status = _normalized_token(str(block.get("status") or ""))
    if status not in valid_statuses:
        errors.append(f"{source}.metrics.ab_provenance.{condition}.status is invalid.")
    window_ids = block.get("window_ids")
    if (
        not isinstance(window_ids, list)
        or not window_ids
        or any(not isinstance(item, str) or not item for item in window_ids)
        or len(set(window_ids)) != len(window_ids)
    ):
        errors.append(
            f"{source}.metrics.ab_provenance.{condition}.window_ids must be "
            "non-empty and unique."
        )
    elif coverage is not None and len(window_ids) != coverage:
        errors.append(
            f"{source}.metrics.ab_provenance.{condition}.window_ids must match "
            "calibration.coverage."
        )
    window_count = _nonnegative_int(block.get("window_count"))
    if not isinstance(window_ids, list) or window_count != len(window_ids):
        errors.append(
            f"{source}.metrics.ab_provenance.{condition}.window_count must "
            "match window_ids."
        )
    for key in (
        "pairing_digest",
        "consumed_pairing_digest",
        "dataset_hash",
        "tokenizer_hash",
        "target_fingerprint",
        "model_id",
    ):
        if not isinstance(block.get(key), str) or not block.get(key):
            errors.append(
                f"{source}.metrics.ab_provenance.{condition}.{key} is required."
            )
    if _normalized_token(str(block.get("tag") or "")) != "post-edit":
        errors.append(
            f"{source}.metrics.ab_provenance.{condition}.tag must be post_edit."
        )
    if _nonnegative_int(block.get("run_seed")) is None:
        errors.append(
            f"{source}.metrics.ab_provenance.{condition}.run_seed must be an integer."
        )
    if isinstance(window_ids, list) and window_ids:
        expected_consumed_digest = hashlib.blake2s(
            "||".join(window_ids).encode("utf-8"), digest_size=16
        ).hexdigest()
        if block.get("consumed_pairing_digest") != expected_consumed_digest:
            errors.append(
                f"{source}.metrics.ab_provenance.{condition}."
                "consumed_pairing_digest must match consumed window_ids."
            )
    return errors


def _variance_ab_provenance_errors(
    metrics: Mapping[str, Any],
    coverage: int | None,
    *,
    source: str,
    top_provenance: Mapping[str, Any] | None,
    details_stats: Mapping[str, Any] | None,
    condition_b_statuses: frozenset[str],
    expected_provenance: Mapping[str, Any],
) -> list[str]:
    provenance = _mapping(metrics.get("ab_provenance"))
    if provenance is None:
        return [f"{source}.metrics.ab_provenance is required."]
    errors: list[str] = []
    conditions: dict[str, Mapping[str, Any]] = {}
    for condition, expected_mode in (
        ("condition_a", "edited-no-ve"),
        ("condition_b", "virtual-ve"),
    ):
        block = _mapping(provenance.get(condition))
        if block is None:
            errors.append(f"{source}.metrics.ab_provenance.{condition} is required.")
            continue
        conditions[condition] = block
        valid_statuses = (
            frozenset({"evaluated"})
            if condition == "condition_a"
            else condition_b_statuses
        )
        errors.extend(
            _variance_ab_condition_errors(
                block,
                source=source,
                condition=condition,
                expected_mode=expected_mode,
                valid_statuses=valid_statuses,
                coverage=coverage,
            )
        )

    if set(conditions) != {"condition_a", "condition_b"}:
        return errors
    left = conditions["condition_a"]
    right = conditions["condition_b"]
    for key in (
        "window_ids",
        "tag",
        "model_id",
        "run_seed",
        "pairing_digest",
        "consumed_pairing_digest",
        "dataset_hash",
        "tokenizer_hash",
        "target_fingerprint",
    ):
        if left.get(key) != right.get(key):
            errors.append(
                f"{source}.metrics.ab_provenance conditions must share {key}."
            )

    for key in (
        "model_id",
        "run_seed",
        "dataset_hash",
        "tokenizer_hash",
        "pairing_digest",
    ):
        if left.get(key) != expected_provenance.get(key):
            errors.append(
                f"{source}.metrics.ab_provenance.{key} must match report provenance."
            )
    expected_window_ids = expected_provenance.get("window_ids")
    condition_window_ids = left.get("window_ids")
    if (
        not isinstance(expected_window_ids, list)
        or not isinstance(condition_window_ids, list)
        or len(expected_window_ids) < len(condition_window_ids)
        or condition_window_ids != expected_window_ids[: len(condition_window_ids)]
    ):
        errors.append(
            f"{source}.metrics.ab_provenance.window_ids must match the consumed "
            "prefix of the report pairing schedule."
        )

    condition_ids = left.get("window_ids")
    if top_provenance is None or "window_ids" not in top_provenance:
        errors.append("variance.ab_test.provenance.window_ids is required.")
    elif top_provenance.get("window_ids") != condition_ids:
        errors.append(
            "variance.ab_test.provenance.window_ids must match raw condition IDs."
        )
    if details_stats is not None:
        calibration = _mapping(details_stats.get("calibration"))
        if calibration is None:
            errors.append(f"{source}.details.stats.calibration is required.")
        elif calibration.get("window_ids") != condition_ids:
            errors.append(
                f"{source}.details.stats.calibration.window_ids must match A/B IDs."
            )
        target_fingerprint = details_stats.get("target_fingerprint")
        if not isinstance(target_fingerprint, str) or not target_fingerprint:
            errors.append(f"{source}.details.stats.target_fingerprint is required.")
        elif target_fingerprint != left.get("target_fingerprint"):
            errors.append(
                f"{source}.details.stats.target_fingerprint must match A/B provenance."
            )
        pairing_reference = _mapping(details_stats.get("pairing_reference"))
        if pairing_reference is None:
            errors.append(f"{source}.details.stats.pairing_reference is required.")
        elif pairing_reference.get("digest") != left.get("pairing_digest"):
            errors.append(
                f"{source}.details.stats.pairing_reference.digest must match A/B provenance."
            )
        dataset_meta = _mapping(details_stats.get("dataset_meta"))
        if dataset_meta is None:
            errors.append(f"{source}.details.stats.dataset_meta is required.")
        else:
            for key in ("dataset_hash", "tokenizer_hash"):
                if dataset_meta.get(key) != left.get(key):
                    errors.append(
                        f"{source}.details.stats.dataset_meta.{key} must match "
                        "A/B provenance."
                    )
    return errors
