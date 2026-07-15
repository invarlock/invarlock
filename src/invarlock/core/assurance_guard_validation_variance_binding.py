"""Bind strict variance policy and A/B provenance to the enclosing report."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from .assurance_guard_validation_common import (
    _mapping,
    _nonnegative_int,
)


def _variance_policy_semantic_errors(
    policy: Mapping[str, Any] | None,
    metrics: Mapping[str, Any],
    *,
    source: str,
) -> list[str]:
    if policy is None:
        return []
    errors: list[str] = []
    if policy.get("monitor_only", False) is not False:
        errors.append(f"{source}.policy.monitor_only must be false.")
    if metrics.get("monitor_only") is not False:
        errors.append(f"{source}.metrics.monitor_only must be false.")
    if policy.get("predictive_gate") is not True:
        errors.append(f"{source}.policy.predictive_gate must be true.")
    if policy.get("mode") != "ci":
        errors.append(f"{source}.policy.mode must be ci.")
    if metrics.get("mode") != policy.get("mode"):
        errors.append(f"{source}.metrics.mode must match {source}.policy.mode exactly.")
    return errors


def _variance_report_binding_errors(
    report: Mapping[str, Any],
    policy: Mapping[str, Any] | None,
    *,
    source: str,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    resolved_policy = _mapping(report.get("resolved_policy"))
    resolved_variance = (
        _mapping(resolved_policy.get("variance"))
        if resolved_policy is not None
        else None
    )
    if policy is None or resolved_variance is None or not resolved_variance:
        errors.append(
            f"{source}.policy and report.resolved_policy.variance are required."
        )
    elif dict(policy) != dict(resolved_variance):
        errors.append(
            f"{source}.policy must match report.resolved_policy.variance exactly."
        )

    meta = _mapping(report.get("meta")) or {}
    dataset = _mapping(report.get("dataset")) or {}
    dataset_hashes = _mapping(dataset.get("hash")) or {}
    dataset_tokenizer = _mapping(dataset.get("tokenizer")) or {}
    data = _mapping(report.get("data")) or {}
    canonical_dataset_hash = dataset_hashes.get("dataset")
    canonical_tokenizer_hash = meta.get("tokenizer_hash")
    expected = {
        "model_id": meta.get("model_id"),
        "run_seed": meta.get("seed"),
        "dataset_hash": canonical_dataset_hash,
        "tokenizer_hash": canonical_tokenizer_hash,
        "pairing_digest": _report_pairing_digest(report),
        "window_ids": _report_pairing_reference(report),
    }
    for key in ("model_id", "dataset_hash", "tokenizer_hash", "pairing_digest"):
        if not isinstance(expected[key], str) or not expected[key]:
            errors.append(f"report {key} is required for strict variance provenance.")
    if _nonnegative_int(expected["run_seed"]) is None:
        errors.append("report meta.seed is required for strict variance provenance.")
    for label, value in (
        ("dataset.hash.dataset", dataset_hashes.get("dataset")),
        ("data.dataset_hash", data.get("dataset_hash")),
    ):
        if value is not None and value != canonical_dataset_hash:
            errors.append(f"report {label} must match dataset.hash.dataset.")
    for label, value in (
        ("meta.tokenizer_hash", meta.get("tokenizer_hash")),
        ("dataset.tokenizer.hash", dataset_tokenizer.get("hash")),
        ("data.tokenizer_hash", data.get("tokenizer_hash")),
    ):
        if value is not None and value != canonical_tokenizer_hash:
            errors.append(f"report {label} must match meta.tokenizer_hash.")
    return errors, expected


def _report_pairing_digest(report: Mapping[str, Any]) -> str | None:
    references = _report_pairing_reference(report)
    if not references:
        return None
    return hashlib.blake2s(
        "||".join(references).encode("utf-8"), digest_size=16
    ).hexdigest()


def _report_pairing_reference(report: Mapping[str, Any]) -> list[str] | None:
    windows = _mapping(report.get("evaluation_windows"))
    if windows is None:
        return None
    references: list[str] = []
    for phase in ("preview", "final"):
        section = _mapping(windows.get(phase))
        window_ids = (
            section.get("window_ids") or section.get("example_ids")
            if section is not None
            else None
        )
        if not isinstance(window_ids, list):
            return None
        for window_id in window_ids:
            token = str(window_id)
            references.append(token if "::" in token else f"{phase}::{token}")
    if not references:
        return None
    return references


__all__ = ["_variance_policy_semantic_errors", "_variance_report_binding_errors"]
