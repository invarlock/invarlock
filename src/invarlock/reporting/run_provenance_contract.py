"""Owner contract for run evaluation-window and provenance finalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_NON_FATAL_EXCEPTIONS = (AttributeError, KeyError, OSError, TypeError, ValueError)


@dataclass(frozen=True)
class RunProvenanceResult:
    missing_evaluation_windows_for_baseline: bool = False
    missing_evaluation_windows_message: str | None = None


def _window_payload_has_signal(window_payload: Any) -> bool:
    if not isinstance(window_payload, Mapping):
        return False
    for key in (
        "window_ids",
        "example_ids",
        "logloss",
        "input_ids",
        "attention_masks",
        "token_counts",
        "masked_token_counts",
        "actual_token_counts",
        "labels",
        "records",
    ):
        value = window_payload.get(key)
        if isinstance(value, list) and value:
            return True
    return False


def _evaluation_windows_have_signal(serialized_windows: Any) -> bool:
    if not isinstance(serialized_windows, Mapping):
        return False
    return _window_payload_has_signal(serialized_windows.get("preview")) or (
        _window_payload_has_signal(serialized_windows.get("final"))
    )


def finalize_run_provenance(
    *,
    report: dict[str, Any],
    core_report: Any,
    preview_records: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    use_mlm: bool,
    preview_mask_counts: list[int] | None,
    final_mask_counts: list[int] | None,
    had_baseline: bool,
    profile: str | None,
    resolved_split: str | None,
    used_fallback_split: bool,
    baseline_report_data: dict[str, Any] | None,
    serialize_evaluation_windows_fn: Any,
    build_fallback_evaluation_windows_fn: Any,
    compute_provider_digest_fn: Any,
    enforce_provider_parity_fn: Any,
) -> RunProvenanceResult:
    """Finalize evaluation windows plus run provenance and provider parity."""

    serialized_evaluation_windows = serialize_evaluation_windows_fn(
        getattr(core_report, "evaluation_windows", None)
    )
    if _evaluation_windows_have_signal(serialized_evaluation_windows):
        report["evaluation_windows"] = serialized_evaluation_windows
    else:
        try:
            fallback_evaluation_windows = build_fallback_evaluation_windows_fn(
                preview_records,
                final_records,
                use_mlm=use_mlm,
                preview_mask_counts=preview_mask_counts,
                final_mask_counts=final_mask_counts,
            )
            if fallback_evaluation_windows:
                report["evaluation_windows"] = fallback_evaluation_windows
        except _NON_FATAL_EXCEPTIONS:
            pass
        if (
            "evaluation_windows" not in report
            and had_baseline
            and (profile or "").lower() in {"ci", "release"}
        ):
            return RunProvenanceResult(
                missing_evaluation_windows_for_baseline=True,
                missing_evaluation_windows_message=(
                    "[INVARLOCK:E001] PAIRING-SCHEDULE-MISMATCH: baseline pairing "
                    "requested but evaluation windows were not produced. Check "
                    "capacity/pairing config."
                ),
            )

    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
        report["provenance"] = provenance

    try:
        provenance["dataset_split"] = str(resolved_split)
        provenance["split_fallback"] = bool(used_fallback_split)
    except _NON_FATAL_EXCEPTIONS:
        pass

    try:
        provider_digest = compute_provider_digest_fn(report)
    except _NON_FATAL_EXCEPTIONS:
        provider_digest = None
    if not provider_digest:
        return RunProvenanceResult()

    provenance["provider_digest"] = provider_digest
    provenance["digest_version"] = 1

    if not isinstance(baseline_report_data, dict):
        return RunProvenanceResult()

    base_digest = None
    base_provenance = baseline_report_data.get("provenance")
    if isinstance(base_provenance, dict):
        base_provider_digest = base_provenance.get("provider_digest")
        if isinstance(base_provider_digest, dict):
            base_digest = base_provider_digest
    if base_digest is None:
        try:
            base_digest = compute_provider_digest_fn(baseline_report_data)
        except _NON_FATAL_EXCEPTIONS:
            base_digest = None

    enforce_provider_parity_fn(
        provider_digest,
        base_digest,
        profile=(str(profile).lower() if profile else None),
    )
    return RunProvenanceResult()
