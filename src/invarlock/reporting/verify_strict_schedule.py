"""Strict baseline and evaluation-schedule binding helpers."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from invarlock.core.metric_kind_contract import (
    MetricKindContractError,
    is_ppl_metric_kind,
    normalize_metric_kind,
)


def _strict_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _schedule_window_id_key(value: Any) -> tuple[str, int | str]:
    """Canonicalize IDs using the same integer-first rule as report building."""

    try:
        return "integer", int(value)
    except (TypeError, ValueError, OverflowError):
        return "text", str(value)


def _schedule_digest(window_ids: list[Any]) -> str:
    """Recompute the final-window BLAKE2s-16 digest emitted by reports."""

    digest = hashlib.blake2s(digest_size=16)
    for window_id in window_ids:
        try:
            digest.update(int(window_id).to_bytes(8, "little", signed=True))
        except (AttributeError, TypeError, ValueError, OverflowError):
            digest.update(str(window_id).encode("utf-8", "ignore"))
    return digest.hexdigest()


def _schedule_digest_has_fixed_integer_encoding(window_ids: list[Any]) -> bool:
    return all(
        isinstance(window_id, int)
        and not isinstance(window_id, bool)
        and -(2**63) <= window_id < 2**63
        for window_id in window_ids
    )


def _metric_kind_and_final(
    errors: list[str],
    *,
    metric: Any,
    source: str,
) -> tuple[str, float] | None:
    if not isinstance(metric, dict):
        errors.append(f"Strict baseline binding requires {source} as an object.")
        return None
    try:
        kind = normalize_metric_kind(metric.get("kind"))
    except (MetricKindContractError, RuntimeError, TypeError, ValueError):
        kind = None
    if kind is None:
        errors.append(f"Strict baseline binding requires a supported {source}.kind.")
        return None
    final = _strict_finite_number(metric.get("final"))
    if final is None:
        errors.append(f"Strict baseline binding requires finite {source}.final.")
        return None
    if is_ppl_metric_kind(kind) and final < 1.0:
        errors.append(f"Strict baseline binding requires {source}.final >= 1 for PPL.")
        return None
    if kind == "accuracy" and not 0.0 <= final <= 1.0:
        errors.append(
            f"Strict baseline binding requires {source}.final in [0,1] for accuracy."
        )
        return None
    return kind, final


def _supplied_baseline_metric(
    errors: list[str],
    *,
    baseline_payload: dict[str, Any],
    tolerance: float,
) -> tuple[str, float] | None:
    candidates: list[tuple[str, Any]] = []
    if "primary_metric" in baseline_payload:
        candidates.append(
            ("supplied_baseline.primary_metric", baseline_payload.get("primary_metric"))
        )
    metrics = baseline_payload.get("metrics")
    if isinstance(metrics, dict) and "primary_metric" in metrics:
        candidates.append(
            (
                "supplied_baseline.metrics.primary_metric",
                metrics.get("primary_metric"),
            )
        )
    if not candidates:
        errors.append(
            "Strict baseline binding requires an independently supplied baseline "
            "primary metric."
        )
        return None

    parsed: list[tuple[str, float, str]] = []
    for source, candidate in candidates:
        value = _metric_kind_and_final(errors, metric=candidate, source=source)
        if value is not None:
            parsed.append((value[0], value[1], source))
    if len(parsed) != len(candidates):
        return None

    first_kind, first_final, first_source = parsed[0]
    for kind, final, source in parsed[1:]:
        if kind != first_kind:
            errors.append(
                "Supplied baseline metric kind mismatch between "
                f"{first_source} and {source}."
            )
        if not math.isclose(
            final,
            first_final,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            errors.append(
                "Supplied baseline final mismatch between "
                f"{first_source}={first_final:.12f} and {source}={final:.12f}."
            )
    return first_kind, first_final


def _canonical_schedule_ids(
    errors: list[str],
    *,
    value: Any,
    source: str,
) -> tuple[tuple[str, int | str], ...] | None:
    if not isinstance(value, list) or not value:
        errors.append(f"Strict baseline binding requires {source} as a non-empty list.")
        return None
    for index, window_id in enumerate(value):
        if isinstance(window_id, bool) or not isinstance(window_id, int | str):
            errors.append(
                f"{source}[{index}] must be a JSON integer or non-empty string."
            )
            return None
        if isinstance(window_id, str) and not window_id:
            errors.append(
                f"{source}[{index}] must be a JSON integer or non-empty string."
            )
            return None
    canonical = tuple(_schedule_window_id_key(item) for item in value)
    if len(canonical) != len(set(canonical)):
        errors.append(f"{source} contains duplicates.")
        return None
    return canonical


def _declared_schedule_digests(
    errors: list[str],
    *,
    payload: dict[str, Any],
    source_prefix: str,
) -> list[tuple[str, str]]:
    provenance = payload.get("provenance")
    guard_metric_impact = payload.get("guard_metric_impact")
    fields = (
        (provenance, "window_ids_digest", "provenance.window_ids_digest"),
        (provenance, "window_plan_digest", "provenance.window_plan_digest"),
        (guard_metric_impact, "schedule_digest", "guard_metric_impact.schedule_digest"),
    )
    digests: list[tuple[str, str]] = []
    for container, key, suffix in fields:
        if not isinstance(container, dict) or key not in container:
            continue
        value = container.get(key)
        source = f"{source_prefix}.{suffix}"
        normalized = value.strip().lower() if isinstance(value, str) else ""
        if len(normalized) != 32 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            errors.append(f"{source} must be a BLAKE2s-16 hex digest.")
            continue
        digests.append((source, normalized))
    return digests


def _append_strict_supplied_baseline_binding_errors(
    errors: list[str],
    *,
    cert_obj: dict[str, Any],
    baseline_payload: dict[str, Any] | None,
    baseline_supplied: bool,
    tolerance: float,
) -> None:
    """Bind strict report claims to the independently supplied baseline payload."""

    if not baseline_supplied:
        return
    if baseline_payload is None:
        errors.append(
            "Strict baseline binding could not load the independently supplied baseline."
        )
        return

    baseline_metric = _supplied_baseline_metric(
        errors,
        baseline_payload=baseline_payload,
        tolerance=tolerance,
    )
    baseline_ref = cert_obj.get("baseline_ref")
    reference_metric = _metric_kind_and_final(
        errors,
        metric=(
            baseline_ref.get("primary_metric")
            if isinstance(baseline_ref, dict)
            else None
        ),
        source="report.baseline_ref.primary_metric",
    )
    subject_metric = _metric_kind_and_final(
        errors,
        metric=cert_obj.get("primary_metric"),
        source="report.primary_metric",
    )
    if baseline_metric is not None and reference_metric is not None:
        baseline_kind, independent_baseline_final = baseline_metric
        reference_kind, reference_final = reference_metric
        if reference_kind != baseline_kind:
            errors.append(
                "Supplied baseline metric kind mismatch: "
                f"report.baseline_ref={reference_kind} supplied={baseline_kind}."
            )
        if not math.isclose(
            reference_final,
            independent_baseline_final,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            errors.append(
                "Supplied baseline final mismatch: "
                f"report.baseline_ref={reference_final:.12f} "
                f"supplied={independent_baseline_final:.12f}."
            )
        if subject_metric is not None and subject_metric[0] != baseline_kind:
            errors.append(
                "Supplied baseline metric kind mismatch: "
                f"report.primary_metric={subject_metric[0]} supplied={baseline_kind}."
            )

    subject_kind = subject_metric[0] if subject_metric is not None else None
    if subject_kind is None or not is_ppl_metric_kind(subject_kind):
        return

    subject_windows = cert_obj.get("evaluation_windows")
    subject_final = (
        subject_windows.get("final") if isinstance(subject_windows, dict) else None
    )
    subject_ids = (
        subject_final.get("window_ids") if isinstance(subject_final, dict) else None
    )
    subject_canonical = _canonical_schedule_ids(
        errors,
        value=subject_ids,
        source="report.evaluation_windows.final.window_ids",
    )
    if subject_canonical is None or not isinstance(subject_ids, list):
        return
    subject_digest = _schedule_digest(subject_ids)

    baseline_windows = baseline_payload.get("evaluation_windows")
    baseline_final_windows = (
        baseline_windows.get("final") if isinstance(baseline_windows, dict) else None
    )
    baseline_ids_present = False
    baseline_canonical = None
    if (
        isinstance(baseline_final_windows, dict)
        and "window_ids" in baseline_final_windows
    ):
        baseline_ids_present = True
        baseline_canonical = _canonical_schedule_ids(
            errors,
            value=baseline_final_windows.get("window_ids"),
            source="supplied_baseline.evaluation_windows.final.window_ids",
        )
        if baseline_canonical is not None and baseline_canonical != subject_canonical:
            errors.append(
                "Supplied baseline final schedule mismatch: canonical window IDs "
                "differ from the subject report."
            )

    declared_digests = _declared_schedule_digests(
        errors,
        payload=baseline_payload,
        source_prefix="supplied_baseline",
    )
    for source, digest in declared_digests:
        if digest != subject_digest:
            errors.append(
                "Supplied baseline final schedule digest mismatch: "
                f"{source}={digest} subject={subject_digest}."
            )

    if (
        not baseline_ids_present
        and declared_digests
        and not _schedule_digest_has_fixed_integer_encoding(subject_ids)
    ):
        errors.append(
            "Strict digest-only baseline schedule binding requires signed 64-bit "
            "integer window IDs; supply raw baseline window IDs for string or "
            "out-of-range identifiers."
        )

    if not baseline_ids_present and not declared_digests:
        errors.append(
            "Strict paired PPL baseline binding requires independently supplied "
            "final window IDs or a final schedule digest."
        )
