"""Generic primary-metric impact contract shared by runtime and strict replay."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

REQUIRED_GUARD_METRIC_IMPACT_CHECKS = frozenset(
    {
        "metric_kind_matches",
        "measurements_valid",
        "guard_metric_impact",
        "arm_facts_replay",
    }
)

SUCCESSFUL_GUARD_METRIC_EXECUTION_STATUSES = frozenset({"success", "completed", "ok"})

GUARD_METRIC_IMPACT_REPORT_FIELDS = frozenset(
    {
        "bare_facts",
        "bare_report",
        "bare_value",
        "checks",
        "degradation",
        "degradation_basis",
        "degradation_limit",
        "diagnostics",
        "direction",
        "display_unit",
        "display_value",
        "evaluated",
        "guarded_facts",
        "guarded_value",
        "metric_kind",
        "mode",
        "passed",
        "schedule_digest",
        "skip_reason",
        "skipped",
        "source",
    }
)


@dataclass(frozen=True)
class GuardMetricImpactMeasurement:
    metric_kind: str
    direction: str
    bare_value: float
    guarded_value: float
    degradation_basis: str
    degradation: float
    display_value: float
    display_unit: str

    def to_metrics(self) -> dict[str, str | float]:
        return {
            "metric_kind": self.metric_kind,
            "direction": self.direction,
            "bare_value": self.bare_value,
            "guarded_value": self.guarded_value,
            "degradation_basis": self.degradation_basis,
            "degradation": self.degradation,
            "display_value": self.display_value,
            "display_unit": self.display_unit,
        }


def _metric_contract(kind: Any) -> tuple[str, str, str] | None:
    normalized = str(kind).strip().lower() if isinstance(kind, str) else ""
    try:
        from .primary_metric import get_metric

        metric = get_metric(normalized)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    direction = str(metric.direction)
    basis = str(metric.guard_degradation_basis)
    if direction == "lower" and basis == "relative_increase":
        return normalized, direction, basis
    if direction == "higher" and basis == "absolute_drop":
        return normalized, direction, basis
    return None


def _finite_measurement(kind: str, value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    resolved = float(value)
    if not math.isfinite(resolved):
        return None
    contract = _metric_contract(kind)
    if contract is None:
        return None
    _, direction, _ = contract
    if direction == "lower":
        return resolved if resolved > 0.0 else None
    if direction == "higher":
        return resolved if 0.0 <= resolved <= 1.0 else None
    return None


def compute_guard_metric_impact(
    metric_kind: Any,
    bare_value: Any,
    guarded_value: Any,
) -> GuardMetricImpactMeasurement | None:
    """Return the canonical impact measurement for one supported metric kind."""

    kind = str(metric_kind).strip().lower() if isinstance(metric_kind, str) else ""
    contract = _metric_contract(kind)
    bare = _finite_measurement(kind, bare_value)
    guarded = _finite_measurement(kind, guarded_value)
    if contract is None or bare is None or guarded is None:
        return None
    _, direction, basis = contract
    if direction == "lower":
        degradation = (guarded / bare) - 1.0
        display_unit = "percent"
    else:
        degradation = bare - guarded
        display_unit = "percentage_points"
    return GuardMetricImpactMeasurement(
        metric_kind=kind,
        direction=direction,
        bare_value=bare,
        guarded_value=guarded,
        degradation_basis=basis,
        degradation=degradation,
        display_value=degradation * 100.0,
        display_unit=display_unit,
    )


def degradation_within_limit(*, degradation: Any, degradation_limit: Any) -> bool:
    if (
        isinstance(degradation, bool)
        or not isinstance(degradation, int | float)
        or isinstance(degradation_limit, bool)
        or not isinstance(degradation_limit, int | float)
    ):
        return False
    value = float(degradation)
    limit = float(degradation_limit)
    if not math.isfinite(value) or not math.isfinite(limit) or limit < 0.0:
        return False
    return value <= limit or math.isclose(value, limit, rel_tol=1e-12, abs_tol=1e-12)


def _report_mapping(report: Any, field: str) -> Mapping[str, Any] | None:
    value = (
        report.get(field)
        if isinstance(report, Mapping)
        else getattr(report, field, None)
    )
    return value if isinstance(value, Mapping) else None


def normalize_guard_metric_execution_status(status: Any) -> str | None:
    """Normalize a retained execution status without inventing a fallback."""

    if not isinstance(status, str):
        return None
    normalized = status.strip().lower()
    return normalized or None


def guard_metric_execution_status_is_successful(status: Any) -> bool:
    """Return whether an execution status is an accepted completed outcome."""

    return (
        normalize_guard_metric_execution_status(status)
        in SUCCESSFUL_GUARD_METRIC_EXECUTION_STATUSES
    )


def _ids_digest(raw_ids: Any) -> str | None:
    if not isinstance(raw_ids, list) or not raw_ids:
        return None
    encoded = json.dumps(raw_ids, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def ordered_guard_metric_final_ids(report: Any, metric_kind: Any) -> list[Any] | None:
    """Return the ordered final-arm IDs appropriate for the primary metric."""

    kind = str(metric_kind).strip().lower() if isinstance(metric_kind, str) else ""
    windows = _report_mapping(report, "evaluation_windows") or {}
    final = windows.get("final")
    if not isinstance(final, Mapping) and isinstance(report, Mapping):
        final = report.get("final")
    if not isinstance(final, Mapping):
        return None
    key = "example_ids" if kind == "accuracy" else "window_ids"
    values = final.get(key)
    return list(values) if isinstance(values, list) and values else None


def guard_metric_schedule_digest(report: Any, metric_kind: Any) -> str | None:
    ids = ordered_guard_metric_final_ids(report, metric_kind)
    if not ids:
        return None
    digest = hashlib.blake2s(digest_size=16)
    for value in ids:
        try:
            digest.update(int(value).to_bytes(8, "little", signed=True))
        except (AttributeError, TypeError, ValueError, OverflowError):
            digest.update(str(value).encode("utf-8", "ignore"))
    return digest.hexdigest()


def extract_guard_metric_arm_facts(
    report: Any,
    metric_kind: Any,
) -> dict[str, int | float | str] | None:
    """Retain sufficient final-arm facts to independently replay its metric value."""

    kind = str(metric_kind).strip().lower() if isinstance(metric_kind, str) else ""
    if _metric_contract(kind) is None:
        return None
    metrics = _report_mapping(report, "metrics") or {}
    windows = _report_mapping(report, "evaluation_windows") or {}
    final_windows = windows.get("final")
    if not isinstance(final_windows, Mapping) and isinstance(report, Mapping):
        final_windows = report.get("final")
    final_windows = final_windows if isinstance(final_windows, Mapping) else {}
    ids_key = "example_ids" if kind == "accuracy" else "window_ids"
    ids_digest = _ids_digest(final_windows.get(ids_key))

    if kind == "accuracy":
        classification = metrics.get("classification")
        classification = classification if isinstance(classification, Mapping) else {}
        final = classification.get("final")
        final = final if isinstance(final, Mapping) else final_windows
        correct = final.get("correct_total")
        total = final.get("total")
        if (
            isinstance(correct, bool)
            or not isinstance(correct, int)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total <= 0
            or correct < 0
            or correct > total
        ):
            return None
        facts: dict[str, int | float | str] = {
            "correct": correct,
            "total": total,
        }
    else:
        observations: Iterable[tuple[Any, Any]]
        raw_losses = final_windows.get("logloss")
        count_key = "masked_token_counts" if kind == "ppl_mlm" else "token_counts"
        raw_counts = final_windows.get(count_key)
        weighted_terms: list[float] = []
        token_count = 0
        if (
            isinstance(raw_losses, list)
            and isinstance(raw_counts, list)
            and raw_losses
            and len(raw_losses) == len(raw_counts)
        ):
            observations = zip(raw_losses, raw_counts, strict=True)
        else:
            mean_logloss = metrics.get("logloss_final")
            aggregate_count = metrics.get(
                "masked_tokens_final" if kind == "ppl_mlm" else "final_total_tokens"
            )
            observations = ((mean_logloss, aggregate_count),)
        for raw_loss, raw_count in observations:
            if (
                isinstance(raw_loss, bool)
                or not isinstance(raw_loss, int | float)
                or not math.isfinite(float(raw_loss))
                or float(raw_loss) < 0.0
                or isinstance(raw_count, bool)
                or not isinstance(raw_count, int)
                or raw_count <= 0
            ):
                return None
            weighted_terms.append(float(raw_loss) * raw_count)
            token_count += raw_count
        facts = {
            "weighted_logloss_sum": math.fsum(weighted_terms),
            "token_count": token_count,
        }
    if ids_digest is not None:
        facts["example_ids_digest"] = ids_digest
    return facts


def build_guard_metric_bare_report(
    report: Any,
    metric_kind: Any,
) -> dict[str, Any] | None:
    """Build the closed minimal bare-arm envelope retained by measured reports."""

    kind = str(metric_kind).strip().lower() if isinstance(metric_kind, str) else ""
    facts = extract_guard_metric_arm_facts(report, kind)
    value = metric_value_from_arm_facts(kind, facts)
    if facts is None or value is None:
        return None
    windows = _report_mapping(report, "evaluation_windows") or {}
    final_windows = windows.get("final")
    if not isinstance(final_windows, Mapping) and isinstance(report, Mapping):
        final_windows = report.get("final")
    final_windows = final_windows if isinstance(final_windows, Mapping) else {}
    final: dict[str, Any]
    if kind == "accuracy":
        final = {"correct_total": facts["correct"], "total": facts["total"]}
        raw_ids = final_windows.get("example_ids")
        if isinstance(raw_ids, list) and raw_ids:
            final["example_ids"] = list(raw_ids)
    else:
        count_key = "masked_token_counts" if kind == "ppl_mlm" else "token_counts"
        losses = final_windows.get("logloss")
        counts = final_windows.get(count_key)
        if isinstance(losses, list) and isinstance(counts, list) and losses and counts:
            final = {"logloss": list(losses), count_key: list(counts)}
        else:
            token_count = int(facts["token_count"])
            final = {
                "logloss": [float(facts["weighted_logloss_sum"]) / token_count],
                count_key: [token_count],
            }
        raw_ids = final_windows.get("window_ids")
        if isinstance(raw_ids, list) and raw_ids:
            final["window_ids"] = list(raw_ids)
    envelope: dict[str, Any] = {
        "primary_metric": {"kind": kind, "final": value},
        "final": final,
    }
    status = normalize_guard_metric_execution_status(
        report.get("status")
        if isinstance(report, Mapping)
        else getattr(report, "status", None)
    )
    if status is not None:
        envelope["status"] = status
    return envelope


def metric_value_from_arm_facts(
    metric_kind: Any,
    facts: Any,
) -> float | None:
    """Recompute a final primary-metric point from canonical retained arm facts."""

    kind = str(metric_kind).strip().lower() if isinstance(metric_kind, str) else ""
    if not isinstance(facts, Mapping) or _metric_contract(kind) is None:
        return None
    if kind == "accuracy":
        correct = facts.get("correct")
        total = facts.get("total")
        if (
            isinstance(correct, bool)
            or not isinstance(correct, int)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total <= 0
            or correct < 0
            or correct > total
        ):
            return None
        return correct / total
    weighted_sum = facts.get("weighted_logloss_sum")
    token_count = facts.get("token_count")
    if (
        isinstance(weighted_sum, bool)
        or not isinstance(weighted_sum, int | float)
        or not math.isfinite(float(weighted_sum))
        or float(weighted_sum) < 0.0
        or isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or token_count <= 0
    ):
        return None
    try:
        value = math.exp(float(weighted_sum) / token_count)
    except OverflowError:
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def arm_facts_match_measurements(
    metric_kind: Any,
    bare_facts: Any,
    guarded_facts: Any,
    bare_value: Any,
    guarded_value: Any,
) -> bool:
    """Check arm-fact replay and optional paired example-identity binding."""

    bare_replayed = metric_value_from_arm_facts(metric_kind, bare_facts)
    guarded_replayed = metric_value_from_arm_facts(metric_kind, guarded_facts)
    bare = _finite_measurement(str(metric_kind), bare_value)
    guarded = _finite_measurement(str(metric_kind), guarded_value)
    if (
        bare_replayed is None
        or guarded_replayed is None
        or bare is None
        or guarded is None
        or not math.isclose(bare_replayed, bare, rel_tol=1e-9, abs_tol=1e-9)
        or not math.isclose(guarded_replayed, guarded, rel_tol=1e-9, abs_tol=1e-9)
    ):
        return False
    assert isinstance(bare_facts, Mapping)
    assert isinstance(guarded_facts, Mapping)
    bare_digest = bare_facts.get("example_ids_digest")
    guarded_digest = guarded_facts.get("example_ids_digest")
    return bool(
        isinstance(bare_digest, str)
        and bare_digest
        and isinstance(guarded_digest, str)
        and guarded_digest == bare_digest
    )


def _primary_metric_value(report: Any, metric_kind: str) -> float | None:
    metrics = _report_mapping(report, "metrics") or {}
    primary = metrics.get("primary_metric")
    if not isinstance(primary, Mapping) and isinstance(report, Mapping):
        primary = report.get("primary_metric")
    if not isinstance(primary, Mapping) or primary.get("kind") != metric_kind:
        return None
    return _finite_measurement(metric_kind, primary.get("final"))


def _facts_equal(left: Any, right: Any) -> bool:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    if set(left) != set(right):
        return False
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, int | float) and not isinstance(left_value, bool):
            if (
                isinstance(right_value, bool)
                or not isinstance(right_value, int | float)
                or not math.isclose(
                    float(left_value),
                    float(right_value),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                return False
        elif left_value != right_value:
            return False
    return True


def guard_metric_impact_payload_errors(
    payload: Any,
    *,
    subject_report: Any | None = None,
    require_bare_report: bool = False,
) -> list[str]:
    """Replay a measured payload and optionally bind it to both retained arms."""

    if not isinstance(payload, Mapping):
        return ["guard metric impact payload must be an object"]
    errors: list[str] = []
    unsupported_fields = sorted(set(payload) - GUARD_METRIC_IMPACT_REPORT_FIELDS)
    if unsupported_fields:
        errors.append(
            "guard metric impact payload contains unsupported fields: "
            + ", ".join(unsupported_fields)
        )
    metric_kind = payload.get("metric_kind")
    measurement = compute_guard_metric_impact(
        metric_kind,
        payload.get("bare_value"),
        payload.get("guarded_value"),
    )
    if measurement is None:
        return ["guard metric impact measurements are invalid"]
    for field, expected in (
        ("direction", measurement.direction),
        ("degradation_basis", measurement.degradation_basis),
        ("display_unit", measurement.display_unit),
    ):
        if payload.get(field) != expected:
            errors.append(f"guard metric impact {field} mismatch")
    for field, expected_numeric in (
        ("degradation", measurement.degradation),
        ("display_value", measurement.display_value),
    ):
        observed = payload.get(field)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int | float)
            or not math.isfinite(float(observed))
            or not math.isclose(
                float(observed), expected_numeric, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            errors.append(f"guard metric impact {field} mismatch")
    if not arm_facts_match_measurements(
        metric_kind,
        payload.get("bare_facts"),
        payload.get("guarded_facts"),
        payload.get("bare_value"),
        payload.get("guarded_value"),
    ):
        errors.append("guard metric impact arm facts mismatch")
    checks = payload.get("checks")
    if (
        not isinstance(checks, Mapping)
        or not REQUIRED_GUARD_METRIC_IMPACT_CHECKS <= set(checks)
        or any(value is not True for value in checks.values())
    ):
        errors.append("guard metric impact checks are incomplete or failed")
    if not degradation_within_limit(
        degradation=payload.get("degradation"),
        degradation_limit=payload.get("degradation_limit"),
    ):
        errors.append("guard metric impact degradation exceeds its limit")
    if payload.get("evaluated") is not True or payload.get("passed") is not True:
        errors.append("guard metric impact outcome is not passing")

    if subject_report is not None and isinstance(metric_kind, str):
        subject_value = _primary_metric_value(subject_report, metric_kind)
        subject_facts = extract_guard_metric_arm_facts(subject_report, metric_kind)
        guarded_value = _finite_measurement(metric_kind, payload.get("guarded_value"))
        if (
            subject_value is None
            or guarded_value is None
            or not math.isclose(
                subject_value, guarded_value, rel_tol=1e-12, abs_tol=1e-12
            )
            or not _facts_equal(subject_facts, payload.get("guarded_facts"))
        ):
            errors.append("guard metric impact guarded arm is not bound to the report")

        observed_schedule_digest = payload.get("schedule_digest")
        expected_schedule_digest = guard_metric_schedule_digest(
            subject_report,
            metric_kind,
        )
        if (
            not isinstance(observed_schedule_digest, str)
            or expected_schedule_digest is None
            or observed_schedule_digest != expected_schedule_digest
        ):
            errors.append(
                "guard metric impact schedule digest is not bound to the report"
            )

    bare_report = payload.get("bare_report")
    if require_bare_report or bare_report is not None:
        if not isinstance(metric_kind, str):
            errors.append("guard metric impact bare report metric kind is invalid")
        else:
            if not guard_metric_execution_status_is_successful(
                bare_report.get("status") if isinstance(bare_report, Mapping) else None
            ):
                errors.append(
                    "guard metric impact bare arm does not retain a successful execution status"
                )
            bare_value = _primary_metric_value(bare_report, metric_kind)
            retained_bare = _finite_measurement(metric_kind, payload.get("bare_value"))
            bare_facts = extract_guard_metric_arm_facts(bare_report, metric_kind)
            if (
                bare_value is None
                or retained_bare is None
                or not math.isclose(
                    bare_value, retained_bare, rel_tol=1e-12, abs_tol=1e-12
                )
                or not _facts_equal(bare_facts, payload.get("bare_facts"))
            ):
                errors.append(
                    "guard metric impact bare arm is not bound to retained evidence"
                )
    return errors


__all__ = [
    "GuardMetricImpactMeasurement",
    "GUARD_METRIC_IMPACT_REPORT_FIELDS",
    "REQUIRED_GUARD_METRIC_IMPACT_CHECKS",
    "SUCCESSFUL_GUARD_METRIC_EXECUTION_STATUSES",
    "arm_facts_match_measurements",
    "build_guard_metric_bare_report",
    "compute_guard_metric_impact",
    "degradation_within_limit",
    "extract_guard_metric_arm_facts",
    "guard_metric_execution_status_is_successful",
    "guard_metric_impact_payload_errors",
    "guard_metric_schedule_digest",
    "metric_value_from_arm_facts",
    "normalize_guard_metric_execution_status",
    "ordered_guard_metric_final_ids",
]
