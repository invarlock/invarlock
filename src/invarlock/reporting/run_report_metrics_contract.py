from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from invarlock.core.metric_kind_contract import normalize_metric_kind

from .run_metric_utils import format_debug_metric_diffs, merge_primary_metric_health

_PARSE_EXCEPTIONS = (AttributeError, KeyError, OverflowError, TypeError, ValueError)


@dataclass(frozen=True)
class RunReportPolicyViolation:
    code: str
    message: str
    details: dict[str, Any]


def _coerce_report_count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _coerce_fraction(
    metric_name: str,
    value: Any,
) -> tuple[float | None, RunReportPolicyViolation | None]:
    try:
        return float(value), None
    except (TypeError, ValueError, OverflowError):
        return None, RunReportPolicyViolation(
            code="E001",
            message=(
                "PAIRING-SCHEDULE-INVALID: "
                f"{metric_name}={value!r} is not a finite numeric fraction"
            ),
            details={metric_name: value},
        )


def validate_pairing_report_metrics(
    metrics_section: Mapping[str, Any] | None,
    *,
    baseline_requested: bool,
    profile: str | None,
    preview_count_report: Any,
    final_count_report: Any,
    expected_preview: Any,
    expected_final: Any,
) -> list[RunReportPolicyViolation]:
    metrics = dict(metrics_section) if isinstance(metrics_section, Mapping) else {}
    violations: list[RunReportPolicyViolation] = []

    match_fraction = metrics.get("window_match_fraction")
    if match_fraction is not None:
        resolved_match_fraction, violation = _coerce_fraction(
            "window_match_fraction",
            match_fraction,
        )
        if violation is not None:
            violations.append(violation)
        elif resolved_match_fraction != 1.0:
            violations.append(
                RunReportPolicyViolation(
                    code="E001",
                    message=(
                        "PAIRING-SCHEDULE-MISMATCH: "
                        f"window_match_fraction={resolved_match_fraction:.3f}"
                    ),
                    details={"window_match_fraction": resolved_match_fraction},
                )
            )

    overlap_fraction = metrics.get("window_overlap_fraction")
    if overlap_fraction is not None:
        resolved_overlap_fraction, violation = _coerce_fraction(
            "window_overlap_fraction",
            overlap_fraction,
        )
        if violation is not None:
            violations.append(violation)
        elif resolved_overlap_fraction is not None and resolved_overlap_fraction > 1e-9:
            violations.append(
                RunReportPolicyViolation(
                    code="E001",
                    message=(
                        "PAIRING-SCHEDULE-MISMATCH: "
                        f"window_overlap_fraction={resolved_overlap_fraction:.3f}"
                    ),
                    details={"window_overlap_fraction": resolved_overlap_fraction},
                )
            )

    profile_normalized = (profile or "").strip().lower()
    if baseline_requested and profile_normalized in {"ci", "release"}:
        pairing_reason = metrics.get("window_pairing_reason")
        if pairing_reason is not None:
            violations.append(
                RunReportPolicyViolation(
                    code="E001",
                    message=(
                        "PAIRING-SCHEDULE-MISMATCH: baseline pairing requested but "
                        f"run was not paired (window_pairing_reason={pairing_reason})"
                    ),
                    details={"window_pairing_reason": pairing_reason},
                )
            )

        paired_windows_val = metrics.get("paired_windows")
        paired_windows_int = _coerce_report_count(paired_windows_val)
        if paired_windows_int is None or paired_windows_int <= 0:
            violations.append(
                RunReportPolicyViolation(
                    code="E001",
                    message=(
                        "PAIRED-WINDOWS-COLLAPSED: paired_windows<=0 under paired "
                        "baseline. Check device stability, dataset windows, or "
                        "edit scope."
                    ),
                    details={
                        "paired_windows": paired_windows_val,
                        "profile": profile_normalized,
                    },
                )
            )

    preview_used = _coerce_report_count(preview_count_report)
    preview_expected = _coerce_report_count(expected_preview)
    final_used = _coerce_report_count(final_count_report)
    final_expected = _coerce_report_count(expected_final)
    if (
        preview_used is not None
        and preview_expected is not None
        and preview_used != preview_expected
    ) or (
        final_used is not None
        and final_expected is not None
        and final_used != final_expected
    ):
        violations.append(
            RunReportPolicyViolation(
                code="E001",
                message=(
                    "PAIRING-SCHEDULE-MISMATCH: counts do not match configuration "
                    "after stratification"
                ),
                details={
                    "preview_used": preview_used if preview_used is not None else -1,
                    "preview_expected": (
                        preview_expected if preview_expected is not None else -1
                    ),
                    "final_used": final_used if final_used is not None else -1,
                    "final_expected": (
                        final_expected if final_expected is not None else -1
                    ),
                },
            )
        )

    return violations


def build_dataset_window_stats(
    *,
    match_fraction: Any,
    overlap_fraction: Any,
    window_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    if match_fraction is not None:
        resolved_match_fraction, violation = _coerce_fraction(
            "window_match_fraction",
            match_fraction,
        )
        if violation is not None:
            raise ValueError(violation.message)
        stats["window_match_fraction"] = resolved_match_fraction
    if overlap_fraction is not None:
        resolved_overlap_fraction, violation = _coerce_fraction(
            "window_overlap_fraction",
            overlap_fraction,
        )
        if violation is not None:
            raise ValueError(violation.message)
        stats["window_overlap_fraction"] = resolved_overlap_fraction

    if isinstance(window_plan, Mapping) and "coverage_ok" in window_plan:
        stats["coverage"] = bool(window_plan.get("coverage_ok"))
        stats["preview_total_tokens"] = window_plan.get("preview_total_tokens")
        stats["final_total_tokens"] = window_plan.get("final_total_tokens")
        stats["min_tokens_target"] = window_plan.get("min_tokens_target")
        stats["tokens_floor_met"] = window_plan.get("tokens_floor_met")

    return stats


@dataclass(frozen=True)
class RunReportMetricsEnrichmentResult:
    report: dict[str, Any]
    pairing_violations: tuple[RunReportPolicyViolation, ...]
    debug_diffs_line: str
    match_fraction: float | None
    overlap_fraction: float | None


def _coerce_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return coerced if math.isfinite(coerced) else None


def _pseudo_accuracy_allowed(profile: str, run_config: Any) -> bool:
    if str(profile or "").strip().lower() == "dev":
        return True
    env_value = str(os.environ.get("INVARLOCK_ALLOW_PSEUDO_ACCURACY", "")).lower()
    if env_value in {"1", "true", "yes"}:
        return True
    context = getattr(run_config, "context", None)
    eval_context = context.get("eval") if isinstance(context, Mapping) else None
    return bool(
        isinstance(eval_context, Mapping)
        and eval_context.get("allow_pseudo_accuracy") is True
    )


def _classification_records(arm_payload: Any) -> list[dict[str, Any]]:
    if not isinstance(arm_payload, Mapping):
        return []
    records = arm_payload.get("records", []) or []
    if isinstance(records, list):
        measured = [dict(record) for record in records if isinstance(record, Mapping)]
        if measured:
            return measured
    example_correct = arm_payload.get("example_correct", []) or []
    if isinstance(example_correct, list):
        measured = [
            {"correct": bool(value)}
            for value in example_correct
            if isinstance(value, bool | int | float)
        ]
        if measured:
            return measured
    sequences = arm_payload.get("input_ids", []) or []
    if not isinstance(sequences, list):
        return []
    return [{"input_ids": seq} for seq in sequences if isinstance(seq, list)]


def _loss_type_from_context(run_config: Any) -> str | None:
    try:
        loss_type_ctx = (
            run_config.context.get("eval", {}).get("loss", {}).get("resolved_type")
        )
    except (AttributeError, TypeError, KeyError):
        return None
    return str(loss_type_ctx).lower()


def _classification_counts_from_primary_metric(
    primary_metric: Any,
) -> tuple[int, int, int, int] | None:
    if not isinstance(primary_metric, Mapping):
        return None
    try:
        kind = str(primary_metric.get("kind", "")).lower()
    except (AttributeError, TypeError, ValueError):
        return None
    if kind != "accuracy":
        return None
    preview = _coerce_finite_float(primary_metric.get("preview"))
    final = _coerce_finite_float(primary_metric.get("final"))
    try:
        n_preview = int(primary_metric.get("n_preview", 0))
        n_final = int(primary_metric.get("n_final", 0))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if preview is None or final is None:
        return None
    if not (
        0.0 <= preview <= 1.0 and 0.0 <= final <= 1.0 and n_preview > 0 and n_final > 0
    ):
        return None
    correct_preview = int(round(preview * n_preview))
    correct_final = int(round(final * n_final))
    if abs(preview - (correct_preview / n_preview)) > 1e-9:
        return None
    if abs(final - (correct_final / n_final)) > 1e-9:
        return None
    return correct_preview, n_preview, correct_final, n_final


def _existing_classification_metrics(
    report: Mapping[str, Any], core_report: Any
) -> dict[str, Any] | None:
    report_metrics = report.get("metrics")
    existing = (
        report_metrics.get("classification")
        if isinstance(report_metrics, Mapping)
        else None
    )
    if isinstance(existing, dict):
        return existing
    core_metrics = getattr(core_report, "metrics", None)
    if not isinstance(core_metrics, Mapping):
        return None
    classification = core_metrics.get("classification")
    return dict(classification) if isinstance(classification, Mapping) else None


def _measured_classification_is_complete(classification: Mapping[str, Any]) -> bool:
    final = classification.get("final")
    return (
        str(classification.get("counts_source", "")).lower() == "measured"
        and isinstance(final, Mapping)
        and isinstance(final.get("total"), int | float)
        and int(final.get("total", 0)) > 0
    )


def _evaluation_windows(report: Mapping[str, Any], core_report: Any) -> dict[str, Any]:
    core_windows = getattr(core_report, "evaluation_windows", None)
    if isinstance(core_windows, dict) and core_windows:
        return core_windows
    report_windows = report.get("evaluation_windows")
    return report_windows if isinstance(report_windows, dict) else {}


def _fallback_classification_counts(
    *,
    report: Mapping[str, Any],
    core_report: Any,
    cfg: Any,
    preview_count_report: Any,
    final_count_report: Any,
) -> tuple[int, int, int, int, bool]:
    report_metrics = report.get("metrics")
    seed = (
        report_metrics.get("primary_metric")
        if isinstance(report_metrics, Mapping)
        else None
    )
    if not isinstance(seed, Mapping):
        core_metrics = getattr(core_report, "metrics", None)
        seed = (
            core_metrics.get("primary_metric")
            if isinstance(core_metrics, Mapping)
            else None
        )
    derived = _classification_counts_from_primary_metric(seed)
    if derived is not None:
        return *derived, False

    dataset = getattr(cfg, "dataset", None)
    preview_configured = getattr(dataset, "preview_n", None)
    final_configured = getattr(dataset, "final_n", None)
    try:
        preview_total = int(preview_count_report or preview_configured or 0)
        final_total = int(final_count_report or final_configured or 0)
    except (TypeError, ValueError, OverflowError):
        preview_total = 0
        final_total = 0
    preview_pair = (preview_total, preview_total) if preview_total > 0 else (0, 0)
    final_pair = (final_total, final_total) if final_total > 0 else (0, 0)
    return *preview_pair, *final_pair, preview_total > 0 or final_total > 0


def _enrich_classification_metrics(
    *,
    report: dict[str, Any],
    core_report: Any,
    run_config: Any,
    cfg: Any,
    profile: str,
    preview_count_report: Any,
    final_count_report: Any,
) -> None:
    existing = _existing_classification_metrics(report, core_report)
    if existing and _measured_classification_is_complete(existing):
        report.setdefault("metrics", {})["classification"] = dict(existing)
        return

    from invarlock.eval.primary_metric import compute_accuracy_counts

    windows = _evaluation_windows(report, core_report)
    preview_records = _classification_records(windows.get("preview", {}))
    final_records = _classification_records(windows.get("final", {}))
    correct_preview, total_preview = compute_accuracy_counts(preview_records)
    correct_final, total_final = compute_accuracy_counts(final_records)
    pseudo = False
    if total_preview == 0 and total_final == 0:
        (
            correct_preview,
            total_preview,
            correct_final,
            total_final,
            pseudo,
        ) = _fallback_classification_counts(
            report=report,
            core_report=core_report,
            cfg=cfg,
            preview_count_report=preview_count_report,
            final_count_report=final_count_report,
        )
        if pseudo and not _pseudo_accuracy_allowed(profile, run_config):
            raise ValueError(
                "pseudo accuracy is only allowed in dev profile or when "
                "INVARLOCK_ALLOW_PSEUDO_ACCURACY=1 / "
                "eval.allow_pseudo_accuracy=true is set"
            )

    classification = {
        "preview": {
            "correct_total": int(correct_preview),
            "total": int(total_preview),
        },
        "final": {"correct_total": int(correct_final), "total": int(total_final)},
        "n_correct": int(correct_final),
        "n_total": int(total_final),
        "counts_source": "pseudo_config" if pseudo else "measured",
        "estimated": pseudo,
    }
    report.setdefault("metrics", {})["classification"] = classification
    if total_final > 0:
        report["metrics"]["accuracy"] = float(correct_final / total_final)
    if pseudo:
        provenance = report.setdefault("provenance", {})
        notes = provenance.setdefault("metric_notes", [])
        if isinstance(notes, list):
            notes.append("accuracy: pseudo counts from preview_n/final_n")


def _recompute_primary_metric(
    *,
    report: dict[str, Any],
    core_report: Any,
    cfg: Any,
    model_profile: Any,
    baseline_report_data: Mapping[str, Any] | None,
    metric_kind: str | None,
    resolved_loss_type: str,
    debug_metric_diffs_enabled: bool,
    resolve_metric_and_provider_fn: Any,
) -> str:
    metric_kind_resolved, _provider_kind, metric_opts = resolve_metric_and_provider_fn(
        cfg,
        model_profile,
        resolved_loss_type=resolved_loss_type,
        metric_kind_override=metric_kind,
    )
    if not metric_kind_resolved:
        return ""
    normalized = normalize_metric_kind(metric_kind_resolved)
    if normalized is None:
        raise TypeError("metric kind could not be normalized")

    from invarlock.eval.primary_metric import compute_primary_metric_from_report

    baseline_report = (
        dict(baseline_report_data)
        if isinstance(baseline_report_data, Mapping)
        else None
    )
    primary_metric = compute_primary_metric_from_report(
        report, kind=normalized, baseline=baseline_report
    )
    core_metrics = getattr(core_report, "metrics", None)
    core_primary = (
        core_metrics.get("primary_metric")
        if isinstance(core_metrics, Mapping)
        else None
    )
    if isinstance(core_primary, dict):
        core_values = (
            _coerce_finite_float(core_primary.get("preview")),
            _coerce_finite_float(core_primary.get("final")),
        )
        computed_values = (
            _coerce_finite_float(primary_metric.get("preview")),
            _coerce_finite_float(primary_metric.get("final")),
        )
        if None not in core_values and None in computed_values:
            primary_metric = dict(core_primary)
    primary_metric = merge_primary_metric_health(primary_metric, core_primary)
    report.setdefault("metrics", {})["primary_metric"] = primary_metric
    try:
        if "reps" in metric_opts:
            primary_metric["reps"] = int(metric_opts["reps"])
        if "ci_level" in metric_opts:
            primary_metric["ci_level"] = float(metric_opts["ci_level"])
    except (KeyError, TypeError, ValueError):
        pass

    if debug_metric_diffs_enabled and str(primary_metric.get("kind", "")).startswith(
        "ppl"
    ):
        return format_debug_metric_diffs(
            primary_metric, report.get("metrics", {}), baseline_report
        )
    return ""


def enrich_run_report_metrics(
    *,
    report: dict[str, Any],
    core_report: Any,
    run_config: Any,
    cfg: Any,
    model_profile: Any,
    baseline_requested: bool,
    baseline_report_data: Mapping[str, Any] | None,
    metric_kind: str | None,
    resolved_loss_type: str,
    effective_preview: Any,
    effective_final: Any,
    profile_normalized: str | None,
    window_plan: Mapping[str, Any] | None,
    debug_metric_diffs_enabled: bool,
    resolve_metric_and_provider_fn: Any,
) -> RunReportMetricsEnrichmentResult:
    metrics_section = report.get("metrics", {}) or {}
    data_section = report.get("data", {}) or {}
    preview_count_report = data_section.get("preview_n")
    final_count_report = data_section.get("final_n")
    match_fraction = metrics_section.get("window_match_fraction")
    overlap_fraction = metrics_section.get("window_overlap_fraction")

    loss_type = (
        _loss_type_from_context(run_config) or str(resolved_loss_type or "").lower()
    )
    if loss_type == "classification":
        try:
            _enrich_classification_metrics(
                report=report,
                core_report=core_report,
                run_config=run_config,
                cfg=cfg,
                profile=profile_normalized or "",
                preview_count_report=preview_count_report,
                final_count_report=final_count_report,
            )
        except (
            AttributeError,
            ImportError,
            ModuleNotFoundError,
            RuntimeError,
            TypeError,
        ):
            pass

    expected_preview = effective_preview or getattr(
        cfg.dataset, "preview_n", preview_count_report
    )
    expected_final = effective_final or getattr(
        cfg.dataset, "final_n", final_count_report
    )
    pairing_violations = tuple(
        validate_pairing_report_metrics(
            metrics_section,
            baseline_requested=bool(baseline_requested),
            profile=profile_normalized,
            preview_count_report=preview_count_report,
            final_count_report=final_count_report,
            expected_preview=expected_preview,
            expected_final=expected_final,
        )
    )

    debug_diffs_line = ""
    try:
        debug_diffs_line = _recompute_primary_metric(
            report=report,
            core_report=core_report,
            cfg=cfg,
            model_profile=model_profile,
            baseline_report_data=baseline_report_data,
            metric_kind=metric_kind,
            resolved_loss_type=resolved_loss_type,
            debug_metric_diffs_enabled=debug_metric_diffs_enabled,
            resolve_metric_and_provider_fn=resolve_metric_and_provider_fn,
        )
    except (
        AttributeError,
        ImportError,
        ModuleNotFoundError,
        RuntimeError,
        TypeError,
    ):
        pass

    try:
        dataset_windows = report.setdefault("dataset", {}).setdefault("windows", {})
        dataset_windows["stats"] = build_dataset_window_stats(
            match_fraction=match_fraction,
            overlap_fraction=overlap_fraction,
            window_plan=window_plan,
        )
    except (AttributeError, KeyError, TypeError):
        pass

    return RunReportMetricsEnrichmentResult(
        report=report,
        pairing_violations=pairing_violations,
        debug_diffs_line=debug_diffs_line,
        match_fraction=match_fraction
        if match_fraction is None
        else float(match_fraction),
        overlap_fraction=(
            overlap_fraction if overlap_fraction is None else float(overlap_fraction)
        ),
    )
