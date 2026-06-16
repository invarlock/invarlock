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
            existing_classification = (
                report.get("metrics", {}).get("classification")
                if isinstance(report.get("metrics"), dict)
                else None
            )
            if not isinstance(existing_classification, dict) and hasattr(
                core_report, "metrics"
            ):
                core_metrics = (
                    core_report.metrics if isinstance(core_report.metrics, dict) else {}
                )
                existing_classification = (
                    core_metrics.get("classification")
                    if isinstance(core_metrics.get("classification"), dict)
                    else None
                )
            if isinstance(existing_classification, dict) and existing_classification:
                counts_source = str(
                    existing_classification.get("counts_source", "")
                ).lower()
                final_existing = existing_classification.get("final", {})
                if (
                    counts_source == "measured"
                    and isinstance(final_existing, Mapping)
                    and isinstance(final_existing.get("total"), (int, float))
                    and int(final_existing.get("total", 0)) > 0
                ):
                    report.setdefault("metrics", {})["classification"] = dict(
                        existing_classification
                    )
                    raise StopIteration

            from invarlock.eval.primary_metric import compute_accuracy_counts

            evaluation_windows = {}
            try:
                if hasattr(core_report, "evaluation_windows") and isinstance(
                    core_report.evaluation_windows, dict
                ):
                    evaluation_windows = core_report.evaluation_windows
            except (AttributeError, TypeError):
                evaluation_windows = {}
            if not evaluation_windows:
                report_windows = report.get("evaluation_windows")
                if isinstance(report_windows, dict):
                    evaluation_windows = report_windows

            preview_records = _classification_records(
                evaluation_windows.get("preview", {})
                if isinstance(evaluation_windows, dict)
                else {}
            )
            final_records = _classification_records(
                evaluation_windows.get("final", {})
                if isinstance(evaluation_windows, dict)
                else {}
            )
            c_prev, n_prev = compute_accuracy_counts(preview_records)
            c_fin, n_fin = compute_accuracy_counts(final_records)

            used_pseudo_counts = False
            if n_prev == 0 and n_fin == 0:
                primary_metric_seed = (
                    report.get("metrics", {}).get("primary_metric")
                    if isinstance(report.get("metrics"), dict)
                    else None
                )
                if not isinstance(primary_metric_seed, Mapping) and hasattr(
                    core_report, "metrics"
                ):
                    core_metrics = (
                        core_report.metrics
                        if isinstance(core_report.metrics, dict)
                        else {}
                    )
                    primary_metric_seed = (
                        core_metrics.get("primary_metric")
                        if isinstance(core_metrics.get("primary_metric"), Mapping)
                        else None
                    )
                derived_counts = _classification_counts_from_primary_metric(
                    primary_metric_seed
                )
                if derived_counts is not None:
                    c_prev, n_prev, c_fin, n_fin = derived_counts
                else:
                    try:
                        prev_n_cfg = getattr(cfg.dataset, "preview_n", None)
                        fin_n_cfg = getattr(cfg.dataset, "final_n", None)
                    except (AttributeError, TypeError):
                        prev_n_cfg = None
                        fin_n_cfg = None
                    try:
                        prev_n = int(preview_count_report or prev_n_cfg or 0)
                        fin_n = int(final_count_report or fin_n_cfg or 0)
                    except (TypeError, ValueError, OverflowError):
                        prev_n = 0
                        fin_n = 0
                    c_prev, n_prev = (prev_n, prev_n) if prev_n > 0 else (0, 0)
                    c_fin, n_fin = (fin_n, fin_n) if fin_n > 0 else (0, 0)
                    used_pseudo_counts = prev_n > 0 or fin_n > 0
                    if used_pseudo_counts and not _pseudo_accuracy_allowed(
                        profile_normalized or "",
                        run_config,
                    ):
                        raise ValueError(
                            "pseudo accuracy is only allowed in dev profile or when "
                            "INVARLOCK_ALLOW_PSEUDO_ACCURACY=1 / "
                            "eval.allow_pseudo_accuracy=true is set"
                        )

            classification_metrics = {
                "preview": {"correct_total": int(c_prev), "total": int(n_prev)},
                "final": {"correct_total": int(c_fin), "total": int(n_fin)},
                "n_correct": int(c_fin),
                "n_total": int(n_fin),
                "counts_source": "pseudo_config" if used_pseudo_counts else "measured",
                "estimated": bool(used_pseudo_counts),
            }
            if used_pseudo_counts:
                try:
                    provenance = report.setdefault("provenance", {})
                    notes = provenance.setdefault("metric_notes", [])
                    if isinstance(notes, list):
                        notes.append("accuracy: pseudo counts from preview_n/final_n")
                except (AttributeError, KeyError, TypeError):
                    pass

            report.setdefault("metrics", {})["classification"] = classification_metrics
            if n_fin > 0:
                report["metrics"]["accuracy"] = float(c_fin / n_fin)
        except StopIteration:
            pass
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
        metric_kind_resolved, _provider_kind, metric_opts = (
            resolve_metric_and_provider_fn(
                cfg,
                model_profile,
                resolved_loss_type=resolved_loss_type,
                metric_kind_override=metric_kind,
            )
        )
        if metric_kind_resolved:
            metric_kind_normalized = normalize_metric_kind(metric_kind_resolved)
            if metric_kind_normalized is None:
                raise TypeError("metric kind could not be normalized")
            from invarlock.eval.primary_metric import compute_primary_metric_from_report

            baseline_report = (
                dict(baseline_report_data)
                if isinstance(baseline_report_data, Mapping)
                else None
            )
            primary_metric = compute_primary_metric_from_report(
                report,
                kind=metric_kind_normalized,
                baseline=baseline_report,
            )
            core_primary_metric = None
            if hasattr(core_report, "metrics") and isinstance(
                core_report.metrics, dict
            ):
                core_primary_metric = core_report.metrics.get("primary_metric")
            primary_metric = merge_primary_metric_health(
                primary_metric, core_primary_metric
            )
            report.setdefault("metrics", {})["primary_metric"] = primary_metric
            if metric_opts:
                try:
                    if "reps" in metric_opts:
                        report["metrics"]["primary_metric"]["reps"] = int(
                            metric_opts["reps"]
                        )
                    if "ci_level" in metric_opts:
                        report["metrics"]["primary_metric"]["ci_level"] = float(
                            metric_opts["ci_level"]
                        )
                except (KeyError, TypeError, ValueError):
                    pass

            try:
                primary_metric_block = report.get("metrics", {}).get(
                    "primary_metric", {}
                )
                ppl_final_v1 = float(primary_metric_block.get("final"))
                ppl_final_v2 = float(primary_metric.get("final", float("nan")))
                if math.isfinite(ppl_final_v1) and math.isfinite(ppl_final_v2):
                    if not math.isclose(
                        ppl_final_v1, ppl_final_v2, rel_tol=1e-9, abs_tol=1e-9
                    ):
                        report.setdefault("metrics", {}).setdefault(
                            "_metric_v1_mismatch", {}
                        )["ppl_final_diff"] = ppl_final_v2 - ppl_final_v1
                if debug_metric_diffs_enabled and str(
                    primary_metric.get("kind", "")
                ).startswith("ppl"):
                    debug_diffs_line = format_debug_metric_diffs(
                        primary_metric,
                        report.get("metrics", {}),
                        baseline_report,
                    )
            except (AttributeError, TypeError, ValueError):
                pass
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
