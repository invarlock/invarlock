from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from invarlock.reporting.run_pairing_contract import (
    RunReportPolicyViolation,
    build_dataset_window_stats,
    validate_pairing_report_metrics,
)

ResolveMetricAndProviderFn = Callable[..., tuple[str | None, Any, dict[str, Any]]]
MergePrimaryMetricHealthFn = Callable[[dict[str, Any], Any], dict[str, Any]]
FormatDebugMetricDiffsFn = Callable[
    [dict[str, Any], Mapping[str, Any] | None, Mapping[str, Any] | None], str
]


@dataclass(frozen=True)
class RunReportMetricsEnrichmentResult:
    report: dict[str, Any]
    pairing_violations: tuple[RunReportPolicyViolation, ...]
    debug_diffs_line: str
    match_fraction: float | None
    overlap_fraction: float | None


def _classification_records(arm_payload: Any) -> list[dict[str, Any]]:
    if not isinstance(arm_payload, Mapping):
        return []
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
    resolve_metric_and_provider_fn: ResolveMetricAndProviderFn,
    merge_primary_metric_health_fn: MergePrimaryMetricHealthFn,
    format_debug_metric_diffs_fn: FormatDebugMetricDiffsFn,
) -> RunReportMetricsEnrichmentResult:
    metrics_section = report.get("metrics", {}) or {}
    data_section = report.get("data", {}) or {}
    preview_count_report = data_section.get("preview_n")
    final_count_report = data_section.get("final_n")
    match_fraction = metrics_section.get("window_match_fraction")
    overlap_fraction = metrics_section.get("window_overlap_fraction")

    if _loss_type_from_context(run_config) == "classification":
        try:
            from invarlock.eval.primary_metric import compute_accuracy_counts

            evaluation_windows = {}
            try:
                if hasattr(core_report, "evaluation_windows") and isinstance(
                    core_report.evaluation_windows, dict
                ):
                    evaluation_windows = core_report.evaluation_windows  # type: ignore[assignment]
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

            classification_metrics = {
                "preview": {"correct_total": int(c_prev), "total": int(n_prev)},
                "final": {"correct_total": int(c_fin), "total": int(n_fin)},
                "counts_source": "pseudo_config" if used_pseudo_counts else "measured",
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
        except (
            AttributeError,
            ImportError,
            ModuleNotFoundError,
            RuntimeError,
            TypeError,
            ValueError,
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
            from invarlock.eval.primary_metric import compute_primary_metric_from_report

            primary_metric = compute_primary_metric_from_report(
                report,
                kind=metric_kind_resolved,
                baseline=baseline_report_data,
            )
            core_primary_metric = None
            if hasattr(core_report, "metrics") and isinstance(
                core_report.metrics, dict
            ):
                core_primary_metric = core_report.metrics.get("primary_metric")
            primary_metric = merge_primary_metric_health_fn(
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
                    debug_diffs_line = format_debug_metric_diffs_fn(
                        primary_metric,
                        report.get("metrics", {}),
                        baseline_report_data,
                    )
            except (AttributeError, TypeError, ValueError):
                pass
    except (
        AttributeError,
        ImportError,
        ModuleNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
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
