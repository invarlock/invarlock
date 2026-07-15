from __future__ import annotations

import math
from typing import Any

_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    RuntimeError,
    OSError,
)


def _report_metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics") if isinstance(report, dict) else None
    return metrics if isinstance(metrics, dict) else {}


def _as_count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value) if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value):
        if abs(value - round(value)) > 1e-9 or value < 0:
            return None
        return int(round(value))
    return None


def _count_examples(section: Any) -> int | None:
    if not isinstance(section, dict):
        return None
    records = section.get("records")
    if isinstance(records, list):
        measured = len([record for record in records if isinstance(record, dict)])
        if measured > 0:
            return measured
    example_ids = section.get("example_ids")
    if isinstance(example_ids, list) and example_ids:
        return int(len(example_ids))
    ids = section.get("window_ids")
    if isinstance(ids, list) and ids:
        return int(len(ids))
    return None


def _classification_total(
    classification_metrics: dict[str, Any],
    arm: str,
) -> int | None:
    section = (
        classification_metrics.get(arm)
        if isinstance(classification_metrics, dict)
        else None
    )
    if isinstance(section, dict):
        total = _as_count(section.get("total"))
        if total is not None:
            return total
    return None


def _populate_stats_with_counts_and_coverage(
    report: dict[str, Any],
    dataset_info: dict[str, Any],
    coverage_summary: dict[str, Any],
    ppl_analysis: dict[str, Any],
    paired_windows: int,
    paired_windows_explicit: bool,
) -> int:
    _ = paired_windows_explicit
    try:
        stats_obj = ppl_analysis.get("stats", {})
        if isinstance(stats_obj, dict):
            classification_metrics = _report_metrics(report).get("classification")
            classification_metrics = (
                classification_metrics
                if isinstance(classification_metrics, dict)
                else {}
            )

            data_cfg = report.get("data", {}) if isinstance(report, dict) else {}
            data_cfg = data_cfg if isinstance(data_cfg, dict) else {}
            windows_cfg = (
                dataset_info.get("windows", {})
                if isinstance(dataset_info, dict)
                else {}
            )
            windows_cfg = windows_cfg if isinstance(windows_cfg, dict) else {}

            req_prev = _as_count(stats_obj.get("requested_preview"))
            if req_prev is None:
                req_prev = _as_count(data_cfg.get("preview_n"))
            if req_prev is None:
                req_prev = _as_count(windows_cfg.get("preview"))

            req_fin = _as_count(stats_obj.get("requested_final"))
            if req_fin is None:
                req_fin = _as_count(data_cfg.get("final_n"))
            if req_fin is None:
                req_fin = _as_count(windows_cfg.get("final"))

            eval_windows = (
                report.get("evaluation_windows", {}) if isinstance(report, dict) else {}
            )
            eval_windows = eval_windows if isinstance(eval_windows, dict) else {}

            act_prev = _as_count(stats_obj.get("actual_preview"))
            if act_prev is None:
                act_prev = _count_examples(eval_windows.get("preview"))
            if act_prev is None:
                act_prev = _classification_total(classification_metrics, "preview")
            if act_prev is None:
                cov_prev = (
                    coverage_summary.get("preview")
                    if isinstance(coverage_summary, dict)
                    else None
                )
                if isinstance(cov_prev, dict):
                    act_prev = _as_count(cov_prev.get("used"))
            if act_prev is None:
                act_prev = req_prev

            act_fin = _as_count(stats_obj.get("actual_final"))
            if act_fin is None:
                act_fin = _count_examples(eval_windows.get("final"))
            if act_fin is None:
                act_fin = _classification_total(classification_metrics, "final")
            if act_fin is None:
                cov_fin = (
                    coverage_summary.get("final")
                    if isinstance(coverage_summary, dict)
                    else None
                )
                if isinstance(cov_fin, dict):
                    act_fin = _as_count(cov_fin.get("used"))
                elif isinstance(coverage_summary, dict):
                    act_fin = _as_count(coverage_summary.get("used"))
            if act_fin is None:
                act_fin = req_fin

            if req_prev is not None:
                stats_obj["requested_preview"] = req_prev
            if req_fin is not None:
                stats_obj["requested_final"] = req_fin
            if act_prev is not None:
                stats_obj["actual_preview"] = act_prev
            if act_fin is not None:
                stats_obj["actual_final"] = act_fin

            coverage_summary_map: dict[str, Any] = coverage_summary
            if isinstance(act_prev, int):
                preview_cov_raw = coverage_summary_map.get("preview")
                preview_cov: dict[str, Any] = (
                    dict(preview_cov_raw) if isinstance(preview_cov_raw, dict) else {}
                )
                preview_cov.setdefault("used", act_prev)
                if isinstance(req_prev, int):
                    preview_cov.setdefault("required", req_prev)
                    preview_cov.setdefault("ok", act_prev >= req_prev)
                coverage_summary_map["preview"] = preview_cov
            if isinstance(act_fin, int):
                final_cov_raw = coverage_summary_map.get("final")
                final_cov: dict[str, Any] = (
                    dict(final_cov_raw) if isinstance(final_cov_raw, dict) else {}
                )
                final_cov.setdefault("used", act_fin)
                if isinstance(req_fin, int):
                    final_cov.setdefault("required", req_fin)
                    final_cov.setdefault("ok", act_fin >= req_fin)
                coverage_summary_map["final"] = final_cov

            # Preview/final coverage does not prove baseline/subject pairing.
            # Only the identical-ID pairing path may populate paired_windows.
            if paired_windows > 0:
                stats_obj["paired_windows"] = paired_windows

            if "coverage_ok" not in stats_obj:
                if (
                    isinstance(req_prev, int)
                    and isinstance(req_fin, int)
                    and isinstance(act_prev, int)
                    and isinstance(act_fin, int)
                ):
                    stats_obj["coverage_ok"] = (act_prev >= req_prev) and (
                        act_fin >= req_fin
                    )
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    return paired_windows


__all__ = [
    "_as_count",
    "_classification_total",
    "_count_examples",
    "_populate_stats_with_counts_and_coverage",
]
