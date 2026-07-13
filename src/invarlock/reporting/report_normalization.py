"""Normalization helpers for evaluation report inputs and baselines."""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, cast

from invarlock.core.metric_kind_contract import is_ppl_metric_kind

from .report_types import RunReport, validate_report

_PARSE_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _finite_float_or_none(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    try:
        result = float(value)
    except _PARSE_EXCEPTIONS:
        return None
    return result if math.isfinite(result) else None


def _is_ppl_kind(name: Any) -> bool:
    """Return True when the metric kind is a ppl-like metric."""

    return is_ppl_metric_kind(name)


def _generate_run_id(report: RunReport | dict[str, Any]) -> str:
    """Generate a stable run ID from report metadata when one is not present."""

    meta = report.get("meta", {}) if isinstance(report, dict) else {}
    if isinstance(meta, dict):
        existing = meta.get("run_id")
        if isinstance(existing, str) and existing:
            return existing
        timestamp = str(meta.get("ts", meta.get("start_time", "")))
        model_id = str(meta.get("model_id", "unknown"))
        commit = str(meta.get("commit", meta.get("commit_sha", "")))[:16]
        base_str = f"{timestamp}{model_id}{commit}"
    else:
        base_str = str(meta or report)
    return hashlib.sha256(base_str.encode()).hexdigest()[:16]


def normalize_and_validate_run_report(
    report: RunReport | dict[str, Any],
) -> RunReport:
    """Accept only the current canonical run-report structure."""

    if not validate_report(report):
        raise ValueError("Invalid canonical RunReport structure")
    return cast(RunReport, copy.deepcopy(report))


def validated_run_report_view(
    report: RunReport | dict[str, Any],
) -> RunReport:
    """Validate and return a read-only-by-contract view for report assembly.

    This internal hot-path helper deliberately avoids cloning the complete report.
    Callers must not mutate the returned object or retain nested values in outputs
    without copying them first.  Public normalization keeps its defensive-copy
    semantics through :func:`normalize_and_validate_run_report`.
    """

    if not validate_report(report):
        raise ValueError("Invalid canonical RunReport structure")
    return cast(RunReport, report)


def validate_canonical_run_report(report: object) -> RunReport:
    """Accept only the current canonical run-report contract."""

    if not validate_report(report):
        raise ValueError("Invalid canonical RunReport structure")
    return cast(RunReport, copy.deepcopy(report))


def _baseline_guard_metrics_block(
    baseline: dict[str, Any], guard_name: str
) -> dict[str, Any]:
    for guard in baseline.get("guards", []) or []:
        if str(guard.get("name", "")).lower() != guard_name:
            continue
        metrics = guard.get("metrics")
        if isinstance(metrics, dict) and metrics:
            return dict(metrics)
        return {}
    return {}


def _baseline_merged_guard_metrics(
    baseline: dict[str, Any], guard_name: str, metrics_value: Any
) -> dict[str, Any]:
    merged = dict(metrics_value) if isinstance(metrics_value, dict) else {}
    guard_metrics = _baseline_guard_metrics_block(baseline, guard_name)
    if guard_metrics:
        merged.update(guard_metrics)
    return merged


def _baseline_coerce_valid_ppl(value: Any, *, label: str) -> float:
    if not (isinstance(value, int | float) and math.isfinite(float(value))):
        raise ValueError(f"Invalid baseline {label}: expected finite numeric value.")
    out = float(value)
    if out < 1.0:
        raise ValueError(
            f"Invalid baseline {label}: expected value >= 1.0, observed {out}."
        )
    return out


def _baseline_normalize_kind(value: Any) -> str:
    try:
        return str(value or "").strip().lower()
    except _PARSE_EXCEPTIONS:
        return ""


def _baseline_derive_ppl_from_logloss_block(block: Any) -> float | None:
    if not isinstance(block, dict):
        return None
    logloss = block.get("logloss")
    if not isinstance(logloss, list) or not logloss:
        return None
    values = [float(x) for x in logloss if isinstance(x, int | float)]
    if not values:
        return None
    token_counts = block.get("token_counts")
    mean_ll: float
    if (
        isinstance(token_counts, list)
        and token_counts
        and len(token_counts) == len(values)
    ):
        try:
            numerator = 0.0
            denominator = 0.0
            for loss_value, token_value in zip(values, token_counts, strict=False):
                if not isinstance(token_value, int | float):
                    continue
                token_float = float(token_value)
                if token_float <= 0.0:
                    continue
                numerator += float(loss_value) * token_float
                denominator += token_float
            mean_ll = (
                numerator / denominator
                if denominator > 0.0
                else float(sum(values) / len(values))
            )
        except _PARSE_EXCEPTIONS:
            mean_ll = float(sum(values) / len(values))
    else:
        mean_ll = float(sum(values) / len(values))
    if not math.isfinite(mean_ll):
        return None
    ppl = math.exp(mean_ll)
    return float(ppl) if math.isfinite(ppl) else None


def _baseline_comparison_output(
    baseline: dict[str, Any],
    *,
    pm: dict[str, Any] | Any,
    pm_is_ppl: bool,
    metrics_ppl_final: float | None,
    metrics_ppl_preview: float | None,
) -> dict[str, Any]:
    evaluation_windows = baseline.get("evaluation_windows", {})
    final_windows = (
        evaluation_windows.get("final", {})
        if isinstance(evaluation_windows, dict)
        else {}
    )
    preview_windows = (
        evaluation_windows.get("preview", {})
        if isinstance(evaluation_windows, dict)
        else {}
    )
    if metrics_ppl_final is None:
        metrics_ppl_final = _baseline_derive_ppl_from_logloss_block(final_windows)
    if metrics_ppl_preview is None:
        metrics_ppl_preview = _baseline_derive_ppl_from_logloss_block(preview_windows)
    if metrics_ppl_preview is None:
        metrics_ppl_preview = metrics_ppl_final

    non_ppl_without_ppl_metrics = metrics_ppl_final is None and not pm_is_ppl

    baseline_eval_windows = {
        "final": {
            "window_ids": list(final_windows.get("window_ids", [])),
            "logloss": [
                float(x)
                for x in final_windows.get("logloss", [])
                if isinstance(x, int | float)
            ],
        }
    }
    bootstrap_info = (
        baseline["metrics"].get("bootstrap", {})
        if isinstance(baseline.get("metrics"), dict)
        else {}
    )
    window_overlap = baseline["metrics"].get("window_overlap_fraction")
    window_match = baseline["metrics"].get("window_match_fraction")

    baseline_tokenizer_hash = None
    try:
        baseline_tokenizer_hash = baseline.get("meta", {}).get(
            "tokenizer_hash"
        ) or baseline.get("data", {}).get("tokenizer_hash")
    except _PARSE_EXCEPTIONS:  # pragma: no cover
        baseline_tokenizer_hash = None

    baseline_out: dict[str, Any] = {
        "run_id": _generate_run_id(baseline),
        "model_id": baseline["meta"]["model_id"],
        "adapter": baseline["meta"].get("adapter"),
        "spectral": _baseline_merged_guard_metrics(
            baseline, "spectral", baseline["metrics"].get("spectral", {})
        ),
        "rmt": _baseline_merged_guard_metrics(
            baseline, "rmt", baseline["metrics"].get("rmt", {})
        ),
        "invariants": baseline["metrics"].get("invariants", {}),
        "moe": baseline["metrics"].get("moe", {}),
        "evaluation_windows": baseline_eval_windows,
        "bootstrap": bootstrap_info,
        "tokenizer_hash": baseline_tokenizer_hash,
    }
    if _finite_float_or_none(window_overlap) is not None:
        baseline_out["window_overlap_fraction"] = float(window_overlap)
    if _finite_float_or_none(window_match) is not None:
        baseline_out["window_match_fraction"] = float(window_match)
    if isinstance(pm, dict) and pm:
        baseline_out["primary_metric"] = copy.deepcopy(pm)

    if non_ppl_without_ppl_metrics:
        return baseline_out

    normalized_ppl_final = _baseline_coerce_valid_ppl(
        metrics_ppl_final,
        label="metrics.ppl_final",
    )
    normalized_ppl_preview = _baseline_coerce_valid_ppl(
        metrics_ppl_preview
        if metrics_ppl_preview is not None
        else normalized_ppl_final,
        label="metrics.ppl_preview",
    )
    baseline_out["ppl_final"] = normalized_ppl_final
    baseline_out["ppl_preview"] = normalized_ppl_preview
    return baseline_out


def _canonical_baseline_output(baseline: dict[str, Any]) -> dict[str, Any]:
    run_id = baseline.get("run_id")
    model_id = baseline.get("model_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Invalid canonical baseline: run_id must be non-empty.")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Invalid canonical baseline: model_id must be non-empty.")

    baseline_out = copy.deepcopy(baseline)
    baseline_out["run_id"] = run_id.strip()
    baseline_out["model_id"] = model_id.strip()
    primary_metric = baseline_out.get("primary_metric")
    if not isinstance(primary_metric, dict) or not primary_metric:
        raise ValueError(
            "Invalid canonical baseline: primary_metric must be a non-empty mapping."
        )
    pm_kind = _baseline_normalize_kind(primary_metric.get("kind"))
    pm_is_ppl = _is_ppl_kind(pm_kind)
    if not pm_is_ppl and pm_kind != "accuracy":
        raise ValueError("Invalid canonical baseline: unsupported metric kind.")
    if pm_is_ppl and "delta_vs_baseline_pp" in primary_metric:
        raise ValueError("PPL baselines cannot contain delta_vs_baseline_pp.")
    if pm_kind == "accuracy" and "ratio_vs_baseline" in primary_metric:
        raise ValueError("Accuracy baselines cannot contain ratio_vs_baseline.")
    metric_final = _finite_float_or_none(primary_metric.get("final"))
    metric_preview = _finite_float_or_none(primary_metric.get("preview"))
    ratio = primary_metric.get("ratio_vs_baseline")
    if pm_is_ppl and ratio is not None:
        ratio_value = _finite_float_or_none(ratio)
        if ratio_value is None or ratio_value <= 0.0:
            raise ValueError("PPL baseline ratio must be finite and positive.")
    if pm_kind == "accuracy":
        if metric_final is None or not 0.0 <= metric_final <= 1.0:
            raise ValueError("Accuracy baseline final must be finite in [0, 1].")
        if metric_preview is not None and not 0.0 <= metric_preview <= 1.0:
            raise ValueError("Accuracy baseline preview must be finite in [0, 1].")
        delta_pp = primary_metric.get("delta_vs_baseline_pp")
        if delta_pp is not None and _finite_float_or_none(delta_pp) is None:
            raise ValueError("Accuracy baseline delta must be finite when present.")
        baseline_out.pop("ppl_final", None)
        baseline_out.pop("ppl_preview", None)
        return baseline_out
    ppl_final = _finite_float_or_none(baseline_out.get("ppl_final"))
    ppl_preview = _finite_float_or_none(baseline_out.get("ppl_preview"))
    if ppl_final is None:
        ppl_final = metric_final
        if ppl_preview is None:
            ppl_preview = metric_preview if metric_preview is not None else metric_final
    if ppl_final is None:
        baseline_out.pop("ppl_final", None)
        baseline_out.pop("ppl_preview", None)
        raise ValueError(
            "Invalid canonical baseline: ppl metrics require finite ppl_final."
        )
    baseline_out["ppl_final"] = _baseline_coerce_valid_ppl(ppl_final, label="ppl_final")
    if ppl_preview is None:
        baseline_out.pop("ppl_preview", None)
    else:
        baseline_out["ppl_preview"] = _baseline_coerce_valid_ppl(
            ppl_preview, label="ppl_preview"
        )
    return baseline_out


def normalize_baseline(baseline: RunReport | dict[str, Any]) -> dict[str, Any]:
    """Normalize a canonical RunReport or canonical comparison baseline."""

    if not isinstance(baseline, dict):
        raise ValueError("Invalid canonical baseline or RunReport structure")
    if "schema_version" in baseline:
        raise ValueError("Versioned legacy baseline schemas are not accepted")
    run_report_sections = {
        "meta",
        "data",
        "edit",
        "guards",
        "metrics",
        "artifacts",
        "flags",
    }
    if not run_report_sections.issubset(baseline):
        return _canonical_baseline_output(cast(dict[str, Any], baseline))
    canonical = normalize_and_validate_run_report(baseline)
    baseline_dict = cast(dict[str, Any], canonical)
    metrics_blk = baseline_dict["metrics"]
    pm = metrics_blk.get("primary_metric", {})
    if not isinstance(pm, dict) or not pm:
        raise ValueError("Canonical RunReport baseline requires primary_metric.")
    pm_kind = _baseline_normalize_kind(pm.get("kind"))
    pm_is_ppl = _is_ppl_kind(pm_kind)
    if not pm_is_ppl and pm_kind != "accuracy":
        raise ValueError("Canonical RunReport baseline has unsupported metric kind.")
    if pm_is_ppl and "delta_vs_baseline_pp" in pm:
        raise ValueError("PPL baselines cannot contain delta_vs_baseline_pp.")
    if pm_kind == "accuracy" and "ratio_vs_baseline" in pm:
        raise ValueError("Accuracy baselines cannot contain ratio_vs_baseline.")
    pm_final = _finite_float_or_none(pm.get("final"))
    pm_preview = _finite_float_or_none(pm.get("preview"))
    if pm_kind == "accuracy":
        if pm_final is None or not 0.0 <= pm_final <= 1.0:
            raise ValueError("Accuracy baseline final must be finite in [0, 1].")
        if pm_preview is not None and not 0.0 <= pm_preview <= 1.0:
            raise ValueError("Accuracy baseline preview must be finite in [0, 1].")
    else:
        if pm_final is None or pm_final < 1.0:
            raise ValueError("PPL baseline final must be finite and at least 1.0.")
        if pm_preview is not None and pm_preview < 1.0:
            raise ValueError("PPL baseline preview must be finite and at least 1.0.")
    metrics_ppl_final = _finite_float_or_none(metrics_blk.get("ppl_final"))
    metrics_ppl_preview = _finite_float_or_none(metrics_blk.get("ppl_preview"))
    if metrics_ppl_final is None and pm_is_ppl:
        metrics_ppl_final = pm_final
        metrics_ppl_preview = pm_preview if pm_preview is not None else pm_final
    return _baseline_comparison_output(
        baseline_dict,
        pm=pm,
        pm_is_ppl=pm_is_ppl,
        metrics_ppl_final=metrics_ppl_final,
        metrics_ppl_preview=metrics_ppl_preview,
    )


__all__ = ["normalize_and_validate_run_report", "normalize_baseline"]
