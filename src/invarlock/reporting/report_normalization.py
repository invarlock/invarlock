"""Normalization helpers for evaluation report inputs and baselines."""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

from .normalizer import normalize_run_report
from .report_types import RunReport, validate_report

_PARSE_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _is_ppl_kind(name: Any) -> bool:
    """Return True when the metric kind is a ppl-like metric."""

    try:
        normalized = str(name or "").lower()
    except _PARSE_EXCEPTIONS:  # pragma: no cover
        normalized = ""
    return normalized in {
        "ppl",
        "perplexity",
        "ppl_causal",
        "causal_ppl",
        "ppl_mlm",
        "mlm_ppl",
        "ppl_masked",
        "ppl_seq2seq",
        "seq2seq_ppl",
    }


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
    """Normalize a run report and fail closed on invalid structure."""

    if isinstance(report, dict):
        report = normalize_run_report(report)
    if not validate_report(report):
        raise ValueError("Invalid RunReport structure")
    return report


def normalize_baseline(baseline: RunReport | dict[str, Any]) -> dict[str, Any]:
    """Normalize baseline payloads into a canonical comparison dictionary."""

    if not isinstance(baseline, dict):
        raise ValueError(
            "Baseline must be a RunReport dict or normalized baseline dict"
        )

    schema_version = baseline.get("schema_version")
    if (
        schema_version is not None
        and schema_version != "baseline-v1"
        and not ("meta" in baseline and "metrics" in baseline and "edit" in baseline)
    ):
        raise ValueError(f"Unsupported baseline schema_version: {schema_version!r}")

    def _guard_metrics_block(guard_name: str) -> dict[str, Any]:
        for guard in baseline.get("guards", []) or []:
            if str(guard.get("name", "")).lower() != guard_name:
                continue
            metrics = guard.get("metrics")
            if isinstance(metrics, dict) and metrics:
                return dict(metrics)
            return {}
        return {}

    def _merged_guard_metrics(guard_name: str, metrics_value: Any) -> dict[str, Any]:
        merged = dict(metrics_value) if isinstance(metrics_value, dict) else {}
        guard_metrics = _guard_metrics_block(guard_name)
        if guard_metrics:
            merged.update(guard_metrics)
        return merged

    def _coerce_valid_ppl(value: Any, *, label: str) -> float:
        if not (isinstance(value, int | float) and math.isfinite(float(value))):
            raise ValueError(
                f"Invalid baseline {label}: expected finite numeric value."
            )
        out = float(value)
        if out <= 0.0:
            raise ValueError(
                f"Invalid baseline {label}: expected value > 0.0, observed {out}."
            )
        return out

    def _normalize_kind(value: Any) -> str:
        try:
            return str(value or "").strip().lower()
        except _PARSE_EXCEPTIONS:
            return ""

    def _derive_ppl_from_logloss_block(block: Any) -> float | None:
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

    if baseline.get("schema_version") == "baseline-v1":
        metrics_blk = baseline.get("metrics", {}) or {}
        pm = (
            metrics_blk.get("primary_metric", {})
            if isinstance(metrics_blk, dict)
            else {}
        )
        pm_kind = _normalize_kind(pm.get("kind")) if isinstance(pm, dict) else ""
        pm_is_ppl = _is_ppl_kind(pm_kind)
        ppl_final_raw = (
            metrics_blk.get("ppl_final") if isinstance(metrics_blk, dict) else None
        )
        if (
            not (
                isinstance(ppl_final_raw, int | float)
                and math.isfinite(float(ppl_final_raw))
            )
            and pm_is_ppl
            and isinstance(pm, dict)
        ):
            final_value = pm.get("final")
            if isinstance(final_value, int | float):
                ppl_final_raw = float(final_value)
        if not (
            isinstance(ppl_final_raw, int | float)
            and math.isfinite(float(ppl_final_raw))
        ):
            evaluation_windows = baseline.get("evaluation_windows")
            if isinstance(evaluation_windows, dict):
                ppl_final_raw = _derive_ppl_from_logloss_block(
                    evaluation_windows.get("final")
                )
        if (
            not (
                isinstance(ppl_final_raw, int | float)
                and math.isfinite(float(ppl_final_raw))
            )
            and not pm_is_ppl
        ):
            evaluation_windows = baseline.get("evaluation_windows")
            has_window_payload = isinstance(evaluation_windows, dict) and bool(
                evaluation_windows.get("final") or evaluation_windows.get("preview")
            )
            if not isinstance(pm, dict) or not pm:
                if not has_window_payload:
                    raise ValueError(
                        "Invalid baseline metrics.primary_metric: expected a non-empty "
                        "primary metric block or finite ppl_final."
                    )
                return {
                    "run_id": baseline.get("meta", {}).get("commit_sha", "unknown")[
                        :16
                    ],
                    "model_id": baseline.get("meta", {}).get("model_id", "unknown"),
                    "spectral": baseline.get("spectral_base", {}),
                    "rmt": baseline.get("rmt_base", {}),
                    "invariants": baseline.get("invariants", {}),
                }
            out: dict[str, Any] = {
                "run_id": baseline.get("meta", {}).get("commit_sha", "unknown")[:16],
                "model_id": baseline.get("meta", {}).get("model_id", "unknown"),
                "spectral": baseline.get("spectral_base", {}),
                "rmt": baseline.get("rmt_base", {}),
                "invariants": baseline.get("invariants", {}),
            }
            if isinstance(pm, dict) and pm:
                out["primary_metric"] = copy.deepcopy(pm)
            return out
        ppl_final = _coerce_valid_ppl(ppl_final_raw, label="metrics.ppl_final")
        return {
            "run_id": baseline.get("meta", {}).get("commit_sha", "unknown")[:16],
            "model_id": baseline.get("meta", {}).get("model_id", "unknown"),
            "ppl_final": ppl_final,
            "spectral": baseline.get("spectral_base", {}),
            "rmt": baseline.get("rmt_base", {}),
            "invariants": baseline.get("invariants", {}),
        }

    if "meta" in baseline and "metrics" in baseline and "edit" in baseline:
        metrics_blk = baseline.get("metrics", {}) or {}
        pm = (
            metrics_blk.get("primary_metric", {})
            if isinstance(metrics_blk, dict)
            else {}
        )
        pm_kind = _normalize_kind(pm.get("kind")) if isinstance(pm, dict) else ""
        pm_is_ppl = _is_ppl_kind(pm_kind)
        ppl_final = metrics_blk.get("ppl_final")
        ppl_preview = metrics_blk.get("ppl_preview")
        if (
            not (isinstance(ppl_final, int | float) and math.isfinite(float(ppl_final)))
            and pm_is_ppl
        ):
            try:
                final_value = pm.get("final")
                preview_value = pm.get("preview", final_value)
                if isinstance(final_value, int | float):
                    ppl_final = float(final_value)
                if isinstance(preview_value, int | float):
                    ppl_preview = float(preview_value)
            except _PARSE_EXCEPTIONS:  # pragma: no cover
                pass

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
        if ppl_final is None:
            ppl_final = _derive_ppl_from_logloss_block(final_windows)
        if ppl_preview is None:
            ppl_preview = _derive_ppl_from_logloss_block(preview_windows)
        if ppl_preview is None:
            ppl_preview = ppl_final

        non_ppl_without_ppl_metrics = (
            not (isinstance(ppl_final, int | float) and math.isfinite(float(ppl_final)))
            and not pm_is_ppl
        )

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
        window_overlap = baseline["metrics"].get(
            "window_overlap_fraction", float("nan")
        )
        window_match = baseline["metrics"].get("window_match_fraction", float("nan"))

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
            "spectral": _merged_guard_metrics(
                "spectral", baseline["metrics"].get("spectral", {})
            ),
            "rmt": _merged_guard_metrics("rmt", baseline["metrics"].get("rmt", {})),
            "invariants": baseline["metrics"].get("invariants", {}),
            "moe": baseline["metrics"].get("moe", {}),
            "evaluation_windows": baseline_eval_windows,
            "bootstrap": bootstrap_info,
            "window_overlap_fraction": window_overlap,
            "window_match_fraction": window_match,
            "tokenizer_hash": baseline_tokenizer_hash,
        }
        if isinstance(pm, dict) and pm:
            baseline_out["primary_metric"] = copy.deepcopy(pm)

        if non_ppl_without_ppl_metrics:
            return baseline_out

        ppl_final = _coerce_valid_ppl(ppl_final, label="metrics.ppl_final")
        ppl_preview = _coerce_valid_ppl(
            ppl_preview if ppl_preview is not None else ppl_final,
            label="metrics.ppl_preview",
        )
        baseline_out["ppl_final"] = ppl_final
        baseline_out["ppl_preview"] = ppl_preview
        return baseline_out

    baseline_out = baseline.copy()
    metrics_blk = baseline_out.get("metrics", {})
    pm_kind = ""
    if isinstance(metrics_blk, dict):
        pm_metrics = metrics_blk.get("primary_metric")
        if isinstance(pm_metrics, dict):
            pm_kind = _normalize_kind(pm_metrics.get("kind"))
    if not pm_kind:
        pm_top = baseline_out.get("primary_metric", {})
        if isinstance(pm_top, dict):
            pm_kind = _normalize_kind(pm_top.get("kind"))
    pm_is_ppl = _is_ppl_kind(pm_kind)

    ppl_final = baseline_out.get("ppl_final")
    ppl_preview = baseline_out.get("ppl_preview")

    if not (isinstance(ppl_final, int | float) and math.isfinite(float(ppl_final))):
        if isinstance(metrics_blk, dict):
            direct_final = metrics_blk.get("ppl_final")
            direct_preview = metrics_blk.get("ppl_preview", direct_final)
            if isinstance(direct_final, int | float) and math.isfinite(
                float(direct_final)
            ):
                ppl_final = float(direct_final)
            if isinstance(direct_preview, int | float) and math.isfinite(
                float(direct_preview)
            ):
                ppl_preview = float(direct_preview)
            if (
                not (
                    isinstance(ppl_final, int | float)
                    and math.isfinite(float(ppl_final))
                )
                and pm_is_ppl
            ):
                pm = metrics_blk.get("primary_metric", {})
                if isinstance(pm, dict):
                    final_value = pm.get("final")
                    preview_value = pm.get("preview", final_value)
                    if isinstance(final_value, int | float):
                        ppl_final = float(final_value)
                    if isinstance(preview_value, int | float):
                        ppl_preview = float(preview_value)
        if (
            not (isinstance(ppl_final, int | float) and math.isfinite(float(ppl_final)))
            and pm_is_ppl
        ):
            pm_top = baseline_out.get("primary_metric", {})
            if isinstance(pm_top, dict):
                final_value = pm_top.get("final")
                preview_value = pm_top.get("preview", final_value)
                if isinstance(final_value, int | float):
                    ppl_final = float(final_value)
                if isinstance(preview_value, int | float):
                    ppl_preview = float(preview_value)
        if not (isinstance(ppl_final, int | float) and math.isfinite(float(ppl_final))):
            evaluation_windows = baseline_out.get("evaluation_windows")
            if isinstance(evaluation_windows, dict):
                ppl_final = _derive_ppl_from_logloss_block(
                    evaluation_windows.get("final")
                )
                if ppl_preview is None:
                    ppl_preview = _derive_ppl_from_logloss_block(
                        evaluation_windows.get("preview")
                    )

    if (
        not (isinstance(ppl_final, int | float) and math.isfinite(float(ppl_final)))
        and not pm_is_ppl
    ):
        if "ppl_final" in baseline_out:
            try:
                ppl_final_float = float(baseline_out.get("ppl_final"))
            except _PARSE_EXCEPTIONS:
                baseline_out.pop("ppl_final", None)
            else:
                if not math.isfinite(ppl_final_float) or ppl_final_float <= 0.0:
                    baseline_out.pop("ppl_final", None)
        if "ppl_preview" in baseline_out:
            try:
                ppl_preview_float = float(baseline_out.get("ppl_preview"))
            except _PARSE_EXCEPTIONS:
                baseline_out.pop("ppl_preview", None)
            else:
                if not math.isfinite(ppl_preview_float) or ppl_preview_float <= 0.0:
                    baseline_out.pop("ppl_preview", None)
        return baseline_out

    ppl_final = _coerce_valid_ppl(ppl_final, label="ppl_final")
    if ppl_preview is None:
        ppl_preview = ppl_final
    else:
        ppl_preview = _coerce_valid_ppl(ppl_preview, label="ppl_preview")

    baseline_out["ppl_final"] = ppl_final
    if "ppl_preview" in baseline_out or not math.isclose(
        float(ppl_preview), float(ppl_final), rel_tol=0.0, abs_tol=0.0
    ):
        baseline_out["ppl_preview"] = ppl_preview
    return baseline_out


__all__ = ["normalize_and_validate_run_report", "normalize_baseline"]
