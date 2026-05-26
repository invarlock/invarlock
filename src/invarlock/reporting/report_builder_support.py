from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

from invarlock.core.exceptions import MetricsError
from invarlock.eval.primary_metric import compute_primary_metric_from_report

from .report_types import RunReport
from .utils import _coerce_int, _sanitize_seed_bundle

_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    RuntimeError,
    OSError,
)


def optional_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def append_build_diagnostic(
    diagnostics: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    severity: str = "warning",
) -> None:
    entry: dict[str, Any] = {
        "code": code,
        "message": message,
        "severity": severity,
    }
    if details:
        entry["details"] = details
    diagnostics.append(entry)


def extract_report_meta(
    report: RunReport, diagnostics: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Extract the evaluation report metadata block with a full seed bundle."""

    def _note(code: str, message: str) -> None:
        if diagnostics is not None:
            append_build_diagnostic(diagnostics, code=code, message=message)

    raw_meta_section = report.get("meta")
    meta_section: dict[str, Any] = (
        dict(raw_meta_section) if isinstance(raw_meta_section, dict) else {}
    )
    seed_value = _coerce_int(meta_section.get("seed"))
    seeds_bundle = _sanitize_seed_bundle(meta_section.get("seeds"), seed_value)
    primary_seed = (
        seeds_bundle.get("python") if isinstance(seeds_bundle, dict) else None
    )
    if primary_seed is None:
        primary_seed = 0
    model_id = optional_text(meta_section.get("model_id"))
    if model_id is None:
        _note(
            "meta.model_id_unavailable",
            "Run metadata is missing a usable model_id; evaluation report metadata leaves it null.",
        )
    adapter = optional_text(meta_section.get("adapter"))
    if adapter is None:
        _note(
            "meta.adapter_unavailable",
            "Run metadata is missing a usable adapter; evaluation report metadata leaves it null.",
        )
    device = optional_text(meta_section.get("device"))
    if device is None:
        _note(
            "meta.device_unavailable",
            "Run metadata is missing a usable device; evaluation report metadata leaves it null.",
        )
    out = {
        "model_id": model_id,
        "adapter": adapter,
        "device": device,
        "ts": meta_section.get("ts"),
        "commit": meta_section.get("commit"),
        "seed": primary_seed,
        "seeds": seeds_bundle,
    }
    for key in ("pm_acceptance_range", "pm_drift_band"):
        value = meta_section.get(key)
        if isinstance(value, dict) and value:
            out[key] = copy.deepcopy(value)
    return out


def generate_run_id(report: RunReport) -> str:
    """Generate a unique run ID from report metadata."""
    raw_meta = (
        report.get("meta") if isinstance(report, dict) else getattr(report, "meta", {})
    )
    meta = raw_meta if isinstance(raw_meta, dict) else {}
    existing = meta.get("run_id")
    if isinstance(existing, str) and existing:
        return existing
    timestamp = meta.get("ts", meta.get("start_time", ""))
    timestamp_str = str(timestamp) if timestamp is not None else ""
    model_id = optional_text(meta.get("model_id")) or ""
    commit = str(meta.get("commit", meta.get("commit_sha", "")) or "")[:16]
    base_str = f"{timestamp_str}{model_id}{commit}"
    return hashlib.sha256(base_str.encode()).hexdigest()[:16]


def _direct_baseline_metric(report: RunReport, payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    def _is_finite_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(float(value))

    def _coerce_finite_float(value: Any) -> float | None:
        if not _is_finite_number(value):
            return None
        return float(value)

    report_kind = (
        report.get("metrics", {}).get("primary_metric", {}).get("kind")
        if isinstance(report.get("metrics"), dict)
        else None
    )
    default_kind = str(report_kind or "ppl_causal")

    direct_pm = payload.get("primary_metric")
    if isinstance(direct_pm, dict) and _is_finite_number(direct_pm.get("final")):
        return copy.deepcopy(direct_pm)
    if _is_finite_number(payload.get("ppl_final")):
        preview_value_raw = payload.get("ppl_preview")
        if not _is_finite_number(preview_value_raw):
            preview_value_raw = payload.get("ppl_final")
        preview_value = _coerce_finite_float(preview_value_raw)
        final_value = _coerce_finite_float(payload.get("ppl_final"))
        if preview_value is None or final_value is None:
            return None
        return {
            "kind": default_kind,
            "preview": preview_value,
            "final": final_value,
        }

    metrics_block = payload.get("metrics")
    if not isinstance(metrics_block, dict):
        return None
    block_pm = metrics_block.get("primary_metric")
    if isinstance(block_pm, dict) and _is_finite_number(block_pm.get("final")):
        return copy.deepcopy(block_pm)
    ppl_final = metrics_block.get("ppl_final")
    if _is_finite_number(ppl_final):
        ppl_preview_raw = metrics_block.get("ppl_preview", ppl_final)
        if not _is_finite_number(ppl_preview_raw):
            ppl_preview_raw = ppl_final
        ppl_preview = _coerce_finite_float(ppl_preview_raw)
        ppl_final_value = _coerce_finite_float(ppl_final)
        if ppl_preview is None or ppl_final_value is None:
            return None
        return {
            "kind": default_kind,
            "preview": ppl_preview,
            "final": ppl_final_value,
        }
    return None


def build_baseline_reference(
    report: RunReport,
    baseline_raw: RunReport | dict[str, Any],
    baseline_normalized: dict[str, Any],
    *,
    compute_primary_metric_from_report_fn: Any = compute_primary_metric_from_report,
) -> dict[str, Any]:
    baseline_raw_map: dict[str, Any] = (
        dict(baseline_raw) if isinstance(baseline_raw, dict) else {}
    )
    baseline_pm = None
    try:
        raw_metrics = baseline_raw_map.get("metrics")
        bm = (
            raw_metrics.get("primary_metric") if isinstance(raw_metrics, dict) else None
        )
        if (
            isinstance(bm, dict)
            and bm
            and "final" in bm
            and bm.get("final") is not None
        ):
            baseline_pm = bm
    except _NON_FATAL_EXCEPTIONS as exc:
        raise MetricsError(
            code="E233",
            message=(
                "Evaluation report assembly requires a concrete baseline metric; "
                "baseline primary metric lookup failed."
            ),
            details={"error": str(exc)},
        ) from exc
    if not isinstance(baseline_pm, dict) or not baseline_pm:
        baseline_pm = _direct_baseline_metric(report, baseline_raw_map)
    if not isinstance(baseline_pm, dict) or not baseline_pm:
        baseline_pm = _direct_baseline_metric(report, baseline_normalized)
    if not isinstance(baseline_pm, dict) or not baseline_pm:
        try:
            baseline_metric_source: dict[str, Any] = baseline_normalized
            if (
                isinstance(baseline_raw_map.get("evaluation_windows"), dict)
                and _direct_baseline_metric(report, baseline_normalized) is None
            ):
                baseline_metric_source = baseline_raw_map
            baseline_pm = compute_primary_metric_from_report_fn(baseline_metric_source)
        except _NON_FATAL_EXCEPTIONS as exc:
            raise MetricsError(
                code="E234",
                message=(
                    "Evaluation report assembly requires a concrete baseline metric; "
                    "baseline primary metric could not be derived."
                ),
                details={"error": str(exc)},
            ) from exc
    baseline_final = baseline_pm.get("final") if isinstance(baseline_pm, dict) else None
    if not isinstance(baseline_final, (int, float)) or not math.isfinite(
        float(baseline_final)
    ):
        raise MetricsError(
            code="E235",
            message=(
                "Evaluation report assembly requires a concrete finite `final` value "
                "for the baseline primary metric."
            ),
            details={"baseline_primary_metric": baseline_pm},
        )
    baseline_ref: dict[str, Any] = {
        "run_id": optional_text(baseline_normalized.get("run_id")),
        "model_id": optional_text(baseline_normalized.get("model_id")),
        "adapter": optional_text(baseline_normalized.get("adapter")),
        "primary_metric": {
            "kind": baseline_pm.get("kind", "ppl_causal"),
            "final": float(baseline_final),
        },
    }
    baseline_metrics = (
        baseline_raw_map.get("metrics")
        if isinstance(baseline_raw_map.get("metrics"), dict)
        else None
    )
    if isinstance(baseline_metrics, dict):
        classification_metrics = baseline_metrics.get("classification")
        if isinstance(classification_metrics, dict) and classification_metrics:
            baseline_ref["metrics"] = {
                "classification": copy.deepcopy(classification_metrics)
            }
    baseline_tok_hash = baseline_normalized.get("tokenizer_hash")
    if isinstance(baseline_tok_hash, str) and baseline_tok_hash:
        baseline_ref["tokenizer_hash"] = baseline_tok_hash
    return baseline_ref
