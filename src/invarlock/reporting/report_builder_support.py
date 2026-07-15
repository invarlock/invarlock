from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from invarlock.core.exceptions import MetricsError
from invarlock.core.metric_kind_contract import is_ppl_metric_kind
from invarlock.core.retry import RetryDiagnostic
from invarlock.eval.guard_metric_impact import guard_metric_schedule_digest
from invarlock.eval.primary_metric import compute_primary_metric_from_report

from . import report_builder_telemetry as _report_builder_telemetry
from .report_build_context import EvaluationReportBuilder, ReportBuildContext
from .report_build_evidence import (
    ensure_report_build_evidence,
    record_report_build_event,
    report_build_has_evidence_events,
)
from .report_types import RunReport
from .utils import _coerce_int, _sanitize_seed_bundle

build_telemetry_payload = _report_builder_telemetry.build_telemetry_payload
save_telemetry_report = _report_builder_telemetry.save_telemetry_report
telemetry_summary_line = _report_builder_telemetry.telemetry_summary_line
telemetry_output_enabled = _report_builder_telemetry.telemetry_output_enabled

__all__ = [
    "EvaluationReportBuilder",
    "ReportBuildContext",
    "RetryReportValidationResult",
    "attach_schedule_digest",
    "build_artifacts_payload",
    "build_baseline_reference",
    "build_moe_section",
    "build_telemetry_payload",
    "ensure_report_build_evidence",
    "evaluate_primary_metric_tail",
    "extract_report_meta",
    "extract_telemetry",
    "record_report_build_event",
    "report_build_has_evidence_events",
    "resolve_capacity_context",
    "save_telemetry_report",
    "telemetry_output_enabled",
    "telemetry_summary_line",
    "validate_retry_evaluation_report",
]

_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    RuntimeError,
    OSError,
)
_NUMERIC_EXCEPTIONS = (OverflowError, TypeError, ValueError)
_VALIDATION_EXCEPTIONS = (
    AttributeError,
    FileNotFoundError,
    json.JSONDecodeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True)
class RetryReportValidationResult:
    status: str
    passed: bool
    validation: dict[str, Any]
    validation_gates: tuple[str, ...]
    attempt_summary: dict[str, Any]
    evaluation_report: dict[str, Any] | None = None
    telemetry_summary: str | None = None
    diagnostic: RetryDiagnostic | None = None


def validate_retry_evaluation_report(
    *,
    report: dict[str, Any],
    baseline_report_data: dict[str, Any] | None,
    baseline_path: Path | None,
    build_retry_result_summary_fn: Any,
    make_report_fn: Any | None = None,
    telemetry_output_enabled_fn: Any = telemetry_output_enabled,
    telemetry_summary_line_fn: Any = telemetry_summary_line,
) -> RetryReportValidationResult:
    try:
        baseline_report = baseline_report_data
        if baseline_report is None and baseline_path is not None:
            with baseline_path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                baseline_report = loaded

        if baseline_report is None:
            raise FileNotFoundError("Baseline report unavailable")

        report_factory: Any = make_report_fn
        if report_factory is None:
            from .report_make import make_report

            report_factory = make_report

        evaluation_report = report_factory(report, baseline_report)
        telemetry_summary = None
        if telemetry_output_enabled_fn():
            telemetry_summary = telemetry_summary_line_fn(evaluation_report)
        validation = (
            evaluation_report.get("validation", {})
            if isinstance(evaluation_report, dict)
            else {}
        )
        if not isinstance(validation, dict):
            validation = {}
        attempt_summary = build_retry_result_summary_fn(validation)
        validation_gates = tuple(attempt_summary.get("failures", []) or [])
        passed = bool(attempt_summary.get("passed"))
        return RetryReportValidationResult(
            status="passed" if passed else "failed",
            passed=passed,
            validation=validation,
            validation_gates=validation_gates,
            attempt_summary=attempt_summary,
            evaluation_report=evaluation_report,
            telemetry_summary=telemetry_summary,
            diagnostic=None,
        )
    except _VALIDATION_EXCEPTIONS as exc:
        return RetryReportValidationResult(
            status="error",
            passed=False,
            validation={},
            validation_gates=("report_error",),
            attempt_summary={
                "passed": False,
                "failures": ["report_error"],
                "validation": {},
            },
            evaluation_report=None,
            telemetry_summary=None,
            diagnostic=RetryDiagnostic(
                code="retry.validation_report_error",
                message=str(exc),
                severity="error",
                details={"validation_gates": ("report_error",)},
            ),
        )


def extract_telemetry(report: RunReport, device_name: Any) -> dict[str, Any]:
    telemetry: dict[str, Any] = {}
    metrics_section = report.get("metrics", {})
    if isinstance(metrics_section, dict):
        for key in (
            "latency_ms_per_tok",
            "memory_mb_peak",
            "gpu_memory_mb_peak",
            "gpu_memory_reserved_mb_peak",
            "throughput_tok_per_s",
        ):
            value = metrics_section.get(key)
            if isinstance(value, int | float) and math.isfinite(value):
                telemetry[key] = float(value)

        for key in ("preview_total_tokens", "final_total_tokens"):
            value = metrics_section.get(key)
            if isinstance(value, int | float) and value >= 0:
                telemetry[key] = float(value)
        for key in (
            "masked_tokens_total",
            "masked_tokens_preview",
            "masked_tokens_final",
        ):
            value = metrics_section.get(key)
            if isinstance(value, int | float) and value >= 0:
                telemetry[key] = float(value)

        edge_ctx = metrics_section.get("edge_device")
        if isinstance(edge_ctx, dict):
            telemetry["edge_device"] = edge_ctx

    if device_name:
        telemetry.setdefault("device", device_name)
    return telemetry


def build_artifacts_payload(report: RunReport) -> dict[str, Any]:
    raw_artifacts = report.get("artifacts", {})
    report_artifacts = dict(raw_artifacts) if isinstance(raw_artifacts, dict) else {}
    artifacts_payload: dict[str, Any] = {
        "events_path": report_artifacts.get("events_path", ""),
        "logs_path": report_artifacts.get("logs_path", ""),
        "checkpoint_path": report_artifacts.get("checkpoint_path", ""),
        "report_path": report_artifacts.get(
            "report_path", report_artifacts.get("logs_path", "")
        ),
        "generated_at": datetime.now().isoformat(),
    }
    masks_path = report_artifacts.get("masks_path")
    if isinstance(masks_path, str) and masks_path:
        artifacts_payload["masks_path"] = masks_path
    return artifacts_payload


def attach_schedule_digest(
    report: RunReport, guard_metric_impact_section: dict[str, Any]
) -> str | None:
    schedule_digest = None
    try:
        schedule_digest = guard_metric_schedule_digest(
            report,
            guard_metric_impact_section.get("metric_kind"),
        )
        if schedule_digest is not None:
            guard_metric_impact_section["schedule_digest"] = schedule_digest
    except _NON_FATAL_EXCEPTIONS:
        schedule_digest = None
    return schedule_digest


def build_moe_section(
    report: RunReport,
    baseline_raw: RunReport | dict[str, Any],
    baseline_normalized: RunReport | dict[str, Any],
) -> dict[str, Any]:
    moe_section: dict[str, Any] = {}
    try:
        run_moe = (
            report.get("metrics", {}).get("moe")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        base_moe = None
        if isinstance(baseline_raw, dict):
            try:
                base_moe = baseline_raw.get("moe")
            except _NON_FATAL_EXCEPTIONS:
                base_moe = None
        if (not isinstance(base_moe, dict) or not base_moe) and isinstance(
            baseline_normalized, dict
        ):
            try:
                bm = baseline_normalized.get("moe")
                if isinstance(bm, dict) and bm:
                    base_moe = bm
                else:
                    metrics = (
                        baseline_normalized.get("metrics")
                        if isinstance(baseline_normalized.get("metrics"), dict)
                        else None
                    )
                    if isinstance(metrics, dict):
                        base_moe = metrics.get("moe")
            except _NON_FATAL_EXCEPTIONS:
                pass
        if isinstance(run_moe, dict) and run_moe:
            for key in (
                "top_k",
                "capacity_factor",
                "expert_drop_rate",
                "load_balance_loss",
                "router_entropy",
            ):
                val = run_moe.get(key)
                if isinstance(val, int | float):
                    moe_section[key] = float(val)
            util = run_moe.get("utilization")
            if isinstance(util, list) and util:
                try:
                    util_vals = [float(x) for x in util]
                    moe_section["utilization_mean"] = float(
                        sum(util_vals) / max(1, len(util_vals))
                    )
                    moe_section["utilization_count"] = int(len(util_vals))
                except _NON_FATAL_EXCEPTIONS:
                    pass
            if isinstance(base_moe, dict) and base_moe:
                for key in ("load_balance_loss", "router_entropy"):
                    run_value = run_moe.get(key)
                    base_value = base_moe.get(key)
                    if isinstance(run_value, int | float) and isinstance(
                        base_value, int | float
                    ):
                        moe_section[f"delta_{key}"] = float(run_value) - float(
                            base_value
                        )
                bu = base_moe.get("utilization")
                if isinstance(util, list) and isinstance(bu, list) and util and bu:
                    try:
                        util_vals = [float(x) for x in util]
                        base_vals = [float(x) for x in bu]
                        mu = float(sum(util_vals) / len(util_vals))
                        mb = float(sum(base_vals) / len(base_vals))
                        moe_section["delta_utilization_mean"] = mu - mb
                    except _NON_FATAL_EXCEPTIONS:
                        pass
    except _NON_FATAL_EXCEPTIONS:
        moe_section = {}
    return moe_section


def resolve_capacity_context(
    window_capacity_ctx: Any, dataset_info: dict[str, Any]
) -> tuple[int | None, int | None]:
    capacity_tokens: int | None = None
    capacity_examples: int | None = None
    try:
        if isinstance(window_capacity_ctx, dict):
            token_value = window_capacity_ctx.get("total_tokens")
            if isinstance(token_value, int | float):
                capacity_tokens = int(token_value)
            examples = (
                window_capacity_ctx.get("available_unique")
                or window_capacity_ctx.get("available_nonoverlap")
                or window_capacity_ctx.get("candidate_limit")
            )
            if isinstance(examples, int | float):
                capacity_examples = int(examples)
        if capacity_examples is None:
            try:
                capacity_examples = int(
                    dataset_info.get("windows", {}).get("preview", 0)
                ) + int(dataset_info.get("windows", {}).get("final", 0))
            except _NON_FATAL_EXCEPTIONS:
                capacity_examples = None
    except _NON_FATAL_EXCEPTIONS:
        capacity_tokens = None
        capacity_examples = None
    return capacity_tokens, capacity_examples


def evaluate_primary_metric_tail(
    report: RunReport,
    baseline_normalized: RunReport | dict[str, Any],
    resolved_policy: dict[str, Any],
    evaluate_metric_tail_fn: Any,
) -> dict[str, Any]:
    pm_tail_result: dict[str, Any] = {}
    try:
        pm_kind = None
        try:
            pm_block = (
                report.get("metrics", {}).get("primary_metric")
                if isinstance(report.get("metrics"), dict)
                else None
            )
            if isinstance(pm_block, dict):
                pm_kind = pm_block.get("kind")
        except _NON_FATAL_EXCEPTIONS:
            pm_kind = None

        pm_tail_policy: dict[str, Any] = {}
        try:
            metrics_policy = (
                resolved_policy.get("metrics", {})
                if isinstance(resolved_policy, dict)
                else {}
            )
            if isinstance(metrics_policy, dict) and isinstance(
                metrics_policy.get("pm_tail"), dict
            ):
                pm_tail_policy = dict(metrics_policy.get("pm_tail") or {})
        except _NON_FATAL_EXCEPTIONS:
            pm_tail_policy = {}

        deltas: list[float] = []
        weights: list[float] = []
        if is_ppl_metric_kind(pm_kind):
            run_windows = (
                report.get("evaluation_windows", {}).get("final", {})
                if isinstance(report.get("evaluation_windows"), dict)
                else {}
            )
            base_windows = (
                baseline_normalized.get("evaluation_windows", {}).get("final", {})
                if isinstance(baseline_normalized, dict)
                else {}
            )
            run_ids = (
                run_windows.get("window_ids") if isinstance(run_windows, dict) else None
            )
            run_ll = (
                run_windows.get("logloss") if isinstance(run_windows, dict) else None
            )
            run_tc = (
                run_windows.get("token_counts")
                if isinstance(run_windows, dict)
                else None
            )
            base_ids = (
                base_windows.get("window_ids")
                if isinstance(base_windows, dict)
                else None
            )
            base_ll = (
                base_windows.get("logloss") if isinstance(base_windows, dict) else None
            )
            if (
                isinstance(run_ids, list)
                and isinstance(run_ll, list)
                and isinstance(base_ids, list)
                and isinstance(base_ll, list)
            ):
                base_map: dict[int, float] = {}
                for b_id, b_val in zip(base_ids, base_ll, strict=False):
                    if isinstance(b_id, int | float) and isinstance(b_val, int | float):
                        base_map[int(b_id)] = float(b_val)
                for idx, (r_id, r_val) in enumerate(zip(run_ids, run_ll, strict=False)):
                    if not (
                        isinstance(r_id, int | float) and isinstance(r_val, int | float)
                    ):
                        continue
                    key = int(r_id)
                    if key not in base_map:
                        continue
                    delta_value = float(r_val) - base_map[key]
                    if math.isfinite(delta_value):
                        deltas.append(float(delta_value))
                        if isinstance(run_tc, list) and idx < len(run_tc):
                            try:
                                weight_value = float(run_tc[idx])
                            except _NUMERIC_EXCEPTIONS:
                                weight_value = 0.0
                            weights.append(float(max(weight_value, 0.0)))

        pm_tail_result = evaluate_metric_tail_fn(
            deltas=deltas,
            weights=weights if (weights and len(weights) == len(deltas)) else None,
            policy=pm_tail_policy,
        )
        pm_tail_result["source"] = "paired_baseline.final"
    except _NON_FATAL_EXCEPTIONS:
        pm_tail_result = {"mode": "warn", "evaluated": False, "passed": False}
    return pm_tail_result


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

    report_kind = (
        report.get("metrics", {}).get("primary_metric", {}).get("kind")
        if isinstance(report.get("metrics"), dict)
        else None
    )
    subject_kind = str(report_kind or "").strip().lower()
    if not subject_kind:
        return None

    direct_pm = payload.get("primary_metric")
    if (
        isinstance(direct_pm, dict)
        and str(direct_pm.get("kind") or "").strip().lower() == subject_kind
        and _is_finite_number(direct_pm.get("final"))
    ):
        return copy.deepcopy(direct_pm)

    metrics_block = payload.get("metrics")
    if not isinstance(metrics_block, dict):
        return None
    block_pm = metrics_block.get("primary_metric")
    if (
        isinstance(block_pm, dict)
        and str(block_pm.get("kind") or "").strip().lower() == subject_kind
        and _is_finite_number(block_pm.get("final"))
    ):
        return copy.deepcopy(block_pm)
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
    report_pm = (
        report.get("metrics", {}).get("primary_metric")
        if isinstance(report.get("metrics"), dict)
        else None
    )
    subject_kind = (
        str(report_pm.get("kind") or "").strip().lower()
        if isinstance(report_pm, dict)
        else ""
    )
    raw_baseline_pm = baseline_raw_map.get("primary_metric")
    if not isinstance(raw_baseline_pm, dict):
        raw_metrics = baseline_raw_map.get("metrics")
        raw_baseline_pm = (
            raw_metrics.get("primary_metric") if isinstance(raw_metrics, dict) else None
        )
    declared_baseline_kind = (
        str(raw_baseline_pm.get("kind") or "").strip().lower()
        if isinstance(raw_baseline_pm, dict)
        else ""
    )
    if declared_baseline_kind and declared_baseline_kind != subject_kind:
        raise MetricsError(
            code="E236",
            message="Subject and baseline primary metrics must use the same kind.",
            details={
                "subject_kind": subject_kind,
                "baseline_kind": declared_baseline_kind,
            },
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
    baseline_kind = str(baseline_pm.get("kind") or "").strip().lower()
    if not subject_kind or baseline_kind != subject_kind:
        raise MetricsError(
            code="E236",
            message="Subject and baseline primary metrics must use the same kind.",
            details={"subject_kind": subject_kind, "baseline_kind": baseline_kind},
        )
    baseline_final_value = float(baseline_final)
    if is_ppl_metric_kind(subject_kind):
        if baseline_final_value < 1.0:
            raise MetricsError(
                code="E237",
                message="PPL baseline final must be at least 1.0.",
                details={"baseline_final": baseline_final_value},
            )
    elif subject_kind == "accuracy":
        if not 0.0 <= baseline_final_value <= 1.0:
            raise MetricsError(
                code="E237",
                message="Accuracy baseline final must be in [0, 1].",
                details={"baseline_final": baseline_final_value},
            )
    else:
        raise MetricsError(
            code="E236",
            message="Unsupported primary metric kind for baseline comparison.",
            details={"subject_kind": subject_kind},
        )
    baseline_ref: dict[str, Any] = {
        "run_id": optional_text(baseline_normalized.get("run_id")),
        "model_id": optional_text(baseline_normalized.get("model_id")),
        "adapter": optional_text(baseline_normalized.get("adapter")),
        "primary_metric": {
            "kind": baseline_kind,
            "final": baseline_final_value,
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
