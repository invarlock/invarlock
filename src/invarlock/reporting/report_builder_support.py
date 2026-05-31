from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from invarlock.core.assurance_contract import (
    REPORT_BUILD_EVENT_CATEGORIES,
    build_assurance_section,
    report_build_has_blocking_evidence_events,
    resolve_report_runtime_provenance_declared,
)
from invarlock.core.exceptions import MetricsError
from invarlock.core.metric_kind_contract import is_ppl_metric_kind
from invarlock.core.retry import RetryDiagnostic
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


def build_telemetry_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Build a structured telemetry payload from a run report."""
    meta_in = report.get("meta", {}) if isinstance(report, dict) else {}
    metrics_in = report.get("metrics", {}) if isinstance(report, dict) else {}

    payload: dict[str, Any] = {"generated_at": datetime.now().isoformat()}

    if isinstance(meta_in, dict):
        payload["meta"] = {
            "model_id": meta_in.get("model_id"),
            "adapter": meta_in.get("adapter"),
            "device": meta_in.get("device"),
            "run_id": meta_in.get("run_id"),
            "profile": meta_in.get("profile"),
        }

    if isinstance(metrics_in, dict):
        timings = metrics_in.get("timings")
        if isinstance(timings, dict):
            payload["timings"] = timings

        guard_timings = metrics_in.get("guard_timings")
        if isinstance(guard_timings, dict):
            payload["guard_timings"] = guard_timings

        memory_snapshots = metrics_in.get("memory_snapshots")
        if isinstance(memory_snapshots, list):
            payload["memory_snapshots"] = memory_snapshots

        memory_summary: dict[str, Any] = {}
        for key in (
            "memory_mb_peak",
            "gpu_memory_mb_peak",
            "gpu_memory_reserved_mb_peak",
        ):
            value = metrics_in.get(key)
            if isinstance(value, int | float):
                memory_summary[key] = float(value)
        if memory_summary:
            payload["memory"] = memory_summary

        perf_metrics: dict[str, Any] = {}
        for key in (
            "latency_ms_per_tok",
            "throughput_tok_per_s",
            "eval_samples",
            "total_tokens",
        ):
            value = metrics_in.get(key)
            if isinstance(value, int | float):
                perf_metrics[key] = float(value)
        if perf_metrics:
            payload["performance"] = perf_metrics

    return payload


def save_telemetry_report(
    report: dict[str, Any],
    output_dir: Path,
    *,
    filename: str = "telemetry.json",
) -> Path:
    """Write telemetry JSON payload to the output directory."""
    payload = build_telemetry_payload(report)
    filename_str = str(filename)
    filename_path = Path(filename_str)
    if (
        not filename_str
        or filename_path.is_absolute()
        or filename_path.name != filename_str
    ):
        raise ValueError("telemetry filename must be a plain file name")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename_path.name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def telemetry_summary_line(evaluation_report: dict[str, Any]) -> str | None:
    telemetry = evaluation_report.get("telemetry")
    if not isinstance(telemetry, dict):
        return None
    summary = telemetry.get("summary_line")
    if isinstance(summary, str) and summary.strip():
        return summary
    return None


def telemetry_output_enabled() -> bool:
    return str(os.environ.get("INVARLOCK_TELEMETRY", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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


def ensure_report_build_evidence(report: dict[str, Any]) -> dict[str, Any]:
    section = report.setdefault("report_build", {})
    if not isinstance(section, dict):
        section = {}
        report["report_build"] = section
    for category in REPORT_BUILD_EVENT_CATEGORIES:
        events = section.get(category)
        if not isinstance(events, list):
            section[category] = []
    return section


def record_report_build_event(
    report: dict[str, Any],
    *,
    category: str,
    field: str,
    reason: str,
    source: str,
) -> None:
    if category not in REPORT_BUILD_EVENT_CATEGORIES:
        raise ValueError(f"Unknown report-build event category: {category}")
    section = ensure_report_build_evidence(report)
    events = section[category]
    events.append(
        {
            "field": str(field),
            "reason": str(reason),
            "source": str(source),
        }
    )


def report_build_has_evidence_events(report: dict[str, Any]) -> bool:
    section = report.get("report_build")
    if not isinstance(section, dict):
        return False
    for category in REPORT_BUILD_EVENT_CATEGORIES:
        events = section.get(category)
        if isinstance(events, list) and bool(events):
            return True
    return False


@dataclass
class ReportBuildContext:
    evaluation_report: dict[str, Any]

    def ensure_evidence(self) -> dict[str, Any]:
        return ensure_report_build_evidence(self.evaluation_report)

    def has_repair_or_fallback_events(self) -> bool:
        self.ensure_evidence()
        return report_build_has_blocking_evidence_events(self.evaluation_report)

    def attach_pending_assurance(self) -> dict[str, Any]:
        self.ensure_evidence()
        assurance = build_assurance_section(
            self.evaluation_report,
            fallback_fields_used=self.has_repair_or_fallback_events(),
            runtime_provenance_verified=None,
            runtime_provenance_declared=resolve_report_runtime_provenance_declared(
                self.evaluation_report
            ),
            runtime_provenance_verification_status="pending",
        )
        self.evaluation_report["assurance"] = assurance
        return assurance


class EvaluationReportBuilder:
    def __init__(self, evaluation_report: dict[str, Any]) -> None:
        self.context = ReportBuildContext(evaluation_report=evaluation_report)

    def finalize_assurance(self) -> dict[str, Any]:
        return self.context.attach_pending_assurance()


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
    report: RunReport, guard_overhead_section: dict[str, Any]
) -> str | None:
    schedule_digest = None
    try:
        final_windows_ctx = (
            report.get("evaluation_windows", {}).get("final", {})
            if isinstance(report.get("evaluation_windows"), dict)
            else {}
        )
        window_ids = final_windows_ctx.get("window_ids")
        if isinstance(window_ids, list) and window_ids:
            digest = hashlib.blake2s(digest_size=16)
            for wid in window_ids:
                try:
                    digest.update(int(wid).to_bytes(8, "little", signed=True))
                except _NON_FATAL_EXCEPTIONS:
                    digest.update(str(wid).encode("utf-8", "ignore"))
            schedule_digest = digest.hexdigest()
            guard_overhead_section["schedule_digest"] = schedule_digest
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
        pm_tail_result = {"mode": "warn", "evaluated": False, "passed": True}
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
