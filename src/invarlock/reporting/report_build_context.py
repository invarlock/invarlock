from __future__ import annotations

import hashlib
import math
from datetime import datetime
from typing import Any

from .report_primary_metric_policy import is_ppl_kind
from .report_types import RunReport

_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OverflowError,
    TypeError,
    ValueError,
)
_NUMERIC_EXCEPTIONS = (OverflowError, TypeError, ValueError)


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
        if is_ppl_kind(pm_kind):
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
