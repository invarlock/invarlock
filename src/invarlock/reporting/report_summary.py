from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .report_console import compute_console_validation_block


@dataclass(frozen=True)
class SafetyDashboardRow:
    label: str
    status: str
    summary: str


@dataclass(frozen=True)
class SafetyDashboardSummary:
    overall_pass: bool
    overall_status: str
    rows: tuple[SafetyDashboardRow, ...]


@dataclass(frozen=True)
class QualityGateRow:
    label: str
    status: str
    measured: str
    threshold: str
    basis: str
    description: str


@dataclass(frozen=True)
class QualityGatesSummary:
    overall_pass: bool
    overall_status: str
    rows: tuple[QualityGateRow, ...]
    hysteresis_applied: bool


@dataclass(frozen=True)
class ReportManifestSummary:
    run_model: str | None
    device: str | None
    seed: Any | None
    overall_status: str
    primary_metric_ratio: float | None
    gates_passed: int
    gates_total: int


def _format_gate_status(
    validation: dict[str, Any] | None,
    key: str,
    ok_default: bool | None = None,
) -> str:
    if not isinstance(validation, dict):
        ok = ok_default
    elif key not in validation:
        ok = ok_default
    else:
        ok = bool(validation.get(key))
    if ok is None:
        return "ℹ️ N/A"
    return "✅ PASS" if ok else "❌ FAIL"


def build_safety_dashboard_summary(
    evaluation_report: dict[str, Any],
) -> SafetyDashboardSummary:
    """Build the executive dashboard rows for an evaluation report."""
    block = compute_console_validation_block(evaluation_report)
    overall_pass = bool(block.get("overall_pass"))
    overall_status = (
        f"{'✅' if overall_pass else '❌'} {'PASS' if overall_pass else 'FAIL'}"
    )

    validation = evaluation_report.get("validation", {}) or {}
    pm = evaluation_report.get("primary_metric", {}) or {}
    auto = evaluation_report.get("auto", {}) or {}
    tier = str(auto.get("tier") or "balanced").lower()

    pm_kind = str(pm.get("kind", "")).lower()
    pm_basis = pm.get("gating_basis") or pm.get("basis") or "point"
    if isinstance(validation, dict) and "primary_metric_acceptable" in validation:
        pm_ok: bool | None = bool(validation.get("primary_metric_acceptable"))
    else:
        pm_ok = None
    pm_value = pm.get("ratio_vs_baseline")

    if pm_kind in {"accuracy", "vqa_accuracy"}:
        measured = f"{pm_value:+.2f} pp" if isinstance(pm_value, int | float) else "N/A"
        th_map = {
            "conservative": -0.5,
            "balanced": -1.0,
            "aggressive": -2.0,
            "none": -1.0,
        }
        th = th_map.get(tier, -1.0)
        threshold = f"≥ {th:+.2f} pp ({pm_basis})"
    else:
        measured = f"{pm_value:.3f}×" if isinstance(pm_value, int | float) else "N/A"
        tier_thresholds = {
            "conservative": 1.05,
            "balanced": 1.10,
            "aggressive": 1.20,
            "none": 1.10,
        }
        ratio_limit = tier_thresholds.get(tier, 1.10)
        target_ratio = auto.get("target_pm_ratio")
        if isinstance(target_ratio, int | float) and target_ratio > 0:
            ratio_limit = min(ratio_limit, float(target_ratio))
        threshold = f"≤ {ratio_limit:.2f}× ({pm_basis})"

    if isinstance(pm_ok, bool):
        pm_status = f"{'✅' if pm_ok else '❌'} {measured}"
    else:
        pm_status = f"ℹ️ {measured}"

    if isinstance(validation, dict) and "preview_final_drift_acceptable" in validation:
        drift_ok: bool | None = bool(validation.get("preview_final_drift_acceptable"))
    else:
        drift_ok = None
    drift_val = "N/A"
    try:
        pv = (
            float(pm.get("preview"))
            if isinstance(pm.get("preview"), int | float)
            else float("nan")
        )
        fv = (
            float(pm.get("final"))
            if isinstance(pm.get("final"), int | float)
            else float("nan")
        )
        drift = (
            fv / pv
            if (math.isfinite(pv) and pv > 0 and math.isfinite(fv))
            else float("nan")
        )
        if math.isfinite(drift):
            drift_val = f"{drift:.3f}×"
    except Exception:
        drift_val = "N/A"
    if isinstance(drift_ok, bool):
        drift_status = f"{'✅' if drift_ok else '❌'} {drift_val}"
    else:
        drift_status = f"ℹ️ {drift_val}"

    rows: list[SafetyDashboardRow] = [
        SafetyDashboardRow("Primary Metric", pm_status, threshold),
        SafetyDashboardRow("Drift", drift_status, "0.95–1.05× band"),
        SafetyDashboardRow(
            "Invariants",
            _format_gate_status(validation, "invariants_pass"),
            "Model integrity checks",
        ),
        SafetyDashboardRow(
            "Spectral",
            _format_gate_status(validation, "spectral_stable"),
            "Weight matrix spectral norms",
        ),
        SafetyDashboardRow(
            "RMT",
            _format_gate_status(validation, "rmt_stable"),
            "Random Matrix Theory guard",
        ),
    ]

    overhead_ctx = evaluation_report.get("guard_overhead", {}) or {}
    overhead_evaluated = (
        bool(overhead_ctx.get("evaluated")) if isinstance(overhead_ctx, dict) else False
    )
    if overhead_evaluated:
        overhead_pct = overhead_ctx.get("overhead_percent")
        overhead_ratio = overhead_ctx.get("overhead_ratio")
        if isinstance(overhead_pct, int | float) and math.isfinite(float(overhead_pct)):
            overhead_measured = f"{float(overhead_pct):+.2f}%"
        elif isinstance(overhead_ratio, int | float) and math.isfinite(
            float(overhead_ratio)
        ):
            overhead_measured = f"{float(overhead_ratio):.3f}×"
        else:
            overhead_measured = "N/A"
        threshold_pct = overhead_ctx.get("threshold_percent")
        if isinstance(threshold_pct, int | float) and math.isfinite(
            float(threshold_pct)
        ):
            threshold_str = f"≤ +{float(threshold_pct):.1f}%"
        else:
            threshold_str = "≤ +1.0%"
        rows.append(
            SafetyDashboardRow(
                "Overhead",
                (
                    f"{'✅' if bool(validation.get('guard_overhead_acceptable', True)) else '❌'} {overhead_measured}"
                    if isinstance(validation, dict)
                    else f"ℹ️ {overhead_measured}"
                ),
                threshold_str,
            )
        )

    return SafetyDashboardSummary(
        overall_pass=overall_pass,
        overall_status=overall_status,
        rows=tuple(rows),
    )


def build_quality_gates_summary(
    evaluation_report: dict[str, Any],
) -> QualityGatesSummary:
    """Build the canonical quality-gates view model for Markdown rendering."""
    block = compute_console_validation_block(evaluation_report)
    overall_pass = bool(block.get("overall_pass"))
    overall_status = (
        f"{'✅' if overall_pass else '❌'} {'PASS' if overall_pass else 'FAIL'}"
    )

    pm_block = evaluation_report.get("primary_metric", {}) or {}
    has_pm = isinstance(pm_block, dict) and bool(pm_block)
    auto_info = evaluation_report.get("auto", {})
    tier = (auto_info.get("tier") or "balanced").lower()
    validation = evaluation_report.get("validation", {}) or {}

    rows: list[QualityGateRow] = []
    if has_pm:
        pm_kind = str(pm_block.get("kind", "")).lower()
        value = pm_block.get("ratio_vs_baseline")
        gating_basis = pm_block.get("gating_basis") or "point"
        pm_ok = bool(validation.get("primary_metric_acceptable", True))
        status = "✅ PASS" if pm_ok else "❌ FAIL"
        if pm_kind in {"accuracy", "vqa_accuracy"}:
            measured = f"{value:+.2f} pp" if isinstance(value, int | float) else "N/A"
            th_map = {
                "conservative": -0.5,
                "balanced": -1.0,
                "aggressive": -2.0,
                "none": -1.0,
            }
            th = th_map.get(tier, -1.0)
            threshold = f"≥ {th:+.2f} pp"
            description = "Δ accuracy vs baseline"
        else:
            tier_thresholds = {
                "conservative": 1.05,
                "balanced": 1.10,
                "aggressive": 1.20,
                "none": 1.10,
            }
            ratio_limit = tier_thresholds.get(tier, 1.10)
            target_ratio = auto_info.get("target_pm_ratio")
            if isinstance(target_ratio, int | float) and target_ratio > 0:
                ratio_limit = min(ratio_limit, float(target_ratio))
            measured = f"{value:.3f}x" if isinstance(value, int | float) else "N/A"
            threshold = f"≤ {ratio_limit:.2f}x"
            description = "Ratio vs baseline"
        rows.append(
            QualityGateRow(
                label="Primary Metric Acceptable",
                status=status,
                measured=measured,
                threshold=threshold,
                basis=str(gating_basis),
                description=description,
            )
        )

        drift_ok = bool(validation.get("preview_final_drift_acceptable", True))
        drift_min = 0.95
        drift_max = 1.05
        try:
            drift_band = (
                pm_block.get("drift_band") if isinstance(pm_block, dict) else None
            )
            if isinstance(drift_band, dict):
                lo = drift_band.get("min")
                hi = drift_band.get("max")
                if isinstance(lo, int | float) and isinstance(hi, int | float):
                    lo_f = float(lo)
                    hi_f = float(hi)
                    if math.isfinite(lo_f) and math.isfinite(hi_f) and 0 < lo_f < hi_f:
                        drift_min = lo_f
                        drift_max = hi_f
            elif isinstance(drift_band, list | tuple) and len(drift_band) == 2:
                lo_raw, hi_raw = drift_band[0], drift_band[1]
                if isinstance(lo_raw, int | float) and isinstance(hi_raw, int | float):
                    lo_f = float(lo_raw)
                    hi_f = float(hi_raw)
                    if math.isfinite(lo_f) and math.isfinite(hi_f) and 0 < lo_f < hi_f:
                        drift_min = lo_f
                        drift_max = hi_f
        except Exception:
            pass
        try:
            pv = (
                float(pm_block.get("preview"))
                if isinstance(pm_block.get("preview"), int | float)
                else float("nan")
            )
            fv = (
                float(pm_block.get("final"))
                if isinstance(pm_block.get("final"), int | float)
                else float("nan")
            )
            drift = (
                fv / pv
                if (math.isfinite(pv) and pv > 0 and math.isfinite(fv))
                else float("nan")
            )
        except Exception:
            drift = float("nan")
        measured = f"{drift:.3f}x" if math.isfinite(drift) else "N/A"
        rows.append(
            QualityGateRow(
                label="Preview Final Drift Acceptable",
                status="✅ PASS" if drift_ok else "❌ FAIL",
                measured=measured,
                threshold=f"{drift_min:.2f}–{drift_max:.2f}x",
                basis="point",
                description="Final/Preview ratio stability",
            )
        )

        guard_overhead = evaluation_report.get("guard_overhead", {}) or {}
        evaluated = bool(guard_overhead.get("evaluated"))
        if evaluated:
            overhead_ok = bool(validation.get("guard_overhead_acceptable", True))
            overhead_pct = guard_overhead.get("overhead_percent")
            overhead_ratio = guard_overhead.get("overhead_ratio")
            if isinstance(overhead_pct, int | float) and math.isfinite(
                float(overhead_pct)
            ):
                measured = f"{float(overhead_pct):+.2f}%"
            elif isinstance(overhead_ratio, int | float) and math.isfinite(
                float(overhead_ratio)
            ):
                measured = f"{float(overhead_ratio):.3f}x"
            else:
                measured = "N/A"
            threshold_pct = guard_overhead.get("threshold_percent")
            if not (
                isinstance(threshold_pct, int | float)
                and math.isfinite(float(threshold_pct))
            ):
                threshold_val = guard_overhead.get("overhead_threshold", 0.01)
                try:
                    threshold_pct = float(threshold_val) * 100.0
                except Exception:
                    threshold_pct = 1.0
            rows.append(
                QualityGateRow(
                    label="Guard Overhead Acceptable",
                    status="✅ PASS" if overhead_ok else "❌ FAIL",
                    measured=measured,
                    threshold=f"≤ +{float(threshold_pct):.1f}%",
                    basis="point",
                    description="Guarded vs bare PM overhead",
                )
            )

        pm_tail = evaluation_report.get("primary_metric_tail", {}) or {}
        if isinstance(pm_tail, dict) and pm_tail:
            evaluated = bool(pm_tail.get("evaluated", False))
            mode = str(pm_tail.get("mode", "warn") or "warn").strip().lower()
            passed = bool(pm_tail.get("passed", True))
            warned = bool(pm_tail.get("warned", False))

            if not evaluated:
                status = "ℹ️ INFO"
            elif passed:
                status = "✅ PASS"
            elif mode == "fail":
                status = "❌ FAIL"
            else:
                status = "⚠️ WARN" if warned else "⚠️ WARN"

            policy = (
                pm_tail.get("policy", {})
                if isinstance(pm_tail.get("policy"), dict)
                else {}
            )
            stats = (
                pm_tail.get("stats", {})
                if isinstance(pm_tail.get("stats"), dict)
                else {}
            )

            q = policy.get("quantile", 0.95)
            try:
                qf = float(q)
            except Exception:
                qf = 0.95
            qf = max(0.0, min(1.0, qf))
            q_key = f"q{int(round(100.0 * qf))}"
            q_name = f"P{int(round(100.0 * qf))}"
            q_val = stats.get(q_key)
            mass_val = stats.get("tail_mass")
            eps = policy.get("epsilon", stats.get("epsilon"))

            measured_parts: list[str] = []
            if isinstance(q_val, int | float) and math.isfinite(float(q_val)):
                measured_parts.append(f"{q_name}={float(q_val):.3f}")
            if isinstance(mass_val, int | float) and math.isfinite(float(mass_val)):
                measured_parts.append(f"mass={float(mass_val):.3f}")
            measured = ", ".join(measured_parts) if measured_parts else "N/A"

            thr_parts: list[str] = []
            qmax = policy.get("quantile_max")
            if isinstance(qmax, int | float) and math.isfinite(float(qmax)):
                thr_parts.append(f"{q_name}≤{float(qmax):.3f}")
            mmax = policy.get("mass_max")
            if isinstance(mmax, int | float) and math.isfinite(float(mmax)):
                thr_parts.append(f"mass≤{float(mmax):.3f}")
            if isinstance(eps, int | float) and math.isfinite(float(eps)):
                thr_parts.append(f"ε={float(eps):.1e}")
            threshold = "; ".join(thr_parts) if thr_parts else "policy"

            rows.append(
                QualityGateRow(
                    label="Primary Metric Tail",
                    status=status,
                    measured=measured,
                    threshold=threshold,
                    basis=q_name.lower(),
                    description="Tail regression vs baseline (ΔlogNLL)",
                )
            )

    return QualityGatesSummary(
        overall_pass=overall_pass,
        overall_status=overall_status,
        rows=tuple(rows),
        hysteresis_applied=bool(validation.get("hysteresis_applied")),
    )


def build_report_manifest_summary(
    run_report: dict[str, Any],
    evaluation_report: dict[str, Any],
) -> ReportManifestSummary:
    """Build the manifest summary payload for evaluation bundle persistence."""
    meta_obj: object = run_report.get("meta")
    meta_dict: dict[str, Any] = meta_obj if isinstance(meta_obj, dict) else {}

    block = compute_console_validation_block(evaluation_report)
    rows = block.get("rows", []) or []
    gates_total = len(rows)
    gates_passed = sum(
        1 for row in rows if isinstance(row, dict) and bool(row.get("ok"))
    )
    overall_status = "PASS" if block.get("overall_pass") else "FAIL"

    primary_metric_ratio = None
    pm = evaluation_report.get("primary_metric", {}) or {}
    if isinstance(pm, dict):
        ratio = pm.get("ratio_vs_baseline")
        if isinstance(ratio, int | float):
            primary_metric_ratio = float(ratio)

    return ReportManifestSummary(
        run_model=meta_dict.get("model_id"),
        device=meta_dict.get("device"),
        seed=meta_dict.get("seed"),
        overall_status=overall_status,
        primary_metric_ratio=primary_metric_ratio,
        gates_passed=gates_passed,
        gates_total=gates_total,
    )


__all__ = [
    "QualityGateRow",
    "QualityGatesSummary",
    "SafetyDashboardRow",
    "SafetyDashboardSummary",
    "ReportManifestSummary",
    "build_quality_gates_summary",
    "build_safety_dashboard_summary",
    "build_report_manifest_summary",
]
