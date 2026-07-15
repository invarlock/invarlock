from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from invarlock.core.auto_tuning import get_tier_policies
from invarlock.json_serialization import normalize_optional_nonfinite_json
from invarlock.public_contracts import load_json_contract

_PARSE_EXCEPTIONS = (AttributeError, KeyError, OverflowError, TypeError, ValueError)
_CONSOLE_LABELS_DEFAULT = [
    "Primary Metric Acceptable",
    "Preview Final Drift Acceptable",
    "Guard Metric Impact Acceptable",
    "Invariants Pass",
    "Spectral Stable",
    "Rmt Stable",
]


def _finite_float_or_nan(value: Any) -> float:
    if not isinstance(value, int | float):
        return float("nan")
    try:
        result = float(value)
    except _PARSE_EXCEPTIONS:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def load_console_labels() -> list[str]:
    """Load console labels allow-list from contracts with a safe fallback."""
    try:
        data = load_json_contract("console_labels.json")
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return list(data)
    except (OSError, TypeError, ValueError):
        pass
    return list(_CONSOLE_LABELS_DEFAULT)


def compute_console_validation_block(
    evaluation_report: dict[str, Any],
) -> dict[str, Any]:
    """Produce a normalized console validation block from an evaluation report."""
    labels = load_console_labels()
    validation_raw = evaluation_report.get("validation")
    validation = validation_raw if isinstance(validation_raw, dict) else {}
    guard_ctx = evaluation_report.get("guard_metric_impact", {}) or {}
    guard_evaluated = (
        bool(guard_ctx.get("evaluated")) if isinstance(guard_ctx, dict) else False
    )

    def _to_key(label: str) -> str:
        return label.strip().lower().replace(" ", "_")

    rows: list[dict[str, Any]] = []
    ok_map: dict[str, bool] = {}
    effective_labels: list[str] = []
    for label in labels:
        key = _to_key(label)
        ok = validation.get(key) is True
        if key == "guard_metric_impact_acceptable" and not guard_evaluated:
            continue
        rows.append(
            {
                "label": label,
                "status": "✅ PASS" if ok else "❌ FAIL",
                "evaluated": key != "guard_metric_impact_acceptable" or guard_evaluated,
                "ok": ok,
            }
        )
        effective_labels.append(label)
        ok_map[key] = ok

    keys_for_overall = [
        "primary_metric_acceptable",
        "preview_final_drift_acceptable",
        "invariants_pass",
        "spectral_stable",
        "rmt_stable",
    ]
    if guard_evaluated:
        keys_for_overall.append("guard_metric_impact_acceptable")

    overall_pass = all(ok_map.get(key, False) for key in keys_for_overall)
    return {"labels": effective_labels, "rows": rows, "overall_pass": overall_pass}


def compute_report_hash(evaluation_report: dict[str, Any]) -> str:
    """Compute a stable integrity hash for an evaluation report."""
    cert_copy = dict(evaluation_report or {})
    cert_copy.pop("artifacts", None)
    cert_str = json.dumps(
        normalize_optional_nonfinite_json(cert_copy),
        sort_keys=True,
        allow_nan=False,
    )
    import hashlib as _hash

    return _hash.sha256(cert_str.encode()).hexdigest()[:16]


def build_console_summary_pack(evaluation_report: dict[str, Any]) -> dict[str, Any]:
    """Build a reusable console summary from an evaluation report."""
    block = compute_console_validation_block(evaluation_report)
    overall_pass = bool(block.get("overall_pass"))
    emoji = "✅" if overall_pass else "❌"
    overall_line = f"Overall Status: {emoji} {'PASS' if overall_pass else 'FAIL'}"

    gate_lines: list[str] = []
    for row in block.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        label = row.get("label", "Gate")
        status = row.get("status", "")
        gate_lines.append(f"{label}: {status}")

    return {
        "overall_pass": overall_pass,
        "overall_line": overall_line,
        "gate_lines": gate_lines,
        "labels": block.get("labels", []),
    }


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


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_finite_float(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    try:
        numeric = float(value)
    except _PARSE_EXCEPTIONS:
        return None
    return numeric if math.isfinite(numeric) else None


def _guard_metric_impact_display(payload: dict[str, Any]) -> tuple[str, str]:
    """Format canonical metric degradation and its policy limit."""

    display_value = _coerce_finite_float(payload.get("display_value"))
    degradation_limit = _coerce_finite_float(payload.get("degradation_limit"))
    display_unit = payload.get("display_unit")
    if display_value is None:
        measured = "N/A"
    elif display_unit == "percentage_points":
        measured = f"{display_value:+.2f} pp"
    elif display_unit == "percent":
        measured = f"{display_value:+.2f}%"
    else:
        measured = "N/A"

    if degradation_limit is None:
        threshold = "N/A"
    elif display_unit == "percentage_points":
        threshold = f"≤ +{degradation_limit * 100.0:.1f} pp"
    elif display_unit == "percent":
        threshold = f"≤ +{degradation_limit * 100.0:.1f}%"
    else:
        threshold = "N/A"
    return measured, threshold


def _resolved_metrics_policy(evaluation_report: dict[str, Any]) -> dict[str, Any]:
    auto = _mapping(evaluation_report.get("auto"))
    tier = str(auto.get("tier") or "balanced").lower()
    resolved = _mapping(evaluation_report.get("resolved_policy"))
    metrics = _mapping(resolved.get("metrics"))
    if metrics:
        return metrics
    defaults = get_tier_policies().get(tier, {})
    return _mapping(_mapping(defaults).get("metrics"))


def _pm_acceptance_range(evaluation_report: dict[str, Any]) -> dict[str, float]:
    meta = _mapping(evaluation_report.get("meta"))
    raw = _mapping(meta.get("pm_acceptance_range"))
    lo = _coerce_finite_float(raw.get("min"))
    hi = _coerce_finite_float(raw.get("max"))
    if lo is None and hi is None:
        return {}
    out: dict[str, float] = {}
    if lo is not None:
        out["min"] = lo
    if hi is not None:
        out["max"] = hi
    return out


def _primary_metric_measured_and_threshold(
    evaluation_report: dict[str, Any],
) -> tuple[str, str, str]:
    pm = _mapping(evaluation_report.get("primary_metric"))
    auto = _mapping(evaluation_report.get("auto"))
    tiny_relax = bool(auto.get("tiny_relax"))
    tier = "aggressive" if tiny_relax else str(auto.get("tier") or "balanced").lower()
    kind = str(pm.get("kind") or "").lower()
    basis = str(pm.get("gating_basis") or pm.get("basis") or "point")
    metrics_policy = {} if tiny_relax else _resolved_metrics_policy(evaluation_report)
    if kind == "accuracy":
        delta = _coerce_finite_float(pm.get("delta_vs_baseline_pp"))
        measured = f"{delta:+.2f} pp" if delta is not None else "N/A"
        acc_policy = _mapping(metrics_policy.get("accuracy"))
        delta_min = _coerce_finite_float(acc_policy.get("delta_min_pp"))
        if delta_min is None:
            delta_min = {
                "conservative": -0.5,
                "balanced": -1.0,
                "aggressive": -2.0,
            }.get(
                tier,
                -1.0,
            )
        hysteresis = _coerce_finite_float(acc_policy.get("hysteresis_delta_pp")) or 0.0
        threshold = f">= {delta_min:+.2f} pp"
        if hysteresis > 0.0:
            threshold += f" (+{hysteresis:.2f} pp hysteresis)"
        return measured, threshold.replace(">=", "≥"), basis

    value = _coerce_finite_float(pm.get("ratio_vs_baseline"))
    measured = f"{value:.3f}x" if value is not None else "N/A"
    acceptance = _pm_acceptance_range(evaluation_report)
    lo = acceptance.get("min")
    hi = acceptance.get("max")
    if hi is None:
        pm_policy = _mapping(metrics_policy.get("pm_ratio"))
        hi = _coerce_finite_float(pm_policy.get("ratio_limit_base"))
    if hi is None:
        hi = {"conservative": 1.05, "balanced": 1.10, "aggressive": 1.20}.get(
            tier,
            1.10,
        )
    if lo is not None:
        threshold = f"{lo:.2f}x to {hi:.2f}x"
    else:
        threshold = f"≤ {hi:.2f}x"
    return measured, threshold, basis


def _primary_metric_drift_measured_and_threshold(
    evaluation_report: dict[str, Any],
) -> tuple[str, str, str]:
    pm = _mapping(evaluation_report.get("primary_metric"))
    kind = str(pm.get("kind") or "").lower()
    if kind == "accuracy":
        preview = _coerce_finite_float(pm.get("preview"))
        final = _coerce_finite_float(pm.get("final"))
        if preview is None or final is None:
            measured = "N/A"
        else:
            measured = f"{(final - preview) * 100.0:+.2f} pp"
        acc_policy = _mapping(
            _resolved_metrics_policy(evaluation_report).get("accuracy")
        )
        limit = _coerce_finite_float(acc_policy.get("preview_final_delta_pp_max"))
        if limit is None:
            limit = _coerce_finite_float(acc_policy.get("hysteresis_delta_pp"))
        if limit is None:
            limit = 0.1
        return measured, f"≤ ±{limit * 100.0:.2f} pp", "absolute-delta"

    try:
        pv = _finite_float_or_nan(pm.get("preview"))
        fv = _finite_float_or_nan(pm.get("final"))
        drift = (
            fv / pv
            if (math.isfinite(pv) and pv > 0 and math.isfinite(fv))
            else float("nan")
        )
    except _PARSE_EXCEPTIONS:
        drift = float("nan")
    measured = f"{drift:.3f}x" if math.isfinite(drift) else "N/A"
    drift_min = 0.95
    drift_max = 1.05
    try:
        drift_band = pm.get("drift_band") if isinstance(pm, dict) else None
        if isinstance(drift_band, dict):
            lo = _coerce_finite_float(drift_band.get("min"))
            hi = _coerce_finite_float(drift_band.get("max"))
            if lo is not None and hi is not None and 0 < lo < hi:
                drift_min, drift_max = lo, hi
        elif isinstance(drift_band, list | tuple) and len(drift_band) == 2:
            lo = _coerce_finite_float(drift_band[0])
            hi = _coerce_finite_float(drift_band[1])
            if lo is not None and hi is not None and 0 < lo < hi:
                drift_min, drift_max = lo, hi
    except _PARSE_EXCEPTIONS:
        pass
    return measured, f"{drift_min:.2f}–{drift_max:.2f}x", "point"


def derive_report_manifest_evidence_level(
    summary: ReportManifestSummary,
    *,
    has_guard_evidence: bool,
) -> str:
    """Classify evaluation-bundle audit strength from shipped sidecars."""
    if has_guard_evidence and summary.gates_total > 0:
        return "high"
    if summary.gates_total > 0:
        return "medium"
    return "low"


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
        value = validation.get(key)
        ok = value if isinstance(value, bool) else None
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

    validation_raw = evaluation_report.get("validation")
    validation = validation_raw if isinstance(validation_raw, dict) else {}
    pm_value = validation.get("primary_metric_acceptable")
    pm_ok: bool | None = pm_value if isinstance(pm_value, bool) else None
    measured, threshold, pm_basis = _primary_metric_measured_and_threshold(
        evaluation_report
    )
    threshold = f"{threshold} ({pm_basis})"

    if isinstance(pm_ok, bool):
        pm_status = f"{'✅' if pm_ok else '❌'} {measured}"
    else:
        pm_status = f"ℹ️ {measured}"

    drift_value = validation.get("preview_final_drift_acceptable")
    drift_ok: bool | None = drift_value if isinstance(drift_value, bool) else None
    drift_val, drift_threshold, _drift_basis = (
        _primary_metric_drift_measured_and_threshold(evaluation_report)
    )
    if isinstance(drift_ok, bool):
        drift_status = f"{'✅' if drift_ok else '❌'} {drift_val}"
    else:
        drift_status = f"ℹ️ {drift_val}"

    rows: list[SafetyDashboardRow] = [
        SafetyDashboardRow("Primary Metric", pm_status, threshold),
        SafetyDashboardRow("Drift", drift_status, drift_threshold),
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

    metric_impact_ctx = evaluation_report.get("guard_metric_impact", {}) or {}
    metric_impact_evaluated = (
        bool(metric_impact_ctx.get("evaluated"))
        if isinstance(metric_impact_ctx, dict)
        else False
    )
    if metric_impact_evaluated:
        metric_impact_measured, limit_str = _guard_metric_impact_display(
            metric_impact_ctx
        )
        rows.append(
            SafetyDashboardRow(
                "Guard Metric Impact",
                (
                    f"{'✅' if validation.get('guard_metric_impact_acceptable') is True else '❌'} {metric_impact_measured}"
                    if isinstance(validation, dict)
                    else f"ℹ️ {metric_impact_measured}"
                ),
                limit_str,
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
    validation_raw = evaluation_report.get("validation")
    validation = validation_raw if isinstance(validation_raw, dict) else {}

    rows: list[QualityGateRow] = []
    if has_pm:
        pm_kind = str(pm_block.get("kind", "")).lower()
        measured, threshold, gating_basis = _primary_metric_measured_and_threshold(
            evaluation_report
        )
        pm_value = validation.get("primary_metric_acceptable")
        status = (
            "✅ PASS"
            if pm_value is True
            else "❌ FAIL"
            if pm_value is False
            else "ℹ️ NOT EVALUATED"
        )
        if pm_kind == "accuracy":
            description = "Δ accuracy vs baseline"
        else:
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

        drift_value = validation.get("preview_final_drift_acceptable")
        measured, threshold, drift_basis = _primary_metric_drift_measured_and_threshold(
            evaluation_report
        )
        rows.append(
            QualityGateRow(
                label="Preview Final Drift Acceptable",
                status=(
                    "✅ PASS"
                    if drift_value is True
                    else "❌ FAIL"
                    if drift_value is False
                    else "ℹ️ NOT EVALUATED"
                ),
                measured=measured,
                threshold=threshold,
                basis=drift_basis,
                description=(
                    "Preview/final accuracy delta"
                    if pm_kind == "accuracy"
                    else "Final/Preview ratio stability"
                ),
            )
        )

        guard_metric_impact = evaluation_report.get("guard_metric_impact", {}) or {}
        evaluated = bool(guard_metric_impact.get("evaluated"))
        if evaluated:
            metric_impact_ok = validation.get("guard_metric_impact_acceptable") is True
            measured, degradation_limit = _guard_metric_impact_display(
                guard_metric_impact
            )
            rows.append(
                QualityGateRow(
                    label="Guard Metric Impact Acceptable",
                    status="✅ PASS" if metric_impact_ok else "❌ FAIL",
                    measured=measured,
                    threshold=degradation_limit,
                    basis=str(guard_metric_impact.get("degradation_basis") or ""),
                    description="Guarded-vs-bare primary metric degradation",
                )
            )

        pm_tail = evaluation_report.get("primary_metric_tail", {}) or {}
        if isinstance(pm_tail, dict) and pm_tail:
            evaluated = bool(pm_tail.get("evaluated", False))
            mode = str(pm_tail.get("mode", "warn") or "warn").strip().lower()
            passed = pm_tail.get("passed") is True
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
            except _PARSE_EXCEPTIONS:
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
        hysteresis_applied=validation.get("hysteresis_applied") is True,
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
    "load_console_labels",
    "compute_console_validation_block",
    "compute_report_hash",
    "build_console_summary_pack",
    "derive_report_manifest_evidence_level",
    "build_quality_gates_summary",
    "build_safety_dashboard_summary",
    "build_report_manifest_summary",
]
